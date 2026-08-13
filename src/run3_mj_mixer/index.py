"""index.py - the hemisphere-index schema, and shard read/write.

The index is the whole point of the rework: matching needs a handful of numbers
per hemisphere, so those numbers live in a compact table and everything else
stays in the slimmed files, fetched on demand in stage 3a.

Deliberately absent from this schema:

* **jet four-vectors and any analysis payload.** Inlining them would make the
  index a schema-coupled copy of the slimmed files, and every new systematic
  would force a full index rebuild plus a re-run of the matching.
* **``px/py/pz/energy/pt_par/pt_perp``.** Exact functions of what is stored -
  see ``hemisphere.hemisphere_summary``.
* **any weight.** The normalization denominator is slice-wide, so a per-file
  shard job cannot compute it. Weights are assigned in stage 1b, which is also
  what lets the same shards serve ``--mode mc`` and ``--mode data``.

``jet_mask`` *is* here, and is not payload: it defines which jets are the
hemisphere, and recording it in the one place the split is known exactly is what
makes the old float-comparison re-derivation impossible.
"""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np

SCHEMA_VERSION = 1
SHARD_TREE = "hindex"

#: One row per usable hemisphere. Order is the on-disk branch order.
SHARD_COLUMNS: dict[str, np.dtype] = {
    # identity / provenance
    "file_hash":       np.dtype(np.uint64),
    "entry":           np.dtype(np.int64),
    "side":            np.dtype(np.int8),
    "hemi_slot":       np.dtype(np.uint8),
    "run":             np.dtype(np.uint32),
    "luminosityBlock": np.dtype(np.uint32),
    "event":           np.dtype(np.uint64),
    # matching
    "pt":              np.dtype(np.float32),
    "eta":             np.dtype(np.float32),
    "partner_eta":     np.dtype(np.float32),
    "thrust_phi":      np.dtype(np.float32),
    "n_jets":          np.dtype(np.uint8),
    "jet_mask":        np.dtype(np.uint16),
    # hemisphere summary
    "phi":             np.dtype(np.float32),
    "mass":            np.dtype(np.float32),
    "thrust":          np.dtype(np.float32),
}

#: Written last, so stage 1b can reject a truncated shard.
COMPLETE_MARKER = "complete"

#: Bins of the per-shard cutflow.
SHARD_CUTFLOW_BINS = [
    "events read",
    "events kept",
    "hemispheres total",
    "hemispheres written",
]

_REDIRECTOR_RE = re.compile(r"^root://[^/]+/+")


def row_nbytes() -> int:
    """Bytes per shard row, ignoring compression."""
    return sum(dt.itemsize for dt in SHARD_COLUMNS.values())


def normalize_path(path: str) -> str:
    """Canonical form of an input path, for hashing and for file-table keys.

    Strips an XRootD redirector so the same logical file hashes identically
    whether it was named as a bare ``/store/...`` LFN or a full URL, and
    collapses duplicate slashes (the double-slash convention for XRootD paths
    would otherwise produce two hashes for one file).
    """
    p = _REDIRECTOR_RE.sub("/", str(path))
    return re.sub(r"/{2,}", "/", p)


def file_hash(path: str) -> int:
    """A stable 64-bit id for an input file, derived from its normalized path.

    A shard must be producible without consulting any global registry - a
    condor job knows only its own input - so the shard carries this hash and
    stage 1b resolves it to a dense ``file_id``. Collision probability over a
    few thousand paths is ~1e-11.
    """
    digest = hashlib.blake2b(
        normalize_path(path).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def empty_columns(n: int = 0) -> dict[str, np.ndarray]:
    """Zero-filled columns, for the cutflow-only path."""
    return {name: np.zeros(n, dtype=dt) for name, dt in SHARD_COLUMNS.items()}


def check_columns(cols: dict[str, np.ndarray]) -> int:
    """Validate a column dict against the schema and return its row count."""
    missing = set(SHARD_COLUMNS) - set(cols)
    if missing:
        raise ValueError(f"index shard is missing columns: {sorted(missing)}")
    extra = set(cols) - set(SHARD_COLUMNS)
    if extra:
        raise ValueError(f"index shard has unexpected columns: {sorted(extra)}")
    lengths = {len(v) for v in cols.values()}
    if len(lengths) != 1:
        raise ValueError(
            "index shard columns have mismatched lengths: "
            + repr({k: len(v) for k, v in cols.items()})
        )
    return lengths.pop()


def cast_columns(cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Columns cast to their schema dtypes, preserving float32 bit patterns.

    Values copied out of a ROOT file are already float32; casting float32 ->
    float32 is a no-op, so a round trip through the shard is lossless.
    """
    check_columns(cols)
    return {name: np.asarray(cols[name], dtype=dt)
            for name, dt in SHARD_COLUMNS.items()}


def _label_hist(label, value=1.0):
    """A one-bin StrCategory histogram carrying a string.

    The repo's way of putting a string in a ROOT file (see the old
    ``_write_version_and_meta``); ``.view()`` assignment rather than label
    indexing, which boost-histogram does not support for setting.
    """
    import boost_histogram as bh

    h = bh.Histogram(bh.axis.StrCategory([label]), storage=bh.storage.Double())
    h.view()[0] = float(value)
    return h


def _label_of(hist):
    """The single category label of a ``_label_hist``."""
    labels = list(hist.axes[0])
    return str(labels[0]) if labels else None


def write_shard(path, cols, *, dataset, source_path, version,
                cutflow=None, schema_version=SCHEMA_VERSION):
    """Write one index shard.

    The ``complete`` marker goes last, so a shard truncated by a dying job is
    detectable rather than silently short.
    """
    import boost_histogram as bh
    import uproot

    cols = cast_columns(cols)
    n = check_columns(cols)

    out_dir = os.path.dirname(str(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with uproot.recreate(path) as f:
        if n > 0:
            f.mktree(SHARD_TREE, {k: v.dtype for k, v in cols.items()})
            f[SHARD_TREE].extend(cols)
        else:
            # uproot's extend chokes on zero-entry chunks, and an empty tree is
            # not writable; the row count lives in the cutflow instead. A shard
            # with no tree and a complete marker means "this file legitimately
            # produced no hemispheres".
            pass

        cf = bh.Histogram(
            bh.axis.StrCategory(SHARD_CUTFLOW_BINS), storage=bh.storage.Double()
        )
        counts = cutflow or {}
        cf.view()[:] = [float(counts.get(label, 0.0)) for label in SHARD_CUTFLOW_BINS]
        f["index_cutflow"] = cf

        f["dataset"] = _label_hist(dataset or "UNKNOWN", float(n))
        f["source_path"] = _label_hist(str(source_path))
        f["schema_version"] = _label_hist(str(schema_version))
        f["version"] = _label_hist(str(version))
        f[COMPLETE_MARKER] = _label_hist("ok")


class IncompleteShard(RuntimeError):
    """A shard without its ``complete`` marker - truncated, or a dead job."""


def read_shard_header(path, *, require_complete=True):
    """Just the small objects: no TTree columns read at all.

    Stage 1b needs two things from every shard before it can allocate the merged
    columns - which file it came from, and how many rows it has - and both are in
    the metadata. Reading the full columns to find that out doubles the xrootd
    traffic over thousands of shards for nothing.

    Returns ``(n_rows, meta)``. ``n_rows`` is the tree's ``num_entries``, and
    the merge re-checks it against the real column length when it reads them.
    """
    import uproot

    with uproot.open(path) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        if require_complete and COMPLETE_MARKER not in keys:
            raise IncompleteShard(
                f"{path} has no '{COMPLETE_MARKER}' marker - it was truncated or "
                "the job that wrote it died. Re-run that job; including it would "
                "silently shorten the index."
            )

        def _label(name, cast=str):
            if name not in keys:
                return None
            lbl = _label_of(f[name].to_boost())
            return None if lbl is None else cast(lbl)

        meta = {
            "source_path": _label("source_path"),
            "schema_version": _label("schema_version", int),
            "version": _label("version"),
            "dataset": _label("dataset"),
            "cutflow": {},
        }
        if "index_cutflow" in keys:
            vals = f["index_cutflow"].values()
            meta["cutflow"] = {
                label: float(v) for label, v in zip(SHARD_CUTFLOW_BINS, vals)
            }
        # num_entries, NOT the cutflow: it is the authoritative row count and is
        # just as cheap (tree metadata, no columns). Trusting the cutflow means a
        # shard with a tree but a missing or wrong cutflow reports zero rows and
        # gets silently dropped from the merge.
        n_rows = int(f[SHARD_TREE].num_entries) if SHARD_TREE in keys else 0

        # The shard also carries file_hash per row. Prefer source_path (free),
        # but fall back to one value of that column so a shard without the
        # metadata objects is still placeable - it is one branch, one entry.
        if meta["source_path"] is None and n_rows:
            meta["file_hash"] = int(
                f[SHARD_TREE]["file_hash"].array(entry_stop=1, library="np")[0])
        else:
            meta["file_hash"] = (file_hash(meta["source_path"])
                                 if meta["source_path"] else None)
        return n_rows, meta


def read_shard(path, *, require_complete=True):
    """Read one index shard.

    Returns ``(columns, meta)``. ``columns`` is a dict of numpy arrays in schema
    dtypes, empty (length 0) for a legitimately empty shard.
    """
    import uproot

    with uproot.open(path) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        if require_complete and COMPLETE_MARKER not in keys:
            raise IncompleteShard(
                f"{path} has no '{COMPLETE_MARKER}' marker - it was truncated or "
                "the job that wrote it died. Re-run that job; including it would "
                "silently shorten the index."
            )

        def _label(name, cast=str):
            if name not in keys:
                return None
            lbl = _label_of(f[name].to_boost())
            return None if lbl is None else cast(lbl)

        meta = {
            "source_path": _label("source_path"),
            "schema_version": _label("schema_version", int),
            "version": _label("version"),
            "dataset": _label("dataset"),
            "cutflow": {},
        }
        if "index_cutflow" in keys:
            vals = f["index_cutflow"].values()
            meta["cutflow"] = {
                label: float(v) for label, v in zip(SHARD_CUTFLOW_BINS, vals)
            }

        if SHARD_TREE not in keys:
            return empty_columns(0), meta

        tree = f[SHARD_TREE]
        cols = {name: np.asarray(tree[name].array(library="np"), dtype=dt)
                for name, dt in SHARD_COLUMNS.items()}
        check_columns(cols)
        return cols, meta

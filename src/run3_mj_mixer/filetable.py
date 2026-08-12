"""filetable.py - the file table, and the SOLE owner of dataset inference.

Two jobs:

1. **Normalize paths out of the index.** A path string on every hemisphere row
   would cost more than the entire rest of the index, so rows carry an int
   ``file_id`` and the mapping lives here.

2. **Assign weights.** ``w_rel = xs_pb / n_original_sum``, where the denominator
   is summed over *every* file of the slice. The old mixer divided by one
   file's ``cutflow[0]``, inflating every weight by roughly the slice's file
   count and non-uniformly across slices - so it distorted shape, not just
   scale. That denominator is slice-wide, which is precisely why stage 1a cannot
   compute it and stage 1b exists.

Dataset inference used to be copied into three places that had already drifted:
``mix.py::infer_dataset`` and the analyzer's copy returned ``None`` on no match,
while ``submit_mixer.py::slice_of`` fell back to a regex and then to a sentinel.
So a file's output directory and its weight were decided by two functions that
disagreed about the failure case. Here there is one rule and two explicit
policies:

* ``infer_dataset``    - the rule. ``None`` means no match. Weighting uses this
                         and fails loudly in mc mode.
* ``slice_or_unknown`` - never fails. Output-directory routing uses this, so an
                         unrecognized filename cannot block a job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from .index import file_hash, normalize_path

SCHEMA = "run3-mj-file-table"
SCHEMA_VERSION = 1

UNKNOWN_SLICE = "unknown_slice"

# Matches the HT-slice fragment of a QCD dataset name, e.g. pulls 'HT-400to600'
# out of 'slimmed_QCD-4Jets_HT-400to600_TuneCP5_...'. The 'to<hi>' is optional so
# the open-ended top bin ('HT-2000') is labelled too.
_HT_SLICE_RE = re.compile(r"HT-\d+(?:to(?:\d+|Inf))?")

_XS_FILENAME = "mj_samples_xs.json"
_AUX_DIRNAME = "run3-mj-pass-the-aux"

MODES = ("mc", "data")


# ---------------------------------------------------------------------------
# Locating and reading the cross-section JSON
# ---------------------------------------------------------------------------

def locate_xs_json(explicit=None):
    """Locate mj_samples_xs.json. Search order:

    1. ``explicit`` (the --xs-json argument),
    2. ``$RUN3_MJ_XS_JSON``,
    3. ``./mj_samples_xs.json`` (a copy transferred into a condor job dir),
    4. a ``run3-mj-pass-the-aux/`` sibling found by walking up from this module
       or the CWD - i.e. the aux repo assumed to live in the same parent dir.

    Returns a path string, or None if nothing was found.
    """
    cands = []
    if explicit:
        cands.append(Path(explicit))
    env = os.environ.get("RUN3_MJ_XS_JSON")
    if env:
        cands.append(Path(env))
    cands.append(Path.cwd() / _XS_FILENAME)
    for base in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in base.parents:
            cands.append(parent / _AUX_DIRNAME / _XS_FILENAME)
    for c in cands:
        if c.is_file():
            return str(c)
    return None


def load_xs(xs_json):
    """``{dataset: {xs_pb: ...}}`` from the cross-section JSON."""
    with open(xs_json) as f:
        return json.load(f)


def xs_keys_of(xs):
    """Dataset names usable as slice labels."""
    return list(xs.keys()) if xs else []


# ---------------------------------------------------------------------------
# Dataset inference - ONE rule, two policies
# ---------------------------------------------------------------------------

def infer_dataset(path, xs_keys):
    """The xs-JSON key that best matches a file path, or None.

    The rule: the longest key that is a substring of the basename. Slimmed
    filenames embed the dataset name (the slimmer's ``--output-tag``), so this
    recovers the HT slice.

    Returns ``None`` when nothing matches. Callers that need a weight must treat
    that as an error; callers that need a directory name should use
    ``slice_or_unknown`` instead.
    """
    base = os.path.basename(normalize_path(path))
    matches = [k for k in xs_keys if k in base]
    return max(matches, key=len) if matches else None


def slice_or_unknown(path, xs_keys):
    """A slice label for a file, never failing.

    ``infer_dataset`` first, then the HT-slice regex, then ``UNKNOWN_SLICE``.
    Used for output-directory routing, where an unrecognized name must not block
    a job. Agrees with ``infer_dataset`` on every path that does match - see
    ``tests/test_filetable.py``.
    """
    hit = infer_dataset(path, xs_keys)
    if hit is not None:
        return hit
    found = _HT_SLICE_RE.search(os.path.basename(normalize_path(path)))
    return found.group(0) if found else UNKNOWN_SLICE


# ---------------------------------------------------------------------------
# n_original: the slice-wide denominator
# ---------------------------------------------------------------------------

def read_cutflow0(path):
    """A file's pre-selection **event** count, ``cutflow[0]``.

    Not the file count - the number of generated events the slimmer saw before
    any cut. Summed over a slice this is the xsec-normalization denominator.
    """
    import uproot

    with uproot.open(path) as f:
        if "cutflow" not in f:
            raise ValueError(
                f"{path} has no 'cutflow' histogram (partial or truncated "
                "slimmer output?) - re-run that file; silently including it "
                "would undercount n_original and over-weight the whole slice."
            )
        return float(f["cutflow"].values()[0])


def scan_n_original(paths, xs_keys, *, workers=1, progress=False):
    """Sum ``cutflow[0]`` per dataset over ``paths``.

    Ported from the analyzer's ``make_hemisphere_histograms.scan_root_inputs``,
    keeping its fail-loudly behaviour: an un-inferable dataset or a missing
    cutflow raises rather than being skipped, because skipping either biases the
    weights silently.

    Returns ``(datasets, n_original)``: a path -> dataset map, and a
    dataset -> summed-events map. Cutflow-only files (no ``events`` tree) are
    included - they contribute to the denominator even though they contribute no
    hemispheres.
    """
    datasets = {}
    for p in paths:
        ds = infer_dataset(p, xs_keys)
        if ds is None:
            raise ValueError(
                f"Cannot infer the HT slice of {p} from its filename (no "
                "cross-section key matches). Use --mode data, or fix the name."
            )
        datasets[p] = ds

    iterator = paths
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=workers)
        values = list(pool.map(read_cutflow0, paths))
        pool.shutdown()
    else:
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(paths, unit="file", desc="cutflow scan")
            except ImportError:
                pass
        values = [read_cutflow0(p) for p in iterator]

    n_original = {}
    for p, n0 in zip(paths, values):
        n_original[datasets[p]] = n_original.get(datasets[p], 0.0) + n0
    return datasets, n_original


# ---------------------------------------------------------------------------
# Building the table
# ---------------------------------------------------------------------------

def build_file_table(paths, *, mode="mc", xs_json=None, workers=1,
                     progress=False, extend=None):
    """Build (or extend) the file table.

    ``file_id`` comes from a lexicographic sort of normalized paths, so it is a
    function of the input set alone and reproducible regardless of the order
    files were discovered in. ``extend`` preserves existing ids and appends new
    files after them - append-only, so ``hemi_id`` stays stable across
    reprocessings.

    In ``--mode data`` no cross sections and no cutflow scan are involved:
    ``w_rel`` is 1 for every file. ``dataset`` is still recorded for provenance,
    via ``slice_or_unknown`` so an unmatched data filename is not an error.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    norm = [normalize_path(p) for p in paths]
    if len(set(norm)) != len(norm):
        dupes = sorted({p for p in norm if norm.count(p) > 1})
        raise ValueError(f"duplicate input paths after normalization: {dupes}")

    xs, xs_path, keys = None, None, []
    if mode == "mc":
        xs_path = locate_xs_json(xs_json)
        if xs_path is None:
            raise ValueError(
                "No mj_samples_xs.json found and --mode mc needs one. Pass "
                "--xs-json, set $RUN3_MJ_XS_JSON, or use --mode data."
            )
        xs = load_xs(xs_path)
        keys = xs_keys_of(xs)

    # existing ids first, then new paths in sorted order
    files, next_id = [], 0
    seen = {}
    if extend:
        for rec in extend["files"]:
            seen[rec["path"]] = rec
            files.append(dict(rec))
            next_id = max(next_id, int(rec["file_id"]) + 1)
    new = sorted(p for p in norm if p not in seen)

    if mode == "mc":
        datasets, n_original = scan_n_original(
            norm, keys, workers=workers, progress=progress)
    else:
        datasets = {p: slice_or_unknown(p, keys) for p in norm}
        n_original = {}

    for p in new:
        files.append({
            "file_id": next_id,
            "path": p,
            "file_hash": file_hash(p),
            "dataset": datasets[p],
            "n_original_file": (read_cutflow0(p) if mode == "mc" else None),
        })
        next_id += 1

    if mode == "mc":
        per_dataset = {}
        for ds, n0 in sorted(n_original.items()):
            xs_pb = xs.get(ds, {}).get("xs_pb")
            if xs_pb is None:
                raise ValueError(
                    f"dataset {ds!r} has no 'xs_pb' in {xs_path}."
                )
            if n0 <= 0:
                raise ValueError(
                    f"dataset {ds!r} has n_original_sum={n0}; cannot weight it."
                )
            per_dataset[ds] = {
                "xs_pb": float(xs_pb),
                "n_original_sum": float(n0),
                "w_rel": float(xs_pb) / float(n0),
                "n_files": sum(1 for f in files if f["dataset"] == ds),
            }
    else:
        per_dataset = {
            ds: {"xs_pb": None, "n_original_sum": None, "w_rel": 1.0,
                 "n_files": sum(1 for f in files if f["dataset"] == ds)}
            for ds in sorted(set(datasets.values()))
        }

    table = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "n_files": len(files),
        "xs_json": xs_path,
        "xs_json_sha256": (_sha256(xs_path) if xs_path else None),
        "datasets": per_dataset,
        "files": files,
    }
    return table


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def save_file_table(table, path):
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(table, f, indent=2)
        f.write("\n")


def load_file_table(path):
    with open(path) as f:
        table = json.load(f)
    if table.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} (got {table.get('schema')!r})")
    return table


def hash_to_id(table):
    """``file_hash -> file_id``, the map stage 1b resolves shards through."""
    out = {}
    for rec in table["files"]:
        h = int(rec["file_hash"])
        if h in out:
            raise ValueError(
                f"file_hash collision between file_id {out[h]} and "
                f"{rec['file_id']} - refusing to guess."
            )
        out[h] = int(rec["file_id"])
    return out


def w_rel_by_id(table):
    """``file_id -> w_rel`` as a dense float64 array indexed by file_id."""
    n = max((int(r["file_id"]) for r in table["files"]), default=-1) + 1
    out = np.ones(n, dtype=np.float64)
    for rec in table["files"]:
        out[int(rec["file_id"])] = float(
            table["datasets"][rec["dataset"]]["w_rel"])
    return out


def dataset_by_id(table):
    """``file_id -> dataset name``."""
    n = max((int(r["file_id"]) for r in table["files"]), default=-1) + 1
    out = [None] * n
    for rec in table["files"]:
        out[int(rec["file_id"])] = rec["dataset"]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_paths(source):
    """Input paths from a fileset JSON, a directory of them, a .txt, or a path.

    Accepts what ``make_eos_filelists.py`` writes
    (``{dataset: {files: {url: tree}}}``) as well as plain coffea filesets and
    bare lists, so its output feeds straight into both the file table and the
    stage-1a submitter without a manual flatten step. ``submit_index.py`` uses
    this same function, so the two cannot disagree about what an input list is.
    """
    import glob

    if os.path.isdir(source):
        out = []
        for p in sorted(glob.glob(os.path.join(source, "*.json"))):
            out += load_paths(p)
        return out
    if str(source).endswith(".txt"):
        with open(source) as f:
            return [ln.strip() for ln in f
                    if ln.strip() and not ln.strip().startswith("#")]
    if str(source).endswith(".json"):
        with open(source) as f:
            blob = json.load(f)
        out = []
        if isinstance(blob, dict):
            for key, val in blob.items():
                if key == "unused":          # the retired grouper's leftovers
                    continue
                if isinstance(val, dict) and "files" in val:
                    out += list(val["files"])
                elif isinstance(val, dict):
                    out += list(val)
                elif isinstance(val, list):
                    out += list(val)
        elif isinstance(blob, list):
            out += blob
        return out
    return [str(source)]


def _read_paths(args):
    paths = list(args.inputs)
    for src in (args.from_file or []):
        paths += load_paths(src)
    if not paths:
        sys.exit("No input files given (pass paths, --from-file, or a "
                 "fileset JSON / directory of them).")
    return paths


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="run3-mj-file-table",
        description="Build the file table: file_id, dataset, and w_rel.",
    )
    p.add_argument("inputs", nargs="*", help="slimmed ROOT files")
    p.add_argument("--from-file", action="append", metavar="SRC",
                   help="a .txt of paths, a fileset JSON, or a directory of "
                        "fileset JSONs (repeatable)")
    p.add_argument("-o", "--output", default="file_table.json")
    p.add_argument("--mode", choices=MODES, default="mc",
                   help="mc: weight from cross sections. data: w_rel = 1, and "
                        "no xs JSON or cutflow scan is read at all.")
    p.add_argument("--xs-json", default=None)
    p.add_argument("--extend", default=None, metavar="TABLE",
                   help="an existing table to extend append-only, preserving "
                        "its file_ids so hemi_ids stay stable")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--progress", action="store_true")
    args = p.parse_args(argv)

    paths = _read_paths(args)
    extend = load_file_table(args.extend) if args.extend else None
    try:
        table = build_file_table(
            paths, mode=args.mode, xs_json=args.xs_json,
            workers=args.workers, progress=args.progress, extend=extend,
        )
    except ValueError as exc:
        sys.exit(f"file table: {exc}")

    save_file_table(table, args.output)
    print(f"mode:    {table['mode']}")
    print(f"files:   {table['n_files']}")
    print(f"xs json: {table['xs_json']}")
    print(f"{'dataset':<62} {'n_original_sum':>15} {'w_rel':>12}")
    for ds, d in sorted(table["datasets"].items()):
        n0 = "-" if d["n_original_sum"] is None else f"{d['n_original_sum']:,.0f}"
        print(f"{ds:<62} {n0:>15} {d['w_rel']:>12.4g}")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

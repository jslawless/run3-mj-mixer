"""index_build.py - stage 1b: merge index shards into one global index.

Reads every shard, resolves each shard's ``file_hash`` to a dense ``file_id``
through the file table, and writes the result as one memory-mappable ``.npy``
per column plus a manifest.

**Canonical order is ``(file_id, entry, hemi_slot)``.** Each shard is already
sorted by ``(entry, hemi_slot)`` (stage 1a emits events in order, both slots per
event), so merging shards in ``file_id`` order is a pure concatenation - O(N),
no global sort of 10^7 rows. The order is therefore a function of the *file
table* alone, which is what makes the merge reproducible no matter what order
shards were discovered in. ``hemi_id`` is just the row number, so it costs
nothing to store, and ``hemi_id_offsets`` gives every file a contiguous range -
the property stage 3a relies on to turn a random-access join into a sequential
one.

Two columns are deliberately **not** stored:

* ``dir_phi`` - float64-from-float32; see ``hemisphere.directed_phi_array``.
* ``w_rel``   - a per-*dataset* constant, so it lives once in the file table and
  is looked up by ``file_id``. Storing it per row would be 10^7 copies of a
  handful of numbers, and two places to keep in agreement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

from . import filetable as ft
from . import index as idx
from .filetable import _progress

SCHEMA = "run3-mj-hemisphere-index"
SCHEMA_VERSION = 1

MANIFEST_NAME = "index_manifest.json"
OFFSETS_NAME = "hemi_id_offsets.npy"

#: Columns carried through from the shards, in canonical order.
CARRIED = {
    "file_id":         np.dtype(np.int32),
    "entry":           np.dtype(np.int32),
    "side":            np.dtype(np.int8),
    "hemi_slot":       np.dtype(np.uint8),
    "n_jets":          np.dtype(np.uint8),
    "jet_mask":        np.dtype(np.uint16),
    "pt":              np.dtype(np.float32),
    "eta":             np.dtype(np.float32),
    "phi":             np.dtype(np.float32),
    "mass":            np.dtype(np.float32),
    "partner_eta":     np.dtype(np.float32),
    "thrust_phi":      np.dtype(np.float32),
    "thrust":          np.dtype(np.float32),
    "run":             np.dtype(np.uint32),
    "luminosityBlock": np.dtype(np.uint32),
    "event":           np.dtype(np.uint64),
}

#: Derived and stored, because the matcher wants it on every run.
DERIVED = {
    "order_by_pt": np.dtype(np.int64),
}

#: Identity columns only - the index_id hash covers these, so adding a derived
#: column does not invalidate every downstream file that quotes the id.
IDENTITY_COLUMNS = ("file_id", "entry", "hemi_slot")


def _xs_keys(table):
    """Dataset names present in the table, for labelling skipped shards."""
    return list(table.get("datasets", {}))


def _npy(out_dir, name):
    return os.path.join(str(out_dir), f"{name}.npy")


def scan_shards(paths, table, *, require_complete=True, skip_unlisted=False,
                progress=False):
    """First pass: work out where each shard lands, from its METADATA only.

    Reads no TTree columns - the file it came from and its row count are both in
    the small objects, so a full read here would double the xrootd traffic over
    thousands of shards. Checks that need the columns (internal sort order, and
    that the row count is honest) happen in the second pass, where they are read
    anyway.

    Returns ``(plan, skipped)``; ``plan`` is sorted by ``file_id``, which is the
    order the second pass concatenates in.
    """
    h2id = ft.hash_to_id(table)
    plan, seen_ids, skipped = [], {}, []
    for p in _progress(paths, total=len(paths), enabled=progress,
                       desc="scanning shards", unit="shard"):
        n, meta = idx.read_shard_header(p, require_complete=require_complete)
        if n == 0:
            # A legitimately empty slice. It contributes no rows but its file is
            # still in the table and still counted in the denominator.
            fid = None
            if meta.get("file_hash") is not None:
                fid = h2id.get(meta["file_hash"])
            if fid is None and skip_unlisted:
                # Also unlisted, just with nothing in it - count it as skipped so
                # the reported totals add up.
                skipped.append((str(p), meta["source_path"]))
                continue
            plan.append({"path": str(p), "file_id": fid, "n_rows": 0})
            continue

        fh = meta.get("file_hash")
        if fh is None:
            raise ValueError(
                f"{p} records neither a source_path nor a file_hash, so it "
                "cannot be matched to the file table. Re-run stage 1a for it."
            )
        if fh not in h2id:
            # Two very different situations, so do not guess between them:
            # the shard is for a file deliberately left out of the sample
            # (--skip-unlisted), or the table is genuinely incomplete (a bug).
            if skip_unlisted:
                skipped.append((str(p), meta["source_path"]))
                continue
            raise ValueError(
                f"{p}\n  has file_hash {fh:#018x}\n  source {meta['source_path']!r}\n"
                "  which is not in the file table.\n"
                "  If you indexed more files than the sample should contain - e.g. "
                "all HT slices but a table over only some - pass --skip-unlisted "
                "to build over the table's files and report the rest.\n"
                "  If the shard should be in the sample, the table is incomplete: "
                "rebuild it over the same input set."
            )
        fid = h2id[fh]
        if fid in seen_ids:
            raise ValueError(
                f"file_id {fid} is covered by two shards: {seen_ids[fid]} and "
                f"{p}. That usually means the same input was indexed twice under "
                "different output tags - drop one."
            )
        seen_ids[fid] = str(p)

        plan.append({"path": str(p), "file_id": fid, "n_rows": n})

    plan.sort(key=lambda r: (r["file_id"] is None, r["file_id"]))
    return plan, skipped


def build(shard_paths, table, out_dir, *, require_complete=True,
          progress=False, file_table_path=None, skip_unlisted=False):
    """Merge shards into ``out_dir``, writing the manifest. Returns it too.

    The manifest is written here rather than by the CLI so that an index built
    through the library API is complete and self-describing - anything less and
    ``HemisphereIndex`` cannot open its own output.
    """
    os.makedirs(str(out_dir), exist_ok=True)
    plan, skipped = scan_shards(shard_paths, table,
                                require_complete=require_complete,
                                skip_unlisted=skip_unlisted,
                                progress=progress)
    if skipped:
        # Loud, and grouped, so "I meant to drop those" and "my table is short"
        # look different at a glance.
        from collections import Counter
        by_ds = Counter(ft.slice_or_unknown(src or path, _xs_keys(table))
                        for path, src in skipped)
        print(f"skipped {len(skipped):,} shard(s) not in the file table:")
        for ds, k in sorted(by_ds.items()):
            print(f"    {ds:<62} {k:>6,}")
    n_rows = int(sum(r["n_rows"] for r in plan))

    mm = {name: np.lib.format.open_memmap(
              _npy(out_dir, name), mode="w+", dtype=dt, shape=(n_rows,))
          for name, dt in CARRIED.items()}

    n_files = max((int(r["file_id"]) for r in table["files"]), default=-1) + 1
    offsets = np.zeros(n_files + 1, dtype=np.int64)

    it = _progress(plan, total=len(plan), enabled=progress,
                   desc="merging shards", unit="shard")

    pos = 0
    for rec in it:
        if rec["n_rows"] == 0:
            continue
        cols, _ = idx.read_shard(rec["path"], require_complete=require_complete)
        n = len(cols["entry"])
        # The row count came from the cutflow in pass 1 and sized the output, so
        # check the columns actually agree rather than writing a short block.
        if n != rec["n_rows"]:
            raise ValueError(
                f"{rec['path']}: header said {rec['n_rows']} rows but the "
                f"columns have {n}. The shard changed between passes, or is "
                "inconsistent - re-run stage 1a for it."
            )
        # Pass 1 reads no columns, so the internal-order check lands here. The
        # merge is a pure concatenation, which is only correct if it holds.
        key = cols["entry"].astype(np.int64) * 4 + cols["hemi_slot"]
        if np.any(np.diff(key) < 0):
            raise ValueError(
                f"{rec['path']} is not sorted by (entry, hemi_slot); the merge "
                "assumes shard-internal order."
            )
        fid = int(rec["file_id"])
        sl = slice(pos, pos + n)
        for name, dt in CARRIED.items():
            if name == "file_id":
                mm[name][sl] = fid
            else:
                mm[name][sl] = cols[name].astype(dt, copy=False)
        offsets[fid + 1] = n
        pos += n
    assert pos == n_rows, f"filled {pos} of {n_rows} rows"

    # offsets[i+1] currently holds file i's row count; make it cumulative so
    # offsets[i]:offsets[i+1] is file i's contiguous hemi_id range.
    np.cumsum(offsets, out=offsets)
    np.save(_npy(out_dir, "hemi_id_offsets"), offsets)

    # entry is narrowed to int32 at merge; make the limit explicit rather than
    # silently wrapping on a pathologically large input file.
    if n_rows and int(mm["entry"][:].max()) >= 2 ** 31 - 1:
        raise ValueError("entry exceeds int32; a single input file has too many entries.")

    order = np.argsort(mm["pt"][:], kind="stable").astype(np.int64)
    np.save(_npy(out_dir, "order_by_pt"), order)

    for m in mm.values():
        m.flush()

    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "index_id": None,
        "mode": table["mode"],
        "n_rows": n_rows,
        "n_files": n_files,
        "n_files_with_rows": sum(1 for r in plan if r["n_rows"]),
        "canonical_sort": list(IDENTITY_COLUMNS),
        "file_table": (os.path.basename(file_table_path) if file_table_path
                       else "file_table.json"),
        "file_table_sha256": (ft._sha256(file_table_path) if file_table_path
                              else None),
        "columns": {name: {"file": f"{name}.npy", "dtype": dt.name}
                    for name, dt in {**CARRIED, **DERIVED}.items()},
        "not_stored": {
            "dir_phi": "derive with hemisphere.directed_phi_array(thrust_phi, side)",
            "w_rel": "per-dataset; look up by file_id in the file table",
        },
        "hemi_id_offsets": OFFSETS_NAME,
        "skipped_unlisted": len(skipped),
        "counters": {
            "shards_read": len(plan),
            "shards_empty": sum(1 for r in plan if not r["n_rows"]),
            "rows": n_rows,
        },
    }
    manifest["index_id"] = index_id(out_dir, manifest)
    save_manifest(manifest, out_dir)
    return manifest


def save_manifest(manifest, out_dir):
    with open(os.path.join(str(out_dir), MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def index_id(out_dir, manifest):
    """A hash over the identity columns only.

    Deliberately not over every column: adding a derived column should not
    invalidate every pair chunk and stitched file that quotes the id.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(str(manifest["schema_version"]).encode())
    h.update(str(manifest["n_rows"]).encode())
    for name in IDENTITY_COLUMNS:
        with open(_npy(out_dir, name), "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


class HemisphereIndex:
    """Read access to a merged index, memory-mapped."""

    def __init__(self, out_dir, table=None, *, mmap=True):
        self.dir = str(out_dir)
        with open(os.path.join(self.dir, MANIFEST_NAME)) as f:
            self.manifest = json.load(f)
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError(f"{out_dir} is not a {SCHEMA}")
        self._mode = "r" if mmap else None
        self._cache = {}
        self.table = table
        self.offsets = np.load(_npy(self.dir, "hemi_id_offsets"),
                               mmap_mode=self._mode)

    @property
    def n_rows(self):
        return int(self.manifest["n_rows"])

    def __len__(self):
        return self.n_rows

    def __getitem__(self, name):
        if name not in self._cache:
            self._cache[name] = np.load(_npy(self.dir, name),
                                        mmap_mode=self._mode)
        return self._cache[name]

    def dir_phi(self):
        """The matching direction, float64, derived not stored."""
        from .hemisphere import directed_phi_array
        return directed_phi_array(self["thrust_phi"][:], self["side"][:])

    def w_rel(self):
        """Per-row relative weight, looked up from the file table by file_id."""
        if self.table is None:
            raise ValueError("w_rel needs the file table; pass table=")
        return ft.w_rel_by_id(self.table)[self["file_id"][:]]

    def file_id_of(self, hemi_id):
        """Which file a hemi_id came from, via the offsets table."""
        return int(np.searchsorted(self.offsets, hemi_id, side="right") - 1)


def load_index(out_dir, table_path=None, *, mmap=True):
    table = ft.load_file_table(table_path) if table_path else None
    return HemisphereIndex(out_dir, table, mmap=mmap)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="run3-mj-index-build",
        description="Stage 1b: merge index shards into one global index.",
    )
    p.add_argument("shards", nargs="*", help="index shard ROOT files")
    p.add_argument("--from-file", help="a text file of shard paths, one per line")
    p.add_argument("-t", "--file-table", required=True)
    p.add_argument("-o", "--out-dir", default="global_index")
    p.add_argument("--skip-unlisted", action="store_true",
                   help="build over the file table's files and REPORT shards for "
                        "anything else, instead of refusing. Use when stage 1a "
                        "indexed more than the sample should contain - e.g. every "
                        "HT slice, with a table over only some. The file table "
                        "defines the sample.")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="accept shards with no completion marker (debug only; "
                        "a truncated shard silently shortens the index)")
    p.add_argument("--progress", action="store_true")
    args = p.parse_args(argv)

    shards = list(args.shards)
    if args.from_file:
        with open(args.from_file) as f:
            shards += [ln.strip() for ln in f
                       if ln.strip() and not ln.startswith("#")]
    if not shards:
        sys.exit("No shards given (pass paths, or --from-file).")

    table = ft.load_file_table(args.file_table)
    try:
        manifest = build(shards, table, args.out_dir,
                         require_complete=not args.allow_incomplete,
                         progress=args.progress,
                         file_table_path=args.file_table,
                         skip_unlisted=args.skip_unlisted)
    except ValueError as exc:
        sys.exit(f"index build: {exc}")

    print(f"mode:      {manifest['mode']}")
    print(f"shards:    {manifest['counters']['shards_read']} "
          f"({manifest['counters']['shards_empty']} empty)")
    print(f"rows:      {manifest['n_rows']:,}")
    print(f"files:     {manifest['n_files']} "
          f"({manifest['n_files_with_rows']} contributed rows)")
    print(f"index_id:  {manifest['index_id']}")
    print(f"-> {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

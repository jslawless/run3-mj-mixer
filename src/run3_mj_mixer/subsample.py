"""subsample.py - a data-like hemisphere pool at a fixed luminosity.

Takes an existing global index (MC mode) and writes a new one that keeps each
*event* with probability ``p = min(1, w_rel * L)``, where ``w_rel = sigma/N``
(pb per event) comes from the file table and ``L`` is in pb^-1. The kept rows
are the pool a dataset of luminosity ``L`` would give, with the natural HT
spectrum, so the pseudo-events matched from it are filled with unit weight
(``make_histograms.py --weight-power 0``). This is study 1b.

**The draw is a hash, not an RNG stream.** Each row's uniform number is a
function of ``(run, luminosityBlock, event, seed)`` alone, so

* reruns are byte-identical, whatever the block size or row order;
* the subsample is nested in ``L``: ``u < p`` with a fixed ``u`` per event, so a
  larger ``L`` keeps a superset of a smaller one;
* both hemispheres of one event (``hemi_slot`` 0 and 1) share ``u`` and are kept
  or dropped together - the unit is the event, as it would be in data.

``L`` is deliberately NOT part of the hash: including it would break nesting.

**The output is an ordinary index.** It is written through the same helpers as
``run3-mj-index-build`` (``open_columns``/``finish_columns``/``core_manifest``),
keeps the parent's file table (same ``file_id``s, same ``w_rel``), canonical
order (a subset of sorted rows is sorted) and ``hemi_id_offsets`` semantics
(dropped files simply get empty ranges), and gets its own ``index_id``. Match,
gather, assemble and QA take it unchanged. The manifest's ``subsample`` block
records L, the seed, the dropped datasets, and per-dataset kept/total rows.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import filetable as ft
from . import index_build as ib

# splitmix64 constants (Steele, Lea & Flood 2014). Any good 64-bit mixer would
# do; this one is short, vectorises in numpy, and is well studied.
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_MASK64 = (1 << 64) - 1

#: Rows per block when scanning the parent index; bounds peak memory.
BLOCK = 1 << 22


def _mix(z):
    """splitmix64 finaliser, elementwise on a uint64 array (wrapping)."""
    with np.errstate(over="ignore"):
        z = z + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        return z ^ (z >> np.uint64(31))


def event_uniform(run, lumi, event, seed):
    """A uniform number in [0, 1) per row, from (run, lumi, event, seed) only."""
    run = np.asarray(run).astype(np.uint64)
    lumi = np.asarray(lumi).astype(np.uint64)
    event = np.asarray(event).astype(np.uint64)
    h = _mix(np.full(run.shape, int(seed) & _MASK64, dtype=np.uint64))
    h = _mix(h ^ ((run << np.uint64(32)) | lumi))
    h = _mix(h ^ event)
    # top 53 bits -> an exactly representable double in [0, 1)
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def resolve_drops(table, patterns):
    """Datasets matched by any substring in ``patterns``. A pattern that matches
    nothing is an error - it is almost always a typo, and silently keeping the
    slice would change the study."""
    names = sorted(table["datasets"])
    dropped = set()
    for pat in patterns or ():
        hit = [d for d in names if pat in d]
        if not hit:
            raise ValueError(f"--drop-dataset {pat!r} matches no dataset in the "
                             f"file table; datasets are: {names}")
        dropped.update(hit)
    return sorted(dropped)


def _rows_per_dataset(offsets, ds_by_id):
    per_file = np.diff(np.asarray(offsets, dtype=np.int64))
    out = {}
    for fid, n in enumerate(per_file):
        if n:
            ds = ds_by_id[fid]
            out[ds] = out.get(ds, 0) + int(n)
    return out


def subsample(index_dir, table, out_dir, *, lumi_pb, seed=0,
              drop_datasets=(), table_path=None, block=BLOCK, progress=False):
    """Write the subsampled index to ``out_dir`` and return its manifest.

    ``lumi_pb`` is L in pb^-1, or ``"max"`` for the largest L at which no
    retained dataset is capped: ``1 / max(w_rel)`` over retained datasets that
    have rows. ``table_path``, when given, is checked against the parent's
    ``file_table_sha256``.
    """
    if os.path.abspath(str(index_dir)) == os.path.abspath(str(out_dir)):
        raise ValueError("the output directory must not be the parent index")
    parent = ib.HemisphereIndex(index_dir, table)
    pm = parent.manifest
    if pm.get("mode") != "mc" or table.get("mode") != "mc":
        raise ValueError(
            "subsampling by w_rel*L needs an MC index and an MC file table "
            f"(index mode {pm.get('mode')!r}, table mode {table.get('mode')!r})")
    want = pm.get("file_table_sha256")
    if table_path is not None and want is not None \
            and ft._sha256(table_path) != want:
        raise ValueError(
            f"{table_path} is not the file table {index_dir} was built from "
            f"(sha256 {want}); file_ids and weights would not line up")

    n_files = int(pm["n_files"])
    ds_by_id = ft.dataset_by_id(table)
    if len(ds_by_id) != n_files:
        raise ValueError(f"file table has {len(ds_by_id)} file_ids, index "
                         f"{n_files}")
    w_rel = ft.w_rel_by_id(table)
    rows_total = _rows_per_dataset(parent.offsets, ds_by_id)
    dropped = resolve_drops(table, drop_datasets)

    kept_ds = [d for d in rows_total if d not in dropped]
    if isinstance(lumi_pb, str):
        if lumi_pb != "max":
            raise ValueError(f"lumi_pb must be a number or 'max', got {lumi_pb!r}")
        if not kept_ds:
            raise ValueError("every dataset with rows is dropped")
        lumi_mode = "max"
        L = 1.0 / max(float(table["datasets"][d]["w_rel"]) for d in kept_ds)
    else:
        lumi_mode = "given"
        L = float(lumi_pb)
        if not L > 0:
            raise ValueError(f"lumi_pb must be positive, got {L}")

    p_file = np.minimum(1.0, w_rel * L)
    drop_file = np.array([d in dropped for d in ds_by_id], dtype=bool)
    p_file[drop_file] = 0.0

    # pass 1: which rows survive. Only the hash inputs and file_id are read.
    n = len(parent)
    kept = []
    rng_blocks = range(0, n, block)
    for lo in ft._progress(rng_blocks, total=len(rng_blocks), enabled=progress,
                           desc="drawing", unit="block"):
        hi = min(lo + block, n)
        fid = np.asarray(parent["file_id"][lo:hi])
        u = event_uniform(parent["run"][lo:hi], parent["luminosityBlock"][lo:hi],
                          parent["event"][lo:hi], seed)
        kept.append(np.flatnonzero(u < p_file[fid]).astype(np.int64) + lo)
    rows = np.concatenate(kept) if kept else np.zeros(0, dtype=np.int64)
    del kept
    n_keep = len(rows)

    # pass 2: copy the kept rows column by column, in blocks.
    os.makedirs(str(out_dir), exist_ok=True)
    mm = ib.open_columns(out_dir, n_keep)
    for name in ib.CARRIED:
        src = parent[name]
        dst = mm[name]
        for lo in range(0, n_keep, block):
            hi = min(lo + block, n_keep)
            dst[lo:hi] = src[rows[lo:hi]]
    fid_out = np.asarray(mm["file_id"][:])
    offsets = np.zeros(n_files + 1, dtype=np.int64)
    offsets[1:] = np.bincount(fid_out, minlength=n_files)
    ib.finish_columns(out_dir, mm, offsets)

    rows_kept = _rows_per_dataset(offsets, ds_by_id)
    per_ds = {}
    for ds in sorted(table["datasets"]):
        w = float(table["datasets"][ds]["w_rel"])
        p = 0.0 if ds in dropped else min(1.0, w * L)
        tot = rows_total.get(ds, 0)
        per_ds[ds] = {
            "w_rel": w,
            "equiv_lumi_pb": 1.0 / w,
            "p_keep": p,
            "capped": bool(ds not in dropped and w * L > 1.0),
            "dropped": ds in dropped,
            "rows_total": tot,
            "rows_expected": tot * p,
            "rows_kept": rows_kept.get(ds, 0),
        }

    manifest = ib.core_manifest(
        mode=pm["mode"], n_rows=n_keep, n_files=n_files,
        n_files_with_rows=int(np.count_nonzero(np.diff(offsets))),
        file_table=pm.get("file_table", "file_table.json"),
        file_table_sha256=want)
    manifest["counters"] = {"rows": n_keep, "parent_rows": n}
    manifest["parent_index"] = {"dir": os.path.abspath(str(index_dir)),
                                "index_id": pm["index_id"],
                                "n_rows": n}
    manifest["subsample"] = {
        "lumi_pb": L,
        "lumi_mode": lumi_mode,
        "seed": int(seed),
        "draw": "keep event iff u < min(1, w_rel*lumi_pb); u = splitmix64 hash "
                "of (seed, run<<32|luminosityBlock, event) -> [0,1); nested in "
                "lumi_pb",
        "drop_patterns": list(drop_datasets or ()),
        "dropped_datasets": dropped,
        "fill_with": "unit weight (make_histograms.py --weight-power 0); each "
                     "pseudo-event then counts events at lumi_pb",
        "datasets": per_ds,
    }
    if "max_rows_per_event" in pm:
        manifest["max_rows_per_event"] = pm["max_rows_per_event"]
    manifest["index_id"] = ib.index_id(out_dir, manifest)
    ib.save_manifest(manifest, out_dir)
    return manifest


def _lumi_arg(s):
    if s == "max":
        return s
    try:
        return float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number or 'max', got {s!r}")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="run3-mj-index-subsample",
        description="Write a data-like subsample of an MC index: keep each "
                    "event with probability min(1, w_rel * L).",
    )
    p.add_argument("index_dir", help="the parent (full) index")
    p.add_argument("-t", "--file-table", required=True,
                   help="the file table the parent was built from (checked by "
                        "sha256); the subsample keeps using it")
    p.add_argument("-o", "--out-dir", required=True)
    p.add_argument("--lumi-pb", required=True, type=_lumi_arg, metavar="L",
                   help="luminosity in pb^-1 (0.2 fb^-1 = 200), or 'max' for "
                        "1/max(w_rel) over the retained datasets, the largest L "
                        "at which no retained slice is capped at p = 1")
    p.add_argument("--seed", type=int, default=0,
                   help="hash seed (default 0). The same seed and a larger L "
                        "give a superset.")
    p.add_argument("--drop-dataset", action="append", default=[],
                   metavar="SUBSTR",
                   help="drop every dataset whose name contains SUBSTR "
                        "(repeatable), e.g. HT-200to400. Careful: HT-200 "
                        "would also match HT-2000.")
    p.add_argument("--progress", action="store_true")
    args = p.parse_args(argv)

    table = ft.load_file_table(args.file_table)
    try:
        m = subsample(args.index_dir, table, args.out_dir, lumi_pb=args.lumi_pb,
                      seed=args.seed, drop_datasets=args.drop_dataset,
                      table_path=args.file_table, progress=args.progress)
    except ValueError as exc:
        sys.exit(f"index subsample: {exc}")

    s = m["subsample"]
    print(f"parent:    {m['parent_index']['index_id']} "
          f"({m['parent_index']['n_rows']:,} rows)")
    print(f"L:         {s['lumi_pb']:.6g} pb^-1 ({s['lumi_mode']}), "
          f"seed {s['seed']}")
    print(f"{'dataset':<48} {'p_keep':>9} {'total':>11} {'expected':>12} "
          f"{'kept':>11}")
    for ds, r in s["datasets"].items():
        flag = " DROPPED" if r["dropped"] else (" CAPPED" if r["capped"] else "")
        print(f"{ds[:48]:<48} {r['p_keep']:9.4g} {r['rows_total']:11,} "
              f"{r['rows_expected']:12,.0f} {r['rows_kept']:11,}{flag}")
    print(f"rows:      {m['n_rows']:,}")
    print(f"index_id:  {m['index_id']}")
    print(f"-> {args.out_dir}/")
    capped = [d for d, r in s["datasets"].items() if r["capped"]]
    if capped:
        print(f"WARNING: {len(capped)} dataset(s) capped at p = 1 - they are "
              f"under-represented relative to L: {capped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

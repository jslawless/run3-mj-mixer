"""gather.py - stage 3a: fetch the event data each pair needs.

**Jobs own files, not pairs.** That inversion is the whole point of this stage.
Every pseudo-event needs jets from two arbitrary entries in two arbitrary files,
and ROOT stores a jet branch as one basket for the whole file - so there is no
entry-level granularity, and touching a file at all costs its full basket set.
Walking the pair list and fetching as you go re-opens every file once per job:
the cost is ``n_jobs x n_files``, which does not scale. Walking the *files*
instead, once each, costs ``n_files`` total, independent of how many stage-3b
jobs there are.

Each job reads its files sequentially and emits one record per referenced
(pair, leg) - the payload for that hemisphere's jets, selected by ``jet_mask``.
Records are bucketed by destination chunk, so a stage-3b job reads only its own
share instead of sifting the whole intermediate.

**The payload is a config list of branch names**, not a hardcoded schema. That is
what keeps the index independent of the analysis: adding a systematic is a config
edit and a re-run of 3a/3b, with stage 1 and stage 2 untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import awkward as ak
import numpy as np

from . import filetable as ft
from . import match as M
from .hemi_io import JET_BRANCH, branch_names, missing_branches
from .hemisphere import unpack_jet_mask

LEGS_TREE = "legs"

#: What to carry through when the config says nothing. The four kinematics the
#: output has always had, plus the JES/JER variations - the first thing anyone
#: actually needs and the reason the payload is configurable at all.
DEFAULT_JET_BRANCHES = [
    "pt", "eta", "phi", "m",
    "pt_jesUp", "pt_jesDown", "pt_jerUp", "pt_jerDown",
    "m_jesUp", "m_jesDown", "m_jerUp", "m_jerDown",
]

#: Always required: stage 3b cannot build a pseudo-event without these.
REQUIRED_JET_BRANCHES = ("pt", "eta", "phi", "m")


def payload_branches(config=None):
    """The per-jet branch list, from config or the default."""
    if config and "payload" in config:
        fields = list(config["payload"].get("jet_branches", DEFAULT_JET_BRANCHES))
    else:
        fields = list(DEFAULT_JET_BRANCHES)
    for req in REQUIRED_JET_BRANCHES:
        if req not in fields:
            raise ValueError(
                f"payload.jet_branches must include {req!r}; stage 3b cannot "
                "assemble a pseudo-event without the jet kinematics."
            )
    return fields


# ---------------------------------------------------------------------------
# Which legs does this job owe?
# ---------------------------------------------------------------------------

def load_pair_manifest(pairs_dir):
    with open(os.path.join(str(pairs_dir), M.MANIFEST_NAME)) as f:
        return json.load(f)


def chunk_of_pair(manifest):
    """A function mapping pair_id -> destination chunk id."""
    los = np.array([c["pair_lo"] for c in manifest["chunks"]], dtype=np.int64)
    ids = np.array([c["chunk_id"] for c in manifest["chunks"]], dtype=np.int64)
    order = np.argsort(los)
    los, ids = los[order], ids[order]

    def f(pair_ids):
        k = np.searchsorted(los, pair_ids, side="right") - 1
        return ids[np.clip(k, 0, len(ids) - 1)]

    return f


def scan_legs(pairs_dir, manifest, owned):
    """Every (pair_id, leg, entry, jet_mask, chunk) this job's files must supply.

    Reads only the provenance columns of each pair chunk, not the whole row.
    """
    import uproot

    to_chunk = chunk_of_pair(manifest)
    owned = set(int(x) for x in owned)
    per_file = defaultdict(list)

    want = ["pair_id",
            "seed_file_id", "seed_entry", "seed_jet_mask",
            "match_file_id", "match_entry", "match_jet_mask"]
    for c in manifest["chunks"]:
        path = os.path.join(str(pairs_dir), c["file"])
        with uproot.open(path) as f:
            a = f[M.PAIRS_TREE].arrays(want, library="np")
        cid = to_chunk(a["pair_id"])
        for leg, tag in ((0, "seed"), (1, "match")):
            fid = a[f"{tag}_file_id"]
            keep = np.isin(fid, list(owned)) if owned else np.zeros(len(fid), bool)
            if not keep.any():
                continue
            for pid, fi, en, mk, ch in zip(a["pair_id"][keep], fid[keep],
                                           a[f"{tag}_entry"][keep],
                                           a[f"{tag}_jet_mask"][keep],
                                           cid[keep]):
                per_file[int(fi)].append(
                    (int(pid), leg, int(en), int(mk), int(ch)))
    return per_file


# ---------------------------------------------------------------------------
# Reading one file
# ---------------------------------------------------------------------------

def read_file_legs(slimmed_path, requests, fields, *, tree_name="events",
                   n_jets_event=None):
    """Payload for every requested (pair, leg) in one file, read in ONE pass.

    ``requests`` is a list of ``(pair_id, leg, entry, jet_mask, chunk)``. The file
    is opened once and its payload branches read whole - which is not wasteful,
    because a branch is a single basket, so a partial read costs the same.
    """
    import uproot

    with uproot.open(slimmed_path) as f:
        tree = f[tree_name]
        keys = list(tree.keys())
        absent = missing_branches(keys, fields)
        if absent:
            raise ValueError(
                f"{slimmed_path} lacks payload branches {absent} "
                f"(as {JET_BRANCH}_<name> or {JET_BRANCH}.<name>). Fix "
                "payload.jet_branches, or use a slimmer output that has them."
            )
        mapping = branch_names(keys, fields)
        arrays = tree.arrays(list(mapping.values()), library="ak")

    n_counts = ak.to_numpy(ak.num(arrays[mapping["pt"]], axis=1))
    out = {"pair_id": [], "leg": [], "chunk": []}
    for fld in fields:
        out[f"jet_{fld}"] = []

    for pid, leg, entry, mask, chunk in requests:
        n_ev = int(n_counts[entry]) if n_jets_event is None else n_jets_event
        sel = unpack_jet_mask(mask, n_ev)
        out["pair_id"].append(pid)
        out["leg"].append(leg)
        out["chunk"].append(chunk)
        for fld in fields:
            vals = np.asarray(arrays[mapping[fld]][entry])
            out[f"jet_{fld}"].append(vals[sel])
    return out


def write_legs(out_dir, job_id, records, fields):
    """Write one file per destination chunk: legA_<job>_<chunk>.root.

    Bucketing by destination is what lets a stage-3b job read only its own share.
    The alternative - one file per gather job - would make every 3b job read the
    entire intermediate.
    """
    import uproot

    os.makedirs(str(out_dir), exist_ok=True)
    by_chunk = defaultdict(lambda: {k: [] for k in
                                    ["pair_id", "leg"] + [f"jet_{f}" for f in fields]})
    for k in range(len(records["pair_id"])):
        b = by_chunk[int(records["chunk"][k])]
        b["pair_id"].append(records["pair_id"][k])
        b["leg"].append(records["leg"][k])
        for fld in fields:
            b[f"jet_{fld}"].append(records[f"jet_{fld}"][k])

    written = []
    for chunk, b in sorted(by_chunk.items()):
        name = f"legA_{job_id:04d}_{chunk:05d}.root"
        path = os.path.join(str(out_dir), name)
        cols = {
            "pair_id": np.asarray(b["pair_id"], dtype=np.int64),
            "leg": np.asarray(b["leg"], dtype=np.uint8),
        }
        for fld in fields:
            cols[f"jet_{fld}"] = ak.Array(
                [np.asarray(v, dtype=np.float32) for v in b[f"jet_{fld}"]])
        tmp = path + ".tmp"
        with uproot.recreate(tmp) as f:
            f[LEGS_TREE] = cols
            f["job_id"] = M._label(str(job_id))
            f["chunk_id"] = M._label(str(chunk))
            f["complete"] = M._label("ok")
        os.replace(tmp, path)
        written.append({"file": name, "chunk_id": chunk,
                        "n_legs": len(b["pair_id"])})
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def assign_files(n_files, job, n_jobs):
    """Which file_ids this job owns - a contiguous block, so a job's reads are
    as local on EOS as the file naming allows."""
    per = -(-n_files // n_jobs)
    lo = job * per
    hi = min(lo + per, n_files)
    return list(range(lo, hi))


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="run3-mj-gather",
        description="Stage 3a: fetch the jets each pair needs, by file.",
    )
    p.add_argument("pairs_dir")
    p.add_argument("-t", "--file-table", required=True)
    p.add_argument("-o", "--out-dir", default="legs")
    p.add_argument("--job", type=int, default=0, help="this job's index")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--file-ids", default=None,
                   help="explicit comma-separated file_ids instead of --job/--n-jobs")
    p.add_argument("--config", default=None, help="mixer config, for payload.jet_branches")
    p.add_argument("--tree", default="events")
    p.add_argument("--progress", action="store_true")
    args = p.parse_args(argv)

    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    try:
        fields = payload_branches(config)
    except ValueError as exc:
        sys.exit(f"gather: {exc}")

    table = ft.load_file_table(args.file_table)
    manifest = load_pair_manifest(args.pairs_dir)
    paths = {int(r["file_id"]): r["path"] for r in table["files"]}

    if args.file_ids:
        owned = [int(x) for x in args.file_ids.split(",") if x.strip()]
    else:
        owned = assign_files(table["n_files"], args.job, args.n_jobs)

    print(f"job {args.job}/{args.n_jobs}: {len(owned)} file(s)")
    print(f"payload: {len(fields)} per-jet branches {fields}")

    per_file = scan_legs(args.pairs_dir, manifest, owned)
    n_want = sum(len(v) for v in per_file.values())
    print(f"legs owed: {n_want:,} across {len(per_file)} file(s) with references")

    merged = {"pair_id": [], "leg": [], "chunk": []}
    for fld in fields:
        merged[f"jet_{fld}"] = []

    items = sorted(per_file.items())
    if args.progress:
        try:
            from tqdm import tqdm
            items = tqdm(items, unit="file", desc="gathering")
        except ImportError:
            pass

    for fid, reqs in items:
        path = paths.get(fid)
        if path is None:
            sys.exit(f"file_id {fid} is not in the file table.")
        try:
            got = read_file_legs(path, reqs, fields, tree_name=args.tree)
        except ValueError as exc:
            sys.exit(f"gather: {exc}")
        for k, v in got.items():
            merged[k].extend(v)

    written = write_legs(args.out_dir, args.job, merged, fields)
    print(f"wrote {len(written)} leg file(s), {len(merged['pair_id']):,} legs")
    for w in written:
        print(f"    {w['file']}  {w['n_legs']:,} legs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

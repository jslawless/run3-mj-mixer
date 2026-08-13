#!/usr/bin/env python3
"""submit_assemble.py - condor submission for stage 3b (assemble).

One job per pair chunk. Each reads its chunk plus that chunk's leg shards,
joins on (pair_id, leg), and writes one stitched file. The join is required to
be exact: a missing leg aborts the job naming the pair, rather than quietly
writing a short file.

Usage
-----
    python scripts/submit_assemble.py -p /store/user/<you>/mixing/pairs \\
        -l /store/user/<you>/mixing/legs -t file_table.json \\
        -o /store/user/<you>/mixing --wheel run3_mj_mixer-*.whl -P 60 --exec

``--check`` instead lists which outputs are missing or short, as a paste-ready
queue block - the ledger is each output file's own metadata, so no database.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# scripts/<group>/x.py -> ../../src
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))

from run3_mj_mixer import condor
from run3_mj_mixer import match as M

STITCHED_SUBDIR = "stitched"


def load_manifest(pairs_dir):
    with open(os.path.join(pairs_dir, M.MANIFEST_NAME)) as f:
        return json.load(f)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-p", "--pairs-dir", required=True)
    p.add_argument("-l", "--legs-dir", required=True)
    p.add_argument("-t", "--file-table", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("-P", "--n-gather-jobs", type=int, default=None, metavar="P",
                   help="how many gather jobs ran. MUST match stage 3a: leg "
                        "filenames are reconstructed as legA_<job>_<chunk>.root "
                        "over range(P), so too small silently misses shards that "
                        "exist and too large references files that do not. "
                        "Normally omitted - it is read from "
                        "--gather-manifest instead.")
    p.add_argument("--gather-manifest", default="batch_gather/gather_manifest.json",
                   metavar="JSON",
                   help="the manifest submit_gather.py wrote, which records P "
                        "(default batch_gather/gather_manifest.json)")
    p.add_argument("--manifest", default=None,
                   help="local copy of pairs_manifest.json (default: read from "
                        "--pairs-dir)")
    condor.add_common_args(p, ram="4GB", disk="4GB")
    args = p.parse_args(argv)

    man_dir = args.manifest and os.path.dirname(args.manifest) or args.pairs_dir
    manifest = load_manifest(man_dir)
    chunks = manifest["chunks"]
    if not chunks:
        sys.exit("the pair manifest lists no chunks")

    # P must equal what stage 3a used. Prefer the recorded value over a retyped
    # flag: a mismatch fails in a way that looks like missing gather output.
    n_gather = args.n_gather_jobs
    if n_gather is None:
        try:
            with open(args.gather_manifest) as f:
                n_gather = int(json.load(f)["n_jobs"])
            print(f"read P={n_gather} from {args.gather_manifest}")
        except (OSError, KeyError, ValueError) as exc:
            sys.exit(f"could not read P from {args.gather_manifest} ({exc}); "
                     "pass -P explicitly with the value stage 3a used.")

    ft_base = os.path.basename(args.file_table)
    cfg_base = os.path.basename(args.config) if args.config else None

    jobs = {}
    for c in chunks:
        cid = int(c["chunk_id"])
        legs = " ".join(
            f"{args.legs_dir}/legA_{j:04d}_{cid:05d}.root"
            for j in range(n_gather))
        out = f"{STITCHED_SUBDIR}/stitched_{cid:05d}.root"
        cmd = (f"run3-mj-assemble {args.pairs_dir}/{c['file']} {legs} "
               f"-t {ft_base} -o {out} "
               f"--expect-index-id {manifest['index_id']}")
        if cfg_base:
            cmd += f" --config {cfg_base}"
        jobs[f"assemble_{cid:05d}"] = f"mkdir -p {STITCHED_SUBDIR}\n{cmd}"

    transfer = condor.transfer_list(args.wheel, args.file_table, args.config)
    sub = condor.write_jobs(args.logdir, jobs, transfer, args.eosoutdir,
                            args.wheel, cpu=args.cpu, queue=args.queue,
                            ram=args.memory, disk=args.disk,
                            redirector=args.redirector,
                            lcg_view=args.lcg_view)
    print(f"{len(chunks)} chunk(s) -> {len(jobs)} assemble job(s)")
    print(f"each expects {n_gather} leg shard(s) from {args.legs_dir}")
    print(f"outputs land under {args.eosoutdir}/{STITCHED_SUBDIR}/")
    print(f"wrote {sub}")
    if not args.do_exec:
        print(f"  dry run; submit with: condor_submit {sub}")
        return 0
    return condor.submit(sub)


if __name__ == "__main__":
    raise SystemExit(main())

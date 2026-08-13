#!/usr/bin/env python3
"""submit_gather.py - condor submission for stage 3a (gather by file).

Jobs own FILES, not pairs. That is the whole point of the stage: every
pseudo-event needs two arbitrary entries from two arbitrary files, and a jet
branch is one basket per file, so walking the pair list re-reads every file once
per job (cost ~ n_jobs x n_files). Walking the files instead costs n_files once,
independent of how many stage-3b jobs there are.

Usage
-----
    python scripts/submit_gather.py -p /store/user/<you>/mixing/pairs \\
        -t file_table.json -o /store/user/<you>/mixing \\
        --config config/config.json --wheel run3_mj_mixer-*.whl -P 60 --exec
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
from run3_mj_mixer.filetable import load_file_table

LEGS_SUBDIR = "legs"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-p", "--pairs-dir", required=True,
                   help="EOS or local dir holding pairs_*.root + the manifest")
    p.add_argument("-t", "--file-table", required=True)
    p.add_argument("--config", default=None,
                   help="mixer config, for payload.jet_branches")
    p.add_argument("--manifest", default=None, metavar="JSON",
                   help="local pairs_manifest.json to transfer into the jobs "
                        "(default <pairs-dir>/pairs_manifest.json when that is "
                        "a local path). The jobs read the pair CHUNKS from EOS "
                        "over xrootd, but the manifest is plain JSON and must "
                        "be transferred.")
    p.add_argument("-P", "--n-jobs", type=int, default=60, metavar="P",
                   help="number of gather jobs; each owns a contiguous block of "
                        "file_ids (default 60). Stage 3b must use the SAME value, "
                        "because it reconstructs leg filenames as "
                        "legA_<job>_<chunk>.root over range(P) - so this is "
                        "recorded in <logdir>/gather_manifest.json and read back "
                        "by submit_assemble.py rather than retyped.")
    p.add_argument("--tree", default="events",
                   help="tree name in the slimmed files")
    condor.add_common_args(p, ram="4GB", disk="4GB")
    args = p.parse_args(argv)

    # The manifest has to reach the worker as a real file.
    manifest = args.manifest
    if manifest is None:
        guess = os.path.join(args.pairs_dir, "pairs_manifest.json")
        if not args.pairs_dir.startswith("root://") and os.path.exists(guess):
            manifest = guess
    if manifest is None or not os.path.exists(manifest):
        sys.exit(
            "need a local pairs_manifest.json to transfer into the jobs; pass "
            "--manifest <path>. (--pairs-dir may be a root:// URL for the "
            "chunks, but plain JSON cannot be read over xrootd.)")

    table = load_file_table(args.file_table)
    n_files = int(table["n_files"])
    n_jobs = max(1, min(args.n_jobs, n_files))

    ft_base = os.path.basename(args.file_table)
    cfg_base = os.path.basename(args.config) if args.config else None
    man_base = os.path.basename(manifest)

    jobs = {}
    for j in range(n_jobs):
        cmd = (f"run3-mj-gather {args.pairs_dir} -t {ft_base} "
               f"-o {LEGS_SUBDIR} --job {j} --n-jobs {n_jobs} "
               f"--tree {args.tree} --manifest {man_base}")
        if cfg_base:
            cmd += f" --config {cfg_base}"
        jobs[f"gather_{j:04d}"] = f"mkdir -p {LEGS_SUBDIR}\n{cmd}"

    transfer = condor.transfer_list(args.wheel, args.file_table, args.config,
                                    manifest)
    sub = condor.write_jobs(args.logdir, jobs, transfer, args.eosoutdir,
                            args.wheel, cpu=args.cpu, queue=args.queue,
                            ram=args.memory, disk=args.disk,
                            redirector=args.redirector,
                            lcg_view=args.lcg_view)
    # Record P so stage 3b does not have to be told again. Written at submit
    # time, not by the jobs: N parallel jobs cannot safely share one manifest.
    man = {"schema": "run3-mj-gather-submit", "n_jobs": n_jobs,
           "legs_subdir": LEGS_SUBDIR, "pairs_dir": args.pairs_dir,
           "eosoutdir": args.eosoutdir}
    man_path = os.path.join(args.logdir, "gather_manifest.json")
    with open(man_path, "w") as f:
        json.dump(man, f, indent=2)
        f.write("\n")

    print(f"{n_files} file(s) -> {n_jobs} gather job(s), "
          f"~{-(-n_files // n_jobs)} file(s) each")
    print(f"recorded P={n_jobs} in {man_path} (stage 3b reads it)")
    print(f"outputs land under {args.eosoutdir}/{LEGS_SUBDIR}/")
    print(f"wrote {sub}")
    if not args.do_exec:
        print(f"  dry run; submit with: condor_submit {sub}")
        return 0
    return condor.submit(sub)


if __name__ == "__main__":
    raise SystemExit(main())

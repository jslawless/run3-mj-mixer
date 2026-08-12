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
    p.add_argument("-P", "--n-jobs", type=int, default=60,
                   help="number of gather jobs; each owns a contiguous block "
                        "of file_ids")
    p.add_argument("--tree", default="events")
    condor.add_common_args(p, ram="4GB", disk="4GB")
    args = p.parse_args(argv)

    table = load_file_table(args.file_table)
    n_files = int(table["n_files"])
    n_jobs = max(1, min(args.n_jobs, n_files))

    ft_base = os.path.basename(args.file_table)
    cfg_base = os.path.basename(args.config) if args.config else None

    jobs = {}
    for j in range(n_jobs):
        cmd = (f"run3-mj-gather {args.pairs_dir} -t {ft_base} "
               f"-o {LEGS_SUBDIR} --job {j} --n-jobs {n_jobs} "
               f"--tree {args.tree}")
        if cfg_base:
            cmd += f" --config {cfg_base}"
        jobs[f"gather_{j:04d}"] = f"mkdir -p {LEGS_SUBDIR}\n{cmd}"

    transfer = condor.transfer_list(args.wheel, args.file_table, args.config)
    sub = condor.write_jobs(args.logdir, jobs, transfer, args.eosoutdir,
                            args.wheel, cpu=args.cpu, queue=args.queue,
                            ram=args.memory, disk=args.disk,
                            redirector=args.redirector)
    print(f"{n_files} file(s) -> {n_jobs} gather job(s), "
          f"~{-(-n_files // n_jobs)} file(s) each")
    print(f"outputs land under {args.eosoutdir}/{LEGS_SUBDIR}/")
    return condor.maybe_submit(sub, args.do_exec)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""submit_index.py - condor submission for stage 1a (index shards).

Replaces the old ``submit_mixer.py``. Two things are gone with it:

* the in-job stitch block, along with ``--no-stitch``/``--max-distance``/
  ``--pt-tolerance`` - matching is stage 2 now, and it is a single central
  process, not something a per-file job does;
* the slice-balanced job grouper (``make_mixing_jobs.py``). With a global library
  a job group no longer has to be a miniature representative sample, so files are
  grouped by slice for read locality instead - which also stops discarding the
  leftovers the grouper used to drop.

Usage
-----
    voms-proxy-init --rfc --voms cms -valid 192:00
    pip wheel . -w .
    python scripts/submit_index.py -i filelists/ -o /store/user/<you>/mixing \\
        --config config/config.json --wheel run3_mj_mixer-*.whl -n 20 --exec
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# scripts/<group>/x.py -> ../../src
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))

from run3_mj_mixer import condor
from run3_mj_mixer.filetable import (UNKNOWN_SLICE, load_paths, load_xs,
                                     locate_xs_json, slice_or_unknown,
                                     xs_keys_of)

SHARD_SUBDIR = "index"


def group_by_slice(files, xs_keys, per_job):
    """Jobs of up to ``per_job`` files, never mixing slices.

    Grouping by slice keeps a job's reads local; it is no longer required to be
    representative, because the library is global now.
    """
    by_slice = {}
    for f in files:
        by_slice.setdefault(slice_or_unknown(f, xs_keys), []).append(f)
    jobs = {}
    for sl, fs in sorted(by_slice.items()):
        fs = sorted(fs)
        for k in range(0, len(fs), per_job):
            tag = f"{sl}_{k // per_job}" if len(fs) > per_job else sl
            jobs[tag] = (sl, fs[k:k + per_job])
    return jobs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--inFile", required=True,
                   help="fileset JSON, a directory of them, or a .txt of paths")
    p.add_argument("--config", required=True)
    p.add_argument("--xs-json", default=None,
                   help="only used to name output subdirectories by slice")
    p.add_argument("-n", "--nfPerJob", type=int, default=20)
    p.add_argument("--tree", default="events")
    p.add_argument("--index-n-jets", type=int, default=3)
    condor.add_common_args(p, ram="3GB", disk="3GB")
    args = p.parse_args(argv)

    files = load_paths(args.inFile)
    if not files:
        sys.exit(f"No input files found in {args.inFile}")

    xs_path = locate_xs_json(args.xs_json)
    xs_keys = xs_keys_of(load_xs(xs_path)) if xs_path else []
    groups = group_by_slice(files, xs_keys, args.nfPerJob)

    cfg = os.path.basename(args.config)
    jobs = {}
    for tag, (sl, fs) in groups.items():
        outdir = f"{SHARD_SUBDIR}/{sl}"
        lines = [f"mkdir -p {outdir}"]
        for f in fs:
            # condor flattens transfers into the CWD, so the config is a basename
            lines.append(
                f"run3-mj-index {f} {cfg} --tree {args.tree} "
                f"--index-n-jets {args.index_n_jets} --outdir {outdir}")
        jobs[tag] = "\n".join(lines)

    transfer = condor.transfer_list(args.wheel, args.config)
    sub = condor.write_jobs(args.logdir, jobs, transfer, args.eosoutdir,
                            args.wheel, cpu=args.cpu, queue=args.queue,
                            ram=args.memory, disk=args.disk,
                            redirector=args.redirector)

    n_unknown = sum(len(fs) for sl, fs in groups.values() if sl == UNKNOWN_SLICE)
    print(f"{len(files)} file(s) -> {len(jobs)} job(s), "
          f"<= {args.nfPerJob} file(s) each")
    if n_unknown:
        print(f"  note: {n_unknown} file(s) matched no slice and go to "
              f"{SHARD_SUBDIR}/{UNKNOWN_SLICE}/")
    print(f"outputs land under {args.eosoutdir}/{SHARD_SUBDIR}/<slice>/")
    return condor.maybe_submit(sub, args.do_exec)


if __name__ == "__main__":
    raise SystemExit(main())

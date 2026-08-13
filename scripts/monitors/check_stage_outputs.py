#!/usr/bin/env python3
"""check_stage_outputs.py - which stage outputs are missing or short?

The ledger is each output file's own metadata, so there is no database to keep in
sync. Prints any gaps as a paste-ready `queue name from` block for resubmission.

    python scripts/check_stage_outputs.py legs   -p pairs/ -d legs/  -P 60
    python scripts/check_stage_outputs.py stitched -p pairs/ -d stitched/
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

from run3_mj_mixer import match as M


def load_manifest(pairs_dir):
    with open(os.path.join(pairs_dir, M.MANIFEST_NAME)) as f:
        return json.load(f)


def check_legs(manifest, legs_dir, n_jobs):
    want, missing = [], []
    for c in manifest["chunks"]:
        for j in range(n_jobs):
            name = f"legA_{j:04d}_{int(c['chunk_id']):05d}.root"
            want.append((j, name))
            if not os.path.exists(os.path.join(legs_dir, name)):
                missing.append(f"gather_{j:04d}")
    return sorted(set(missing)), len(want)


def check_stitched(manifest, out_dir):
    import uproot
    missing = []
    for c in manifest["chunks"]:
        cid = int(c["chunk_id"])
        path = os.path.join(out_dir, f"stitched_{cid:05d}.root")
        name = f"assemble_{cid:05d}"
        if not os.path.exists(path):
            missing.append((name, "absent"))
            continue
        try:
            with uproot.open(path) as f:
                keys = {k.split(";")[0] for k in f.keys()}
                if "complete" not in keys:
                    missing.append((name, "no completion marker"))
                    continue
                n = f["events"].num_entries
        except Exception as exc:
            missing.append((name, f"unreadable: {exc}"))
            continue
        want = int(c["pair_hi"]) - int(c["pair_lo"])
        if n != want:
            missing.append((name, f"short: {n} of {want}"))
    return missing, len(manifest["chunks"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["legs", "stitched"])
    p.add_argument("-p", "--pairs-dir", required=True)
    p.add_argument("-d", "--dir", required=True, help="the stage's output dir")
    p.add_argument("-P", "--n-gather-jobs", type=int, default=None, metavar="P",
                   help="how many gather jobs ran; needed to know which leg files "
                        "to expect. Read from --gather-manifest when omitted.")
    p.add_argument("--gather-manifest", default="batch_gather/gather_manifest.json",
                   metavar="JSON", help="the manifest submit_gather.py wrote")
    args = p.parse_args(argv)

    manifest = load_manifest(args.pairs_dir)
    n_gather = args.n_gather_jobs
    if args.stage == "legs" and n_gather is None:
        try:
            with open(args.gather_manifest) as f:
                n_gather = int(json.load(f)["n_jobs"])
        except (OSError, KeyError, ValueError) as exc:
            sys.exit(f"could not read P from {args.gather_manifest} ({exc}); "
                     "pass -P explicitly with the value stage 3a used.")
    if args.stage == "legs":
        bad, total = check_legs(manifest, args.dir, n_gather)
        bad = [(b, "absent") for b in bad]
    else:
        bad, total = check_stitched(manifest, args.dir)

    print(f"{args.stage}: {total - len(bad)} of {total} present and complete")
    if not bad:
        print("nothing to resubmit")
        return 0
    for name, why in bad:
        print(f"  {name}: {why}")
    print("\nqueue name from (")
    for name, _ in bad:
        print(f"\t{name}")
    print(")")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

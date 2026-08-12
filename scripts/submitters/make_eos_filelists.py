#!/usr/bin/env python3
"""make_eos_filelists.py - Build per-dataset coffea filelists from an EOS dir.

Recursively lists the *.root files under each dataset subdirectory of an EOS
path (via `xrdfs <host> ls -R`) and writes one mixer-style filelist JSON per
dataset:

    { "<dataset>": { "files": { "root://<host>//store/.../file.root": "<tree>" } } }

Point --base at the slimmer's EOS output directory (the dir you passed as the
slimmer's -o/--eosoutdir, whose subdirs are the per-dataset slimmed outputs).
The recorded tree defaults to 'events' (what the slimmer writes). Run it where
`xrdfs` can reach the host (e.g. lxplus for eoscms.cern.ch, or an LPC node for
cmseos.fnal.gov).

Example:
    python scripts/make_eos_filelists.py \\
        --host cmseos.fnal.gov \\
        --base /store/user/you/slimmed \\
        -o filelists/
"""

import argparse
import json
import os
import subprocess
import sys


def xrdfs_ls(host, path, recursive=False):
    """Return absolute entry paths under an EOS dir via `xrdfs <host> ls [-R]`."""
    cmd = ["xrdfs", host, "ls"] + (["-R"] if recursive else []) + [path]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit("'xrdfs' not found - run where the XRootD client is available (e.g. lxplus/LPC).")
    except subprocess.CalledProcessError as e:
        sys.exit(f"`{' '.join(cmd)}` failed:\n{e.stderr.strip()}")
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


def dataset_name(subdir, strip_at):
    """Short dataset name from a sample subdir, e.g.
    'TTto4Q_TuneCP5_13p6TeV_powheg-pythia8' -> 'TTto4Q' (strip_at='_Tune')."""
    base = os.path.basename(subdir.rstrip("/"))
    if strip_at and strip_at in base:
        return base.split(strip_at)[0]
    return base


def main():
    p = argparse.ArgumentParser(
        description="Build per-dataset coffea filelists from an XRootD EOS directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="cmseos.fnal.gov",
                   help="XRootD host; also the redirector written into the file paths.")
    p.add_argument("--base", required=True,
                   help="EOS dir whose immediate subdirs are per-dataset slimmed outputs.")
    p.add_argument("-o", "--outdir", default=".", help="Output dir for the filelists.")
    p.add_argument("--only", action="append", default=[], metavar="SUBSTR",
                   help="Keep only sample subdirs whose name contains SUBSTR (repeatable).")
    p.add_argument("--tree", default="events",
                   help="Tree name recorded for each file (slimmed files use 'events').")
    p.add_argument("--strip-at", default="", metavar="MARKER",
                   help="Shorten dataset names to the part before MARKER (e.g. '_Tune'). "
                        "Default: keep the full subdir name (matches the slimmer's -o/$tag layout).")
    args = p.parse_args()

    base = args.base.rstrip("/")
    samples = [d for d in xrdfs_ls(args.host, base) if not d.endswith(".root")]
    if args.only:
        samples = [d for d in samples if any(s in os.path.basename(d) for s in args.only)]
    if not samples:
        sys.exit(f"No sample subdirs under {base} (after --only {args.only}).")

    os.makedirs(args.outdir, exist_ok=True)
    host_prefix = f"root://{args.host}/"   # trailing / + leading / of path => '//store'
    total = 0
    written = 0
    for sub in samples:
        roots = [f for f in xrdfs_ls(args.host, sub, recursive=True) if f.endswith(".root")]
        if not roots:
            print(f"  skip (no .root files): {sub}", file=sys.stderr)
            continue
        name = dataset_name(sub, args.strip_at)
        files = {f"{host_prefix}/{f.lstrip('/')}": args.tree for f in sorted(roots)}
        outpath = os.path.join(args.outdir, f"{name}.json")
        with open(outpath, "w") as fh:
            json.dump({name: {"files": files}}, fh, indent=4)
        total += len(files)
        written += 1
        print(f"  {name}: {len(files)} files -> {outpath}")

    print(f"\nWrote {written} filelist(s) ({total} files) to {args.outdir}/")


if __name__ == "__main__":
    main()

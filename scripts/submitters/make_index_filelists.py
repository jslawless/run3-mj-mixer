#!/usr/bin/env python3
"""make_index_filelists.py - list index shards on EOS, per slice.

A thin wrapper around `xrdfs <host> ls -R`, matching what make_eos_filelists.py
does for slimmed inputs. Stage 1a writes shards to <base>/index/<slice>/, so the
per-slice layout is already what this expects.

    python scripts/make_index_filelists.py --host cmseos.fnal.gov \
        --base /store/user/<you>/mixing/index -o shards.txt
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


def xrdfs_ls_r(host, base):
    cmd = shlex.split(f"xrdfs {host} ls -R {base}")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        sys.exit("xrdfs not found - run this on cmslpc/lxplus.")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"xrdfs failed: {exc.stderr.strip()}")
    return [ln.strip() for ln in out.splitlines() if ln.strip().endswith(".root")]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="cmseos.fnal.gov")
    p.add_argument("--base", required=True, help="e.g. /store/user/<you>/mixing/index")
    p.add_argument("-o", "--output", default="shards.txt")
    p.add_argument("--url", action="store_true",
                   help="write full root:// URLs instead of bare /store paths")
    args = p.parse_args(argv)

    paths = xrdfs_ls_r(args.host, args.base)
    if not paths:
        sys.exit(f"no .root files under {args.base}")
    if args.url:
        # the DOUBLE slash matters: root://host/store/... is a different path
        paths = [f"root://{args.host}/{p}" for p in paths]
    with open(args.output, "w") as f:
        f.write("\n".join(paths) + "\n")
    print(f"{len(paths)} shard(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

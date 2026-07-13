#!/usr/bin/env python3
"""make_mixing_jobs.py - Group per-slice QCD filelists into mixed condor jobs.

Unlike make_fileset.py / make_eos_filelists.py (which produce ONE fileset per
dataset), this builds jobs that each span ALL QCD HT slices, so every condor run
sees a representative slice of the full QCD spectrum - the sample the hemisphere
library is built from.

Input is one or more coffea-style per-slice filelist JSONs (or directories of
them), i.e. the output of make_eos_filelists.py:

    { "<HT-slice dataset>": { "files": { "root://.../slimmed_x.root": "events" } } }

Grouping: each job takes ``--per-slice`` files (default 5) from EVERY slice, in
sorted order. job_1 gets files 0-4 of each slice, job_2 gets files 5-9, and so
on. Jobs keep forming while EVERY slice can still supply its quota; as soon as
one slice runs out (the smallest slice sets the limit), the remaining files
from every slice are leftovers.

    n_jobs = min_over_slices( n_files_in_slice // per_slice )

Output is a coffea-style fileset JSON whose top-level keys are the jobs:

    {
        "job_1":  { "files": { path: tree, ... } },   # per_slice * n_slices files
        ...
        "job_N":  { "files": { ... } }
    }

which submit_mixer.py consumes directly. Leftover files are written to a
sidecar ``<output>_unused.json`` for bookkeeping only - they are deliberately
NOT part of the submittable fileset (they cannot form slice-balanced jobs). To make each job_k a SINGLE condor run
(rather than one sub-job per file), submit with ``-n`` >= the files-per-job the
summary prints, e.g.:

    python scripts/submit_mixer.py -i mixing_jobs.json -o /store/.../mixed \\
        --config config/config.json --wheel run3_mj_mixer-*.whl -n <files_per_job>

Usage:
    python scripts/make_mixing_jobs.py filelists/ -o mixing_jobs.json
    python scripts/make_mixing_jobs.py filelists/ --per-slice 5 --only QCD
"""

import argparse
import glob
import json
import os
import sys
from collections import OrderedDict


def _iter_json_paths(inputs):
    """Yield every JSON path from a list of files and/or directories."""
    for item in inputs:
        if os.path.isdir(item):
            yield from sorted(glob.glob(os.path.join(item, "*.json")))
        elif os.path.isfile(item):
            yield item
        else:
            print(f"  Warning: not a file or directory, skipping: {item}", file=sys.stderr)


def load_slices(inputs, only):
    """Load per-slice coffea filelists into {slice_name: [(path, tree), ...]}.

    Each input JSON may hold one or more datasets; every dataset is treated as a
    slice. Files within a slice are sorted by path for reproducible grouping.
    """
    slices = OrderedDict()
    for jpath in _iter_json_paths(inputs):
        try:
            with open(jpath) as f:
                blob = json.load(f)
        except json.JSONDecodeError as exc:
            sys.exit(f"Invalid JSON in {jpath}: {exc}")
        for name, ds in blob.items():
            if only and not any(s in name for s in only):
                continue
            files = ds.get("files", {})
            if name in slices:
                print(f"  Warning: slice '{name}' seen in multiple inputs; merging.",
                      file=sys.stderr)
                existing = dict(slices[name])
                existing.update(files)
                slices[name] = sorted(existing.items())
            else:
                slices[name] = sorted(files.items())
    # Deterministic slice order regardless of input discovery order.
    return OrderedDict(sorted(slices.items()))


def build_jobs(slices, per_slice, job_prefix):
    """Round-robin ``per_slice`` files/slice into jobs.

    Returns (jobs_ordereddict, unused_ordereddict, n_jobs, files_per_job).
    ``unused`` holds the leftover {path: tree} once the smallest slice runs
    out; it is bookkeeping, NOT a job group, and is kept out of ``jobs`` so
    submit_mixer.py never turns it into condor runs.
    """
    if not slices:
        sys.exit("No slices found (check the input paths and --only filter).")

    counts = {name: len(files) for name, files in slices.items()}
    n_jobs = min(n // per_slice for n in counts.values())

    jobs = OrderedDict()
    for j in range(1, n_jobs + 1):
        lo, hi = (j - 1) * per_slice, j * per_slice
        files = OrderedDict()
        for name, slice_files in slices.items():
            for path, tree in slice_files[lo:hi]:
                files[path] = tree
        jobs[f"{job_prefix}_{j}"] = {"files": files}

    unused = OrderedDict()
    cut = n_jobs * per_slice
    for name, slice_files in slices.items():
        for path, tree in slice_files[cut:]:
            unused[path] = tree

    files_per_job = per_slice * len(slices)
    return jobs, unused, n_jobs, files_per_job


def main():
    p = argparse.ArgumentParser(
        description="Group per-slice QCD filelists into mixed condor jobs "
                    "(job_1, job_2, ...; leftovers -> <output>_unused.json).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", metavar="JSON_OR_DIR",
                   help="Per-slice coffea filelist JSON(s) or directories of them.")
    p.add_argument("-o", "--output", default="mixing_jobs.json", help="Output JSON path.")
    p.add_argument("--per-slice", type=int, default=5,
                   help="Files taken from each slice per job.")
    p.add_argument("--only", action="append", default=[], metavar="SUBSTR",
                   help="Keep only slices whose name contains SUBSTR (repeatable). "
                        "e.g. --only QCD")
    p.add_argument("--job-prefix", default="job", help="Prefix for the job tags.")
    args = p.parse_args()

    if args.per_slice < 1:
        sys.exit("--per-slice must be >= 1.")

    slices = load_slices(args.inputs, args.only)
    jobs, unused, n_jobs, files_per_job = build_jobs(
        slices, args.per_slice, args.job_prefix
    )

    print(f"\nSlices: {len(slices)} (>= {args.per_slice} files/slice/job)")
    for name, files in slices.items():
        n = len(files)
        used = n_jobs * args.per_slice
        print(f"  {name:<62s} {n:>5} files  ({used} used, {n - used} unused)")

    if n_jobs == 0:
        print("\nWARNING: at least one slice has fewer than --per-slice files; "
              "no complete jobs were formed - every file is a leftover.",
              file=sys.stderr)

    with open(args.output, "w") as f:
        json.dump(jobs, f, indent=4)

    print(f"\nWrote {n_jobs} job(s) x {files_per_job} files -> {args.output}")

    if unused:
        stem, ext = os.path.splitext(args.output)
        unused_path = f"{stem}_unused{ext or '.json'}"
        with open(unused_path, "w") as f:
            json.dump({"unused": {"files": unused}}, f, indent=4)
        print(f"Leftover files (bookkeeping only, do NOT submit): "
              f"{len(unused)} -> {unused_path}")

    if n_jobs:
        print(f"\nSubmit each job as ONE condor run with:  -n {files_per_job}")


if __name__ == "__main__":
    main()

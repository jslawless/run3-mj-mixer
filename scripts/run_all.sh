#!/bin/bash
# Usage: bash scripts/run_all.sh <filelists-dir-or-json> <wheel> <eos-outdir> [config]
#
# The first argument may be either a DIRECTORY of fileset JSONs (every *.json in
# it is submitted) or a SINGLE fileset JSON (only that one is submitted). These
# filesets should point at SLIMMED files (tree 'events') - build them with
# scripts/make_eos_filelists.py against the slimmer's EOS output dir.
#
# The config defaults to config/config.json.
CONFIG="${4:-config/config.json}"

# Collect the fileset(s): a directory -> all *.json in it; a single file -> just it.
filesets=()
if [ -d "$1" ]; then
    while IFS= read -r f; do filesets+=("$f"); done \
        < <(find "$1" -maxdepth 1 -name '*.json' | sort)
elif [ -f "$1" ]; then
    filesets=("$1")
fi
if [ ${#filesets[@]} -eq 0 ]; then
    echo "ERROR: no .json fileset(s) for '$1' (pass a directory of JSONs or a single JSON)" >&2
fi

for i in "${filesets[@]}"; do
    filename=$(basename "$i")
    IFS='.' read -ra arrIN <<< "$filename"

    tag=${arrIN[0]}
    # --no-stitch: this layout is one FILE per condor run, so an in-job stitch
    # would see a degenerate single-file library. Stitch separately if needed.
    python scripts/submit_mixer.py -i $i -o $3/$tag --config "$CONFIG" --wheel $2 --logdir ${tag}_log --no-stitch
    condor_submit ${tag}_log/submit.sub
    sleep 2
done

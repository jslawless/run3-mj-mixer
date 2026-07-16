# How to run on the LPC Condor cluster

The mixer runs on **slimmed** files (the output of `run3-mj-slimmer`, tree
`events`) and writes `mixed_*.root` files back to EOS.

## 1. Activate your voms proxy
`voms-proxy-init --rfc --voms cms -valid 192:00`

## 2. Build the project wheel
`pip wheel . -w .`

Each condor job `pip install`s this wheel with `--no-deps` into a venv that
inherits the cvmfs LCG view's `uproot`/`awkward`/`numpy`/`boost-histogram`, so
no large PyPI downloads happen on the worker node. The mixer needs no coffea or
onnxruntime, so the job is light.

## 3. Build filelists of the slimmed inputs
Point `make_eos_filelists.py` at the slimmer's EOS output directory (the dir you
passed as the slimmer's `-o/--eosoutdir`), whose subdirs are the per-dataset
slimmed outputs:

```
python scripts/make_eos_filelists.py \
    --host cmseos.fnal.gov \
    --base /store/user/<you>/slimmed \
    -o filelists/
```

This writes one `filelists/<dataset>.json` per dataset, each recording tree
`events`. (`filelists/EXAMPLE_*.json` shows the schema; `filelists_local/` holds
a local-file example for quick tests.)

## 4. Group into mixed jobs
Each condor run should span all QCD HT slices so the hemisphere library sees the
full spectrum. Build the job-grouped fileset:

```
python scripts/make_mixing_jobs.py filelists/ --only QCD -o mixing_jobs.json
```

This makes `job_1`, `job_2`, ... (5 files/slice each by `--per-slice`). Once the
smallest slice runs out, the leftover files go to a `mixing_jobs_unused.json`
sidecar — bookkeeping only, they are not submitted (they can't form
slice-balanced jobs). The summary prints the files-per-job for the `-n` below.

## 5. Submit (mix + stitch in one condor run)
```
python scripts/submit_mixer.py -i mixing_jobs.json -o /store/user/<you>/mixed \
    --config config/config.json --wheel run3_mj_mixer-1.0.0-py3-none-any.whl \
    -n <files_per_job>
```

`-n <files_per_job>` makes each `job_k` a single condor run. `<eos-outdir>` is a
bare `/store/...` path; the job adds the `root://cmseos.fnal.gov/` redirector
automatically.

**Stitching runs inside the job by default**: after mixing its files, each job
runs `run3-mj-stitch` over its own `mixed_*.root` (the job group is a complete
slice-balanced library by construction) and delivers BOTH the mixed files and
`stitched_job_<k>.root` to EOS. The stitch parameters pass through:
`--max-distance` (0.5), `--pt-tolerance` (0.10), `--seed` (42). Pass
`--no-stitch` to skip. A stitch failure never discards the mixed outputs —
they are delivered anyway and the job exits nonzero so condor flags it.
The submitter warns if `-n` splits a job group across condor runs (each run
would stitch only a partial library).

Cross-section weighting is on by default: `submit_mixer.py` transfers
`run3-mj-pass-the-aux/mj_samples_xs.json` (the sibling aux repo) into each job,
and the mixer weights every hemisphere by `lumi * xs_pb / n_original`
(inferring the HT slice from each file's name). Pass `--xs-json` to point
elsewhere. The stitched statistics are set by these weights (usage budgets =
stochastic rounding), so normalization choices belong to THIS step.

(`scripts/run_all.sh <filelists-dir-or-json> <wheel> <eos-outdir> [config]`
still works for the one-dataset-per-job layout — it submits each fileset with
the default `-n 1` and `--no-stitch`, since a single-file library is not
meaningful.)

## Run one file locally (no condor)
```
run3-mj-mixer /path/to/slimmed_X.root config/config.json
# -> ./mixed_slimmed_X.root
```

## 6. Re-stitching from EOS (parameter scans)

The mixed files on EOS are the cheap re-run checkpoint: retuning
`--max-distance` / `--pt-tolerance` / `--seed` only needs stitching, not
re-mixing. From the LPC login node, set up an env that sees the LCG view's
uproot/awkward and the wheel (the same trick the condor jobs use):

```
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
python -m venv --system-site-packages stitch-env
source stitch-env/bin/activate
pip install --no-deps run3_mj_mixer-1.0.0-py3-none-any.whl
```

uproot in the LCG view reads `root://` URLs directly, so no local copies are
needed — build each job's URL list with `xrdfs ls` (note: `xrdcp` does NOT glob
remote paths):

```
HOST=root://cmseos.fnal.gov
MIXED=/store/user/<you>/mixed
OUT=/store/user/<you>/stitched
for k in $(seq 1 <NJOBS>); do
    FILES=$(xrdfs ${HOST#root://} ls $MIXED | grep "mixed_job_${k}_" \
            | sed "s|^|$HOST/|")
    run3-mj-stitch $FILES -o stitched_job_${k}.root \
        --max-distance 0.5 --pt-tolerance 0.10 --seed 42
    xrdcp -f stitched_job_${k}.root $HOST/$OUT/ && rm stitched_job_${k}.root
done
```

Each run prints `draws / pseudo-events / failed`; per-event provenance,
`match_distance`, `stitch_cutflow` and a `meta` tree (seed, max_distance,
pt_tolerance) are in the output for QA. The stitched files are
evaluator-compatible and go straight into the evaluator.

Notes:
- **Statistics** come from the usage budgets = stochastic rounding of the
  hemisphere weights baked in at mix time (`lumi * xs_pb / n_original`, with
  lumi = 1.0 and per-file n_original under the condor flow). The total budget
  printed at stitch start bounds the pseudo-event yield (~budget/2 minus
  failed seeds).
- `--max-distance` was tuned for the old 4-coordinate metric; with the
  (directed phi, partner eta) plane + hard pT window it is effectively looser
  and may deserve a retune.

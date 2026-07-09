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

This makes `job_1`, `job_2`, ... (5 files/slice each by `--per-slice`) plus an
`unused` group for the leftovers once the smallest slice runs out. The summary
prints the files-per-job for the `-n` below.

## 5. Submit
```
python scripts/submit_mixer.py -i mixing_jobs.json -o /store/user/<you>/mixed \
    --config config/config.json --wheel run3_mj_mixer-1.0.0-py3-none-any.whl \
    -n <files_per_job>
```

`-n <files_per_job>` makes each `job_k` a single condor run. `<eos-outdir>` is a
bare `/store/...` path; the job adds the `root://cmseos.fnal.gov/` redirector
automatically.

Cross-section weighting is on by default: `submit_mixer.py` transfers
`run3-mj-pass-the-aux/mj_samples_xs.json` (the sibling aux repo) into each job,
and the mixer weights every hemisphere by `lumi * xs_pb / n_original`
(inferring the HT slice from each file's name). Pass `--xs-json` to point
elsewhere.

(`scripts/run_all.sh <filelists-dir-or-json> <wheel> <eos-outdir> [config]`
still works for the one-dataset-per-job layout — it submits each fileset with
the default `-n 1`.)

## Run one file locally (no condor)
```
run3-mj-mixer /path/to/slimmed_X.root config/config.json
# -> ./mixed_slimmed_X.root
```

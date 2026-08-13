# How to run on the LPC Condor cluster

From a bare login node to checked pseudo-events. Assumes you already have a
directory of slimmed files on EOS.

Three logical stages — **index**, **match**, **assemble** — the first and last
split into a parallel half and a serial half:

```
slimmed ──1a──> index shards ──1b──> global index ──2──> pairs ──3a──> legs ──3b──> pseudo-events
        (condor)             (login)              (login)      (condor)     (condor)
```

`-o` is the **same output parent for every condor stage**; each writes its own
subdirectory (`index/`, `legs/`, `stitched/`).

Scripts are grouped by what they do: `scripts/submitters/` build filelists and
submit condor jobs, `scripts/monitors/` inspect what came back,
`scripts/event_displayers/` draw events.

---

## 1. Clone, side by side

The mixer locates `mj_samples_xs.json` by walking up for a
`run3-mj-pass-the-aux/` sibling, so the two repos must share a parent directory.

```bash
mkdir -p ~/nobackup/analysis && cd ~/nobackup/analysis
git clone git@github.com:jslawless/run3-mj-mixer.git
git clone git@github.com:jslawless/run3-mj-pass-the-aux.git
cd run3-mj-mixer
```

## 2. Login-node environment

Stages 1b, 2 and the QA run on the login node and need
uproot/awkward/numpy/boost-histogram. Use the same LCG view the condor jobs use,
and install this package with **`--no-deps`** — pulling PyPI wheels risks an
ABI mismatch against the view's libstdc++.

**The view is pinned**, so a run stays reproducible and a retired view breaks
loudly instead of quietly changing results. The value below is the same constant
the condor jobs use — `LCG_VERSION` / `LCG_PLATFORM` in
`src/run3_mj_mixer/condor.py`. Change it in one place and both follow.

```bash
export LCG_VIEW=/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt
source $LCG_VIEW/setup.sh

python3 -m venv --system-site-packages pkg-env
source pkg-env/bin/activate
pip install --no-deps -e .
```

## 3. Proxy and wheel

```bash
voms-proxy-init --rfc --voms cms -valid 192:00

rm -f run3_mj_mixer-*.whl      # pip wheel does not clean up old builds
pip wheel . -w .
ls run3_mj_mixer-*.whl         # expect exactly one

export WHEEL=$PWD/$(ls run3_mj_mixer-*.whl)
export OUT=/store/user/$USER/mixing
```

**Rebuild the wheel after any source change**

## 4. List the slimmed inputs

```bash
python scripts/submitters/make_eos_filelists.py \
    --host cmseos.fnal.gov \
    --base /store/user/$USER/slimmed \
    -o filelists/
```

## 5. Stage 1a — index shards (condor)

```bash
python scripts/submitters/submit_index.py \
    -i filelists/ -o $OUT \
    --config config/config.json --wheel $WHEEL \
    -n 20 --logdir batch_index --exec

condor_q
```

Omit `--exec` the first time and read `batch_index/submit.sub` plus one `.sh`.

`-n 20` is **input files per condor job**, not files per output. Stage 1a always
writes one shard per input file, so `-n 20` means one job that runs
`run3-mj-index` twenty times and produces twenty shards. It tunes job granularity
only:

- each job pays a fixed setup cost — source the LCG view, build the venv, install
  the wheel — so `-n 1` would spend most of its wall time on setup and queue
  ~5,800 jobs;
- a large `-n` means fewer, longer jobs, but a single failure re-does that many
  files;
- jobs never mix slices, so the real count rounds up per slice (with 11 slices,
  a few more jobs than `n_files / 20`).

Each shard is ~65 bytes per hemisphere and lands in
`$OUT/index/<slice>/hindex_*.root`. Stage 1a does **not** copy the event data —
that stays in the slimmed files and is looked up in 3a. Files are grouped by slice
for read locality; they no longer need to be representative samples, which was
only required when each job built its own private library.

A slimmed file that passes no events still yields a **0-row shard with a
completion marker**, so a missing shard is unambiguously a failed job.

## 6. Stage 1b — file table and global index (login node)

```bash
run3-mj-file-table --from-file filelists/ -o file_table.json --workers 8 --progress

python scripts/submitters/make_index_filelists.py \
    --host cmseos.fnal.gov --base $OUT/index -o shards.txt --url

run3-mj-index-build --from-file shards.txt -t file_table.json \
    -o global_index/ --progress
```

The file table maps `file_id` → path, dataset, weight. **This is where the
normalization is fixed**: `w_rel = xs_pb / Σ cutflow[0]` over *every file of the
slice*. The old mixer divided by one file's count, so two files of the same slice
got different weights.

Check the printed `w_rel` per slice, and note the `index_id` — it is stamped into
every downstream file.

For data: add `--mode data`, which sets `w_rel ≡ 1` and reads no cross-section
JSON at all.

> The file-table step opens every slimmed file to read one histogram. `--workers 8`
> helps; expect minutes at full scale. A file with no `cutflow` fails the step
> loudly rather than under-counting the denominator — deliberate, but it means one
> bad slimmed file blocks you until it is re-slimmed.

## 7. Stage 2 — match (login node)

```bash
run3-mj-match global_index/ -t file_table.json -o pairs/ \
    --max-distance 0.5 --pt-tolerance 0.10 --seed 42 \
    --pairs-per-chunk 100000 --progress
```

Reads **only the index**, never event data. Draws hemispheres uniformly at
random, finds each one's partner, streams `pairs/pairs_*.root`.

**Single use**: a draw consumes the seed and a match consumes the partner, so
each hemisphere appears in at most one pair and each source event in at most one
pseudo-event.

Read the printed reason breakdown before continuing:

| reason | meaning | what helps |
|---|---|---|
| `WINDOW_EMPTY` | nothing in the sample has a compatible pT | more MC |
| `ALL_CONSUMED` | the window's candidates were all used | draw order |
| `TOO_FAR_DEPLETED` | a full pool *would* have matched it | draw order |
| `TOO_FAR_INTRINSIC` | nothing near it even in a full pool | geometry / cut |

`DEPLETED` vs `INTRINSIC` is the depletion measurement. Note depletion surfaces
as `TOO_FAR`, **not** `ALL_CONSUMED`: late in the run the pT window still holds
candidates, they are just further away.

## 8. Copy the pairs to EOS

Stage 2 runs on the login node, but stages 3a/3b are condor jobs that cannot read
your scratch directory.

```bash
xrdfs cmseos.fnal.gov mkdir -p $OUT/pairs
for f in pairs/*.root; do
  xrdcp -f "$f" "root://cmseos.fnal.gov/$OUT/pairs/$(basename $f)"
done
xrdfs cmseos.fnal.gov ls $OUT/pairs | head
```

Keep the local `pairs/pairs_manifest.json` — stage 3b reads it with `--manifest`.

## 9. Stage 3a — gather (condor)

```bash
export NP=60
python scripts/submitters/submit_gather.py \
    -p root://cmseos.fnal.gov/$OUT/pairs \
    -t file_table.json -o $OUT \
    --config config/config.json --wheel $WHEEL \
    -P $NP --logdir batch_gather --exec
```

**Jobs own files, not pairs.** Each reads its slimmed files once, sequentially,
and emits the payload for every hemisphere any pair references. Walking the pair
list would re-open every file once per job — a jet branch is one basket per file,
so there is no cheaper granularity — and that cost scales as `n_jobs × n_files`.

**Check `payload.jet_branches` in `config/config.json` against your slimmer
vintage first.** Older slimmed output lacks the JES/JER variations; gather
refuses up front, naming every missing branch, rather than failing mid-job.

## 10. Stage 3b — assemble (condor)

```bash
python scripts/submitters/submit_assemble.py \
    -p root://cmseos.fnal.gov/$OUT/pairs \
    -l root://cmseos.fnal.gov/$OUT/legs \
    -t file_table.json -o $OUT \
    --config config/config.json --wheel $WHEEL \
    --manifest pairs/pairs_manifest.json \
    --gather-manifest batch_gather/gather_manifest.json \
    --logdir batch_assemble --exec
```

**No `-P` here.** The number of gather jobs must match stage 3a exactly — leg
filenames are reconstructed as `legA_<job>_<chunk>.root` over `range(P)`, so too
small silently misses shards that exist on disk, and too large references files
that never existed. `submit_gather.py` records the value it actually used in
`batch_gather/gather_manifest.json` and this reads it back. That matters because
gather clamps `P` to the file count, so the number it used may differ from the
number you asked for.

One job per pair chunk: read the chunk plus its leg shards, join on
`(pair_id, leg)`, write `$OUT/stitched/stitched_*.root`.

The join is **exact** — a missing leg aborts the job naming the pair rather than
writing a short file, because a silently dropped pseudo-event is invisible
downstream.

`xs_weight` is the **product** of both parents' weights. That is pb²/event², not
a cross section: the sample needs a global rescale, and the sums to derive it are
in the pair manifest and each output's `sum_xs_weight`.

## 11. Find and resubmit gaps

```bash
python scripts/monitors/check_stage_outputs.py legs \
    -p pairs/ -d /eos/uscms$OUT/legs \
    --gather-manifest batch_gather/gather_manifest.json
python scripts/monitors/check_stage_outputs.py stitched \
    -p pairs/ -d /eos/uscms$OUT/stitched
```

The ledger is each output file's own metadata — no database. Gaps print as a
paste-ready `queue name from` block; drop it into the relevant
`batch_*/submit.sub` and resubmit. Every stage is idempotent: names are
deterministic, so a resubmit overwrites via `xrdcp -f`.

## 12. QA

```bash
run3-mj-mixqa all \
    -i global_index/ -t file_table.json \
    -p pairs/ -l /eos/uscms$OUT/legs \
    -s /eos/uscms$OUT/stitched/stitched_*.root \
    --shards $(cat shards.txt) \
    --config config/config.json
echo "exit: $?"
```

Non-zero exit means an invariant failed. `spot` is the check that closes the loop
back to the source: it re-reads slimmed files and verifies `jet_mask` really
selects the jets the index claims — which is what licenses 3a to trust it.

The `unmatched` section is a report, not pass/fail: the reason breakdown, the
failure rate by quartile of the run (end-loaded means depletion), and
matched-vs-unmatched kinematics.

## 13. Clean up the intermediate

Once 3b verifies, the legs are throwaway — `P × Q` small files:

```bash
xrdfs cmseos.fnal.gov rm -r $OUT/legs
```

---

## Parameter scans

Re-run **stage 2 only**. It reads a ~1 GB memory-mapped index, not tens of GB of
ROOT, so there is no re-index and no ROOT re-read:

```bash
for MD in 0.3 0.5 0.7; do
  run3-mj-match global_index/ -t file_table.json -o scan_md${MD}/ --max-distance $MD
  run3-mj-mixqa pairs -p scan_md${MD}/ -t file_table.json
done
```

The match-distance distribution and the `unmatched` reason breakdown tell you
whether a cut change is buying real matches or just admitting bad ones.

## Local smoke test

```bash
export PYTHONPATH=$PWD/src
run3-mj-index <slimmed>.root config/config.json --outdir /tmp/shards
run3-mj-file-table <slimmed>.root -o /tmp/file_table.json
run3-mj-index-build /tmp/shards/*.root -t /tmp/file_table.json -o /tmp/gi
run3-mj-match /tmp/gi -t /tmp/file_table.json -o /tmp/pairs
run3-mj-gather /tmp/pairs -t /tmp/file_table.json -o /tmp/legs --job 0 --n-jobs 1
run3-mj-assemble /tmp/pairs/pairs_00000.root /tmp/legs/*.root \
    -t /tmp/file_table.json -o /tmp/stitched.root
run3-mj-mixqa all -i /tmp/gi -t /tmp/file_table.json -p /tmp/pairs \
    -l /tmp/legs -s /tmp/stitched.root
```

## Merging for the analyzer

Follow `run3-mj-analyzer/scripts/hadd_datasets.py`: hadd to a **local temp**,
then `xrdcp`. Never hadd directly to EOS.

## Notes

- **The slimmed files are a production dependency**, because 3a reads them for
  every payload branch. Retain them as long as you might add a systematic — but
  they are kept anyway, which is why this beats depending on a large intermediate.
- **`--max-distance 0.5` was tuned for an older 4-coordinate metric.** The retune
  inputs are the match-distance distribution *and* the `unmatched` breakdown: a
  pile-up just past the cut means real matches are being discarded, whereas
  `TOO_FAR_INTRINSIC` means nothing was nearby at any radius.
- **Data has no HT slices**, so every data file currently lands in one
  `unknown_slice` bucket. Weighting does not care (`w_rel ≡ 1`) but output
  organisation does; run era is the natural key and is not implemented yet.

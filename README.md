# run3-mj-mixer

Hemisphere-mixing background model for the Run 3 scouting **≥6-jet** multijet
analysis. It sits between the slimmer and the evaluator:

```
slimmer  ->  mixer  ->  evaluator
```

The mixer builds a **data-driven QCD background** by splitting real events on the
transverse thrust axis and (eventually) recombining hemispheres from different
events into synthetic multijet "pseudo-events" that carry QCD kinematics but no
genuine signal correlations. See `../mixer-slides/mixer-method.pdf` and
arXiv:1712.02538 / arXiv:2403.20241 for the method.

## Status: prototype (Steps 1–2)

This implementation covers the per-event **split + hemisphere characterization**.
The library, nearest-neighbor matching and 5→6 stitching are not implemented yet.

For every event in a slimmed file with **exactly `n_jets` jets** (config,
default 5 — others are dropped), `mix.py`:
1. finds the **transverse thrust axis** `n_T` in the x–y plane, by maximizing
   `H(φ) = Σᵢ |p⃗_T,i · n̂(φ)|` over the axis angle (grid scan on `[0, π)` +
   parabolic refine; `H` is π-periodic so the axis lives on `[0, π)`);
2. splits the event into **two hemispheres** by the plane ⊥ `n_T` (a jet joins
   the `+n_T` side when its p_T projection on `n_T` is > 0, else the `−n_T` side);
3. **sums the jet four-vectors** in each hemisphere.

## Output

The output is a faithful **copy of the slimmed input** (every branch passed
through unchanged) plus:

| object | kind | contents |
| ------ | ---- | -------- |
| `events` | TTree | + `thrust_axis_phi` (rad, `[0,π)`), `thrust` (normalized value `(0,1]`), `xs_weight` (per-event MC weight), and a `Hemisphere` collection (2/event, index 0 = `+n_T` side) |
| `thrust_axis_phi` | TH1 | histogram of the per-event thrust-axis angle |
| `hemisphere_njets` | TH1 | histogram of the per-hemisphere jet multiplicity |
| `mixer_cutflow` | TH1 | bin 1 = events read, bin 2 = events with exactly `n_jets` jets (written out) |
| `version`, `dataset` | TH1 | mixer config `metadata.version` string; matched HT-slice name |
| `meta` | TTree | scan parameters + xsec bookkeeping (`lumi`, `xs_pb`, `n_original`, `xs_weight`) |
| `cutflow`, `slimmer_version` | TH1 | passed through from the slimmer if present |

`Hemisphere` fields (per hemisphere): the summed four-vector as both cartesian
(`px, py, pz, energy`) and physics (`pt, eta, phi, mass`) form, its
thrust-projected transverse components (`pt_par` along `n_T`, `pt_perp` ⊥ to it),
`partner_eta` (the eta of the event's *other* hemisphere's summed four-vector),
the jet multiplicity `n_jets`, `side` (+1/−1), and the cross-section `weight`
(see below). These are the variables the next step (library + matching) will use.

## Cross-section weighting

Each hemisphere inherits its source event's MC weight so that a library built
from a mix of QCD HT slices reflects each slice's physical abundance:

    weight = lumi * xs_pb / n_original

- `xs_pb` — from the shared aux repo `run3-mj-pass-the-aux/mj_samples_xs.json`
  (assumed checked out **in the same parent dir as run3-mj-mixer**), keyed by
  the full HT-slice/dataset name, which the mixer infers from the input
  filename (override with `--dataset`).
- `n_original` — this file's `cutflow[0]` by default (its contribution to the
  slice denominator). For absolute per-slice normalization, pass the
  slice-summed `n_original` via `--n-original`.
- `lumi` — `--lumi` (default 1.0).

The weight is stored per hemisphere as `Hemisphere.weight` and per event as
`xs_weight`. If no cross section can be resolved the mixer warns and uses weight
1.0 (`--require-xs` to fail instead; `--no-xs-weight` to disable). `mix.py`
locates the xs JSON from the aux sibling, `$RUN3_MJ_XS_JSON`, `--xs-json`, or a
`./mj_samples_xs.json` transferred into a condor job.

## Building mixed condor jobs

`scripts/make_mixing_jobs.py` groups per-slice filelists so each condor run
spans **all** QCD HT slices (a representative sample for the library). Each job
takes `--per-slice` files (default 5) from every slice; once the smallest slice
runs out, the leftover files are written to a `<output>_unused.json` sidecar
(bookkeeping only — never submitted):

```
python scripts/make_mixing_jobs.py filelists/ --only QCD -o mixing_jobs.json
# -> job_1, job_2, ... (5 files/slice each); leftovers -> mixing_jobs_unused.json
```

The output is a coffea-style fileset consumed by `submit_mixer.py`; submit with
`-n <files_per_job>` (the summary prints it) so each `job_k` is one condor run.

The mixer accepts the jets in either the slimmer's nested `ScoutingPFJet` record
or the flat `ScoutingPFJet_*` layout, and the output stays in the slimmer's
format so pseudo-events flow straight into the existing evaluator.

## Install / run

```
pip install -e .                     # or: pip wheel . -w .
run3-mj-mixer slimmed_X.root config/config.json   # -> ./mixed_slimmed_X.root
```

Condor submission and filelist generation: see `docs/how_to_run.md`.

## Config

```json
{
    "metadata": {"version": "v1"},
    "mixer": {
        "thrust_scan_points": 180,
        "hist_bins":          90,
        "n_jets":             5
    }
}
```

- `thrust_scan_points` — grid points on `[0, π)` for the thrust-axis scan.
- `hist_bins` — bins of the `thrust_axis_phi` histogram over `[0, π)`.
- `n_jets` — keep only events with exactly this many jets (5 for the 5→6
  mixing); the selection is recorded in the `mixer_cutflow` histogram.

## Tests

```
pip install -e ".[test]"
pytest
```

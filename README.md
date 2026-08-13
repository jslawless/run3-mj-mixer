# run3-mj-mixer

Hemisphere-mixing background model for the Run 3 scouting **≥6-jet** multijet
analysis. It sits between the slimmer and the evaluator:

```
slimmer  ->  mixer  ->  evaluator
```

The mixer builds a **data-driven QCD background** by splitting events on the
transverse thrust axis and recombining hemispheres from *different* events into
synthetic multijet "pseudo-events" that carry QCD kinematics but no genuine
signal correlations. See `../mixer-slides/mixer-method.pdf` and
arXiv:1712.02538 / arXiv:2403.20241 for the method, and
`../mixer-slides/mixer-scaleout.pdf` for this architecture.

## The three stages

```
slimmed ──1a──> index shards ──1b──> global index ──2──> pairs ──3a──> legs ──3b──> pseudo-events
        (condor)             (login)              (login)      (condor)     (condor)
```

The organising idea: **matching needs almost nothing about an event** — a
hemisphere's pT, its direction, where its lost partner pointed. So build a
compact index of just that, match on the index alone, and look the real event
data up only for the pairs actually chosen.

| stage | CLI | what it does |
|---|---|---|
| **1a** index | `run3-mj-index` | one slimmed file → one ~65 B/row index shard |
| **1b** merge | `run3-mj-file-table`, `run3-mj-index-build` | shards → one memory-mapped index + the file table |
| **2** match | `run3-mj-match` | index → a list of hemisphere pairs; never reads event data |
| **3a** gather | `run3-mj-gather` | jobs own *files*; fetch the payload each pair needs |
| **3b** assemble | `run3-mj-assemble` | join legs → pseudo-events |
| **4** qa | `run3-mj-mixqa` | invariants and profiles over every stage |

`library.py` is kept as the **reference implementation** the fast matcher is
tested against match-for-match, and is never on the production path.

See [`docs/how_to_run.md`](docs/how_to_run.md) for the full condor flow.

## What each stage guarantees

**Stage 1a** finds the transverse thrust axis by maximizing
`H(φ) = Σᵢ |p⃗_T,i · n̂(φ)|` (grid scan on `[0, π)` + parabolic refine; `H` is
π-periodic so the axis lives on `[0, π)`), splits the event on the plane ⊥ `n_T`,
and sums each half's four-vectors. It records a **`jet_mask`** — which jets are
this hemisphere — computed from the *same* boolean array the summary was summed
from. Nothing downstream re-derives the split, so a jet cannot be mislabelled.

**Stage 2** matching is a hard pT window (candidate within 10% of the seed's pT)
plus nearest-neighbour in (directed φ, partner η). The query is the direction
*opposite* the seed, so the match must **naturally** point back at it. **No
transform of any kind**: jet four-vectors are never modified — no rotations,
reflections or mirroring; φ is detector-physical.

The search is bucketed but **exact**: every cell is at least `max_distance` wide,
so a Euclidean distance within the cut implies an L∞ distance within it, and the
3×3 neighbourhood provably contains every admissible candidate. It is verified
against `library.py`'s brute-force scan for identical partner *and* bit-identical
distance.

**Single use.** A draw consumes the seed and a match consumes the partner, so
each hemisphere appears in at most one pair. Since a 5-jet event yields at most
one 3-jet hemisphere, each source event lands in at most one pseudo-event — so
the output is statistically independent and can be bootstrapped directly. There
is no budget column and no re-use knob.

**Stage 3b** joins legs exactly: a missing one aborts naming the pair rather than
writing a short file.

## Weights

Per hemisphere, a **relative** weight:

```
w_rel = xs_pb[slice] / Σ cutflow[0] over EVERY file of that slice
```

The denominator is a sum over **events**, across the whole slice. (The previous
version divided by one file's count, so two files of the same slice got different
weights — a shape distortion, not just a scale one.)

Per pseudo-event, the **product** of both parents' weights:

```
xs_weight = w_rel[seed] × w_rel[match]
```

Two consequences: the product is pb²/event², so it is **not** a cross section and
the sample needs a global rescale (the sums to derive it are recorded, not left to
be rediscovered); and cross-slice pairs are **suppressed** relative to same-slice
ones by the weight ratio, which QA profiles explicitly.

**Luminosity never enters the mixer.** The analyzer multiplies at fill time, so
it can change without regenerating anything. `--mode data` sets `w_rel ≡ 1` and
reads no cross-section JSON at all.

## Rejections

A seed with no partner is recorded, not just counted. `pairs/unmatched.root`
carries one row per rejection with the reason, the candidate counts before and
after availability, the distance it would have achieved against a *full* pool,
and at which draw:

| reason | meaning |
|---|---|
| `WINDOW_EMPTY` | nothing in the sample has a compatible pT |
| `ALL_CONSUMED` | the window's candidates were all used |
| `TOO_FAR_DEPLETED` | a full pool *would* have matched it — supply |
| `TOO_FAR_INTRINSIC` | nothing near it even in a full pool — geometry |

The last two are the depletion measurement. Note depletion surfaces as
`TOO_FAR`, **not** `ALL_CONSUMED`: late in a run the pT window still holds
candidates, they are just further away.

## Output

Tree `events`, flat branch names (`ScoutingPFJet_pt`, not dotted):

| object | contents |
|---|---|
| `ScoutingPFJet_*` | the 6 stitched jets, pT-sorted, plus `_hemisphere` (0 = seed, 1 = partner) and every configured payload branch in the same order |
| `HT`, `thrust_axis_phi`, `thrust` | recomputed from the 6 jets |
| `xs_weight` | the product weight above |
| `Hemisphere_*` | 2/event, summed **by parentage** — not re-split on the new axis |
| `match_distance` | the matching distance |
| `{seed,match}_*` | full global provenance: `hemi_id`, `file_id`, `entry`, `side`, `slot`, `jet_mask`, `run`, `luminosityBlock`, `event` |
| `pair_id` | position in the pair list |
| `chunk_cutflow` | **per-chunk** counters, so `hadd` sums them correctly |

Global counters live once, in `pairs/pairs_manifest.json` — not smeared across
chunks where summing would multiply them.

## Config

```json
{
  "metadata": { "version": "v1" },
  "mixer":    { "thrust_scan_points": 180, "hist_bins": 90, "n_jets": 5 },
  "payload":  { "jet_branches": ["pt", "eta", "phi", "m",
                                 "pt_jesUp", "pt_jesDown", "pt_jerUp", "pt_jerDown",
                                 "m_jesUp", "m_jesDown", "m_jerUp", "m_jerDown"] }
}
```

`payload.jet_branches` is the **only** place the analysis schema appears. Adding
a systematic re-runs 3a/3b; the index and the matching are untouched. Missing
branches are reported up front, before anything is written.

## Install / run

```bash
pip install -e .            # local
pip wheel . -w .            # for condor
```

## Tests

```bash
python -m pytest tests/ -q
```

The load-bearing ones: `test_match.py` proves the bucketed search returns the
same partner and the same distance as `library.py`'s brute force;
`test_index.py` proves `jet_mask` selects exactly the jets the summary was summed
from; `test_index_build.py` proves the merge is byte-identical under shard
shuffling. `tests/fixtures/oracle/` holds output from the pre-rework chain,
captured before it was deleted.

## Scripts

Grouped by role:

| directory | holds |
|---|---|
| `scripts/submitters/` | filelist builders and the three condor submitters |
| `scripts/monitors/` | `check_stage_outputs.py` - what came back, what to resubmit; `plot_match_rate.py` for stage 2 and `plot_{index,gather,assemble}_jobs.py` for the condor stages, over `joblogs.py` |
| `scripts/event_displayers/` | the mixing test display and its drawing primitives |

## Diagnostics

`scripts/event_displayers/` holds `mixing_test_display.py` plus the drawing primitives, which
live here rather than in the analyzer, because they are mixer diagnostics — and
because the cross-repo import they used to need silently preferred a stale
installed copy.

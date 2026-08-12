# Global hemisphere index for `run3-mj-mixer`

## Context

The mixer currently builds its `HemisphereLibrary` in RAM inside each condor job, from ~55
slimmed files (5 per HT slice). Libraries are therefore small (~10^5 hemispheres) and
**disjoint** — every job stitches its own private library, so no pseudo-event can pair
hemispheres from different jobs. We want **one global library over all 5,820 slimmed
files**, which is ~2×10^7 three-jet hemispheres and does not fit in one job's RAM in the
current representation (~1.2 kB/hemisphere → ~29 GB).

The shape is three stages — **index**, **match**, **assemble**: build a compact index of every
hemisphere, match pairs using only that index, then look the pairs' event data up and write
pseudo-events. Index and assemble each split into a parallel half and a serial half.
**All of it lives in `run3-mj-mixer`**; the analyzer only imports from it.

Two consequences fall out of that shape and drive most of the design:

- **The intermediate `mixed_*.root` files stop existing.** They were a near-copy of the
  slimmed files with hemisphere columns bolted on. Once matching reads an index and assembly
  looks jets up by `(file, entry)`, the natural thing to look up is the *slimmed* file
  directly. That removes 66 GB from EOS, removes a redundant format that can drift from its
  source, and removes the retention obligation that a lookup architecture would otherwise
  impose on it.
- **Stage 3 is a join, not a lookup.** Every output row needs two inputs at unrelated random
  locations, and ROOT stores each jet branch as **one basket for the whole file** — there is
  no entry-level granularity, so touching a file at all costs its full basket set (measured:
  5.49 MB of jet branches for an 11,462-event slimmed file). Any job with >~10k legs touches
  essentially all 5,820 files, so per-pair lookup costs `Q × 32 GB` (≈3 TB at Q=100 jobs).
  Sorting the pair list by `seed_file` fixes one leg only; a list cannot be sorted by both
  endpoints. Stage 3 must therefore **gather by file, not by pair**.

Two further problems the staged shape does not by itself solve:

- **Time, not memory, is the wall.** The index takes 29 GB → ~800 MB of numpy columns, but
  `findPartnerHemisphere` is O(N) per query with a Python generator loop for same-event
  exclusion (`library.py:287-290`, ~4 s/query at 2×10^7). Stage 2 is a rewrite.
- **The budget model.** Cross sections span 1.2×10^7. Today ~99.95% of HT-2000 hemispheres
  get budget 0 while HT-40to70 ones are reused ~6,000× each — the ≥6-jet signal region is
  the part being discarded.

## Ground rule: no backwards compatibility

Nothing in this plan exists to keep an older reader, format, or code path working.
Concretely:

- **The old budgeting model is removed, not deprecated.** No `budget` column, no `R`
  parameter, no stochastic rounding, no prior-match blacklist, no dedup table, no flag that
  could reintroduce re-use. Single-use is the design.
- **`mixed_*.root`, `stitch.py`, `run3-mj-stitch`, the in-job stitch block and
  `make_mixing_jobs.py` are deleted**, not left as a legacy path.
- **Output branch names are chosen on their merits.** Where a name's meaning changes it is
  renamed, so a stale reader fails loudly rather than silently misreading it.
- **Histogram and counter schemas are designed correctly**, not shaped so that an existing
  summing script happens to produce the right answer.
- Downstream analyzer scripts are **migrated**, not accommodated.
- **Shared logic gets one owner.** Where a rule is currently copied across repos it is
  consolidated into the mixer and the copies deleted — no "keep both in sync" comments, and no
  `try/except ImportError` fallbacks that silently prefer a stale installed copy.

The one thing kept deliberately is `library.py`, and *not* for compatibility: it is the
reference implementation the new matcher is tested against match-for-match. It is never on
the production path.

## Decisions taken

| Decision | Choice |
|---|---|
| Intermediate files | **None.** Stage 1a reads slimmed files and writes only an index shard; `mixed_*.root` is retired outright, along with `stitch.py` and the in-job stitch path. Stage 3a looks payload up from the slimmed files. |
| Event payload | **Thin index + gather by file.** The index carries what matching needs, the hemisphere summary, provenance and `jet_mask`; all analysis payload is looked up in stage 3a as a **config list of branch names**. Rejected inlining the payload: stitched files must eventually carry JES/JER variations, the 30 per-jet slimmer branches, and (for data) trigger/prescale/MET-filter info, so an inline index would become a schema-coupled copy of the slimmed files and every new systematic would force a full index rebuild plus a stage-2 rerun. |
| Budget model | **Single-use: every hemisphere is used at most once.** Not a tunable — there is no budget column and no `R` knob. Each source event therefore appears in at most one pseudo-event, so pseudo-events are statistically independent. Yield ≈ N/2 ≈ 10^7 pseudo-events. |
| Normalization | **No lumi anywhere in the mixer.** Per hemisphere, the *relative* weight `w_rel = xs_pb[ds]/n_original_sum[ds]`; per pseudo-event, the **product** `w_rel[seed] × w_rel[match]`. Lumi and any rescale are applied downstream at fill time, as `make_hemisphere_histograms.py` already does. |
| MC vs data | **`--mode {mc,data}` on stage 1b.** In data mode `w_rel ≡ 1` and `xs_weight ≡ 1`: no xs JSON, no cutflow scan, no dataset inference for weighting. Everything else in the chain is identical. |
| Seed draw order | **Uniform random over available** — today's `returnRandomHemisphere` procedure, unchanged. Free under the grid matcher (candidates are gathered per query, not precomputed). |
| Index format | **ROOT TTree shards + per-column `.npy` merged index.** Shards ride the existing `find`/`xrdcp -f -p` delivery with zero new deps; the merged index is memmapped so the matcher gets out-of-core access and column selection. |
| Matcher structure | **numpy-only bucketed grid**, provably exhaustive (cell width ≥ `max_distance` ⇒ the 3×3 neighbourhood contains every admissible candidate), so it is bit-exact vs `library.py`, not approximate. **Build the grid only.** `scipy.spatial.cKDTree(boxsize=[2π, 0.0])` is verified to work and measures ~9× faster, but the grid already finishes stage 2 in well under the available time, and a second search path behind a flag is exactly the speculative option the ground rule above rejects — two implementations to keep in agreement for a speedup nothing needs. Recorded in the backup material; not built. |

This moves toward the group-meeting item recorded in memory (*"Lawrence: use weights
directly, not copies"*): with single-use there are no copies at all, and the cross-section
information lives entirely in a per-event weight. Its one cost: pseudo-events are no longer
all weight 1 in MC (they are in data).

## Weights

**Per hemisphere (MC), computed once in stage 1b:**

1. `dataset ← infer_dataset(filename)` — the longest `mj_samples_xs.json` key that is a
   substring of the basename (`mix.py:171-177`), the same rule as
   `submit_mixer.py::slice_of` (`:103-118`), so slice and weight cannot disagree.
2. `xs_pb ← mj_samples_xs.json[dataset]["xs_pb"]`.
3. `n_original_sum[dataset] ← Σ cutflow[0]` over **every** slimmed file of that slice,
   including cutflow-only files with no `events` tree. `cutflow[0]` is a file's pre-selection
   **event** count, so this is the slice's total original *events* — order 2×10^7 per slice,
   not its file count.
4. `w_rel = xs_pb / n_original_sum` — **a per-dataset constant**; every hemisphere from the
   same HT slice carries the identical value. E.g. HT-800to1000:
   `3033 / 2.24×10^7 = 1.4×10^-4`. Spread across the contributing slices is ~5,200×, not the
   1.2×10^7 of the raw cross sections: today's per-file denominator (`mix.py` uses *this*
   file's `cutflow[0]`) inflates every weight by ≈ n_files_in_slice ≈ 530, non-uniformly
   across slices. Fixing that denominator is one of the main reasons the file table exists.

**Per pseudo-event:** the **product** of both parents' weights.

```
xs_weight = w_rel[dataset(seed)] × w_rel[dataset(match)]
```

The pseudo-event is treated as a joint sample from two populations, so the weights multiply
the way independent draws do. It is symmetric in the two halves, which matches the fact that
neither half is privileged in the final 6-jet object.

Two consequences to be aware of, both affecting shape rather than correctness:

- **The absolute scale is no longer a cross section.** `w_rel` is pb/event, so the product is
  pb²/event² and the sample needs a global rescale to be interpreted as a yield. Since the
  mixing gives the background *shape* and the normalization is anchored separately, this is
  a bookkeeping step, not a loss — but the rescale factor must be recorded rather than
  rediscovered later.
- **Cross-slice pairs are suppressed.** The ±10% pT window deliberately lets a pair straddle
  HT slices; under the product, such a pair carries `w_A·w_B` against `w_A²` for a same-slice
  pair, i.e. down-weighted by the slice weight ratio (~3-10× for adjacent slices). Seed-only
  weighting was blind to the partner's slice; this is not. Worth watching in the closure
  plots, and a reason for QA to profile `xs_weight` split by same-slice vs cross-slice.

This reverses a previously stated rationale — `mixer-implementation.tex` lists *"no
cross-event weight products"* as an argument for the old unit-weight budget scheme — so the
change is deliberate, not an oversight.

`sqrt(w_A·w_B)` is the dimension-preserving variant (keeps pb/event, still symmetric) and is a
one-line change in `assemble.py` if the scale turns out to matter more than the joint-draw
interpretation.

**Physical normalization** stays downstream, and lumi never enters the mixer. Because the
product is not a cross section, the mixer's job is to make the rescale *derivable* rather than
to pick a convention: the pair manifest records `Σ xs_weight` over all pairs and `Σ w_rel` over
all indexed hemispheres, so whatever anchor is chosen (a control region, the 5-jet yield) can
be applied with both sums in hand.

**Data mode:** steps 1-4 are skipped and `w_rel ≡ 1`, so the product is `1 × 1 = 1` and
`xs_weight ≡ 1` — the product convention costs nothing in data. The file table still
records `dataset` for provenance, but needs no xs JSON and no cutflow scan.

**What single-use removes.** Because every hemisphere is used exactly once, in both modes:

- No stochastic rounding, no budget RNG, no `budget` column; the matcher carries a plain
  `available` bitmap.
- **No prior-match blacklist and no dedup table.** The draw consumes the seed and the match
  consumes the partner, so each draw permanently removes two hemispheres and a repeat pair is
  unrepresentable. The symmetric `_prior_matches` structure (`library.py:341-348`) has nothing
  to do, so neither is built.
- Seed retirement becomes a no-op — the seed's budget is already spent by the draw.
- Checkpoint state shrinks to a bitmap + draw position + RNG state.

And a physics gain: at `n_jets=5` each source event yields **at most one** 3-jet hemisphere
(measured — 4,842 of 6,446 events split (2,3), and never two 3-jet halves). Combined with
single-use that means **each source event appears in at most one pseudo-event**, so pseudo-events are
statistically independent and can be bootstrapped directly.

Single-use is the design, not a setting. There is **no `R` parameter** — no budget column, no
`/budget` in the weight formula, no reuse bookkeeping, no code path that could reintroduce
them. If re-use is ever wanted it is a deliberate new feature with its own validation, not a
flag left lying around.

Same-event exclusion stays: it is cheap, and at `n_jets=6` a (3,3) split gives two indexable
hemispheres per event.

## Stages

Three logical stages. **Index** and **assemble** each split into a parallel half and a serial
half, because in both cases the per-file work and the cross-file work have different shapes.

| # | Stage | Where | Reads | Writes |
|---|---|---|---|---|
| **1. Index** | | | | |
| 1a | shard | condor, one job per ~20 slimmed files | slimmed | `index/<slice>/hindex_*.root` |
| 1b | merge | one central process (LPC login node) | shards | `global_index/` (`.npy` columns + manifest + file table) |
| **2. Match** | | | | |
| 2 | match | one central process, checkpointed | `global_index/` | `pairs/pairs_NNNNN.root` + manifest |
| **3. Assemble** | | | | |
| 3a | gather | condor, jobs own **files** | slimmed + pairs | `legs/legA_<P>_<Q>.root` |
| 3b | assemble | condor, jobs own **pair ranges** | pairs + legs | `stitched/stitched_NNNNN.root` |
| **4. QA** | | | | |
| 4 | qa | central, read-only | all of the above | JSON + ROOT report |

1a and 1b are one job: *build the index*. Splitting them is purely a consequence of the
normalization denominator being slice-wide — a single-file job cannot see it, so the weight
has to be assigned once the whole sample is visible. Likewise 3a and 3b are one job — *turn
pairs into pseudo-events* — split because the data is laid out by file and the output is laid
out by pair.

### Stage 1a — index shards (`run3-mj-index`)

Reads a slimmed file, computes the transverse thrust axis, splits into hemispheres, and
writes **one ~330 kB index shard**. No ROOT copy of the event data. Output per input file
drops ~35×, and EOS goes from ~66 GB of mixed files to **~1.3 GB of index**.

This is `mix.py` minus its writer: `transverse_thrust` (`:310-354`), the `keep` cut, and
`hemisphere_fourvectors` (`:360-435`) all stay; `_passthrough_branches` (`:266-294`) and the
`ak.zip`/`uproot.recreate` block (`:530-625`) go away.

Tree `hindex`, one row per usable (3-jet) hemisphere:

- **Identity** — `file_hash` u64 (`blake2b(normalized_slimmed_path, digest_size=8)`, so a
  shard needs no global registry to be produced), `entry` i64, `side` i8, `hemi_slot` u8,
  `run` u32, `luminosityBlock` u32, `event` u64.
- **Matching** — `pt`, `eta`, `partner_eta`, `thrust_phi` f32, `n_jets` u8,
  **`jet_mask` u16**.
- **Hemisphere summary** — `phi`, `mass` f32 and the event's `thrust` f32.
- 65 B/row → ~315 kB/shard → **~1.2 GB on EOS** for 5,820 shards.

No weight is computed here. Stage 1a has no access to `n_original_sum` (a slice-wide quantity)
and does not need one, so the whole xs machinery moves to stage 1b — which is also what lets
the same shards serve both `--mode mc` and `--mode data`.

`px/py/pz/energy/pt_par/pt_perp` are **not** stored: all six are exactly recoverable from
`(pt, eta, phi, mass, thrust_phi)` — `pt_par = pt·cos(phi − thrust_phi)`,
`pt_perp = pt·sin(phi − thrust_phi)`, etc. A single `hemisphere_summary()` helper in the
mixer package owns those derivations so every consumer agrees.

`jet_mask` is the load-bearing addition: bit *i* set ⟺ jet *i* of the slimmed
`ScoutingPFJet` collection belongs to this hemisphere, recorded in the one place the mask is
exact (`mix.py:387-391`, `pos = proj > 0.0`). `popcount(jet_mask) == n_jets` is asserted at
write time.

**This kills a live silent-corruption bug.** `stitch.py:97-99` currently re-derives the mask
in float64 from float32-stored values. Measured flip probability ≈3.4×10⁻⁸/jet → ~4 per full
run. A flip usually crashes `np.stack` in `build_output` after the whole stitch has
completed, but a 2-jet seed paired with a 4-jet match concatenates to exactly 6 and
**silently mislabels a jet's parentage**, corrupting both `ScoutingPFJet_hemisphere` and the
recomputed `Hemisphere` collection. With `jet_mask` the re-derivation never happens.

Also gone: the `entry_offset` trap. `entry` is simply the slimmed entry, so there is no
post-`keep` renumbering to get wrong.

The cutflow-only early return (`mix.py:466-473`) must still emit a 0-row shard with the
marker histogram, so "missing shard" is unambiguously a job failure. Per-shard objects:
`index_cutflow`, `dataset` StrCategory, `version`, `source_path` (TObjString), and a
`complete` marker written **last** so stage 1b can reject truncated shards.

### Stage 1b — merge into the global index (`run3-mj-index-build`)

`file_table.json` (~2 MB) normalizes the per-hemisphere path string out of the index
(5,820 × ~150 B × 2×10^7 rows would be 3 GB of duplicated strings) and is the home of the
`n_original` fix:

- `file_id` from a lexicographic sort of normalized slimmed paths (strip the redirector with
  `submit_mixer.py:483`'s `re.sub`). `--extend <old>` is append-only so `hemi_id`s stay
  stable across reprocessings.
- Per dataset (**MC mode only**): `xs_pb`, **`n_original_sum` summed over ALL files of the
  slice** including cutflow-only files with no `events` tree — reuse the scan in
  `make_hemisphere_histograms.py:216-256` verbatim, including its fail-loudly-on-missing-
  cutflow guard. Reading the slimmed `cutflow` directly removes a hop. In `--mode data` this
  whole step is skipped and `w_rel ≡ 1`; `dataset` is still recorded for provenance.
- Dataset inference reuses `mix.py::infer_dataset` (`:171-177`), the same rule as
  `submit_mixer.py::slice_of` (`:103-118`), so slice, output dir and weight cannot disagree.

Canonical order: `np.lexsort` by `(file_id, entry, hemi_slot)`. The merge is then a pure
concatenation in `file_id` order — O(N), no global sort — and `hemi_id` is just the row
number (zero bytes stored), with `hemi_id_offsets` i64[5821] giving `file_id ↔ hemi_id`.
Reproducible regardless of shard arrival order, because the order is a function of the file
table alone.

Columns written as one `.npy` each (`np.lib.format.open_memmap`), ~1.6 GB total:

- core 38 B/row: `file_id` i32, `entry` i32, `side` i8, `n_jets` u8, `hemi_slot` u8,
  `jet_mask` u16, `pt`/`eta`/`phi`/`mass`/`partner_eta`/`thrust_phi`/`thrust` f32
- cms 16 B/row: `run` u32, `luminosityBlock` u32, `event` u64
- derived: `dir_phi` f32 (`directed_phi(thrust_phi, side)`), `w_rel` f32, `order_by_pt` i64

No `budget` column: it would be identically 1. The matcher carries an `available` bitmap. The matcher's resident set is core + derived ≈ **900 MB**.

`index_manifest.json` records schema version, `n_rows`, `canonical_sort`, `file_table_sha256`,
`mode` (mc/data), `rng_seed`, per-column dtype/shape, and an `index_id` hashed over
**identity columns only** so adding a derived column doesn't invalidate downstream files.
`index_id` is stamped into every pair chunk and every stitched file.

Prefer a uproot-streamed merge over `hadd` — the merge must touch every row anyway to assign
`file_id`, recompute `w_rel`/`budget` and build the offsets table. Emit `global_index.root`
behind `--also-root` as an inspectable artifact, never on the hot path.

### Stage 2 — the matcher (`run3-mj-match`)

`library.py` is **not** modified. It stays the public API `mixing_test_display.py:49-55`
imports, and becomes the reference implementation the new matcher is tested against.

Bucketed grid over `(pt_slab, dir_phi_cell, eta_cell)`:

- `pt_slab = floor(log(pt/pt_min) / log1p(pt_tolerance))` — slab width is exactly one
  tolerance unit; ~56 slabs for pt ∈ [20, 4000] at tol=0.10.
- `dir_phi_cell` width `max_distance` over [0, 2π) → 13 cells; `eta_cell` width
  `max_distance` over [-5, 5] → 20 cells. ~14,560 buckets, mean occupancy ~1,370.
- Built with `np.argsort(bucket_id)` + `np.bincount` → `bucket_start` i64[nb+1] +
  `bucket_rows` i32[N] (80 MB).

**Exactness**: Euclidean `d ≤ max_distance` implies L∞ `≤ max_distance`, so with cell width
≥ `max_distance` the 3×3 `(dir_phi, eta)` neighbourhood contains every admissible candidate.
The pT window spans `(1+tol)/(1-tol)` = 2.1 slab widths, so gather ≤4 slabs from
`searchsorted` bounds and apply the exact cut on the gathered set. No approximation — which
is what makes the equivalence test against `HemisphereLibrary` an exact-equality test.

Per query: gather ~10-40k rows, compute `d²`, `argmin`. ~100-300 µs → ~10^7 queries in
minutes, 10^8 in a few hours single-threaded.

Semantics preserved verbatim from `library.py`: hard pT window relative to the seed (`:285`),
2π-folded `_direction_delta` (`:90-93`), candidate-`eta` vs seed-`partner_eta` (`:310`),
same-event exclusion (`:286-290`), uniform-random draw over available. Under single-use the symmetric
prior-match blacklist (`:341-348`) and seed retirement (`:334-338`) have nothing to do — the
draw already consumed the seed and a match already consumed the partner — so neither is
built; `test_match.py`'s parametrised oracle still covers them via the unmodified
`HemisphereLibrary`. Distances are always recomputed with the
exact `library.py` expression, never taken from an acceleration structure (scipy's periodic
distance differs by up to 1 ULP in ~32% of cases, which flips boundary decisions and
tie-breaks). Ties resolve to the lowest row index via `np.lexsort((cand_idx, dist))`,
matching the chunked `argmin` with strict `<`.

Not accelerable, route to a brute-force path and assert the routing: `pt_tolerance=None`,
`max_distance=None`. One deliberate difference: on *failure*, `with_distance` returns `inf`
rather than the min distance over allowed candidates — the old `stitch_all` already
discarded it.

**Failed seeds and the depletion tail.** A seed whose nearest in-window candidate exceeds
`max_distance` produces no pseudo-event. Today `stitch_all` drops it and increments one
scalar (`stitch_cutflow["failed matches"]`), so there is no way to tell from the output which
hemispheres were lost or where in phase space. That is not good enough at this scale.

The effect is small but not flat. The hard pT window decomposes the pool into ~56
near-independent slabs, so depletion is per-slab, not global: within a slab, matching is 2-D
over (directed phi, partner eta) spanning ~2π × 5, a `max_distance=0.5` disk covers ~2.5% of
it, and failure sets in only when a slab is down to its last ~40 hemispheres — order
**0.01%** overall, though far worse in the thinly-populated extreme-pT slabs. The larger term
is intrinsic, not depletion: **0.13% of seeds have no in-window candidate within
`max_distance` even against a full library** (measured on a realistic 2×10^7 mock). Both terms
concentrate at high pT and forward eta — i.e. the signal-region-relevant population — so the
loss is shape-distorting rather than a flat efficiency.

The response is to measure it, not to weaken the cut (`max_distance` is what guarantees the
partner genuinely points back, and at a median match distance of 0.0023 the failures are
"nothing anywhere nearby", not "cut too tight"):

- **Differential failure accounting** — 2-D histograms of failed seeds in (pT, eta) and
  (pT, dir_phi), replacing the single scalar bin.
- **`unmatched.root`** — dump the index rows of every unmatched hemisphere at end of run. A
  few thousand rows; lets the residue be checked directly against the estimate above.
- **`--min-fill-rate`** — optional stop condition that halts when the running success rate
  drops below a threshold, rather than draining to zero. Probably unnecessary, but it is the
  lever if the measurement disagrees.
- `max_distance=None` must stay unusable in production: `library.py:326-333` **raises** rather
  than returning `None`, so an unbounded run turns the endgame into a crash.

Pair chunks streamed as written (`fsync` + `rename .tmp → .root`, then atomically rewrite
`pairs_manifest.json`), so a crash leaves *k* complete, usable chunks. Tree `pairs`,
~108 B/pair: `pair_id` i64, `{seed,match}_hemi_id` i64, `match_distance` f32,
`{seed,match}_{file_id, entry, side, slot, jet_mask}`, `{seed,match}_{run,
luminosityBlock, event}`, and a QA-replay group (`pt`, `eta`, `partner_eta`, `thrust_phi`)
that lets stage 4 re-verify the pT window and directed-phi geometry without opening the
index. Default `--pairs-per-chunk 100_000` so each stage-3b job is one chunk; target
`Q ∈ [50, 200]` chunks.

Checkpoint every `--checkpoint-every` chunks: the `available` bitmap (2.5 MB at 2×10^7),
draw position and the bit-generator state → `state_NNNNN.npz`. `--resume` reproduces
byte-identical output and **refuses** if `index_id` or any of
`(seed, max_distance, pt_tolerance)` differ.

### Stage 3a — gather (`run3-mj-gather`)

Jobs own **files**, not pairs. This inversion is what turns a ~3 TB random-access join into
a ~40 GB sequential one, at a cost independent of the number of 3b jobs.

Each of `P` jobs (default 60, ~97 slimmed files each) reads each of its files **once,
sequentially**. `hemi_id_offsets` gives every `file_id` a contiguous `hemi_id` range, so the
set of entries any pair references is a lookup, not a scan. For each referenced
`(hemi_id, leg)` it emits one record: `pair_id`, `leg` (0=seed, 1=match), and the payload
branches for that hemisphere's jets, selected by `jet_mask`.

- **Per job: ~530 MB read, ~140 MB written.** Total: **32 GB read once** (vs `Q × 32 GB` for
  per-pair lookup), ~8 GB of intermediates.
- Output is bucketed by destination: `legs/legA_<P>_<Q>.root`, so `P × Q` = 60 × 100 =
  **6,000 files of ~1.4 MB**. Same order as the 5,820 index shards. Buffering is trivial.

**The payload is a config list of branch names**, defaulting to
`ScoutingPFJet_{pt,eta,phi,m}` plus the JES/JER variations
`pt_{jesUp,jesDown,jerUp,jerDown}` / `m_{...}`, and extensible to the remaining 24 per-jet
slimmer branches, `GenJet`/`GenPart`, event-level quantities (`ScoutingRho_*`,
`ScoutingMET_*`) and, for data, trigger bits / prescales / MET filters — none of which
requires touching the index or rerunning stage 2. Reading the slimmed files directly means
the payload is not limited to what any intermediate chose to carry. Stage 3a validates every
requested branch exists in the first file it opens and fails loudly before writing anything.

### Stage 3b — assemble (`run3-mj-assemble`)

Jobs own a contiguous `pair_id` range: read one pair chunk plus its `P` leg shards (~140 MB
total), join on `(pair_id, leg)`, write one stitched file. Sequential reads only.

The join must be **exact**: assert every pair in the chunk has both legs, and fail loudly
naming the missing `pair_id`s rather than silently dropping pseudo-events. A missing leg
means a 3a job failed, and `--check` is how you find it.

Jets stack to `(n, 6)` **rectangular by construction** — `popcount(jet_mask) == n_jets` was
asserted at index time — so the ragged `np.stack` failure at `stitch.py:172-176` becomes
structurally impossible. `build_output` (`stitch.py:166-217`) is lifted into `assemble.py`
before `stitch.py` is deleted; it keeps its semantics exactly (`ak.argsort(pt, descending)`,
`transverse_thrust(..., N_SCAN=180)`, `hemisphere_fourvectors(..., pos_mask=hemi_flag==0)` —
parentage, not geometry — `side = where(pt_par >= 0, 1, -1)`). Two changes: block
`transverse_thrust` in sub-blocks of ~50k events (it calls `_proj_sum` 183× over the whole
array — ~8.8 GB of awkward intermediates at n=10^6), with a numpy fast path for the
rectangular case; and widen the provenance dtype to i64.

**Output schema is designed on its merits, not for compatibility.** Kept because they are the
right names: `ScoutingPFJet` (pt/eta/phi/m/hemisphere), `HT`, `thrust_axis_phi`, `thrust`,
`xs_weight`, the 14-field `Hemisphere` collection, `match_distance`. Every configured payload
branch is written through as `ScoutingPFJet_<name>` on the 6-jet collection in the same
pT-sorted order as the kinematics (carried through the same `argsort`), so JES/JER variations
line up with their nominal jets. Provenance: `{seed,match}_{hemi_id i64, file_id i32,
entry i32, side i8, slot u8, jet_mask u16}` and `{seed,match}_{run, luminosityBlock, event}`,
plus `pair_id`.

**Renamed, deliberately breaking:** `seed_file`/`match_file` → `seed_file_id`/`match_file_id`.
Their meaning changes from a per-invocation `sys.argv` index to a global `file_id`, and a
same-named branch with different semantics is exactly the trap that quiet compatibility
creates — a stale reader would silently misinterpret it instead of failing.

`xs_weight` is the product `w_rel[dataset(seed)] × w_rel[dataset(match)]`, resolved in-job
from `seed_file_id` and `match_file_id` through the file table (already transferred for
provenance). Both `file_id`s are on every row, so the weight stays fully recomputable and
overridable downstream — the same escape hatch `make_hemisphere_histograms.py` already uses.

**Dropped:** `seed_budget` (always 1 — dead weight), and the bare
`run`/`luminosityBlock`/`event` are deliberately *not* emitted, since a pseudo-event has two
parents and a single event id there would be a lie.

**Counters are split by scope rather than contorted to survive summing.** Per-chunk counts go
in `chunk_cutflow` and sum correctly under `hadd`. Global counts (hemispheres, draws,
unmatched) live once, in the pair manifest and the QA report — not smeared across chunks
where summing would multiply them. The old single `stitch_cutflow` mixed both scopes, and
`make_stitched_histograms.py:349-357` sums it, so global counts came out multiplied by the
chunk count. Fixing that by writing zeros into some bins would have preserved a bad design.

Idempotency needs no database for either pass: the ledger is each output file's own `meta`
tree, and both maps (3a job → file set, 3b job → `pair_id` range) are deterministic, so a
resubmit overwrites via `xrdcp -f`. `--check <eosdir>` does `xrdfs ls`, verifies
`chunk_id`/`job_id`, row counts, `index_id` and the `complete` marker, then prints missing or
short ids as a paste-ready `queue name from` block. Downstream merge follows
`hadd_datasets.py:24-29`: hadd to a local temp, then `xrdcp`, never hadd to EOS. The `P × Q`
leg shards are deletable once 3b verifies — worth an explicit `--rm-legs-after-check`.

## Files (all in `run3-mj-mixer`)

**New** under `src/run3_mj_mixer/`:

| module | role | console script |
|---|---|---|
| `hemisphere.py` | the physics: thrust axis, the split, `jet_mask`, `hemisphere_summary()`. Lifted out of today's `mix.py` and shared by 1a and 3b so the two can never disagree. | — |
| `shard.py` | stage 1a — replaces `mix.py` | `run3-mj-index` |
| `index.py` | index schema constants, shard writer/reader, column dtypes | — |
| `filetable.py` | file table build/load/validate, `file_id`, `n_original` aggregation, `w_rel`, and **sole owner of dataset inference** (see below) | `run3-mj-file-table` |
| `index_build.py` | stage 1b merge → `.npy` columns + manifest | `run3-mj-index-build` |
| `match.py` | stage 2 grid matcher, `available` bitmap, streaming writer, checkpoint/resume | `run3-mj-match` |
| `gather.py` | stage 3a; owns the payload-branch config and `jet_mask` selection | `run3-mj-gather` |
| `assemble.py` | stage 3b; owns the leg join and `build_output` | `run3-mj-assemble` |
| `qa.py` | stage 4 checks | `run3-mj-mixqa` |
| `hemi_io.py` | shared loader: reconstitute a hemisphere (summary + jets) from an index row + its slimmed file. The single API the analyzer imports. | — |

**New** under `scripts/`: `submit_index.py`, `submit_gather.py`, `submit_assemble.py`
(near-clones of `submit_mixer.py` — same `configure_batch`/`EXECUTABLE_TEMPLATE`,
`queue name from`, `transfer_output_files = ""`, the `find`-not-globstar delivery loop,
LCG-view + `--system-site-packages` venv + `--no-deps` wheel); `make_index_filelists.py`;
`check_stage_outputs.py`.

### One owner for dataset inference

The rule that decides which cross section a file gets — the longest xs-JSON key contained in
the basename — is currently **copied into three places, and they have already drifted**:

| location | behaviour on no match |
|---|---|
| `run3-mj-mixer/src/run3_mj_mixer/mix.py:171` `infer_dataset` | returns `None` |
| `run3-mj-mixer/scripts/submit_mixer.py:103` `slice_of` | falls back to an `HT-\d+` regex, then `UNKNOWN_SLICE` — never `None` |
| `run3-mj-analyzer/scripts/make_hemisphere_histograms.py:205` `infer_dataset` | returns `None` |

So a file's **output directory** and its **weight** are decided by two functions that disagree
about the failure case. Since this plan's whole point is fixing the weight denominator, that
cannot stand.

`filetable.py` becomes the single owner, with one rule and two *explicit* policies:

```python
def infer_dataset(path, xs_keys) -> str | None     # the rule; None means no match
def slice_or_unknown(path, xs_keys) -> str          # never fails: regex fallback, then UNKNOWN_SLICE
```

Weight lookup uses `infer_dataset` and fails loudly in `--mode mc`; output-directory routing
uses `slice_or_unknown`, so an unrecognized name never blocks a job. `filetable.py` also
absorbs `locate_xs_json` (`mix.py:143-168`) and the `n_original_sum` scan — ported from
`make_hemisphere_histograms.py::scan_root_inputs` (`:212-256`), keeping its
fail-loudly-on-missing-cutflow guard. All three existing copies are deleted; the analyzer
imports from the mixer.

**The dependency direction matters.** Analyzer `scripts/` is not an importable package (the
existing tests already resort to `importlib.util.spec_from_file_location` to reach
`make_mixing_jobs.build_jobs`), so "reuse it from there" is not available — and a
mixer → analyzer import would be backwards anyway. The mixer owns the shared logic; the
analyzer imports it, which is the direction that already exists.

### Relocating the display cluster into the mixer

`mixing_test_display.py` is a **mixer diagnostic**, not analysis: it builds a
`HemisphereLibrary` and calls `findPartnerHemisphere` directly. It lives in the analyzer only
for historical reasons, and that split is actively harmful — `mixing_test_display.py:48-56`
does

```python
try:
    from run3_mj_mixer.library import ...
except ImportError:                      # fall back to the sibling checkout
    sys.path.insert(0, .../run3-mj-mixer/src); from run3_mj_mixer.library import ...
```

The fallback only fires when the import **fails**, so an installed copy always wins. Verified
in the current tree: `run3-mj-analyzer/pkg-env/.../run3_mj_mixer/` exists and is what Python
resolves; its `mix.py` is 18 lines behind the checkout (pre-`--outdir`) and `stitch.py` 10.
`library.py` happens to be identical, which is the only reason this is not biting today. Memory
records the same trap costing real debugging time on 2026-07-16.

Moving these into `run3-mj-mixer/scripts/` — `mixing_test_display.py` plus the three drawing
primitives `event_display_{transverse,3d,zr}.py` — removes the cross-repo import entirely. The
move is clean: those four files are imported **only by each other** (verified), nothing else in
the analyzer uses them, and they depend on nothing but numpy/matplotlib/uproot. After the move
they import `run3_mj_mixer.hemi_io` as a plain in-package import.

**The `try/except ImportError` pattern is banned generally**, including in the analyzer scripts
that legitimately still import `hemi_io`: import directly and let it fail loudly. A silent
fallback that prefers a stale copy is worse than a missing module.

**Replaced**: `mix.py` → `shard.py` + `hemisphere.py`. Stage 1a no longer mixes anything, and
keeping the old module name to make the diff easier to read would be exactly the kind of
legacy this plan removes. `transverse_thrust`, the split, and `hemisphere_fourvectors` move to
`hemisphere.py`; the ROOT writer and `_passthrough_branches` are dropped entirely; the console
script becomes `run3-mj-index`.

**Deleted**: `stitch.py` (after `build_output` is lifted into `assemble.py`), the
`run3-mj-stitch` console script, `submit_mixer.py` in its entirety — `submit_index.py`
replaces it, rather than inheriting its `--no-stitch`/`--max-distance`/`--pt-tolerance`
passthrough — and `make_mixing_jobs.py`, since with a global library jobs no longer need
slice-balanced groups. Feed `make_eos_filelists.py` output straight to `submit_index.py -n 20`
grouped by slice for locality, which also stops discarding the `mixing_jobs_unused.json`
leftovers.

**Moved in from the analyzer** (into `run3-mj-mixer/scripts/`): `mixing_test_display.py` and
its three drawing primitives `event_display_{transverse,3d,zr}.py` — see above. Repointed at
the index + `hemi_io` while moving, since their current input (`mixed_*.root`) ceases to exist.

**Unmodified**: `library.py` — kept *only* as the oracle `match.py` is tested against, and
because the relocated `mixing_test_display.py` imports it. Never on the production path.

**Analyzer, after the move** — two scripts, both `run3-mj-analyzer`:

| script | change |
|---|---|
| `make_hemisphere_histograms.py` | reads the merged index instead of opening every mixed file (faster, and re-plottable interactively). Its `infer_dataset` and `scan_root_inputs` are deleted in favour of `run3_mj_mixer.filetable`. |
| `make_stitched_histograms.py` | two renames, since the plan no longer bends the output schema to spare it: `seed_file`/`match_file` → `seed_file_id`/`match_file_id` in `BRANCHES` if it ever reads them (it does not today), and `stitch_cutflow` → `chunk_cutflow` in the summing block at `:349-357`. |

**Also dead**: `run3-mj-analyzer/scripts/event_display.py` — the ROOT/EVE display. It targets
`mixed_*.root`, which no longer exists, and EVE was already abandoned (pip-wheel ROOT lacks
it). Delete rather than port.

### Packaging and docs

Easy to forget and wrong the moment this lands:

- **`pyproject.toml`** — `[project.scripts]` currently has exactly two entries,
  `run3-mj-mixer = run3_mj_mixer.mix:main` and `run3-mj-stitch = run3_mj_mixer.stitch:main`.
  Both disappear; six replace them (`run3-mj-index`, `-file-table`, `-index-build`, `-match`,
  `-gather`, `-assemble`, `-mixqa`). **No new runtime dependencies** — uproot / awkward / numpy
  / boost-histogram / fsspec_xrootd is still the whole list, which is what keeps the condor
  `--no-deps` LCG-inherit pattern working.
- **`docs/how_to_run.md`** is a full rewrite, not an edit. Every one of its six steps changes,
  and §4 (`make_mixing_jobs.py`) and §6 (re-stitching from EOS for parameter scans) describe
  machinery that ceases to exist. The replacement flow is: proxy → wheel → filelists →
  `submit_index.py` → `run3-mj-file-table` + `run3-mj-index-build` → `run3-mj-match` →
  `submit_gather.py` → `submit_assemble.py` → `run3-mj-mixqa`. The parameter-scan story gets
  *better* and should be written up: re-running stage 2 off the index is now the cheap
  checkpoint, with no ROOT re-read at all.
- **`README.md`** — the Status section's step-1-to-4 → code mapping, the Output table, and the
  cross-section-weighting section all describe the retired design. The "Building mixed condor
  jobs" section goes away with `make_mixing_jobs.py`.

## Verification

**Capture the oracle first.** Before any deletion, run today's `run3-mj-mixer` +
`run3-mj-stitch` on one `mixing_jobs.json` group (~55 files) and keep both the mixed files
and the stitched output as test fixtures. They are the reference for the `jet_mask`
spot-check (the mixed files carry the `Hemisphere_*` the index now computes itself) and for
the end-to-end comparison. This ordering matters — deleting `stitch.py` first would destroy
the only independent implementation to check against.

**Unit tests** (`tests/`, pytest, `tmp_path`, matching the existing style):

- `test_index.py` — schema round-trip; **`jet_mask` reproduces the hemisphere exactly** (the
  direct replacement for the mask-flip bug, which is now unrepresentable); float32
  bit-exactness through write/read; `hemisphere_summary()` derivations agree with
  `hemisphere_fourvectors` to float32; cutflow-only path emits a 0-row shard with the
  `complete` marker.
- `test_filetable.py` — `file_id` deterministic from the path sort; `--extend` append-only
  stability; slice-sum vs per-file denominator; unknown dataset fails loudly in `--mode mc`
  but is tolerated in `--mode data`; **`--mode data` yields `w_rel ≡ 1` with no xs JSON
  present at all** (i.e. the data path never touches it, rather than defaulting through it).
  Plus the two dataset-inference policies pinned against each other on the same input:
  `infer_dataset` returns `None` on no match while `slice_or_unknown` returns
  `UNKNOWN_SLICE`, and both agree on every name that *does* match — the invariant the three
  drifted copies were silently violating.
- `test_index_build.py` — **shuffle the shard list, assert byte-identical columns and
  identical `index_id`**; `hemi_id_offsets` correctness; duplicate `(file_hash, entry, slot)`
  and incomplete shards both error.
- `test_match.py` — the load-bearing one. Parametrise the 21 existing tests in
  `tests/test_library.py` over `[HemisphereLibrary, IndexLibrary]` via a fixture, so
  `test_library.py` becomes the shared oracle (42 tests). Most likely to break:
  `test_directed_phi_wrap_prefers_correct_side` (the phi seam),
  `test_pt_window_boundary_and_tolerance_override`,
  `test_prior_matches_rejected_until_exhausted`,
  `test_match_is_blacklisted_symmetrically`,
  `test_max_distance_failure_retires_seed_only`,
  `test_record_false_does_not_consume_matches`,
  `test_random_draws_deterministic_for_seed`.
  Plus a randomized differential test (`@pytest.mark.slow`) running the full draw loop
  against the real library and diffing the `(seed_idx, match_idx, distance)` sequence with
  float **equality**. The generator must put two hemispheres per event with sides ±1 sharing
  `thrust_phi` and cross-linked `partner_eta` — random sides collide in
  `_key_to_index[(event_id, side)]` and make the *oracle* retire the wrong hemisphere.
  Include exact coordinate duplicates (tie-break), `thrust_phi` at 0∓/π∓, `|eta| > 3`, and
  pT values placing a window boundary on a slab boundary.
- `test_gather.py` — `jet_mask` selects exactly the right entries from a slimmed file; a
  requested payload branch missing from the input fails loudly before any write; leg records
  round-trip; destination bucketing covers every `pair_id` exactly once.
- `test_assemble.py` — hand-built 2-pair chunk + leg shards → stitched file; branch set
  equals a golden list so a schema regression is a test failure; bit-identical jets;
  `Hemisphere_n_jets == [3,3]`; a **deliberately missing leg raises** rather than dropping
  the pseudo-event; payload branches follow the same pT `argsort` as the kinematics.

**QA stage** (`run3-mj-mixqa`, subcommands `index`/`spot`/`pairs`/`legs`/`stitched`):

- index: `n_rows` == Σ shard cutflows == Σ file-table counts (three independent counts);
  `(file_id, entry, hemi_slot)` unique; **global dedup on `run:lumi:event`** with expected
  multiplicity from the manifest (1 at n_jets=5 — measured, never two 3-jet hemispheres per
  event — 2 at n_jets=6), reporting duplicates *with their `file_id`s*;
  `popcount(jet_mask) == n_jets`; `slot == 0 ⟺ side == +1`; `thrust_phi ∈ [0, π)`.
- spot (`--n-spot 2000`): open the slimmed file, apply `jet_mask`, and assert the selected
  jets reproduce the stored `pt`/`eta`/`phi`/`mass`/`partner_eta`/`n_jets`/`side` and the
  captured-oracle mixed file's `Hemisphere_*[slot]` **bit-for-bit** (`np.array_equal`, not
  `allclose`). This is what validates `jet_mask` as the authoritative hemisphere definition.
- pairs: no same-event pairs by **both** `(file_id, entry)` and `(run, lumi, event)`;
  `|seed_pt - match_pt| ≤ tol·seed_pt` for 100%; `0 ≤ d ≤ max_distance` for 100% plus the
  distribution (a pile-up at `max_distance` means the retune flagged at
  `docs/how_to_run.md:158-160` is due); no same-side landings
  (`cos(dir_phi_seed - dir_phi_match) < 0`); **every `hemi_id` appears at most once across
  both legs of the entire pair list** — the single-use invariant, and the check that catches a
  double-spend; `Σ legs == 2·n_pairs`; no duplicate pairs via `np.unique` on `min·N + max`.
- pairs, weights: profile `xs_weight` **split by same-slice vs cross-slice pairs**, since the
  product suppresses the latter by the slice weight ratio; report the cross-slice fraction per
  pT band and `Σ xs_weight` (the manifest's rescale input).
- pairs, matching efficiency: `n_pairs·2 + n_unmatched == n_hemispheres` (nothing vanished
  unaccounted for); the **failure rate profiled in pT and eta**, with the high-pT tail called
  out explicitly, since that is where the loss concentrates and where it matters most.
- legs (3a): every `pair_id × leg` appears exactly once across the `P × Q` shards —
  `Σ leg rows == 2 × n_pairs`, with missing/duplicated `pair_id`s named.
- stitched: `n_events == pair_hi - pair_lo` (**no pseudo-event silently dropped by an
  incomplete join**); jets a bitwise permutation of the leg shards'; exactly 6 jets with
  `Σ ScoutingPFJet_hemisphere == 3`; **`Hemisphere_n_jets == [3,3]`**; `HT == Σ pt`; every
  configured payload branch present with the same 6-jet length as the kinematics.

**End-to-end**: run the full chain on the captured-oracle group (~55 files, ~10^5
hemispheres) and compare against the saved `run3-mj-stitch` output. This is a **sanity
check, not a compatibility target** — the draw bookkeeping differs, so byte-identity is not
expected or wanted. The binding correctness test is `test_match.py`'s match-for-match equality
against `library.py`; this pass checks the pair population looks the same in aggregate. Then scale to all 5,820 files.

## Notes

- **The slimmed files become the production dependency.** Stage 3a reads them for every
  payload branch, so they must be retained for as long as you might add a systematic — but
  they are kept anyway, which is exactly why this is better than depending on a 66 GB
  intermediate.
- **Data has no HT slices, so `dataset` needs a different meaning in `--mode data`.** Data
  filenames will not match any `mj_samples_xs.json` key, so `slice_or_unknown` returns
  `UNKNOWN_SLICE` for every file and all output lands in one directory. Weighting does not care
  (`w_rel ≡ 1`), but provenance and output organisation do — run era (2022C, 2022D, …) is the
  natural key. Settle this when data first goes through; it is a `filetable.py` argument, not a
  schema change, so it does not block the MC build.
- **`--mode data` composes cleanly with single-use.** Since no budget depends on the weight,
  the only difference between modes is the value of one column, and data pseudo-events come
  out with `xs_weight = 1 × 1 = 1` — genuinely unit-weight, which is what the earlier "mixed MC
  behaves like mixed data" rationale was reaching for.
- **Payload branches must exist in the slimmed files.** Nothing the slimmer drops is
  recoverable. A `qa` check that the configured payload list is a subset of the slimmed
  schema, run once before a full production, is cheap insurance.

## Deferred / out of scope

- **Closure plots** (njets/HT/lead pT/trijet mass; 5-jet MC vs mixed MC vs exactly-6-jet MC)
  remain the user's own task, but they are what validates the capped-budget change — the
  weighted pseudo-event HT spectrum should reproduce the direct QCD MC spectrum.
- **Bootstrap by pseudo-event is valid** — each source event contributes at most one
  hemisphere and each hemisphere is used at most once, so pseudo-events are statistically
  independent. It would cease to hold under any re-use scheme (effective size becomes ~N_hemispheres,
  understating the uncertainty by ~√R), which is a further reason single-use is the design
  rather than a default. `seed_hemi_id`/`match_hemi_id` are emitted regardless, so hemisphere-level
  resampling stays available.
- **The evaluator is still blocked on mixed input** (patched `_batched` comb-solver models
  peak ~3.4 GB at the hardwired `batch_size=2048`, `evaluate.py:403`, vs condor's 4 GB).
  This work multiplies the pseudo-event count substantially, which makes that fix more
  urgent, but it is a separate repo and a separate change.
- `--max-distance 0.5` was tuned for the older 4-coordinate metric. The retune inputs are the
  pair-level distance distribution **and** the differential failure map from the first full
  run — together they say whether the cut is discarding real matches or only flagging seeds
  that had no neighbour at any radius.

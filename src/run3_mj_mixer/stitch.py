#!/usr/bin/env python3
"""stitch.py - production 5 -> 6 hemisphere mixing (Step 4 of the method).

Loads every 3-jet hemisphere of the input ``mixed_*.root`` file(s) into a
``HemisphereLibrary`` (usage budgets = stochastic rounding of each event's
weight; RNG seeded by --seed, default 42), then repeatedly:

  1. draws a random still-available hemisphere (budget - 1),
  2. finds its nearest partner in the 4-coordinate matching space, skipping
     same-event and already-produced pairs (both directions) and exhausted
     hemispheres; a returned partner costs budget - 1,
  3. if the best candidate is farther than --max-distance the seed is
     retired (no output event; a deterministic search would only fail
     again), otherwise the partner's jets are reflected phi -> -phi (the
     ONLY transform - phi is physical) and the two 3-jet hemispheres become
     one 6-jet pseudo-event,

until no budget is left anywhere. All pseudo-events carry weight 1.

Output tree ``events`` is evaluator-compatible (the evaluator only requires
the jet collection): ``ScoutingPFJet`` with pt/eta/phi/m (pT-sorted) plus a
per-jet ``hemisphere`` flag (0 = seed's jets, 1 = partner's), recomputed
``HT``, ``thrust_axis_phi``/``thrust`` (recomputed from the 6 jets), a
recomputed 2-per-event ``Hemisphere`` collection (same fields as the mixer's,
weight 1), and per-event matching diagnostics: ``match_distance`` (the
goodness of match), ``seed_file``/``seed_entry``/``seed_side`` and
``match_file``/``match_entry``/``match_side`` provenance. A
``match_distance`` histogram, a ``stitch_cutflow``, a ``version`` histogram
and a ``meta`` tree (seed, max_distance, counters) are also written.

Usage:
    run3-mj-stitch mixed_A.root mixed_B.root ... -o stitched.root
    run3-mj-stitch mixed_*.root -o stitched.root --max-distance 0.5 --seed 42
"""

import argparse
import math
import sys

import awkward as ak
import boost_histogram as bh
import numpy as np
import uproot

from run3_mj_mixer import __version__
from run3_mj_mixer.library import Hemisphere, HemisphereLibrary
from run3_mj_mixer.mix import hemisphere_fourvectors, transverse_thrust

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BRANCHES = [
    "ScoutingPFJet_pt", "ScoutingPFJet_eta", "ScoutingPFJet_phi",
    "ScoutingPFJet_m",
    "thrust_axis_phi",
    "Hemisphere_pt", "Hemisphere_eta", "Hemisphere_partner_eta",
    "Hemisphere_n_jets", "Hemisphere_side", "Hemisphere_weight",
]

N_SCAN = 180  # thrust grid points for the pseudo-event, as in the mixer


def _bar(iterable=None, **kw):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kw) if iterable is not None else tqdm(**kw)


def load_library(paths, tree_name, lib):
    """Fill ``lib`` with every 3-jet hemisphere of every input file.

    ``event_id`` is ``(file_index, entry)``. Returns the number of usable
    input events (files without an events tree are skipped with a note).
    """
    n_events = 0
    for f_idx, path in enumerate(paths):
        with uproot.open(path) as f:
            if tree_name not in f:
                print(f"[skip] {path}: no '{tree_name}' tree")
                continue
            a = f[tree_name].arrays(BRANCHES, library="np")
        n = len(a["thrust_axis_phi"])
        n_events += n
        desc = f"loading {path.split('/')[-1][:40]}"
        for i in _bar(range(n), unit="evt", desc=desc):
            phi_t = float(a["thrust_axis_phi"][i])
            for h in range(len(a["Hemisphere_n_jets"][i])):
                if int(a["Hemisphere_n_jets"][i][h]) != 3:
                    continue
                side = int(a["Hemisphere_side"][i][h])
                proj = a["ScoutingPFJet_pt"][i] * np.cos(
                    a["ScoutingPFJet_phi"][i] - phi_t)
                mask = proj > 0.0 if side > 0 else proj <= 0.0
                lib.addHemisphere(
                    a["ScoutingPFJet_pt"][i][mask],
                    a["ScoutingPFJet_eta"][i][mask],
                    a["ScoutingPFJet_phi"][i][mask],
                    a["ScoutingPFJet_m"][i][mask],
                    thrust_phi=phi_t,
                    partner_eta=float(a["Hemisphere_partner_eta"][i][h]),
                    pt=float(a["Hemisphere_pt"][i][h]),
                    eta=float(a["Hemisphere_eta"][i][h]),
                    side=side,
                    weight=float(a["Hemisphere_weight"][i][h]),
                    event_id=(f_idx, i),
                )
    return n_events


def stitch_all(lib, max_distance):
    """Draw / match / stitch until the library is exhausted.

    Returns (records, counters): ``records`` holds the per-pseudo-event jet
    lists and provenance, ``counters`` the draw/fill/fail tallies and the
    distances of failed matches.
    """
    jets_pt, jets_eta, jets_phi, jets_m, jets_hemi = [], [], [], [], []
    distance, seed_id, match_id = [], [], []
    n_draw = n_fill = n_fail = 0
    budget0 = lib.total_budget

    bar = _bar(total=budget0, unit="use", desc="stitching")
    while True:
        seed = lib.returnRandomHemisphere()
        if seed is None:
            break
        n_draw += 1
        match, dist = lib.findPartnerHemisphere(
            seed, exclude_event_id=seed.event_id,
            max_distance=max_distance, with_distance=True,
        )
        if match is None:
            n_fail += 1
        else:
            n_fill += 1
            # The one allowed transform: undo the query reflection.
            p_phi = (-match.jet_phi + math.pi) % (2.0 * math.pi) - math.pi
            jets_pt.append(np.concatenate([seed.jet_pt, match.jet_pt]))
            jets_eta.append(np.concatenate([seed.jet_eta, match.jet_eta]))
            jets_phi.append(np.concatenate([seed.jet_phi, p_phi]))
            jets_m.append(np.concatenate([seed.jet_mass, match.jet_mass]))
            jets_hemi.append(np.array([0, 0, 0, 1, 1, 1], dtype=np.int32))
            distance.append(dist)
            seed_id.append((*seed.event_id, seed.side))
            match_id.append((*match.event_id, match.side))
        if bar is not None:
            bar.n = budget0 - lib.total_budget
            bar.refresh()
    if bar is not None:
        bar.close()

    records = (jets_pt, jets_eta, jets_phi, jets_m, jets_hemi,
               distance, seed_id, match_id)
    counters = {"budget0": budget0, "draws": n_draw,
                "filled": n_fill, "failed": n_fail}
    return records, counters


def build_output(records):
    """Vectorized assembly of the output record from the stitched lists:
    pT-sorted jets, recomputed thrust and Hemisphere collection, weights 1."""
    (jets_pt, jets_eta, jets_phi, jets_m, jets_hemi,
     distance, seed_id, match_id) = records

    pt = ak.Array(np.stack(jets_pt))
    eta = ak.Array(np.stack(jets_eta))
    phi = ak.Array(np.stack(jets_phi))
    m = ak.Array(np.stack(jets_m))
    hemi_flag = ak.Array(np.stack(jets_hemi))

    order = ak.argsort(pt, axis=1, ascending=False)
    pt, eta, phi, m = pt[order], eta[order], phi[order], m[order]
    hemi_flag = hemi_flag[order]

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    sum_pt = ak.to_numpy(ak.sum(pt, axis=1))
    phi_t, thrust = transverse_thrust(px, py, sum_pt, N_SCAN)
    # Hemisphere collection by PARENTAGE (index 0 = the seed's three jets,
    # index 1 = the matched partner's), not by re-splitting on the recomputed
    # axis - soft jets can migrate across the new axis, and matching QA needs
    # the stitched halves. pt_par/pt_perp are still w.r.t. the new axis;
    # side is the actual sign of each half's axis projection.
    hemi = hemisphere_fourvectors(pt, eta, phi, m, phi_t,
                                  pos_mask=(hemi_flag == 0))
    n = len(sum_pt)
    hemi["side"] = np.where(hemi["pt_par"] >= 0.0, 1, -1).astype(np.int32)
    hemi["weight"] = np.ones((n, 2), dtype=np.float32)

    seed_id = np.asarray(seed_id, dtype=np.int32)
    match_id = np.asarray(match_id, dtype=np.int32)
    out = {
        "ScoutingPFJet": ak.from_regular(ak.zip({
            "pt": pt, "eta": eta, "phi": phi, "m": m,
            "hemisphere": hemi_flag,
        })),
        "HT": ak.Array(sum_pt.astype(np.float32)),
        "thrust_axis_phi": ak.Array(phi_t),
        "thrust": ak.Array(thrust),
        "xs_weight": ak.Array(np.ones(n, dtype=np.float32)),
        "Hemisphere": ak.from_regular(ak.zip(hemi)),
        "match_distance": ak.Array(np.asarray(distance, dtype=np.float32)),
        "seed_file": ak.Array(seed_id[:, 0]),
        "seed_entry": ak.Array(seed_id[:, 1]),
        "seed_side": ak.Array(seed_id[:, 2]),
        "match_file": ak.Array(match_id[:, 0]),
        "match_entry": ak.Array(match_id[:, 1]),
        "match_side": ak.Array(match_id[:, 2]),
    }
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Stitch 3-jet hemispheres of mixed_*.root files into "
        "6-jet pseudo-events (weight 1, evaluator-compatible output).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="mixed ROOT file(s)")
    parser.add_argument("-o", "--output", default="stitched.root",
                        help="output ROOT file")
    parser.add_argument("--tree", default="events", help="input tree name")
    parser.add_argument("--max-distance", type=float, default=0.5,
                        help="reject matches farther than this in the "
                        "4-coordinate space (the seed is retired)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for budgets and random draws")
    parser.add_argument("--chunk-size", type=int, default=100_000,
                        help="output write chunk (events)")
    args = parser.parse_args()

    lib = HemisphereLibrary(seed=args.seed)
    n_events = load_library(args.inputs, args.tree, lib)
    print(f"library: {len(lib):,} 3-jet hemispheres from {n_events:,} events "
          f"in {len(args.inputs)} file(s) | total budget {lib.total_budget:,} "
          f"(pT_max {lib.pt_max:.1f} GeV, seed {args.seed})")
    if len(lib) == 0:
        sys.exit("No 3-jet hemispheres found - nothing to stitch.")

    records, counters = stitch_all(lib, args.max_distance)
    print(f"draws: {counters['draws']:,} | pseudo-events: "
          f"{counters['filled']:,} | failed (d > {args.max_distance:g}): "
          f"{counters['failed']:,}")
    if counters["filled"] == 0:
        sys.exit("No pseudo-events produced (max-distance too tight?).")

    out = build_output(records)
    n = counters["filled"]

    dist_hist = bh.Histogram(bh.axis.Regular(100, 0.0, args.max_distance),
                             storage=bh.storage.Double())
    dist_hist.fill(ak.to_numpy(out["match_distance"]))

    cutflow = bh.Histogram(
        bh.axis.StrCategory(["hemispheres", "total budget", "draws",
                             "pseudo-events", "failed matches"]),
        storage=bh.storage.Double(),
    )
    cutflow.view()[:] = [len(lib), counters["budget0"], counters["draws"],
                         counters["filled"], counters["failed"]]

    version_hist = bh.Histogram(bh.axis.StrCategory([__version__]),
                                storage=bh.storage.Double())
    version_hist.view()[0] = 1.0

    with uproot.recreate(args.output) as f:
        f.mktree("events", {name: arr.type for name, arr in out.items()})
        for lo in range(0, n, args.chunk_size):
            hi = min(lo + args.chunk_size, n)
            f["events"].extend({k: v[lo:hi] for k, v in out.items()})
        f["match_distance"] = dist_hist
        f["stitch_cutflow"] = cutflow
        f["version"] = version_hist
        f.mktree("meta", {"rng_seed": np.dtype(np.int64),
                          "max_distance": np.dtype(np.float64),
                          "n_hemispheres": np.dtype(np.int64),
                          "budget0": np.dtype(np.int64)})
        f["meta"].extend({
            "rng_seed": np.array([args.seed], dtype=np.int64),
            "max_distance": np.array([args.max_distance], dtype=np.float64),
            "n_hemispheres": np.array([len(lib)], dtype=np.int64),
            "budget0": np.array([counters["budget0"]], dtype=np.int64),
        })
    print(f"Done.  {n:,} pseudo-events  ->  {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""mixing_test_display.py - one-event test of the hemisphere-mixing library.

Loads EVERY 3-jet hemisphere of a ``mixed_*.root`` file into a
``run3_mj_mixer.HemisphereLibrary`` (progress bar), picks a seed event (CLI
argument, or a random 2+3-split event), finds the nearest-neighbor partner
for its 3-jet hemisphere (progress bar; same-event pairing excluded), and
stitches the seed's 3-jet hemisphere + the matched 3-jet hemisphere into a
6-jet pseudo-event. Saves ONE PDF (<outdir>/plots/evt<N>_mixing_display.pdf)
with 13 slides: matching summary, then THREE events per perspective - the
seed's source event, the match's source event, and the stitched pseudo-event
- in the transverse view, the longitudinal z-r view, and 3D in two camera
views.

Matching/stitching convention: the directed-phi query selects a hemisphere
that NATURALLY points opposite the seed, so there is nothing to transform -
no reflections, no rotations; every jet four-vector is used exactly as
recorded (phi is physical in the detector). The pseudo-event's thrust axis
and thrust value are recomputed from its 6 jets for the display.

Usage:
    python scripts/mixing_test_display.py mixed_X.root          # random seed
    python scripts/mixing_test_display.py mixed_X.root 42
    python scripts/mixing_test_display.py mixed_X.root 42 --outdir plots/
"""

import argparse
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from event_display_3d import draw as draw_3d
from event_display_transverse import draw as draw_transverse
from event_display_zr import draw as draw_zr

# In-package import, no try/except fallback. The old form was
#
#     try:    from run3_mj_mixer.library import ...
#     except ImportError:  sys.path.insert(<sibling checkout>); import again
#
# and the fallback only fires when the import FAILS - so a stale pip-installed
# copy always won silently. It did, in July 2026: the analyzer's pkg-env held a
# mixer copy several commits behind and this script ran the old matcher without
# erroring. Now that this lives in the mixer, there is one path to the code.
sys.path.insert(0, str(_SCRIPTS.parents[1] / "src"))
from run3_mj_mixer.library import (
    Hemisphere, HemisphereLibrary, directed_phi, query_direction,
)

BRANCHES = [
    "ScoutingPFJet_pt", "ScoutingPFJet_eta", "ScoutingPFJet_phi",
    "ScoutingPFJet_m",
    "thrust_axis_phi", "thrust",
    "Hemisphere_px", "Hemisphere_py", "Hemisphere_pz", "Hemisphere_pt",
    "Hemisphere_eta", "Hemisphere_n_jets", "Hemisphere_side",
    "Hemisphere_partner_eta", "Hemisphere_weight",
]


def _bar(iterable, **kw):
    return tqdm(iterable, **kw) if tqdm is not None else iterable


def hemisphere_jet_mask(jet_pt, jet_phi, thrust_phi, side):
    """Which jets belong to the ``side`` hemisphere (mixer convention:
    +n_T side <=> pT projection on the thrust axis > 0)."""
    proj = jet_pt * np.cos(jet_phi - thrust_phi)
    return proj > 0.0 if side > 0 else proj <= 0.0


def transverse_thrust(jet_pt, jet_phi, n_scan=720):
    """(axis angle in [0, pi), thrust value) of one jet set - display-grade
    grid scan, same definition as the mixer's."""
    px, py = jet_pt * np.cos(jet_phi), jet_pt * np.sin(jet_phi)
    angs = np.linspace(0.0, math.pi, n_scan, endpoint=False)
    h = np.abs(np.outer(np.cos(angs), px) + np.outer(np.sin(angs), py)).sum(axis=1)
    k = int(np.argmax(h))
    return float(angs[k]), float(h[k] / jet_pt.sum())


def main():
    parser = argparse.ArgumentParser(
        description="Load 3-jet hemispheres into a HemisphereLibrary, match "
        "one seed event, and save before/after event displays as PDFs."
    )
    parser.add_argument("input", help="mixed ROOT file")
    parser.add_argument("event", nargs="?", type=int, default=None,
                        help="seed event number (default: random 2+3 event)")
    parser.add_argument("--tree", default="events")
    parser.add_argument("--outdir", default=".", help="where the PDFs go")
    args = parser.parse_args()

    with uproot.open(args.input) as f:
        if args.tree not in f:
            sys.exit(f"No '{args.tree}' tree in {args.input}")
        a = f[args.tree].arrays(BRANCHES, library="np")
    n_events = len(a["thrust_axis_phi"])

    # ---------------------------------------------------------- the library
    lib = HemisphereLibrary()
    for i in _bar(range(n_events), unit="evt", desc="loading hemispheres"):
        phi_t = float(a["thrust_axis_phi"][i])
        for h in range(len(a["Hemisphere_n_jets"][i])):
            if int(a["Hemisphere_n_jets"][i][h]) != 3:
                continue
            side = int(a["Hemisphere_side"][i][h])
            mask = hemisphere_jet_mask(
                a["ScoutingPFJet_pt"][i], a["ScoutingPFJet_phi"][i],
                phi_t, side,
            )
            lib.addHemisphere(
                a["ScoutingPFJet_pt"][i][mask], a["ScoutingPFJet_eta"][i][mask],
                a["ScoutingPFJet_phi"][i][mask], a["ScoutingPFJet_m"][i][mask],
                thrust_phi=phi_t,
                partner_eta=float(a["Hemisphere_partner_eta"][i][h]),
                pt=float(a["Hemisphere_pt"][i][h]),
                eta=float(a["Hemisphere_eta"][i][h]),
                side=side, weight=float(a["Hemisphere_weight"][i][h]),
                event_id=i,
            )
    print(f"library: {len(lib):,} 3-jet hemispheres from {n_events:,} events "
          f"(pt_max = {lib.pt_max:.1f} GeV)")

    # ------------------------------------------------------------- the seed
    has3 = [i for i in range(n_events)
            if 3 in a["Hemisphere_n_jets"][i].tolist()]
    if not has3:
        sys.exit("No 2+3-split events in this file.")
    if args.event is None:
        seed_i = random.choice(has3)
        print(f"randomly selected seed event {seed_i} (of {n_events})")
    else:
        seed_i = args.event
        if not 0 <= seed_i < n_events:
            sys.exit(f"Event {seed_i} out of range (file has {n_events}).")
        if seed_i not in has3:
            sys.exit(f"Event {seed_i} is not a 2+3 split - no 3-jet "
                     "hemisphere to seed the mixing with.")

    phi_t = float(a["thrust_axis_phi"][seed_i])
    h3 = int(np.argmax(a["Hemisphere_n_jets"][seed_i] == 3))
    side3 = int(a["Hemisphere_side"][seed_i][h3])
    mask3 = hemisphere_jet_mask(
        a["ScoutingPFJet_pt"][seed_i], a["ScoutingPFJet_phi"][seed_i],
        phi_t, side3,
    )
    seed = Hemisphere(
        a["ScoutingPFJet_pt"][seed_i][mask3],
        a["ScoutingPFJet_eta"][seed_i][mask3],
        a["ScoutingPFJet_phi"][seed_i][mask3],
        a["ScoutingPFJet_m"][seed_i][mask3],
        pt=float(a["Hemisphere_pt"][seed_i][h3]), thrust_phi=phi_t,
        partner_eta=float(a["Hemisphere_partner_eta"][seed_i][h3]),
        eta=float(a["Hemisphere_eta"][seed_i][h3]),
        side=side3, event_id=seed_i,
    )

    # ------------------------------------------------------------ the match
    # The directed-phi query is the direction OPPOSITE the seed, so the
    # returned hemisphere naturally points back at it - nothing is ever
    # reflected or rotated (phi is physical in the detector).
    match, dist = lib.findPartnerHemisphere(
        seed, exclude_event_id=seed_i, with_distance=True,
        progress=True, chunk_size=max(1000, len(lib) // 100),
    )
    print(f"matched event {match.event_id}: d = {dist:.4f} | "
          f"pt {seed.pt:.1f} -> {match.pt:.1f} GeV | "
          f"directed phi {directed_phi(seed.thrust_phi, seed.side):.3f} -> "
          f"{directed_phi(match.thrust_phi, match.side):.3f} "
          f"(query {query_direction(seed):.3f})")
    print(f"eta match: match eta {match.eta:+.2f} ~ seed partner_eta "
          f"{seed.partner_eta:+.2f} (the pseudo-event keeps the seed "
          "event's boost)")

    # ------------------------------------------------- stitch: 3 + 3 -> 6 jets
    # NO transform of any kind: the matching selected a hemisphere that
    # NATURALLY points opposite the seed, and every jet four-vector is used
    # exactly as recorded.
    p_phi = match.jet_phi
    ppx = float((match.jet_pt * np.cos(p_phi)).sum())
    ppy = float((match.jet_pt * np.sin(p_phi)).sum())
    spx = float(a["Hemisphere_px"][seed_i][h3])
    spy = float(a["Hemisphere_py"][seed_i][h3])
    if ppx * spx + ppy * spy > 0.0:
        print("WARNING: matched partner points along the seed hemisphere's "
              "side (very poor axis match?) - displayed as-is.")
    print("stitch: no transform - matched jets used exactly as recorded")

    mix_pt = np.concatenate([seed.jet_pt, match.jet_pt])
    mix_eta = np.concatenate([seed.jet_eta, match.jet_eta])
    mix_phi = np.concatenate([seed.jet_phi, p_phi])
    new_phi_t, new_thrust = transverse_thrust(mix_pt, mix_phi)

    ppz = float((match.jet_pt * np.sinh(match.jet_eta)).sum())
    hemis_px = np.array([spx, ppx])
    hemis_py = np.array([spy, ppy])
    hemis_pz = np.array([float(a["Hemisphere_pz"][seed_i][h3]), ppz])
    proj = hemis_px * math.cos(new_phi_t) + hemis_py * math.sin(new_phi_t)
    sides = np.where(proj > 0.0, 1, -1)
    if sides[0] == sides[1]:  # degenerate; force opposite for the display
        sides = np.array([1, -1])

    after = {
        "ScoutingPFJet_pt": mix_pt, "ScoutingPFJet_eta": mix_eta,
        "ScoutingPFJet_phi": mix_phi,
        "thrust_axis_phi": new_phi_t, "thrust": new_thrust,
        "Hemisphere_px": hemis_px, "Hemisphere_py": hemis_py,
        "Hemisphere_pz": hemis_pz,
        "Hemisphere_n_jets": np.array([3, 3]), "Hemisphere_side": sides,
    }
    before = {k: a[k][seed_i] for k in BRANCHES}
    match_i = int(match.event_id)
    before_match = {k: a[k][match_i] for k in BRANCHES}

    # ------------------------------------------- one PDF, six labeled slides
    outdir = Path(args.outdir)
    after_src = f"pseudo-event (seed {seed_i} + match {match.event_id})"
    pdf_path = outdir / f"plots/evt{seed_i}_mixing_display.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Two 3D camera angles as (elevation, azimuth offset from the event's
    # thrust axis); view 1 is the event_display_3d default.
    VIEWS = {1: (15.0, 60.0), 2: (50.0, 150.0)}

    # Fixed references so arrow sizes are comparable across every slide:
    # the seed 3-jet hemisphere's pT (transverse) and |p| (3D and z-r, where
    # the projected arrow length equals the full |p|), plus one shared 3D box
    # size covering the longest hemisphere arrow of ANY of the three
    # displayed events (seed source, match source, pseudo-event).
    spz = float(a["Hemisphere_pz"][seed_i][h3])
    seed_p3 = math.sqrt(spx**2 + spy**2 + spz**2)
    p_all = [
        math.sqrt(float(a["Hemisphere_px"][evt][k]) ** 2
                  + float(a["Hemisphere_py"][evt][k]) ** 2
                  + float(a["Hemisphere_pz"][evt][k]) ** 2)
        for evt in (seed_i, match_i)
        for k in range(len(a["Hemisphere_n_jets"][evt]))
    ] + [math.sqrt(ppx**2 + ppy**2 + ppz**2)]
    box_lim = max(1.1, 1.12 * max(p_all) / seed_p3)

    # The z-r reference: boosted hemispheres can carry |p| far above the
    # seed's, so scale those slides to the largest |p| (jet or hemisphere)
    # of ANY displayed event - shared across the three z-r pages, and
    # everything stays inside the rim.
    def _zr_max(evt):
        jpt = np.asarray(evt["ScoutingPFJet_pt"], dtype=float)
        jeta = np.asarray(evt["ScoutingPFJet_eta"], dtype=float)
        hp = np.hypot(
            np.hypot(np.asarray(evt["Hemisphere_px"], dtype=float),
                     np.asarray(evt["Hemisphere_py"], dtype=float)),
            np.asarray(evt["Hemisphere_pz"], dtype=float),
        )
        return max(float((jpt * np.cosh(jeta)).max()), float(hp.max()))

    zr_ref = 1.02 * max(_zr_max(before), _zr_max(before_match), _zr_max(after))

    def page(pdf, draw, event, source, label, three_d, view=None,
             index=seed_i, figsize=None, **draw_kw):
        if figsize is None:
            figsize = (7.5, 7.5) if three_d else (6.5, 6.5)
        fig = plt.figure(figsize=figsize)
        if three_d:
            ax = fig.add_subplot(projection="3d")
            draw(event, index, source, ax, **draw_kw)
            elev, azim_off = view
            # Camera anchored to the SEED event's axis for every slide: the
            # pseudo-event's recomputed axis (or the match event's own) can
            # differ / wrap across the [0, pi) boundary, which would spin the
            # camera and make unchanged hemispheres appear to swap sides.
            ax.view_init(elev=elev, azim=math.degrees(phi_t) + azim_off)
            fig.subplots_adjust(left=-0.15, right=1.15, bottom=-0.12, top=0.98)
        else:
            ax = fig.add_subplot()
            draw(event, index, source, ax, **draw_kw)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        fig.text(0.5, 0.965, label, ha="center", va="center",
                 fontsize=15, fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)

    def summary_page(pdf):
        h2 = 1 - h3  # the seed event's other (left-behind) hemisphere
        nj2 = int(a["Hemisphere_n_jets"][seed_i][h2])
        pt2 = float(a["Hemisphere_pt"][seed_i][h2])
        eta2 = float(a["Hemisphere_eta"][seed_i][h2])
        pe2 = float(a["Hemisphere_partner_eta"][seed_i][h2])
        side2 = int(a["Hemisphere_side"][seed_i][h2])
        head = (f"{'':26s}{'pT [GeV]':>9s}"
                f"{'dir. phi':>10s}{'eta':>8s}{'partner eta':>13s}")
        rows = [
            f"{'seed 3-jet hemisphere':26s}{seed.pt:9.1f}"
            f"{directed_phi(seed.thrust_phi, seed.side):10.3f}"
            f"{seed.eta:+8.2f}{seed.partner_eta:+13.2f}",
            f"{'   (query, opposite)':26s}{'':9s}"
            f"{query_direction(seed):10.3f}"
            f"{seed.partner_eta:+8.2f}{'':13s}",
            f"{'matched 3-jet hemisphere':26s}{match.pt:9.1f}"
            f"{directed_phi(match.thrust_phi, match.side):10.3f}"
            f"{match.eta:+8.2f}{match.partner_eta:+13.2f}",
            f"{f'left-behind {nj2}-jet hemi.':26s}{pt2:9.1f}"
            f"{directed_phi(seed.thrust_phi, side2):10.3f}"
            f"{eta2:+8.2f}{pe2:+13.2f}",
        ]
        fname = Path(args.input).name
        if len(fname) > 62:  # keep the tail: dataset + file number live there
            fname = "..." + fname[-59:]
        body = "\n".join([
            f"file:          {fname}",
            f"seed event:    {seed_i}",
            f"matched event: {match.event_id}",
            f"library:       {len(lib):,} 3-jet hemispheres",
            f"pT window:     |{match.pt:.1f} - {seed.pt:.1f}| / {seed.pt:.1f}"
            f" = {abs(match.pt - seed.pt) / seed.pt:.3f}  (tolerance 0.10)",
            f"distance:      d = {dist:.4f}  in (dir. phi, eta)",
            "",
            head,
            "-" * len(head),
            *rows,
            "",
            "matching = hard pT window (candidate within 10% of the seed pT)",
            "+ Euclidean NN in (dir. phi, eta).",
            "stitch transform: NONE. The matched hemisphere NATURALLY points",
            "opposite the seed (that is what the query selects); every jet",
            "four-vector is used exactly as recorded.",
            "dir. phi = the transverse direction the hemisphere points along",
            "(axis phi, + pi on the -n_T side; 2pi-periodic); the query is",
            "the direction the seed's lost partner pointed (side built in).",
            "eta: the match's own eta ~ the seed's partner eta, so the",
            "pseudo-event keeps the seed event's longitudinal boost.",
        ])
        fig = plt.figure(figsize=(7.5, 7.5))
        fig.text(0.5, 0.88, f"Hemisphere matching — seed event {seed_i}",
                 ha="center", fontsize=15, fontweight="bold")
        fig.text(0.07, 0.78, body, ha="left", va="top",
                 family="monospace", fontsize=9.5)
        pdf.savefig(fig)
        plt.close(fig)

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(pdf_path) as pdf:
        summary_page(pdf)
        page(pdf, draw_transverse, before, args.input,
             "BEFORE mixing — seed event (transverse)", False, pt_ref=seed.pt)
        page(pdf, draw_transverse, before_match, args.input,
             "BEFORE mixing — matched event (transverse)", False,
             index=match_i, pt_ref=seed.pt)
        page(pdf, draw_transverse, after, after_src,
             "AFTER mixing — pseudo-event (transverse)", False, pt_ref=seed.pt)
        zr_size = (7.0, 4.6)
        page(pdf, draw_zr, before, args.input,
             "BEFORE mixing — seed event (z–r)", False,
             figsize=zr_size, p_ref=zr_ref)
        page(pdf, draw_zr, before_match, args.input,
             "BEFORE mixing — matched event (z–r)", False,
             index=match_i, figsize=zr_size, p_ref=zr_ref)
        page(pdf, draw_zr, after, after_src,
             "AFTER mixing — pseudo-event (z–r)", False,
             figsize=zr_size, p_ref=zr_ref)
        for v, view in VIEWS.items():
            page(pdf, draw_3d, before, args.input,
                 f"BEFORE mixing — seed event (3D, view {v})", True, view,
                 p_ref=seed_p3, box_lim=box_lim)
            page(pdf, draw_3d, before_match, args.input,
                 f"BEFORE mixing — matched event (3D, view {v})",
                 True, view,
                 index=match_i, p_ref=seed_p3, box_lim=box_lim)
            page(pdf, draw_3d, after, after_src,
                 f"AFTER mixing — pseudo-event (3D, view {v})", True, view,
                 p_ref=seed_p3, box_lim=box_lim)
    print(f"saved {pdf_path} (13 slides: matching summary, then seed / "
          "matched / pseudo-event in transverse, z-r, and 3D x 2 views)")


if __name__ == "__main__":
    main()

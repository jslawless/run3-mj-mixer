#!/usr/bin/env python3
"""event_display_zr.py - longitudinal (z-r) event display for mixed files.

Draws one event of a run3-mj-mixer ``mixed_*.root`` file in the z-r plane
(beam axis horizontal, radial/transverse magnitude vertical - the view that
shows the longitudinal boost the eta matching preserves):

  * every ScoutingPFJet as an arrow from the origin at (p_z, p_T), tinted by
    the hemisphere it belongs to (same multiplicity colors as the transverse
    display: 1 jet = light green, 2 = blue, 3 = orange, 4 = red);
  * each hemisphere's summed momentum (from the stored Hemisphere_px/py/pz)
    as a thicker, darker arrow at (p_z, sqrt(px^2+py^2)), labelled with its
    jet multiplicity;
  * the beam axis along r = 0, and dotted eta guide rays at
    eta = -2, -1, 0, +1, +2 labelled at the rim;
  * a |p| scale: the reference momentum touches the semicircle, with dotted
    half-rings at 3/4, 1/2 and 1/4 of it, each labelled in GeV.

In this projection an arrow's length is the full |p| of the object
(|(p_z, p_T)| = |p|), so ``p_ref`` here is directly comparable to the 3D
display's.

Usage:
    python scripts/event_display_zr.py mixed_X.root          # random event
    python scripts/event_display_zr.py mixed_X.root 42
    python scripts/event_display_zr.py mixed_X.root 42 -o evt42.png
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from event_display_transverse import FALLBACK_FILL, FILL_COLORS, darker

# The z-r view needs jet eta on top of the transverse display's branches.
BRANCHES = [
    "ScoutingPFJet_pt", "ScoutingPFJet_eta", "ScoutingPFJet_phi",
    "thrust_axis_phi", "thrust",
    "Hemisphere_px", "Hemisphere_py", "Hemisphere_pz",
    "Hemisphere_n_jets", "Hemisphere_side",
]

ETA_GUIDES = (-2.0, -1.0, 0.0, 1.0, 2.0)


def _eta_direction(eta):
    """Unit (z, r) direction of pseudorapidity ``eta``."""
    theta = 2.0 * math.atan(math.exp(-eta))
    return math.cos(theta), math.sin(theta)


def draw(event, index, source, ax, p_ref=None):
    """``p_ref`` overrides the arrow scale: that |p| maps to the semicircle
    radius (arrows above it stick out past it) and a dashed guide half-ring
    is added at 5/4 of it. Default: the event's largest |p| touches the rim."""
    phi_t = float(event["thrust_axis_phi"])

    jet_pt = np.asarray(event["ScoutingPFJet_pt"], dtype=float)
    jet_eta = np.asarray(event["ScoutingPFJet_eta"], dtype=float)
    jet_phi = np.asarray(event["ScoutingPFJet_phi"], dtype=float)
    hemi_px = np.asarray(event["Hemisphere_px"], dtype=float)
    hemi_py = np.asarray(event["Hemisphere_py"], dtype=float)
    hemi_pz = np.asarray(event["Hemisphere_pz"], dtype=float)
    hemi_nj = np.asarray(event["Hemisphere_n_jets"], dtype=int)
    hemi_side = np.asarray(event["Hemisphere_side"], dtype=int)

    jet_pz = jet_pt * np.sinh(jet_eta)
    jet_p = np.hypot(jet_pz, jet_pt)
    hemi_pr = np.hypot(hemi_px, hemi_py)
    hemi_p = np.hypot(hemi_pz, hemi_pr)

    ref = float(p_ref) if p_ref is not None else max(jet_p.max(), hemi_p.max())
    scale = 1.0 / ref

    # Which hemisphere each jet belongs to (mixer convention: +n_T side <=>
    # pT projection on the thrust axis > 0), for the jet tint.
    proj = jet_pt * np.cos(jet_phi - phi_t)
    jet_side = np.where(proj > 0.0, 1, -1)
    side_fill = {int(s): FILL_COLORS.get(int(nj), FALLBACK_FILL)
                 for s, nj in zip(hemi_side, hemi_nj)}

    # Rim + |p| reference half-rings (labels on the left, away from eta tags).
    ax.add_patch(Arc((0, 0), 2.0, 2.0, theta1=0.0, theta2=180.0,
                     color="black", lw=1.2, zorder=3))
    for frac in (0.75, 0.5, 0.25):
        ax.add_patch(Arc((0, 0), 2 * frac, 2 * frac, theta1=0.0, theta2=180.0,
                         color="silver", ls=(0, (2, 3)), lw=0.9, zorder=2))
        ax.text(0.0, frac + 0.025, f"{frac * ref:.0f} GeV", color="black",
                ha="center", va="bottom", fontsize=7, zorder=10)
    if p_ref is not None:
        ax.add_patch(Arc((0, 0), 2.5, 2.5, theta1=0.0, theta2=180.0,
                         color="silver", ls="--", lw=0.9, zorder=2))
        ax.text(0.0, 1.275, f"{1.25 * ref:.0f} GeV", color="black",
                ha="center", va="bottom", fontsize=7, zorder=10)
    ax.text(0.0, 1.025, f"{ref:.0f} GeV", color="black", ha="center",
            va="bottom", fontsize=8, zorder=10)

    # Beam axis along r = 0.
    ax.plot([-1.0, 1.0], [0.0, 0.0], color="black", lw=1.0, zorder=2)
    ax.text(1.07, 0.0, "+z", ha="left", va="center", fontsize=9)
    ax.text(-1.07, 0.0, "-z", ha="right", va="center", fontsize=9)

    # Eta guide rays, labelled just outside the rim.
    for eta in ETA_GUIDES:
        gz, gr = _eta_direction(eta)
        ax.plot([0.0, gz], [0.0, gr], color="silver", ls=(0, (1, 3)),
                lw=0.9, zorder=1)
        label = "0" if eta == 0 else f"{eta:+.0f}"
        ax.text(1.09 * gz, 1.09 * gr, rf"$\eta={label}$", color="gray",
                ha="center", va="center", fontsize=7, zorder=10)

    # Jets: arrows at (pz, pT), tinted by their hemisphere.
    for pz, pt, s in zip(jet_pz, jet_pt, jet_side):
        color = darker(side_fill.get(int(s), FALLBACK_FILL), 0.75)
        ax.arrow(0.0, 0.0, scale * pz, scale * pt, color=color, width=0.006,
                 head_width=0.035, head_length=0.05,
                 length_includes_head=True, zorder=5)

    # Hemisphere summed-momentum arrows + multiplicity labels at the tips.
    for pz, pr, nj in zip(hemi_pz, hemi_pr, hemi_nj):
        fill = FILL_COLORS.get(int(nj), FALLBACK_FILL)
        ax.arrow(0.0, 0.0, scale * pz, scale * pr, color=darker(fill),
                 alpha=0.55, width=0.012, head_width=0.055, head_length=0.07,
                 length_includes_head=True, zorder=6)
        # Label beside the shaft (perpendicular, upward), not past the tip -
        # a radial offset collides with the eta guide labels when the
        # hemisphere points along a guide ray.
        tip = np.array([scale * pz, scale * pr])
        n = tip / max(np.linalg.norm(tip), 1e-9)
        perp = np.array([-n[1], n[0]])
        if perp[1] < 0.0:
            perp = -perp
        ax.text(*(0.85 * tip + 0.11 * perp),
                f"{nj} jet{'s' if nj != 1 else ''}",
                color=darker(fill), ha="center", va="center", fontsize=9,
                zorder=10)

    split = "+".join(str(n) for n in sorted(hemi_nj))
    ax.set_title(
        f"{Path(source).name}\n"
        f"event {index} | T = {float(event['thrust']):.3f} | {split} split",
        fontsize=8,
    )
    lim = 1.45 if p_ref is not None else 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.12, lim)
    ax.set_aspect("equal")
    ax.axis("off")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Longitudinal (z-r) event display for mixed_*.root files."
    )
    parser.add_argument("input", help="mixed ROOT file")
    parser.add_argument("event", nargs="?", type=int, default=None,
                        help="event number (default: pick one at random)")
    parser.add_argument("--tree", default="events")
    parser.add_argument("-o", "--output", default=None,
                        help="save to this image file instead of showing")
    args = parser.parse_args()

    import uproot

    with uproot.open(args.input) as f:
        if args.tree not in f:
            sys.exit(f"No '{args.tree}' tree in {args.input}")
        tree = f[args.tree]
        n = tree.num_entries
        index = args.event
        if index is None:
            import random
            index = random.randrange(n)
            print(f"randomly selected event {index} (of {n})")
        if not 0 <= index < n:
            sys.exit(f"Event {index} out of range: {args.input} has {n} events.")
        arrays = tree.arrays(BRANCHES, entry_start=index,
                             entry_stop=index + 1, library="np")
    event = {k: v[0] for k, v in arrays.items()}

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    draw(event, index, args.input, ax)
    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"saved {args.output}")
    else:
        plt.show()

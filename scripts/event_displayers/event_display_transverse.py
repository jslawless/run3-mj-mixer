#!/usr/bin/env python3
"""event_display_transverse.py - transverse-plane event display for mixed files.

Draws one event of a run3-mj-mixer ``mixed_*.root`` file, looking down the
beam axis:

  * every ScoutingPFJet as a dark gray arrow from the center, pointing along
    its transverse momentum (px, py), length proportional to pT;
  * the transverse thrust axis as a red line through the center with its
    endpoints on the circle;
  * the two hemisphere half-disks (split by the plane perpendicular to the
    thrust axis) tinted by their jet multiplicity:
        1 jet = light green, 2 = blue, 3 = orange, 4 = red;
  * each hemisphere's summed transverse momentum (read from the stored
    Hemisphere_px/py, not recomputed) as a thicker arrow in a darker version
    of its half-disk color;
  * a pT scale: the largest-pT object (jet or hemisphere sum) touches the
    circle, labelled in GeV at the rim, with light gray dotted rings at 3/4,
    1/2 and 1/4 of that pT, each labelled.

Usage:
    python scripts/event_display_transverse.py mixed_X.root          # random event
    python scripts/event_display_transverse.py mixed_X.root 42
    python scripts/event_display_transverse.py mixed_X.root 42 -o evt42.png
"""

import argparse
import math
import random
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import uproot
from matplotlib.patches import Circle, Wedge

# Half-disk tint per hemisphere jet multiplicity (0/5-jet splits fall back to
# gray). The summed-momentum arrow uses a darker version of the same color.
FILL_COLORS = {1: "lightgreen", 2: "tab:blue", 3: "orange", 4: "red"}
FALLBACK_FILL = "lightgray"

BRANCHES = [
    "ScoutingPFJet_pt", "ScoutingPFJet_phi",
    "thrust_axis_phi", "thrust",
    "Hemisphere_px", "Hemisphere_py", "Hemisphere_n_jets", "Hemisphere_side",
]


def darker(color, factor=0.55):
    r, g, b = mcolors.to_rgb(color)
    return (factor * r, factor * g, factor * b)


def load_event(path, tree_name, index):
    """Read one event; ``index=None`` picks one at random (and prints it)."""
    with uproot.open(path) as f:
        if tree_name not in f:
            sys.exit(f"No '{tree_name}' tree in {path} (empty mixed file?)")
        tree = f[tree_name]
        n = tree.num_entries
        if index is None:
            index = random.randrange(n)
            print(f"randomly selected event {index} (of {n})")
        if not 0 <= index < n:
            sys.exit(f"Event {index} out of range: {path} has {n} events.")
        arrays = tree.arrays(
            BRANCHES, entry_start=index, entry_stop=index + 1, library="np"
        )
    return {k: v[0] for k, v in arrays.items()}, index


def draw(event, index, source, ax, pt_ref=None):
    """``pt_ref`` overrides the arrow scale: that pT maps to the circle radius
    (arrows above it stick out past the circle) and a dashed guide ring is
    added at 5/4 of it. Default: the event's largest pT touches the circle."""
    phi_t = float(event["thrust_axis_phi"])
    nx, ny = math.cos(phi_t), math.sin(phi_t)

    jet_pt = np.asarray(event["ScoutingPFJet_pt"], dtype=float)
    jet_phi = np.asarray(event["ScoutingPFJet_phi"], dtype=float)
    hemi_px = np.asarray(event["Hemisphere_px"], dtype=float)
    hemi_py = np.asarray(event["Hemisphere_py"], dtype=float)
    hemi_nj = np.asarray(event["Hemisphere_n_jets"], dtype=int)
    hemi_side = np.asarray(event["Hemisphere_side"], dtype=int)
    hemi_pt = np.hypot(hemi_px, hemi_py)

    # One momentum scale for every arrow: the reference pT touches the circle
    # (the event's largest pT unless pt_ref pins it externally).
    ref = float(pt_ref) if pt_ref is not None else max(jet_pt.max(), hemi_pt.max())
    scale = 1.0 / ref

    # Hemisphere half-disks: each spans +-90 deg around +-n_T.
    phi_t_deg = math.degrees(phi_t)
    for nj, side in zip(hemi_nj, hemi_side):
        center = phi_t_deg if side > 0 else phi_t_deg + 180.0
        fill = FILL_COLORS.get(int(nj), FALLBACK_FILL)
        ax.add_patch(Wedge((0, 0), 1.0, center - 90.0, center + 90.0,
                           facecolor=fill, alpha=0.25, edgecolor="none",
                           zorder=0))
        # Multiplicity label near the rim, offset from the hemisphere axis so
        # it doesn't sit on the thrust-line endpoint.
        lang = math.radians(center + 28.0)
        ax.text(1.13 * math.cos(lang), 1.13 * math.sin(lang),
                f"{nj} jet{'s' if nj != 1 else ''}",
                color=darker(fill), ha="center", va="center", fontsize=10)

    ax.add_patch(Circle((0, 0), 1.0, fill=False, color="black", lw=1.2,
                        zorder=3))

    # pT reference rings at 3/4, 1/2, 1/4 of the reference pT.
    for frac in (0.75, 0.5, 0.25):
        ax.add_patch(Circle((0, 0), frac, fill=False, color="silver",
                            ls=(0, (2, 3)), lw=0.9, zorder=2))
        ax.text(0.0, frac + 0.025, f"{frac * ref:.0f} GeV", color="black",
                ha="center", va="bottom", fontsize=7, zorder=10)
    if pt_ref is not None:
        # Guide ring outside the circle at 5/4 of the reference pT.
        ax.add_patch(Circle((0, 0), 1.25, fill=False, color="silver",
                            ls="--", lw=0.9, zorder=2))
        ax.text(0.0, 1.275, f"{1.25 * ref:.0f} GeV", color="black",
                ha="center", va="bottom", fontsize=7, zorder=10)

    # The reference pT, near the rim where the object carrying it touches
    # (with pt_ref: the object closest to it in pT), offset a little off its
    # direction so it clears the thrust-line endpoint.
    all_pt = np.concatenate([jet_pt, hemi_pt])
    all_phi = np.concatenate([jet_phi, np.arctan2(hemi_py, hemi_px)])
    phi_ref = float(all_phi[int(np.argmin(np.abs(all_pt - ref)))])
    lang = phi_ref - math.radians(14.0)
    ax.text(1.14 * math.cos(lang), 1.14 * math.sin(lang),
            f"{ref:.0f} GeV", color="black", ha="center", va="center",
            fontsize=8, zorder=10)

    # Hemisphere boundary (the split line, perpendicular to the thrust axis):
    # light gray dashed, so the halves read even when both have the same color.
    ax.plot([ny, -ny], [-nx, nx], color="silver", ls="--", lw=1.3, zorder=2)

    # Thrust axis: red line through the center, endpoints on the circle.
    ax.plot([-nx, nx], [-ny, ny], color="red", lw=2, zorder=4,
            solid_capstyle="round")

    # Jets: dark gray pT arrows from the center.
    for pt, phi in zip(jet_pt, jet_phi):
        dx, dy = scale * pt * math.cos(phi), scale * pt * math.sin(phi)
        ax.arrow(0.0, 0.0, dx, dy, color="dimgray", width=0.006,
                 head_width=0.035, head_length=0.05,
                 length_includes_head=True, zorder=5)

    # Hemisphere summed-pT arrows, straight from the stored four-vectors.
    # Slightly transparent so the jet arrows stay visible underneath.
    for px, py, nj in zip(hemi_px, hemi_py, hemi_nj):
        fill = FILL_COLORS.get(int(nj), FALLBACK_FILL)
        ax.arrow(0.0, 0.0, scale * px, scale * py, color=darker(fill),
                 alpha=0.55, width=0.012, head_width=0.055, head_length=0.07,
                 length_includes_head=True, zorder=6)

    split = "+".join(str(n) for n in sorted(hemi_nj))
    ax.set_title(
        f"{Path(source).name}\n"
        f"event {index} | T = {float(event['thrust']):.3f} | {split} split",
        fontsize=8,
    )
    lim = 1.45 if pt_ref is not None else 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transverse-plane event display for mixed_*.root files."
    )
    parser.add_argument("input", help="mixed ROOT file")
    parser.add_argument("event", nargs="?", type=int, default=None,
                        help="event number (default: pick one at random)")
    parser.add_argument("--tree", default="events")
    parser.add_argument("-o", "--output", default=None,
                        help="save to this image file instead of showing")
    args = parser.parse_args()

    event, index = load_event(args.input, args.tree, args.event)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    draw(event, index, args.input, ax)
    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"saved {args.output}")
    else:
        plt.show()


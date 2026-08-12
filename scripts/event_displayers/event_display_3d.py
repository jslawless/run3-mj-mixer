#!/usr/bin/env python3
"""event_display_3d.py - 3D event display for run3-mj-mixer output.

Draws one event of a ``mixed_*.root`` file in 3D (matplotlib, mouse-rotatable
when shown interactively):

  * every ScoutingPFJet as a dark gray arrow from the origin along its full
    momentum (px, py, pz from pT, eta, phi), length proportional to |p|;
  * the transverse thrust axis as a red line in the z = 0 plane;
  * the hemisphere-splitting plane (contains the beam axis, perpendicular to
    the thrust axis) as a translucent gray slab;
  * each hemisphere's summed momentum (from the stored Hemisphere_px/py/pz)
    as a thicker, slightly transparent arrow in a darker version of the same
    multiplicity colors as the transverse display
    (1 jet = green, 2 = blue, 3 = orange, 4 = red), labelled at the tip;
  * a thin gray beam axis and a light circle marking the transverse plane.

The camera starts looking roughly along the splitting plane, so the two
hemispheres separate left/right; drag to rotate.

Usage:
    python scripts/event_display_3d.py mixed_X.root          # random event
    python scripts/event_display_3d.py mixed_X.root 42
    python scripts/event_display_3d.py mixed_X.root 42 -o evt42_3d.png
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

FILL_COLORS = {1: "lightgreen", 2: "tab:blue", 3: "orange", 4: "red"}
FALLBACK_FILL = "lightgray"

BRANCHES = [
    "ScoutingPFJet_pt", "ScoutingPFJet_eta", "ScoutingPFJet_phi",
    "thrust_axis_phi", "thrust",
    "Hemisphere_px", "Hemisphere_py", "Hemisphere_pz",
    "Hemisphere_n_jets", "Hemisphere_side",
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


def draw(event, index, source, ax, p_ref=None, box_lim=None):
    """``p_ref`` overrides the arrow scale: that |p| maps to unit length
    (arrows above it stick out past the reference circle). Default: the
    event's largest |p| reaches unit length. ``box_lim`` pins the axes
    half-width - pass the same value for displays that must share a zoom."""
    phi_t = float(event["thrust_axis_phi"])
    nx, ny = math.cos(phi_t), math.sin(phi_t)
    tx, ty = -ny, nx  # in-plane direction of the splitting plane

    jet_pt = np.asarray(event["ScoutingPFJet_pt"], dtype=float)
    jet_eta = np.asarray(event["ScoutingPFJet_eta"], dtype=float)
    jet_phi = np.asarray(event["ScoutingPFJet_phi"], dtype=float)
    jx = jet_pt * np.cos(jet_phi)
    jy = jet_pt * np.sin(jet_phi)
    jz = jet_pt * np.sinh(jet_eta)
    jet_p = np.sqrt(jx**2 + jy**2 + jz**2)

    hx = np.asarray(event["Hemisphere_px"], dtype=float)
    hy = np.asarray(event["Hemisphere_py"], dtype=float)
    hz = np.asarray(event["Hemisphere_pz"], dtype=float)
    hemi_nj = np.asarray(event["Hemisphere_n_jets"], dtype=int)
    hemi_p = np.sqrt(hx**2 + hy**2 + hz**2)

    # One momentum scale for every arrow: the reference |p| reaches length 1
    # (the event's largest |p| unless p_ref pins it externally).
    ref = float(p_ref) if p_ref is not None else max(jet_p.max(), hemi_p.max())
    scale = 1.0 / ref

    # Splitting plane: contains the beam axis, perpendicular to n_T.
    corners = [
        (-tx, -ty, -1.0), (tx, ty, -1.0), (tx, ty, 1.0), (-tx, -ty, 1.0),
    ]
    plane = Poly3DCollection([corners], facecolor="gray", alpha=0.15,
                             edgecolor="silver", linewidth=0.8)
    ax.add_collection3d(plane)

    # Beam axis and the transverse-plane circle at z = 0, for orientation.
    ax.plot([0, 0], [0, 0], [-1.1, 1.1], color="gray", lw=1.0)
    ang = np.linspace(0.0, 2.0 * np.pi, 120)
    ax.plot(np.cos(ang), np.sin(ang), np.zeros_like(ang),
            color="silver", lw=0.9, ls=(0, (2, 3)))

    # Transverse thrust axis: red line in the z = 0 plane.
    ax.plot([-nx, nx], [-ny, ny], [0.0, 0.0], color="red", lw=2)

    # Jets: dark gray momentum arrows from the origin.
    for x, y, z in zip(jx, jy, jz):
        ax.quiver(0, 0, 0, scale * x, scale * y, scale * z,
                  color="dimgray", linewidth=2.2, arrow_length_ratio=0.07)

    # Hemisphere summed momenta, from the stored four-vectors, labelled.
    for x, y, z, nj, p in zip(hx, hy, hz, hemi_nj, hemi_p):
        col = darker(FILL_COLORS.get(int(nj), FALLBACK_FILL))
        ax.quiver(0, 0, 0, scale * x, scale * y, scale * z,
                  color=col, alpha=0.65, linewidth=3.5,
                  arrow_length_ratio=0.07)
        lx, ly, lz = (scale * v * 1.22 for v in (x, y, z))
        ax.text(lx, ly, lz,
                f"{nj} jet{'s' if nj != 1 else ''}\n{p:.0f} GeV",
                color=col, fontsize=9, ha="center", va="center")

    split = "+".join(str(n) for n in sorted(hemi_nj))
    ax.set_title(
        f"{Path(source).name}\n"
        f"event {index} | T = {float(event['thrust']):.3f} | {split} split",
        fontsize=8, y=0.92,
    )
    # With an external reference the longest arrow may exceed unit length;
    # widen the box unless the caller pinned a shared value.
    if box_lim is not None:
        lim = float(box_lim)
    else:
        lim = 1.1 if p_ref is None else max(1.1, 1.12 * scale * hemi_p.max())
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    # Camera ~30 deg off edge-on to the splitting plane: the plane reads as a
    # slanted sheet and the two hemispheres separate left/right.
    ax.view_init(elev=15.0, azim=math.degrees(phi_t) + 60.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D event display for mixed_*.root files."
    )
    parser.add_argument("input", help="mixed ROOT file")
    parser.add_argument("event", nargs="?", type=int, default=None,
                        help="event number (default: pick one at random)")
    parser.add_argument("--tree", default="events")
    parser.add_argument("-o", "--output", default=None,
                        help="save to this image file instead of showing")
    args = parser.parse_args()

    event, index = load_event(args.input, args.tree, args.event)
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(projection="3d")
    draw(event, index, args.input, ax)
    # 3D axes leave big default margins; let the scene fill the figure.
    fig.subplots_adjust(left=-0.15, right=1.15, bottom=-0.12, top=1.02)
    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"saved {args.output}")
    else:
        plt.show()

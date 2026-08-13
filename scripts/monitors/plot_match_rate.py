#!/usr/bin/env python3
"""plot_match_rate.py - throughput of stage 2 against depletion and against time.

Reads the CSV that ``run3-mj-match`` writes (``<out-dir>/match_rate.csv``) and
plots the rate two ways, because they answer different questions:

* **vs hemispheres remaining** - is the slowdown caused by the pool draining?
  The x-axis is inverted so the run reads left to right.
* **vs elapsed time** - how long is this actually going to take?

Two more rows carry the *explanation* rather than only the symptom: mean rings
searched per query (the search working harder) and the fraction of draws that
fail (running out of partners). A drop in rate with flat rings and flat failures
means something else entirely - I/O, or another process on the node.

Each panel holds one series against one axis. Rate and rings and failure
fraction are three different units, so they never share a y-axis - a twin axis
would let any of them be scaled to tell whichever story the author preferred.

    # a finished run
    python scripts/monitors/plot_match_rate.py pairs_26_08_13/match_rate.csv

    # live, while it runs (the log is line-buffered, so rows land as written)
    python scripts/monitors/plot_match_rate.py pairs_26_08_13/match_rate.csv --watch

    # for a talk
    python scripts/monitors/plot_match_rate.py .../match_rate.csv -o rate.png --dark
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
import numpy as np

# Colours are slots 1-3 of the validated categorical palette, in fixed order,
# with the dark column stepped for the dark surface rather than flipped. Every
# panel names its own series in the title, so identity never rests on colour.
THEME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
              "grid": "#e3e2df",
              "series": ("#2a78d6", "#eb6834", "#1baf7a")},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
             "grid": "#343431",
             "series": ("#3987e5", "#d95926", "#199e70")},
}

#: (column, row title, y-axis label, series slot)
ROWS = [
    ("hemi_per_s", "Throughput", "hemispheres / s", 0),
    ("mean_rings", "Search effort", "mean rings per query", 1),
    ("frac_failed", "Draws with no partner", "fraction failing", 2),
]


def read_log(path):
    """The log as a dict of float arrays. Tolerates a partial final line."""
    with open(path, newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in row.items() if v != ""})
            except (TypeError, ValueError):
                continue          # the writer is mid-line; it will be there next read
    if not rows:
        return None
    keys = set(rows[0])
    for r in rows:
        keys &= set(r)
    return {k: np.array([r[k] for r in rows]) for k in keys}


def draw(fig, data, theme, logy=False, logx=False):
    t = THEME[theme]
    fig.clear()
    fig.patch.set_facecolor(t["surface"])

    n = len(data["elapsed_s"])
    # Markers only help when the samples are sparse; on a long run they are noise.
    marker = "o" if n <= 60 else None
    # sharey per row: each row is one quantity seen two ways, so the two
    # panels must sit on the same scale to be comparable.
    axes = fig.subplots(len(ROWS), 2, sharex="col", sharey="row")
    if len(ROWS) == 1:
        axes = np.array([axes])

    # (column, label, invert, log-able). Only the depletion axis takes --logx:
    # elapsed time has no reason to be logarithmic, and the interesting part of
    # a run is the last few percent of the pool, which a log axis expands.
    xs = [("remaining", "hemispheres remaining", True, True),
          ("elapsed_s", "elapsed (s)", False, False)]
    n_dropped = 0

    for r, (col, title, ylab, slot) in enumerate(ROWS):
        if col not in data:
            continue
        for c, (xcol, xlab, invert, logable) in enumerate(xs):
            ax = axes[r, c]
            ax.set_facecolor(t["surface"])
            x, y = data[xcol], data[col]
            use_log = logx and logable
            if use_log:
                # A log axis cannot place remaining == 0, which the final sample
                # always is. Drop it and say so, rather than letting matplotlib
                # silently mask the point and look like a complete curve.
                keep = x > 0
                n_dropped = int((~keep).sum())
                x, y = x[keep], y[keep]
            ax.plot(x, y, color=t["series"][slot], lw=2.0,
                    marker=marker, ms=4, solid_capstyle="round")
            if use_log:
                ax.set_xscale("log")
            if invert:
                # so the run reads left to right: starts full, ends empty
                ax.invert_xaxis()
            if logy and col == "hemi_per_s":
                ax.set_yscale("log")
            else:
                ax.set_ylim(bottom=0)
            ax.grid(True, color=t["grid"], lw=0.8, alpha=0.9)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(t["grid"])
            ax.tick_params(colors=t["ink2"], labelsize=9)
            if c == 0:
                ax.set_ylabel(ylab, color=t["ink2"], fontsize=9)
            # Every panel is titled, so no panel depends on colour for identity.
            ax.set_title(title if c else f"{title}", color=t["ink"],
                         fontsize=10.5, loc="left", pad=6)
            if r == len(ROWS) - 1:
                extra = ""
                if use_log:
                    extra = ", log" + (f" — {n_dropped} sample(s) at 0 omitted"
                                       if n_dropped else "")
                ax.set_xlabel(xlab + extra, color=t["ink2"], fontsize=9)

    # Where the run stands, as text - the thing you actually want when watching.
    i = -1
    done = data["consumed"][i]
    total = done + data["remaining"][i]
    head = (f"{data['pairs'][i]:,.0f} pairs   "
            f"{data['unmatched'][i]:,.0f} unmatched   "
            f"{done / max(total, 1) * 100:.1f}% of {total:,.0f} consumed   "
            f"{data['hemi_per_s'][i]:,.0f} hemi/s   "
            f"{data['elapsed_s'][i] / 60:.1f} min elapsed")
    fig.suptitle(head, color=t["ink"], fontsize=12, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="match_rate.csv from run3-mj-match")
    p.add_argument("-o", "--output", default=None,
                   help="write a PNG instead of opening a window")
    p.add_argument("--watch", action="store_true",
                   help="redraw while the run is still going")
    p.add_argument("--every", type=float, default=5.0,
                   help="seconds between redraws with --watch (default 5)")
    p.add_argument("--dark", action="store_true")
    p.add_argument("--logx", action="store_true",
                   help="log the 'hemispheres remaining' axis only. The dropoff "
                        "happens in the last few percent of the pool, which a "
                        "linear axis crushes against the right edge. Elapsed "
                        "time stays linear. The final sample sits at 0 remaining "
                        "and cannot appear on a log axis; it is dropped and the "
                        "axis label says so.")
    p.add_argument("--logy", action="store_true",
                   help="log y on the rate panels; useful when the rate spans "
                        "more than about a decade")
    p.add_argument("--dpi", type=int, default=140)
    args = p.parse_args(argv)

    if args.output and not args.watch:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = "dark" if args.dark else "light"
    if not os.path.exists(args.csv):
        sys.exit(f"{args.csv} not found - has stage 2 started?")

    fig = plt.figure(figsize=(12, 8))

    def once():
        data = read_log(args.csv)
        if data is None or len(data.get("elapsed_s", ())) < 2:
            return False
        draw(fig, data, theme, logy=args.logy, logx=args.logx)
        return True

    if not args.watch:
        if not once():
            sys.exit(f"{args.csv} has no complete samples yet.")
        if args.output:
            fig.savefig(args.output, dpi=args.dpi,
                        facecolor=THEME[theme]["surface"])
            print(f"-> {args.output}")
        else:
            plt.show()
        return 0

    plt.ion()
    plt.show(block=False)
    print(f"watching {args.csv} (Ctrl-C to stop)")
    try:
        while True:
            if once():
                fig.canvas.draw_idle()
                if args.output:
                    fig.savefig(args.output, dpi=args.dpi,
                                facecolor=THEME[theme]["surface"])
            plt.pause(args.every)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""joblogs.py - shared reader and plotter behind the per-stage job monitors.

Not a script. `plot_index_jobs.py`, `plot_gather_jobs.py` and
`plot_assemble_jobs.py` are one per condor stage, the way `plot_match_rate.py`
is the one for stage 2, and each supplies only the thing that differs between
them: what "work" means in its stdout. Everything here - reading condor's log,
joining it to ours, the panels, the fit - is identical across stages and so
lives once.

Two files per job, joined by the job name in the filename:

* ``log_<cluster>_<name>.log`` - condor's record: when the job started and
  ended, peak memory against what was requested, disk, bytes transferred, how
  it ended.
* ``log_<cluster>_<name>.out`` - our stdout: how much work the job did, which
  only the stage itself knows.

**Several log dirs of the same stage overlay as separate runs**, which is the
point of splitting per stage: rerun a stage after a change and the old and new
clouds sit on the same axes.

The three panels answer three separate questions. **Wall time vs work** should
be a line through the origin; a large intercept is a fixed per-job cost that
dominates at small sizes, and vertical spread at fixed work means the workers
are not equivalent. **Peak memory vs work** is what decides the request - the
intercept is the baseline and the slope is what eventually walks a job into the
limit. **Bytes read vs work** says whether reads are proportional or whether
jobs pay to open files they barely touch. Each run is fitted with a dashed
least-squares line labelled by its intercept, since the intercept is the part
you act on - **when the jobs vary enough in size to measure one**. If they are
all the same size, which is what an even split produces, the intercept is an
extrapolation from a narrow band and the legend says so instead of quoting a
number; only the slope is real in that case.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import os
import re
import sys
from collections import namedtuple

import matplotlib.ticker
import numpy as np

#: One work metric a stage's stdout exposes. `pattern` is searched over the
#: whole .out and SUMMED over every match, because a job runs its stage once
#: per input file; `group` is the capturing group holding the number, or None
#: to count matching lines instead. The first metric a stage lists is its
#: default x-axis.
Metric = namedtuple("Metric", "key pattern group label")

# Series colours are slots 1-3 of the validated categorical palette, in fixed
# order, so a run keeps its colour whatever else is on the axes. Failures wear
# the reserved critical status colour - which never carries meaning alone, so
# they also take their own marker and a legend entry.
THEME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
              "grid": "#e3e2df", "bad": "#d03b3b",
              "series": ("#2a78d6", "#eb6834", "#1baf7a")},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
             "grid": "#343431", "bad": "#d03b3b",
             "series": ("#3987e5", "#d95926", "#199e70")},
}

#: (column, panel title, unit, scale from the recorded unit)
PANELS = [("wall_s", "Wall time on the worker", "s", 1.0),
          ("mem_mb", "Peak memory", "MB", 1.0),
          ("bytes_in", "Bytes read", "GB", 1e-9)]

# One HTCondor log event: `005 (12345.000.000) 2026-08-12 10:00:00 Job ...`.
# The timestamp is `MM/DD HH:MM:SS` on older condor and ISO on newer, so both
# are accepted rather than assuming whichever version this pool happens to run.
_EVENT = re.compile(r"^(\d{3}) \((\d+)\.\d+\.\d+\) (\S+ \S+)")
_TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%m/%d %H:%M:%S")

_FIELDS = {
    # The `Partitionable Resources : Usage Request Allocated` block of the
    # terminate event. Usage is the peak, which is the number that matters.
    "mem_mb": re.compile(r"Memory \(MB\)\s+:\s+(\d+)"),
    "mem_req_mb": re.compile(r"Memory \(MB\)\s+:\s+\d+\s+(\d+)"),
    "disk_kb": re.compile(r"Disk \(KB\)\s+:\s+(\d+)"),
    "cpus": re.compile(r"Cpus\s+:\s+([\d.]+)"),
    "bytes_in": re.compile(r"(\d+)\s+-\s+Run Bytes Received By Job"),
    "bytes_out": re.compile(r"(\d+)\s+-\s+Run Bytes Sent By Job"),
    # Written by newer condor only; when absent the event timestamps are used.
    "exec_s": re.compile(r"TimeExecute \(s\)\s+:\s+([\d.]+)"),
    "slot_s": re.compile(r"TimeSlotBusy \(s\)\s+:\s+([\d.]+)"),
}


def _stamp(text):
    for fmt in _TS_FORMATS:
        try:
            # 1900 on the year-less format; only differences are ever used.
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def parse_condor_log(text):
    """One job's condor log -> timing, resources and how it ended.

    Wall time comes from the execute -> terminate timestamps rather than from
    ``TimeExecute``, which only newer condor writes. On the year-less timestamp
    format a job spanning New Year would come out negative; that is reported as
    missing rather than as a wrong number.
    """
    rec = {k: np.nan for k in _FIELDS}
    rec.update(cluster="", exit=np.nan, wall_s=np.nan, queue_s=np.nan,
               n_starts=0, status="running")
    t_submit = t_exec = t_end = None

    for block in text.split("\n...\n"):
        m = _EVENT.match(block.lstrip("\n"))
        if not m:
            continue
        code, when = m.group(1), _stamp(m.group(3))
        rec["cluster"] = m.group(2)
        if code == "000":
            t_submit = when
        elif code == "001":                        # executing
            t_exec = when
            rec["n_starts"] += 1
        elif code == "012":                        # held
            rec["status"] = "held"
        elif code == "009":                        # aborted
            rec["status"] = "aborted"
            t_end = when
        elif code == "005":                        # terminated
            t_end = when
            hit = re.search(r"Normal termination \(return value (\d+)\)", block)
            if hit:
                rec["exit"] = float(hit.group(1))
                rec["status"] = "ok" if rec["exit"] == 0 else "failed"
            else:
                rec["status"] = "signalled"
            for key, pat in _FIELDS.items():
                got = pat.search(block)
                if got:
                    rec[key] = float(got.group(1))

    if t_exec and t_end:
        rec["wall_s"] = (t_end - t_exec).total_seconds()
    if t_submit and t_exec:
        rec["queue_s"] = (t_exec - t_submit).total_seconds()
    for key in ("wall_s", "queue_s"):
        if np.isfinite(rec[key]) and rec[key] < 0:
            rec[key] = np.nan                      # year-less timestamp wrapped
    if not np.isfinite(rec["wall_s"]) and np.isfinite(rec["exec_s"]):
        rec["wall_s"] = rec["exec_s"]
    return rec


def collect(logdir, metrics):
    """One row per job in `logdir`, condor's log joined to our stdout."""
    logs = sorted(glob.glob(os.path.join(str(logdir), "log_*.log")))
    rows = []
    for path in logs:
        with open(path, errors="replace") as f:
            rec = parse_condor_log(f.read())
        rec["job"] = os.path.basename(path)[:-4].split("_", 2)[-1]
        try:
            with open(path[:-4] + ".out", errors="replace") as f:
                out = f.read()
        except OSError:
            out = ""
        for met in metrics:
            hits = list(met.pattern.finditer(out))
            if not hits:
                rec[met.key] = np.nan
            elif met.group is None:
                rec[met.key] = float(len(hits))
            else:
                rec[met.key] = float(sum(
                    float(h.group(met.group).replace(",", "")) for h in hits))
        rows.append(rec)
    return rows


def _fit(x, y):
    """Least-squares (intercept, slope), or None if it would be meaningless."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.ptp(x[m]) <= 0:
        return None
    slope, intercept = np.polyfit(x[m], y[m], 1)
    return intercept, slope


def _si(v):
    """1000 -> 1k, 100000 -> 100k, 1e6 -> 1M. Legends have no room for zeros."""
    for div, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if v >= div:
            return f"{v / div:g}{suffix}"
    return f"{v:g}"


def _fit_label(intercept, slope, lo_x, hi_x, y_scale, unit):
    """Legend text for a fit, quoting the intercept only if it is measured.

    When every job in a submission is the same size - the normal case, since
    the submitters split the work evenly - the x range is a narrow band far
    from zero, and the intercept is an extrapolation the data cannot support:
    fitting jobs clustered at 220k events recovers a "fixed cost" off by 3x.
    The slope is well determined regardless, so it is always quoted; the
    intercept is quoted only when job sizes vary enough to separate fixed cost
    from variable cost. `False` in the second return value asks the caller for
    the figure-level footnote saying why it is missing.
    """
    step = 10.0 ** np.floor(np.log10(max(hi_x, 1.0)))
    if step > hi_x / 2:
        step /= 10.0
    rise = slope * step
    # A slope that rounds to nothing against the data's own scale is flat, and
    # printing it as -3.15e-17 would dress up a zero as a measurement.
    per = ("~flat" if abs(rise) < 5e-3 * max(y_scale, 1e-12)
           else f"{rise:,.3g} {unit}/{_si(step)}")
    if lo_x <= 0.3 * hi_x:
        return f"fit: {intercept:,.3g} {unit} + {per}", True
    return f"fit: {per}", False


def _span(v, origin):
    """Axis limits for `v`. One submission's jobs are near-identical in size,
    so anchoring at zero squeezes them all into one corner and hides exactly
    the spread the plot is for - hence `--origin` is opt-in. The fitted
    intercept is in the legend either way."""
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi > lo:
        pad = 0.07 * (hi - lo)
    else:                            # every job identical on this axis
        pad = abs(hi) * 0.07 or 1.0
    lo, hi = lo - pad, hi + pad
    if origin:
        lo = min(lo, 0.0)
    return lo, hi


def summarize(rows, name, met):
    x = np.array([r[met.key] for r in rows], dtype=float)
    wall = np.array([r["wall_s"] for r in rows], dtype=float)
    mem = np.array([r["mem_mb"] for r in rows], dtype=float)
    bad = [r for r in rows if r["status"] not in ("ok", "running")]
    parts = [f"{name}: {len(rows)} job(s)"]
    if bad:
        parts.append(f"{len(bad)} not ok "
                     f"({', '.join(sorted({r['status'] for r in bad}))})")
    parts += [f"{np.nansum(x):,.0f} {met.label}",
              f"wall median {np.nanmedian(wall) / 60:,.1f} min",
              f"max {np.nanmax(wall) / 60:,.1f} min",
              f"total {np.nansum(wall) / 3600:,.1f} core-h",
              f"peak mem max {np.nanmax(mem):,.0f} MB"]
    return "   ".join(parts)


def draw(fig, runs, met, stage, theme, logxy=False, origin=False):
    """`runs` is [(name, rows), ...] - several are overlaid for comparison."""
    t = THEME[theme]
    comma = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.6g}")
    fig.clear()
    fig.patch.set_facecolor(t["surface"])
    axes = np.atleast_1d(fig.subplots(1, len(PANELS)))
    intercept_shown = True

    for ax, (col, title, unit, scale) in zip(axes, PANELS):
        ax.set_facecolor(t["surface"])
        ys = [np.array([r[col] for r in rows], dtype=float) * scale
              for _, rows in runs]
        # Seconds stop being readable past about ten minutes. Rescale the data
        # rather than reformatting ticks, so the fitted intercept printed in the
        # legend carries the same unit as the axis.
        top = max((np.nanmax(y) for y in ys if np.isfinite(y).any()),
                  default=0.0)
        if unit == "s" and top >= 600:
            ys, unit = [y / 60.0 for y in ys], "min"

        seen_x, seen_y = [], []
        for i, ((name, rows), y) in enumerate(zip(runs, ys)):
            colour = t["series"][i % len(t["series"])]
            x = np.array([r[met.key] for r in rows], dtype=float)
            ok = np.array([r["status"] == "ok" for r in rows])
            m = np.isfinite(x) & np.isfinite(y)
            seen_x.append(x[m])
            seen_y.append(y[m])
            # With one run the label would only repeat the title; with several
            # it is the only thing telling them apart.
            ax.scatter(x[m & ok], y[m & ok], s=26, c=colour, alpha=0.75,
                       edgecolors="none",
                       label=name if len(runs) > 1 else "succeeded")
            if (m & ~ok).any():
                ax.scatter(x[m & ~ok], y[m & ~ok], s=48, c=t["bad"],
                           marker="x", linewidths=1.6,
                           label="not ok" if i == 0 else None)
            fit = _fit(x[m], y[m])
            if fit and not logxy:
                b, a = fit
                # Only over the range that was actually measured. Extending it
                # back to the intercept would both claim an extrapolation the
                # data does not support and, since the line counts as data,
                # drag the axes down to zero and flatten the cloud.
                lo_x, hi_x = np.nanmin(x[m]), np.nanmax(x[m])
                xs = np.array([lo_x, hi_x])
                label, got_b = _fit_label(b, a, lo_x, hi_x,
                                          np.nanmax(np.abs(y[m])), unit)
                intercept_shown &= got_b
                ax.plot(xs, b + a * xs, color=colour, lw=1.1, ls="--",
                        label=label)

        # Limits from the points alone, so no annotation or guide line can
        # rescale the panel out from under the data.
        xa = np.concatenate(seen_x) if seen_x else np.array([])
        ya = np.concatenate(seen_y) if seen_y else np.array([])
        if logxy:
            ax.set_xscale("log")
            ax.set_yscale("log")
        elif xa.size:
            ax.set_xlim(*_span(xa, origin))
            ax.set_ylim(*_span(ya, origin))
            ax.xaxis.set_major_formatter(comma)
            ax.yaxis.set_major_formatter(comma)
            # Six-figure counts collide at matplotlib's default tick density.
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))

        if col == "mem_mb":
            req = np.nanmedian([r["mem_req_mb"]
                                for _, rows in runs for r in rows])
            hi = ya.max() if ya.size else np.nan
            if np.isfinite(req) and np.isfinite(hi):
                if req <= ax.get_ylim()[1]:
                    ax.axhline(req, color=t["bad"], lw=1.0, ls=":")
                    ax.annotate(f"requested {req:,.0f} MB", (0.02, req),
                                xycoords=("axes fraction", "data"),
                                va="bottom", fontsize=8, color=t["bad"])
                else:
                    # Above the panel. Stretching to reach it would flatten the
                    # cloud to a line, so state the headroom instead.
                    ax.annotate(f"requested {req:,.0f} MB, "
                                f"{req / hi:.1f}x the peak job",
                                (0.02, 0.95), xycoords="axes fraction",
                                va="top", fontsize=8, color=t["bad"])
        ax.grid(True, color=t["grid"], lw=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(t["grid"])
        ax.tick_params(colors=t["ink2"], labelsize=9)
        ax.set_title(title, color=t["ink"], fontsize=10.5, loc="left", pad=6)
        ax.set_ylabel(unit, color=t["ink2"], fontsize=9)
        ax.set_xlabel(met.label, color=t["ink2"], fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=t["ink2"], loc="best")

    n_jobs = sum(len(rows) for _, rows in runs)
    n_bad = sum(1 for _, rows in runs for r in rows
                if r["status"] not in ("ok", "running"))
    head = f"stage {stage}   {n_jobs} job(s)"
    if len(runs) > 1:
        head += f" over {len(runs)} runs"
    if n_bad:
        head += f"   {n_bad} not ok"
    fig.suptitle(head, color=t["ink"], fontsize=12, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0.045 if not intercept_shown else 0, 1, 0.93))
    if not intercept_shown:
        # Said once, at the figure level: in a legend it is wider than the panel.
        fig.text(0.012, 0.012,
                 "fits quote a slope only - the jobs in a run are too alike in "
                 "size to separate a fixed per-job cost from it",
                 color=t["ink2"], fontsize=8.5, ha="left")


def write_csv(path, runs):
    allrows = [dict(r, run=name) for name, rows in runs for r in rows]
    keys, seen = [], set()
    for r in allrows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(allrows)
    print(f"-> {path}")


def run_cli(stage, metrics, doc, argv=None):
    """The whole command line for one stage's monitor."""
    p = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logdir", nargs="+",
                   help=f"condor log directory from a stage {stage} "
                        "submission. Give more than one to overlay separate "
                        "runs of this stage on the same axes.")
    p.add_argument("--x", default=metrics[0].key,
                   choices=[m.key for m in metrics],
                   help=f"work metric for the x-axis (default {metrics[0].key})")
    p.add_argument("--label", nargs="+", default=None, metavar="NAME",
                   help="legend name per logdir (default: its basename)")
    p.add_argument("-o", "--output", default=None, help="write a PNG")
    p.add_argument("--csv", default=None,
                   help="also dump the joined per-job table")
    p.add_argument("--dark", action="store_true")
    p.add_argument("--log", dest="logxy", action="store_true",
                   help="log both axes; useful when job sizes span decades. "
                        "Suppresses the fit, which is a linear model.")
    p.add_argument("--origin", action="store_true",
                   help="include (0, 0) on every panel, so the fitted "
                        "intercept can be read off the axis. Off by default: "
                        "one submission's jobs are near-identical in size, so "
                        "this squeezes them all into one corner.")
    p.add_argument("--dpi", type=int, default=140)
    args = p.parse_args(argv)

    if args.label and len(args.label) != len(args.logdir):
        sys.exit(f"--label takes one name per logdir "
                 f"({len(args.logdir)} given)")
    names = args.label or [os.path.basename(str(d).rstrip("/"))
                           for d in args.logdir]
    met = next(m for m in metrics if m.key == args.x)

    runs = []
    for name, logdir in zip(names, args.logdir):
        if not os.path.isdir(logdir):
            sys.exit(f"{logdir} is not a directory")
        rows = collect(logdir, metrics)
        if not rows:
            sys.exit(f"no log_*.log files in {logdir}")
        # A job still running, or one that died before printing, has no work
        # count. Say so rather than letting it vanish out of the scatter.
        n_none = sum(1 for r in rows if not np.isfinite(r[met.key]))
        if n_none:
            print(f"{name}: {n_none}/{len(rows)} job(s) have no '{met.key}' "
                  "count in their .out (still running, or died early)")
        print(summarize(rows, name, met))
        runs.append((name, rows))

    if args.csv:
        write_csv(args.csv, runs)

    if args.output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = "dark" if args.dark else "light"
    fig = plt.figure(figsize=(13.5, 4.4))
    draw(fig, runs, met, stage, theme, logxy=args.logxy, origin=args.origin)
    if args.output:
        fig.savefig(args.output, dpi=args.dpi,
                    facecolor=THEME[theme]["surface"])
        print(f"-> {args.output}")
    else:
        plt.show()
    return 0

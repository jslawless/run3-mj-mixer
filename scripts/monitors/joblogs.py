"""joblogs.py - shared reader and plotter behind the per-stage job monitors.

Not a script. `plot_index_jobs.py`, `plot_gather_jobs.py` and
`plot_assemble_jobs.py` are one per condor stage, the way `plot_match_rate.py`
is the one for stage 2, and each supplies only the thing that differs between
them: what "work" means in its stdout, and optionally how to group its jobs.
Everything here - reading condor's log, joining it to ours, the panels - is
identical across stages and so lives once.

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
rise proportionally; vertical spread at fixed work means the workers are not
equivalent. **Peak memory vs work** is what decides the request - a slope is
what eventually walks a job into the limit. **Bytes read vs work** says whether
reads are proportional or whether jobs pay to open files they barely touch.

A stage may also declare a `Grouper`, which turns its own stdout into a label
per job and adds a flag to colour by it (stage 1a: HT slice). Groups are drawn
from an **ordinal** blue ramp in group order, not from categorical hues: HT
slices are ordered bins of one quantity, so the encoding should carry that
order. The cost is that adjacent slices are deliberately similar - about 4.7
OKLab lightness apart at eight groups, 4.35 under CVD simulation - which is why
grouping also prints an exact per-group table, so no reading depends on telling
two shades apart.
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

#: An optional way for a stage to split its jobs. `fn` reads the job's stdout
#: and returns a label (or None); `order` sorts the labels; `flag` is the
#: command-line switch that turns colouring on; `neutral` names the labels that
#: are outside the ordered sequence, so they are not given a ramp step.
Grouper = namedtuple("Grouper", "flag dest label help fn order neutral")

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

# The blue ramp, for ordinal groups. Light mode runs 250 -> 700 and dark mode
# 600 -> 100: in both, later groups sit further from the surface, so the step
# nearest it still clears the 2:1 floor (2.06:1 on light, 2.15:1 on dark).
# Steps are sampled evenly across the range for the number of groups present.
RAMP = {
    "light": ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
              "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"),
    "dark": ("#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
             "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6",
             "#cde2fb"),
}


def ramp_colours(n, theme):
    """`n` steps spread across the ramp, in group order.

    Past the ramp's length this raises rather than cycling or generating a
    hue: a repeated colour would silently claim two groups are one.
    """
    steps = RAMP[theme]
    if n > len(steps):
        raise SystemExit(
            f"{n} groups is more than the {len(steps)}-step ramp holds. "
            "Colouring would have to repeat a shade, which reads as two "
            "groups being the same one.")
    if n == 1:
        return [steps[len(steps) // 2]]
    idx = np.round(np.linspace(0, len(steps) - 1, n)).astype(int)
    return [steps[i] for i in idx]


def group_colours(groups, theme, neutral=()):
    """[(label, colour), ...] in group order.

    Labels outside the ordered sequence - "several slices", an unrecognised
    name - take neutral ink rather than a ramp step. Given one they would read
    as the sequence's extreme, which is the opposite of what they mean.
    """
    ordered = [g for g in groups if g not in neutral]
    shades = dict(zip(ordered, ramp_colours(len(ordered), theme)))
    return [(g, shades.get(g, THEME[theme]["ink2"])) for g in groups]

#: (column, panel title, unit, scale from the recorded unit)
PANELS = [("wall_s", "Wall time on the worker", "s", 1.0),
          ("mem_mb", "Peak memory", "MB", 1.0),
          ("bytes_in", "Bytes read", "GB", 1e-9)]

# One HTCondor log event: `005 (12345.000.000) 2026-08-12 10:00:00 Job ...`.
# The timestamp is `MM/DD HH:MM:SS` on older condor and ISO on newer, so both
# are accepted rather than assuming whichever version this pool happens to run.
_EVENT = re.compile(r"^(\d{3}) \((\d+)\.\d+\.\d+\) (\S+ \S+)")
# (prefix, format). The year-less form gets an explicit leap year bolted on:
# strptime's own default is 1900, which is not a leap year, so a job that ran
# on Feb 29 would fail to parse - and python 3.15 turns the bare form into an
# error anyway.
_TS_FORMATS = (("", "%Y-%m-%d %H:%M:%S"), ("1904 ", "%Y %m/%d %H:%M:%S"))

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
    for prefix, fmt in _TS_FORMATS:
        try:
            return dt.datetime.strptime(prefix + text, fmt)
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


def collect(logdir, metrics, grouper=None):
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
        if grouper is not None:
            rec["group"] = grouper.fn(out)
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


def _span(v, origin):
    """Axis limits for `v`. One submission's jobs are near-identical in size,
    so anchoring at zero squeezes them all into one corner and hides exactly
    the spread the plot is for - hence `--origin` is opt-in."""
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


def summarize_groups(runs, met, grouper):
    """A per-group table, printed whenever colouring by group.

    The ramp is ordinal, so adjacent groups are close by design; these are the
    numbers, so nothing has to be read off a shade.
    """
    rows = [r for _, rows in runs for r in rows]
    groups = group_order(runs, grouper)
    width = max([len(str(g)) for g in groups] + [len(grouper.label)])
    out = [f"{grouper.label:<{width}}   jobs   {met.label:>18}   "
           f"wall median   wall max   peak mem"]
    for g in groups:
        mine = [r for r in rows if r.get("group") == g]
        x = np.array([r[met.key] for r in mine], dtype=float)
        wall = np.array([r["wall_s"] for r in mine], dtype=float)
        mem = np.array([r["mem_mb"] for r in mine], dtype=float)
        out.append(f"{str(g):<{width}}   {len(mine):>4}   {np.nansum(x):>18,.0f}"
                   f"   {np.nanmedian(wall) / 60:>9,.1f} m"
                   f"   {np.nanmax(wall) / 60:>6,.1f} m"
                   f"   {np.nanmax(mem):>6,.0f} MB")
    ungrouped = sum(1 for r in rows if r.get("group") is None)
    if ungrouped:
        out.append(f"({ungrouped} job(s) with no {grouper.label} in their .out "
                   "- not coloured, not plotted)")
    return "\n".join(out)


def group_order(runs, grouper):
    """The groups present, in the stage's own order."""
    seen = {r.get("group") for _, rows in runs for r in rows}
    return sorted((g for g in seen if g is not None), key=grouper.order)


def draw(fig, runs, met, stage, theme, logxy=False, origin=False, groups=None,
         group_label=None):
    """`runs` is [(name, rows), ...] - several are overlaid for comparison.

    `groups`, when given, is [(label, colour), ...] in group order; points are
    then coloured by group instead of by run.
    """
    t = THEME[theme]
    comma = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.6g}")
    fig.clear()
    fig.patch.set_facecolor(t["surface"])
    axes = np.atleast_1d(fig.subplots(1, len(PANELS)))

    for ax, (col, title, unit, scale) in zip(axes, PANELS):
        ax.set_facecolor(t["surface"])
        ys = [np.array([r[col] for r in rows], dtype=float) * scale
              for _, rows in runs]
        # Seconds stop being readable past about ten minutes, so rescale the
        # data rather than reformatting the ticks.
        top = max((np.nanmax(y) for y in ys if np.isfinite(y).any()),
                  default=0.0)
        if unit == "s" and top >= 600:
            ys, unit = [y / 60.0 for y in ys], "min"

        seen_x, seen_y = [], []
        for i, ((name, rows), y) in enumerate(zip(runs, ys)):
            x = np.array([r[met.key] for r in rows], dtype=float)
            ok = np.array([r["status"] == "ok" for r in rows])
            m = np.isfinite(x) & np.isfinite(y)
            seen_x.append(x[m])
            seen_y.append(y[m])

            if groups:
                # One scatter per group, in group order, so the legend reads in
                # that order too and the shading is interpretable as a sequence.
                who = np.array([r.get("group") for r in rows], dtype=object)
                for g, shade in groups:
                    sel = m & ok & (who == g)
                    if sel.any():
                        ax.scatter(x[sel], y[sel], s=26, c=shade, alpha=0.85,
                                   edgecolors="none", label=g)
            else:
                # A lone unnamed series needs no legend entry - the panel title
                # already says what it is. With several runs the name is the
                # only thing telling them apart.
                ax.scatter(x[m & ok], y[m & ok], s=26,
                           c=t["series"][i % len(t["series"])], alpha=0.75,
                           edgecolors="none",
                           label=name if len(runs) > 1 else None)
            if (m & ~ok).any():
                ax.scatter(x[m & ~ok], y[m & ~ok], s=48, c=t["bad"],
                           marker="x", linewidths=1.6,
                           label="not ok" if i == 0 else None)

        # Limits from the points alone, so nothing else can rescale the panel
        # out from under the data.
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

    n_jobs = sum(len(rows) for _, rows in runs)
    n_bad = sum(1 for _, rows in runs for r in rows
                if r["status"] not in ("ok", "running"))
    head = f"stage {stage}   {n_jobs} job(s)"
    if len(runs) > 1:
        head += f" over {len(runs)} runs"
    if group_label:
        head += f"   coloured by {group_label}"
    if n_bad:
        head += f"   {n_bad} not ok"
    # A job with no work count is not on the scatter, so the header would
    # otherwise promise more points than the panels hold.
    n_off = sum(1 for _, rows in runs for r in rows
                if not np.isfinite(r[met.key]))
    if n_off:
        head += f"   {n_off} with no {met.label} recorded, not plotted"
    fig.suptitle(head, color=t["ink"], fontsize=12, x=0.012, ha="left")

    # One legend for the figure, not one per panel: all three panels share the
    # same encoding, so repeating it three times would only cost plot area.
    pairs = {}
    for handle, label in zip(*axes[0].get_legend_handles_labels()):
        if label and label not in pairs:
            pairs[label] = handle
    if not pairs:
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    elif groups:
        # Down the right, one entry per line: an ordinal ramp has to be read as
        # a sequence, and a wrapped multi-column legend fills column-major, so
        # the order would come out scrambled.
        fig.tight_layout(rect=(0, 0, 0.88, 0.93))
        fig.legend(list(pairs.values()), list(pairs), frameon=False, ncol=1,
                   fontsize=8.5, labelcolor=t["ink2"], loc="center left",
                   handletextpad=0.35, bbox_to_anchor=(0.885, 0.5))
    else:
        fig.tight_layout(rect=(0, 0.1, 1, 0.93))
        fig.legend(list(pairs.values()), list(pairs), frameon=False,
                   fontsize=8.5, labelcolor=t["ink2"], loc="lower center",
                   ncol=min(len(pairs), 6), handletextpad=0.35,
                   columnspacing=1.3, bbox_to_anchor=(0.5, 0.005))


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


def run_cli(stage, metrics, doc, grouper=None, argv=None):
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
    if grouper is not None:
        p.add_argument(grouper.flag, dest=grouper.dest, action="store_true",
                       help=grouper.help)
    p.add_argument("--label", nargs="+", default=None, metavar="NAME",
                   help="legend name per logdir (default: its basename)")
    p.add_argument("-o", "--output", default=None, help="write a PNG")
    p.add_argument("--csv", default=None,
                   help="also dump the joined per-job table")
    p.add_argument("--dark", action="store_true")
    p.add_argument("--log", dest="logxy", action="store_true",
                   help="log both axes; useful when job sizes span decades")
    p.add_argument("--origin", action="store_true",
                   help="include (0, 0) on every panel. Off by default: one "
                        "submission's jobs are near-identical in size, so this "
                        "squeezes them all into one corner.")
    p.add_argument("--dpi", type=int, default=140)
    args = p.parse_args(argv)

    by_group = grouper is not None and getattr(args, grouper.dest)
    if by_group and len(args.logdir) > 1:
        sys.exit(f"{grouper.flag} colours by {grouper.label}, so it cannot "
                 "also separate runs by colour. Give one log directory.")
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
        rows = collect(logdir, metrics, grouper if by_group else None)
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

    groups = None
    if by_group:
        print(summarize_groups(runs, met, grouper))

    if args.csv:
        write_csv(args.csv, runs)

    if args.output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = "dark" if args.dark else "light"
    if by_group:
        groups = group_colours(group_order(runs, grouper), theme,
                               grouper.neutral)
    fig = plt.figure(figsize=(13.5, 4.4))
    draw(fig, runs, met, stage, theme, logxy=args.logxy, origin=args.origin,
         groups=groups, group_label=grouper.label if by_group else None)
    if args.output:
        fig.savefig(args.output, dpi=args.dpi,
                    facecolor=THEME[theme]["surface"])
        print(f"-> {args.output}")
    else:
        plt.show()
    return 0

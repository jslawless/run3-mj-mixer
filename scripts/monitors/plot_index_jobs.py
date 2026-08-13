#!/usr/bin/env python3
"""plot_index_jobs.py - what stage 1a cost per job, against the events it read.

Stage 1a jobs run ``run3-mj-index`` once per input file (``submit_index.py -n``
files per job), so the per-file `Done.` lines are summed and one point is one
condor job.

What to look for here:

* **Wall time** should be close to proportional to events read, since the work
  is one streaming pass per file. A large fitted intercept means the fixed cost
  - LCG view, venv, wheel install, xrdcp of the shard - is a real fraction of
  the job, and `-n` should go up.
* **Peak memory should be flat**, because files are processed one at a time and
  nothing is kept between them. A slope against events read means something
  accumulates across a job's files, which is a bug rather than a tuning issue.
* **Bytes read** is the slimmed input, so it tracks events read almost exactly;
  it is the panel that shows a job that was handed unusually large files.

`--by-slice` keys each job by the HT slice its files came from, which is how to
see whether one slice is carrying the cost - the slices differ by orders of
magnitude in events per file, so "slow job" and "big slice" are easy to confuse
otherwise. Slices are keyed by hue and marker shape together, since a scatter
with eleven groups has more groups than any set of hues separates; the exact
per-slice numbers are printed as a table alongside.

    python scripts/monitors/plot_index_jobs.py batch_index
    python scripts/monitors/plot_index_jobs.py batch_index --by-slice
    python scripts/monitors/plot_index_jobs.py batch_index --x hemispheres
    python scripts/monitors/plot_index_jobs.py batch_index_old batch_index_new
"""

import re

from joblogs import Grouper, Metric, run_cli
# filetable owns dataset inference for the whole pipeline; a monitor deciding
# for itself what counts as a slice would be a fourth copy of that rule.
from run3_mj_mixer.filetable import UNKNOWN_SLICE, slice_or_unknown

# Anchored on shard.py's stdout: the `Done.` summary per file, and the `Input:`
# banner, which is how many files this job was handed.
METRICS = [
    Metric("events", re.compile(r"Done\.\s+[\d,]+ / ([\d,]+) events"), 1,
           "events read"),
    Metric("kept", re.compile(r"Done\.\s+([\d,]+) / [\d,]+ events"), 1,
           "events kept (exactly n jets)"),
    Metric("hemispheres", re.compile(r"-> ([\d,]+) \d+-jet hemispheres"), 1,
           "hemispheres written"),
    Metric("files", re.compile(r"^Input:\s", re.M), None, "input files"),
]

_INPUT = re.compile(r"^Input:\s+(\S+)", re.M)
_MIXED = "several slices"


def ht_slice(out_text):
    """The HT slice a job's files came from, or None if it printed no inputs.

    `submit_index.py` is fed filelists grouped by slice, so a job is normally
    one slice - but `-n` can straddle a boundary, and a job that did is called
    out rather than being filed under whichever slice happened to come first.
    Passing no xs keys makes `slice_or_unknown` fall through to its HT regex,
    which is what is wanted here: a label, not a weight, and no xs JSON to find.
    """
    found = {slice_or_unknown(path, ()) for path in _INPUT.findall(out_text)}
    if not found:
        return None
    return found.pop() if len(found) == 1 else _MIXED


def ht_order(label):
    """Slices sort by their lower HT edge; the odd ones out sort last."""
    if label in (_MIXED, UNKNOWN_SLICE):
        return (1, label)
    edge = re.search(r"HT-(\d+)", label)
    return (0, int(edge.group(1))) if edge else (1, label)


SLICE = Grouper(flag="--by-slice", dest="by_slice", label="HT slice",
                help="colour each job by the HT slice of its input files, "
                     "shaded light to dark in HT order. One log directory "
                     "only - colour cannot carry both slice and run.",
                fn=ht_slice, order=ht_order,
                neutral=(_MIXED, UNKNOWN_SLICE))

if __name__ == "__main__":
    raise SystemExit(run_cli("1a (index)", METRICS, __doc__, grouper=SLICE))

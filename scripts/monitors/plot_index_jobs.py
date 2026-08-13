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

    python scripts/monitors/plot_index_jobs.py batch_index
    python scripts/monitors/plot_index_jobs.py batch_index --x hemispheres
    python scripts/monitors/plot_index_jobs.py batch_index_old batch_index_new
"""

import re

from joblogs import Metric, run_cli

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

if __name__ == "__main__":
    raise SystemExit(run_cli("1a (index)", METRICS, __doc__))

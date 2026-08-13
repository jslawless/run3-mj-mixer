#!/usr/bin/env python3
"""plot_gather_jobs.py - what stage 3a cost per job, against the legs it wrote.

Stage 3a jobs own **files**, not pairs: each reads its slimmed files once,
sequentially, and emits one leg record per referenced hemisphere. One point is
one condor job.

What to look for here:

* **Bytes read is the load-bearing panel.** Gather-by-file exists so that the
  input is read once in total rather than once per consumer, so bytes read
  should track `--x files` and be nearly *flat* against legs written. If it
  slopes with legs, files are being reopened and the layout is not holding.
* **Wall time** splits into a read cost set by the files owned and a write cost
  set by the legs, so it is worth looking at both `--x legs` and `--x files` -
  whichever gives the tighter line is the one actually driving the stage.
* **Peak memory** grows with the legs held in memory before bucketing them to
  the P x Q shards, so this panel is where a raised `--n-jobs` earns its keep.

`legs owed` (`--x owed`) is what the pair list asked for and `legs` is what was
written; they must agree, and a job where they don't is one to open by hand.

    python scripts/monitors/plot_gather_jobs.py batch_gather
    python scripts/monitors/plot_gather_jobs.py batch_gather --x files
    python scripts/monitors/plot_gather_jobs.py batch_gather_60 batch_gather_120
"""

import re

from joblogs import Metric, run_cli

# Anchored on gather.py's stdout.
METRICS = [
    Metric("legs", re.compile(r"leg file\(s\), ([\d,]+) legs"), 1,
           "legs written"),
    Metric("owed", re.compile(r"legs owed: ([\d,]+)"), 1, "legs owed"),
    Metric("files", re.compile(r"^job \d+/\d+: (\d+) file\(s\)", re.M), 1,
           "slimmed files owned"),
    Metric("shards", re.compile(r"wrote (\d+) leg file\(s\)"), 1,
           "leg shards written"),
]

if __name__ == "__main__":
    raise SystemExit(run_cli("3a (gather)", METRICS, __doc__))

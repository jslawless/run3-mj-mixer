#!/usr/bin/env python3
"""plot_assemble_jobs.py - what stage 3b cost per job, per pseudo-event written.

Stage 3b jobs own one pair chunk each: read the chunk plus its P leg shards,
join on `(pair_id, leg)`, write one stitched file. One point is one condor job.

What to look for here:

* **The cloud should be tight**, because `--pairs-per-chunk` fixes the work per
  job by construction. Horizontal spread means chunks came out uneven; vertical
  spread at fixed work is the workers, not the code.
* **Peak memory is the panel that matters.** `build_output` calls
  `transverse_thrust` on the assembled 6-jet events, which is why it runs in
  sub-blocks - a memory slope steeper than the block size explains means the
  blocking is not covering some array. This is also the stage nearest the 4 GB
  limit, so the dotted request line is the one to watch.
* **Wall time's intercept** is the leg-shard read: P shards are opened whatever
  the chunk size, so that cost does not shrink by making chunks smaller.

    python scripts/monitors/plot_assemble_jobs.py batch_assemble
    python scripts/monitors/plot_assemble_jobs.py batch_assemble --x legs
    python scripts/monitors/plot_assemble_jobs.py batch_assemble -o assemble.png
"""

import re

from joblogs import Metric, run_cli

# Anchored on assemble.py's stdout.
METRICS = [
    Metric("events", re.compile(r"chunk \d+: ([\d,]+) pseudo-events"), 1,
           "pseudo-events written"),
    Metric("legs", re.compile(r"pseudo-events from ([\d,]+) legs"), 1,
           "legs joined"),
]

if __name__ == "__main__":
    raise SystemExit(run_cli("3b (assemble)", METRICS, __doc__))

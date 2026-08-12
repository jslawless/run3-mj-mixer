"""condor.py - the shared HTCondor job template.

Stages 1a, 3a and 3b all need the same job wrapper: pick the LCG view matching
the node's OS, build a venv that inherits its packages, install only our wheel,
run something, then stage the output to EOS with xrdcp.

That wrapper lives here **once**. The three existing submitters in this
repository family (mixer, slimmer, evaluator) each carry their own copy and have
already drifted apart - which is the same failure mode that left three subtly
different copies of ``infer_dataset``. A submit script here is a thin CLI that
supplies a job list and the commands to run.

Everything preserved from the working mixer submitter, because each line of it
was a bug once:

* ``transfer_output_files = ""`` - otherwise condor returns every leftover
  sandbox file to the submit dir, which looks like output went local.
* ``LC_ALL=C.UTF-8`` twice - the cvmfs view resets ``LC_*`` when sourced.
* ``--system-site-packages`` + ``pip install --no-deps`` - pulling PyPI wheels
  risks a wheel-vs-libstdc++ ABI mismatch against the LCG runtime; the view's own
  libraries match it.
* ``find``, not a ``**/`` glob - globstar is bash>=4 only and silently collapses
  to one level where it is missing, which would drop every nested output.
* ``xrdcp -f -p`` with paths relative to the job dir, so the local directory
  layout is reproduced under the EOS output dir.
"""

from __future__ import annotations

import os
import re
import stat

LCG_BASE = "/cvmfs/sft.cern.ch/lcg/views/LCG_106"
DEFAULT_REDIRECTOR = "root://cmseos.fnal.gov"


def strip_redirector(path):
    """A bare ``/store/...`` path from either a bare path or a full XRootD URL.

    The job template prepends the redirector itself, so passing a full URL would
    otherwise produce a doubled one.
    """
    return re.sub(r"^root://[^/]+/+", "/", str(path))


def submit_file(logdir, names, transfer, eosoutdir, *, cpu=1,
                queue="tomorrow", ram="4GB", disk="4GB"):
    """The condor .sub file."""
    name_block = "\n".join(f"\t{n}" for n in names)
    return f"""\
universe                = vanilla
executable              = {logdir}/$(name).sh
arguments               = $(ClusterId)$(ProcId)
output                  = {logdir}/log_$(ClusterId)_$(name).out
error                   = {logdir}/log_$(ClusterId)_$(name).err
log                     = {logdir}/log_$(ClusterId)_$(name).log
Should_Transfer_Files   = YES
transfer_input_files    = {transfer}
# The job stages its own output to EOS via xrdcp, so condor must transfer
# NOTHING back. Without this, condor's default returns every leftover top-level
# sandbox file to the submit dir, which looks like "output went local".
transfer_output_files   = ""
RequestCPUs             = {cpu}
+JobFlavour             = {queue}
request_memory          = {ram}
request_disk            = {disk}
use_x509userproxy       = true

queue name from (
{name_block}
)
"""


_HEAD = """\
#!/usr/bin/env bash
echo "Starting job on " `date`
pwd
ls
echo

## The cvmfs LCG view points LC_* at a UTF-8 locale the minimal cms:rhel9
## container lacks. Force an always-present one, and re-assert after sourcing
## the view because the view resets LC_*.
export LC_ALL=C.UTF-8 LANG=C.UTF-8 LC_CTYPE=C.UTF-8

## Pick the view matching this node's OS and the newest gcc available for it.
LCG_BASE={lcg_base}
osmaj=$(rpm -E %{{rhel}} 2>/dev/null || echo 9)
LCG_VIEW=$(ls "$LCG_BASE"/x86_64-el${{osmaj}}-gcc*-opt/setup.sh 2>/dev/null | sort -V | tail -1)
if [ -z "$LCG_VIEW" ] || [ ! -r "$LCG_VIEW" ]; then
  LCG_VIEW=$(ls "$LCG_BASE"/x86_64-el*-gcc*-opt/setup.sh 2>/dev/null | sort -V | tail -1)
fi
echo "Node OS major: $osmaj"
echo "Sourcing LCG view: $LCG_VIEW"
source "$LCG_VIEW"
export LC_ALL=C.UTF-8 LANG=C.UTF-8 LC_CTYPE=C.UTF-8
echo "Base python: $(python3 --version)"

## A venv that INHERITS the view's packages; install ONLY our wheel, no deps.
## Do NOT unset PYTHONPATH - that is how the view exposes uproot/awkward.
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --quiet --no-deps {wheel}
echo "uproot: $(python3 -c 'import uproot; print(uproot.__version__)')"

echo
## Abort before any upload if a command fails, so partial output is never
## delivered to EOS and the job exits non-zero.
set -e
{run_commands}
set +e
"""

_DELIVER = """
echo
echo "outputs:"
find . -type f -name '*.root' | sort
OUTDIR={eos_url}
echo "delivering to $OUTDIR"

# find, not a **/ glob: globstar is bash>=4 only, and where it is missing **
# silently collapses to one level - which would drop every nested output.
root_files=()
while IFS= read -r FOUND; do
  root_files+=( "${FOUND#./}" )
done < <(find . -type f -name '*.root' | sort)
if [[ ${#root_files[@]} -eq 0 ]]; then
  echo "ERROR: job produced no .root output - nothing to deliver." >&2
  exit 1
fi
# FILE is relative to the job dir, so the local layout is reproduced under
# $OUTDIR; xrdcp -p creates the destination directories.
for FILE in "${root_files[@]}"
do
  echo "xrdcp -f -p ${FILE} ${OUTDIR}/${FILE}"
  xrdcp -f -p "${FILE}" "${OUTDIR}/${FILE}" 2>&1
  XRDEXIT=$?
  if [[ $XRDEXIT -ne 0 ]]; then
    echo "ERROR: xrdcp of ${FILE} failed (exit ${XRDEXIT})." >&2
    rm -f -- "${FILE}"
    exit $XRDEXIT
  fi
  rm -f -- "${FILE}"
done

echo
echo "Ending job on " `date`
exit 0
"""


def eos_url(eosoutdir, redirector=DEFAULT_REDIRECTOR):
    """``root://host//store/...`` - with the DOUBLE slash XRootD requires.

    ``root://host/store/...`` (one slash) is a different, wrong path and fails at
    the door. The old submitter got this right only because the slash was baked
    into its template literal; concatenating a redirector with a bare
    ``/store/...`` path silently produces the single-slash form.
    """
    return f"{redirector.rstrip('/')}/{strip_redirector(eosoutdir)}"


def job_script(run_commands, eosoutdir, wheel, *,
               redirector=DEFAULT_REDIRECTOR, lcg_base=LCG_BASE):
    """The per-job .sh: environment, the work, then delivery to EOS."""
    head = _HEAD.format(lcg_base=lcg_base, wheel=os.path.basename(wheel),
                        run_commands=run_commands)
    deliver = _DELIVER.replace("{eos_url}", eos_url(eosoutdir, redirector))
    return head + deliver


def write_jobs(logdir, jobs, transfer, eosoutdir, wheel, *, cpu=1,
               queue="tomorrow", ram="4GB", disk="4GB",
               redirector=DEFAULT_REDIRECTOR, submit_name="submit.sub"):
    """Write one .sh per job plus the .sub. Returns the .sub path.

    ``jobs`` maps job name -> the shell commands that job should run.
    """
    os.makedirs(str(logdir), exist_ok=True)
    for name, cmds in jobs.items():
        path = os.path.join(str(logdir), f"{name}.sh")
        with open(path, "w") as f:
            f.write(job_script(cmds, eosoutdir, wheel, redirector=redirector))
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP
                 | stat.S_IXOTH)
    sub = os.path.join(str(logdir), submit_name)
    with open(sub, "w") as f:
        f.write(submit_file(logdir, sorted(jobs), transfer, eosoutdir,
                            cpu=cpu, queue=queue, ram=ram, disk=disk))
    return sub


def transfer_list(*paths):
    """The comma-joined ``transfer_input_files`` value.

    Condor flattens transfers into the job CWD, so every in-job reference must
    use the basename - which the run commands below all do.
    """
    return ",".join(str(p) for p in paths if p)


def add_common_args(p, *, ram="4GB", disk="4GB"):
    """The submit flags every stage shares."""
    p.add_argument("-o", "--eosoutdir", required=True,
                   help="the SHARED output parent as a bare /store/... path; each "
                        "stage writes its own subdirectory under it (index/, "
                        "legs/, stitched/), so pass the same value to every "
                        "stage. The redirector is added in-job.")
    p.add_argument("--wheel", required=True, help="the built run3-mj-mixer wheel")
    p.add_argument("--logdir", default="batch")
    p.add_argument("--cpu", type=int, default=1)
    p.add_argument("--queue", default="tomorrow")
    p.add_argument("--memory", default=ram)
    p.add_argument("--disk", default=disk)
    p.add_argument("--redirector", default=DEFAULT_REDIRECTOR)
    p.add_argument("--exec", dest="do_exec", action="store_true",
                   help="run condor_submit instead of only writing the files")
    return p


def maybe_submit(sub_path, do_exec):
    print(f"wrote {sub_path}")
    if not do_exec:
        print(f"  (dry run; submit with: condor_submit {sub_path})")
        return 0
    import subprocess
    return subprocess.call(["condor_submit", sub_path])

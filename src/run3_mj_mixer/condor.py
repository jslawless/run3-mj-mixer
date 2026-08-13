"""condor.py - the shared HTCondor job template.

Stages 1a, 3a and 3b all need the same job wrapper: source the pinned LCG view,
build a venv that inherits its packages, install only our wheel, run something,
then stage the output to EOS with xrdcp.

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
* the DOUBLE slash in ``root://host//store/...`` - one slash is a different path.
"""

from __future__ import annotations

import os
import re
import shlex
import stat

# The pinned environment. ONE constant, used by the job template and quoted by
# docs/how_to_run.md, so the login node and the workers cannot drift apart.
#
# Pinned rather than discovered, deliberately: a fixed view makes a run
# reproducible months later, and an LCG view being retired or re-pointed is
# exactly the kind of change that should break loudly at submit time rather than
# silently alter results.
#
# The cost of pinning the PLATFORM (not just the version): the view's binaries are
# built against that OS's glibc/libstdc++, so this string only works on el9
# nodes. cmslpc condor defaults to the cms:rhel9 apptainer, so that is the norm -
# but if jobs start landing on el8, this is the line to change, and
# `--lcg-view` overrides it without an edit.
LCG_VERSION = "LCG_106"
LCG_PLATFORM = "x86_64-el9-gcc13-opt"
LCG_VIEW = f"/cvmfs/sft.cern.ch/lcg/views/{LCG_VERSION}/{LCG_PLATFORM}"

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

## CONTAINER-only fix, not needed on a login node: condor runs inside the
## minimal cms:rhel9 apptainer, which ships almost no locales, and the LCG view
## sets LC_* to a UTF-8 locale that image lacks - so every tool it runs warns
## "Setting LC_CTYPE failed, using C". C.UTF-8 is built into glibc on el8/el9, so
## it is always present. Re-asserted after sourcing the view, which resets LC_*.
export LC_ALL=C.UTF-8 LANG=C.UTF-8 LC_CTYPE=C.UTF-8

## The PINNED LCG view - one constant in condor.py, quoted by the docs, so the
## login node and the workers cannot drift. Fail fast and loudly: 300 jobs dying
## on a cryptic import error is far worse than one clear line here.
LCG_VIEW={lcg_view}
echo "Node OS major: $(rpm -E %{{rhel}} 2>/dev/null || echo unknown)"
echo "Pinned LCG view: $LCG_VIEW"
if [ ! -r "$LCG_VIEW/setup.sh" ]; then
  echo "ERROR: $LCG_VIEW/setup.sh is not readable." >&2
  echo "  Either cvmfs is not mounted on this node, or the pinned view has" >&2
  echo "  moved. Available builds of this version:" >&2
  ls -d "$(dirname "$LCG_VIEW")"/x86_64-el*-gcc*-opt 2>/dev/null >&2 \
    || echo "    (none - the whole version is gone)" >&2
  echo "  Change LCG_VERSION / LCG_PLATFORM in src/run3_mj_mixer/condor.py," >&2
  echo "  or pass --lcg-view to the submitter." >&2
  exit 1
fi
source "$LCG_VIEW/setup.sh"
export LC_ALL=C.UTF-8 LANG=C.UTF-8 LC_CTYPE=C.UTF-8
echo "Base python: $(python3 --version)"

## A venv that INHERITS the view's packages; install ONLY our wheel, no deps.
## Do NOT unset PYTHONPATH - that is how the view exposes uproot/awkward.
python3 -m venv --system-site-packages pkg-env
source pkg-env/bin/activate
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
               redirector=DEFAULT_REDIRECTOR, lcg_view=LCG_VIEW):
    """The per-job .sh: environment, the work, then delivery to EOS."""
    head = _HEAD.format(lcg_view=lcg_view, wheel=os.path.basename(wheel),
                        run_commands=run_commands)
    deliver = _DELIVER.replace("{eos_url}", eos_url(eosoutdir, redirector))
    return head + deliver


def write_jobs(logdir, jobs, transfer, eosoutdir, wheel, *, cpu=1,
               queue="tomorrow", ram="4GB", disk="4GB",
               redirector=DEFAULT_REDIRECTOR, lcg_view=LCG_VIEW,
               submit_name="submit.sub"):
    """Write one .sh per job plus the .sub. Returns the .sub path.

    ``jobs`` maps job name -> the shell commands that job should run.
    """
    os.makedirs(str(logdir), exist_ok=True)
    for name, cmds in jobs.items():
        path = os.path.join(str(logdir), f"{name}.sh")
        with open(path, "w") as f:
            f.write(job_script(cmds, eosoutdir, wheel, redirector=redirector,
                               lcg_view=lcg_view))
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
    p.add_argument("--lcg-view", default=LCG_VIEW,
                   help=f"pinned LCG view (default {LCG_VIEW}); override without "
                        "editing the source if nodes move OS")
    p.add_argument("--allow-stale-wheel", action="store_true",
                   help="submit even if the wheel predates the source (the jobs "
                        "will run whatever the wheel contains)")
    p.add_argument("--exec", dest="do_exec", action="store_true",
                   help="run condor_submit instead of only writing the files")
    return p


def check_wheel_fresh(wheel, *, allow_stale=False):
    """Refuse to submit a wheel older than the source it was built from.

    The submitter imports the package from ``src/`` but the jobs install the
    WHEEL, so editing source without rebuilding leaves the two out of step. The
    symptom is remote and unhelpful - typically argparse rejecting a flag the
    local code has - and it costs a full submission to discover. Comparing
    mtimes catches it in one line, before anything is queued.
    """
    if not os.path.exists(wheel):
        raise SystemExit(f"wheel not found: {wheel}\n  build it with: pip wheel . -w .")
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    newest, newest_f = 0.0, None
    for root, _, files in os.walk(src):
        for f in files:
            if f.endswith(".py"):
                m = os.path.getmtime(os.path.join(root, f))
                if m > newest:
                    newest, newest_f = m, os.path.join(root, f)
    if newest > os.path.getmtime(wheel):
        msg = (f"the wheel is older than the source:\n"
               f"    wheel  {wheel}\n"
               f"    newer  {newest_f}\n"
               "  Jobs install the wheel, not your checkout, so they would run "
               "stale code.\n  Rebuild:  rm -f run3_mj_mixer-*.whl && pip wheel . -w .")
        if not allow_stale:
            raise SystemExit(msg)
        print("WARNING: " + msg)


def submit(sub_path):
    """Run condor_submit on a .sub file. Returns its exit code.

    Through a shell, not a direct exec: ``subprocess`` execs the file itself, so
    it dies with ENOEXEC ("Exec format error") on a shell function, an alias, or
    a wrapper with no ``#!``. The shell handles all three and does the PATH
    search for us - the same reason the filelist scripts call the ``eos`` binary
    rather than the ``eosls`` shell function.
    """
    import subprocess

    cmd = f"condor_submit {shlex.quote(str(sub_path))}"
    rc = subprocess.call(["bash", "-lc", cmd])
    if rc != 0:
        # The .sub file is written and complete, so say that submission failed
        # rather than leaving it looking like the jobs were never generated.
        print(f"  submit failed (exit {rc}); the job files are fine - "
              f"run it yourself:\n      {cmd}")
    return rc

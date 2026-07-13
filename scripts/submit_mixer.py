#!/usr/bin/env python3
"""submit_mixer.py - Submit run3-mj-mixer jobs to HTCondor.

Reads a coffea-style fileset JSON of SLIMMED files (the output of run3-mj-slimmer),
splits files into per-job groups, and writes condor submission files. Each job
installs run3-mj-mixer from a pre-built wheel and runs it on its assigned files.

Build the wheel before submitting:
    pip wheel /path/to/run3-mj-mixer -w .

Each job installs the wheel with --no-deps into a venv that inherits the LCG
view's uproot/awkward/numpy/boost-histogram, so no large PyPI downloads happen
on the worker node (and the view's native libs always match the runtime, which
sidesteps manylinux wheel-vs-libstdc++ ABI crashes). The mixer needs no coffea /
onnxruntime, so the venv is light.

Submit:
    python submit_mixer.py \\
        -i fileset.json \\
        -o /store/user/you/mixed \\
        --config config.json \\
        --wheel run3_mj_mixer-1.0.0-py3-none-any.whl

-o / --eosoutdir is a BARE EOS path (e.g. /store/user/you/mixed); the job
script adds the root://cmseos.fnal.gov/ redirector automatically. A full
root://host//store/... URL is also accepted - the leading redirector is
stripped so it is never doubled in the xrdcp destination.

Fileset JSON format (coffea-style; tree is 'events' for slimmed files):
    {
        "dataset_name": {
            "files": {
                "root://.../slimmed_file.root": "events",
                ...
            }
        }
    }
"""

import os
import re
import argparse
import json

# The cross sections live in the shared aux repo, assumed checked out next to
# run3-mj-mixer (same convention as the analyzer). Transferred into each job so
# mix.py finds ./mj_samples_xs.json on the worker (no sibling dir there).
_DEFAULT_XS_JSON = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "run3-mj-pass-the-aux", "mj_samples_xs.json")
)


def configure_batch(logdir, names, transfer, eosoutdir, cpu, queue, ram, disk):
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
# sandbox file to the submit dir -- e.g. a mixed_*.root from a job that died
# before its xrdcp -- which looks like "output went local instead of EOS".
transfer_output_files   = ""
RequestCPUs             = {cpu}
+JobFlavour             = {queue}
request_memory          = {ram}
request_disk            = {disk}
use_x509userproxy       = true

queue name from (
{names}
)
"""


EXECUTABLE_TEMPLATE = """\
#!/usr/bin/env bash
echo "Starting job on " `date`
echo "Running on: `uname -a`"
echo "System software: `cat /etc/redhat-release`"
workarea=$PWD
echo
echo "Work Area: $workarea"
ls
echo

## The cvmfs LCG view points LC_* at a UTF-8 locale the minimal cms:rhel9
## container lacks, so tools it runs warn "Setting LC_CTYPE failed, using C".
## Force an always-present locale (C.UTF-8 is built into glibc on el8/el9);
## re-asserted after sourcing the view in case it overrides LC_*.
export LC_ALL=C.UTF-8 LANG=C.UTF-8 LC_CTYPE=C.UTF-8

## run3-mj-mixer needs Python >=3.8 plus uproot/awkward/numpy/boost-histogram -
## all already provided by the cvmfs LCG view, compiled against the view's own
## gcc/libstdc++. So source the view and build a venv that INHERITS its
## site-packages (--system-site-packages), then install ONLY our package on top
## with --no-deps.
##
## Do NOT pip-install the deps: pulling PyPI wheels risks a wheel-vs-libstdc++
## ABI mismatch against the LCG runtime. The view's own libs load fine (they
## match its libstdc++), so we just use them. This also makes the job
## OS-agnostic: on el8 or el9 the selected view's native libs always match.
##
## Pick the view matching this node's OS and the newest gcc available for it.
LCG_BASE=/cvmfs/sft.cern.ch/lcg/views/LCG_106
osmaj=$(rpm -E %{{rhel}} 2>/dev/null || echo 9)
LCG_VIEW=$(ls "$LCG_BASE"/x86_64-el${{osmaj}}-gcc*-opt/setup.sh 2>/dev/null | sort -V | tail -1)
if [ -z "$LCG_VIEW" ] || [ ! -r "$LCG_VIEW" ]; then
  # Last resort: newest gcc for any arch this node can run.
  LCG_VIEW=$(ls "$LCG_BASE"/x86_64-el*-gcc*-opt/setup.sh 2>/dev/null | sort -V | tail -1)
fi
echo "Node OS major: $osmaj"
echo "Sourcing LCG view: $LCG_VIEW"
source "$LCG_VIEW"
export LC_ALL=C.UTF-8 LANG=C.UTF-8 LC_CTYPE=C.UTF-8   # re-assert: the view may reset LC_*
echo "Base python: $(python3 --version)"

## Build a venv that inherits the view's packages; install ONLY our wheel, with
## no PyPI deps. Do NOT unset PYTHONPATH - that is how the view exposes its
## uproot/awkward to the venv.
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --quiet --no-deps {WHEEL}
echo "uproot: $(python3 -c 'import uproot; print(uproot.__version__)')"

## Run
echo
# Abort (non-zero exit, nothing uploaded) if any mixer invocation fails, so
# partial / empty outputs are never xrdcp'd to EOS.
set -e
{RUN_COMMANDS}
set +e
echo "what directory am I in?"
pwd
echo "List all root files = "
ls *.root 2>/dev/null || echo "  (no .root output produced)"
echo "List all files"
ls -alh
echo "*******************************************"
OUTDIR=root://cmseos.fnal.gov/{EOSOUTDIR}
echo "xrdcp output for condor to "
"""

EXECUTABLE_TEMPLATE2 = """\
echo $OUTDIR
# Fail loudly (non-zero exit) if the mixer delivered no output, instead of
# the old confusing "xrdcp ... no such file" when the *.root glob is empty.
shopt -s nullglob
root_files=( *.root )
if [[ ${#root_files[@]} -eq 0 ]]; then
  echo "ERROR: mixer produced no .root output - nothing to deliver to EOS." >&2
  exit 1
fi
for FILE in "${root_files[@]}"
do
  echo "xrdcp -f ${FILE} ${OUTDIR}/${FILE}"
  xrdcp -f "${FILE}" "${OUTDIR}/${FILE}" 2>&1
  XRDEXIT=$?
  if [[ $XRDEXIT -ne 0 ]]; then
    echo "ERROR: xrdcp of ${FILE} failed (exit ${XRDEXIT}); output NOT delivered." >&2
    rm -f -- "${FILE}"   # worker scratch only
    exit $XRDEXIT
  fi
  rm -f -- "${FILE}"     # worker scratch only
done

echo
echo "Ending job on " `date`
"""


class Fileset:
    def __init__(self, args):
        self.infile = args.inFile
        self.nf_per_job = args.nfPerJob
        self.eosoutdir = args.eosoutdir
        self.logdir = args.logdir
        self.fileset = {}
        self.jobs = []

        self._read()
        self._split()
        os.makedirs(self.logdir, exist_ok=True)

    def _read(self):
        try:
            with open(self.infile) as f:
                self.fileset = json.load(f)
        except FileNotFoundError:
            raise SystemExit(f"Fileset not found: {self.infile}")
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid JSON in {self.infile}: {e}")

    def _split(self):
        # 'unused' is make_mixing_jobs.py's leftover bucket (older versions put
        # it in the main fileset) - bookkeeping, never a job group.
        if "unused" in self.fileset:
            n_skip = len(self.fileset.pop("unused").get("files", {}))
            print(f"\n  Skipping 'unused' ({n_skip} leftover files, not a job group).")
        print(f"\nDatasets: {len(self.fileset)}")
        total = 0
        for k, (dataset, data) in enumerate(self.fileset.items()):
            files = list(data["files"].items())  # [(path, tree_name), ...]
            n = self.nf_per_job
            subjobs = [files[i:i + n] for i in range(0, len(files), n)]
            self.jobs.append((dataset, subjobs))
            print(f"  {k + 1}: {dataset}  ->  {len(files)} files  ->  {len(subjobs)} jobs")
            total += len(subjobs)
        print(f"\n  Total: {total} jobs\n")


class Batch:
    def __init__(self, jobs, args):
        self.jobs = jobs
        self.eosoutdir = args.eosoutdir
        self.logdir = args.logdir
        self.cpu = args.cpu
        self.queue = args.queue
        self.ram = args.memory
        self.disk = args.disk
        self.config = args.config
        self.wheel = args.wheel
        self.xs_json = args.xs_json
        self.default_tree = args.tree
        self._write_jobs()
        self._write_submit()

    def _write_jobs(self):
        wheel_basename = os.path.basename(self.wheel)
        config_basename = os.path.basename(self.config)
        for dataset, subjobs in self.jobs:
            single = (len(subjobs) == 1)
            for i, files in enumerate(subjobs):
                name = dataset if single else f"{dataset}_{i}"
                run_cmds = []
                for filepath, tree in files:
                    tree_name = tree if tree else self.default_tree
                    # Tag with the dataset name only (not the per-job index):
                    # the input basename already makes each output unique, so
                    # the output is mixed_<dataset>_<input basename>.
                    run_cmds.append(
                        f"run3-mj-mixer {filepath} {config_basename}"
                        f" --tree {tree_name}"
                        f" --output-tag {dataset}"
                    )
                exe = EXECUTABLE_TEMPLATE.format(
                    WHEEL=wheel_basename,
                    RUN_COMMANDS="\n".join(run_cmds),
                    EOSOUTDIR=self.eosoutdir,
                )
                exe = exe + EXECUTABLE_TEMPLATE2
                path = f"{self.logdir}/{name}.sh"
                with open(path, "w") as f:
                    f.write(exe)
                os.chmod(path, 0o755)

    def _write_submit(self):
        names = ""
        for dataset, subjobs in self.jobs:
            single = (len(subjobs) == 1)
            for i in range(len(subjobs)):
                name = dataset if single else f"{dataset}_{i}"
                names += f"\t{name}\n"

        transfer_files = [self.wheel, self.config]
        if self.xs_json and os.path.isfile(self.xs_json):
            # Lands in the job CWD; mix.py's locate_xs_json picks up ./mj_samples_xs.json.
            transfer_files.append(self.xs_json)
        else:
            print(f"  Warning: xs JSON not found ({self.xs_json}); jobs will run "
                  "unweighted (weight 1.0). Pass --xs-json to point at "
                  "mj_samples_xs.json.")
        transfer = ",".join(transfer_files)
        config = configure_batch(
            logdir=self.logdir,
            names=names.strip(),
            transfer=transfer,
            eosoutdir=self.eosoutdir,
            cpu=self.cpu,
            queue=self.queue,
            ram=self.ram,
            disk=self.disk,
        )
        with open(f"{self.logdir}/submit.sub", "w") as f:
            f.write(config)

    def submit(self, execute):
        if execute:
            os.system(f"condor_submit {self.logdir}/submit.sub")
            print()
            print("Your jobs are here:")
            os.system("condor_q")
            print()
        else:
            print()
            print(f"To submit:       condor_submit {self.logdir}/submit.sub")
            print("To check status: condor_q")
            print("To see jobs:     condor_q -nobatch")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit run3-mj-mixer jobs to HTCondor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--inFile",   required=True,  help="Coffea-style fileset JSON of slimmed files")
    parser.add_argument("-o", "--eosoutdir", required=True, help="EOS output dir as a bare /store/... path (cmseos redirector added automatically)")
    parser.add_argument("--config",         required=True,  help="run3-mj-mixer config JSON")
    parser.add_argument("--wheel",          required=True,  help="Pre-built run3-mj-mixer .whl file")
    parser.add_argument("--xs-json",        default=_DEFAULT_XS_JSON,
                        help="Cross-section JSON transferred to each job for hemisphere "
                             "weighting (default: the run3-mj-pass-the-aux sibling repo's "
                             "mj_samples_xs.json).")
    parser.add_argument("-n", "--nfPerJob", type=int, default=1, help="Files per job")
    parser.add_argument("--tree",   default="events",   help="Fallback input tree name (overridden by fileset JSON; slimmed files use 'events')")
    parser.add_argument("--logdir", default="batch",    help="Directory for condor log/sh files")
    parser.add_argument("--cpu",    type=int, default=1, help="CPUs per job")
    parser.add_argument("--queue",  default="tomorrow", help="HTCondor JobFlavour")
    parser.add_argument("--memory", default="4GB",      help="Memory per job")
    parser.add_argument("--disk",   default="3GB",      help="Disk per job (venv inherits LCG packages; mostly root-file I/O headroom)")
    parser.add_argument("--exec",   action="store_true", help="Submit jobs immediately after writing")

    args = parser.parse_args()

    # The job template (EXECUTABLE_TEMPLATE) already prepends the cmseos.fnal.gov
    # redirector to the output dir, so accept a bare /store/... path. If a full
    # root://host//store/... URL is given, strip the leading redirector to avoid
    # a doubled prefix in the xrdcp destination.
    args.eosoutdir = re.sub(r"^root://[^/]+/+", "/", args.eosoutdir)

    fileset = Fileset(args)
    batch = Batch(fileset.jobs, args)
    batch.submit(args.exec)

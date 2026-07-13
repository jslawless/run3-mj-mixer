#!/usr/bin/env python3
"""mix.py - hemisphere-mixing prototype for the 6-jet multijet analysis.

Reads a slimmed ROOT file (produced by run3-mj-slimmer) and, for every event
with exactly ``mixer.n_jets`` jets (others are dropped):
  1. finds the transverse thrust axis n_T (in the transverse x-y plane),
  2. splits the event into two hemispheres by the plane perpendicular to n_T,
  3. sums the jet four-vectors in each hemisphere.

The output ROOT file is a COPY of the slimmed input (all branches passed
through unchanged) plus:
  - TTree 'events'  : + 'thrust_axis_phi' (rad, in [0, pi)), 'thrust'
                      (normalized transverse-thrust value in (0, 1]), and a
                      'Hemisphere' collection (2 entries/event) holding the
                      summed jet four-vector (px, py, pz, energy and the derived
                      pt, eta, phi, mass), its thrust-projected components
                      (pt_par along n_T, pt_perp perpendicular to it), the jet
                      multiplicity n_jets, and the side (+1 = along +n_T).
  - TH1  'thrust_axis_phi'  : histogram of the per-event thrust-axis angle.
  - TH1  'hemisphere_njets' : histogram of the per-hemisphere jet multiplicity.
  - TH1  'mixer_cutflow'    : bin 1 = events read, bin 2 = events with exactly
                              n_jets jets (the ones written out).
  - TH1  'version'          : mixer config metadata.version string.
  - TTree 'meta'            : mixer version, config version, scan parameters.
  - 'cutflow' and 'slimmer_version' are passed through from the input if present.

This is a PROTOTYPE covering Steps 1-2 of the hemisphere-mixing method (split +
per-hemisphere characterization). The library, nearest-neighbor matching and
5->6 stitching steps are not implemented here.

Usage:
    python mix.py input.root config.json
    python mix.py input.root config.json --tree events --chunk-size 50000

Output is written to the current directory as mixed_<input basename>, e.g.
input /path/to/slimmed_X.root -> ./mixed_slimmed_X.root

Config JSON format:
    {
        "metadata": {"version": "v1"},
        "mixer": {
            "thrust_scan_points": 180,
            "hist_bins":          90,
            "n_jets":             5
        }
    }
"""

import argparse
import json
import os
import sys
from pathlib import Path

import awkward as ak
import boost_histogram as bh
import numpy as np
import uproot

_JET_BRANCH      = "ScoutingPFJet"
_GEN_JET_BRANCH  = "GenJet"
_GEN_PART_BRANCH = "GenPart"

# Collections that may be stored either as a nested record ("<B>.pt") or as flat
# NanoAOD-style branches ("<B>_pt"); both are regrouped on pass-through so uproot
# emits one shared offset counter instead of one per field.
_REGROUP_COLLECTIONS = [_JET_BRANCH, _GEN_JET_BRANCH, _GEN_PART_BRANCH]

_REQUIRED_METADATA_KEYS = {
    "version": str,
}

_REQUIRED_MIXER_KEYS = {
    "thrust_scan_points": int,
    "hist_bins":          int,
    "n_jets":             int,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and validate a mixer config JSON file."""
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Config file not found: {config_path}")
    except json.JSONDecodeError as exc:
        sys.exit(f"Invalid JSON in {config_path}: {exc}")

    for section, required in (("metadata", _REQUIRED_METADATA_KEYS),
                              ("mixer", _REQUIRED_MIXER_KEYS)):
        if section not in cfg:
            sys.exit(f"Config missing top-level section: '{section}'")
        for key, expected in required.items():
            if key not in cfg[section]:
                sys.exit(f"Config section '{section}' missing required key: '{key}'")
            # bool is a subclass of int; reject it where a real int is wanted.
            if expected is int and isinstance(cfg[section][key], bool):
                sys.exit(f"Config '{section}.{key}' must be an int, not a bool.")
            if not isinstance(cfg[section][key], expected):
                sys.exit(
                    f"Config '{section}.{key}' has wrong type: "
                    f"expected {expected.__name__}, got {type(cfg[section][key]).__name__}"
                )

    if cfg["mixer"]["thrust_scan_points"] < 2:
        sys.exit("Config 'mixer.thrust_scan_points' must be >= 2.")
    if cfg["mixer"]["hist_bins"] < 1:
        sys.exit("Config 'mixer.hist_bins' must be >= 1.")
    if cfg["mixer"]["n_jets"] < 1:
        sys.exit("Config 'mixer.n_jets' must be >= 1.")

    return cfg


# ---------------------------------------------------------------------------
# Cross-section weighting
# ---------------------------------------------------------------------------
#
# Each hemisphere inherits its source event's MC weight so that when the library
# is built from a mix of QCD HT slices, every slice enters with its physical
# abundance. The convention matches the analyzer (compare_qcd_slimming.py):
#
#     weight = lumi * xs_pb / n_original
#
# with ``xs_pb`` from the shared aux repo's mj_samples_xs.json (keyed by the full
# slice/dataset name) and ``n_original`` the slimmer's cutflow[0] (events read
# before any cut). NOTE: the mixer runs per file, so by default ``n_original`` is
# THIS file's cutflow[0] - the file's own contribution to the slice denominator.
# For absolute per-slice normalization pass the slice-summed n_original via
# ``--n-original`` (the analyzer's load_fileset sums cutflow[0] over a slice).

_XS_FILENAME = "mj_samples_xs.json"
_AUX_DIRNAME = "run3-mj-pass-the-aux"


def locate_xs_json(explicit=None):
    """Locate mj_samples_xs.json. Search order:

    1. ``explicit`` (the --xs-json argument),
    2. ``$RUN3_MJ_XS_JSON``,
    3. ``./mj_samples_xs.json`` (a copy transferred into the condor job dir),
    4. a ``run3-mj-pass-the-aux/`` sibling found by walking up from this module
       or the CWD - i.e. the aux repo assumed to live in the same parent dir as
       run3-mj-mixer.

    Returns a path string, or None if nothing was found.
    """
    cands = []
    if explicit:
        cands.append(Path(explicit))
    env = os.environ.get("RUN3_MJ_XS_JSON")
    if env:
        cands.append(Path(env))
    cands.append(Path.cwd() / _XS_FILENAME)
    for base in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in base.parents:
            cands.append(parent / _AUX_DIRNAME / _XS_FILENAME)
    for c in cands:
        if c.is_file():
            return str(c)
    return None


def infer_dataset(path, xs_keys):
    """Return the xs.json key that best matches a file path (the longest key that
    is a substring of the basename), or None. Slimmed/mixed filenames embed the
    dataset name (the slimmer's --output-tag), so this recovers the HT slice."""
    base = os.path.basename(path)
    matches = [k for k in xs_keys if k in base]
    return max(matches, key=len) if matches else None


def resolve_xs_weight(input_path, xs_json_path, dataset, lumi, n_original,
                      require_xs=False):
    """Compute the per-event xsec weight ``lumi * xs_pb / n_original`` for a file.

    ``dataset`` may be given explicitly or left None to infer from the filename.
    ``n_original`` is the denominator (this file's cutflow[0], or a --n-original
    override). Returns ``(weight, info)``; ``info`` records how the weight was
    resolved. When no cross section can be resolved the weight is 1.0 (a warning
    is printed), unless ``require_xs`` is set, in which case it exits.
    """
    info = {"dataset": dataset, "xs_pb": None, "n_original": n_original,
            "lumi": lumi, "weight": 1.0, "source": xs_json_path}

    def _fail(msg):
        if require_xs:
            sys.exit(f"Cross-section weighting failed: {msg}")
        print(f"WARNING: {msg} -> using weight 1.0 (unweighted).")
        return 1.0, info

    if xs_json_path is None:
        return _fail(f"could not locate {_XS_FILENAME} "
                     f"(pass --xs-json or put {_AUX_DIRNAME}/ beside run3-mj-mixer)")
    try:
        with open(xs_json_path) as f:
            xs = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return _fail(f"could not read {xs_json_path}: {exc}")

    if dataset is None:
        dataset = infer_dataset(input_path, xs.keys())
        info["dataset"] = dataset
    if dataset is None:
        return _fail(f"no dataset in {xs_json_path} matches '{os.path.basename(input_path)}' "
                     "(pass --dataset)")
    if dataset not in xs:
        return _fail(f"dataset '{dataset}' not in {xs_json_path}")
    if "xs_pb" not in xs[dataset]:
        return _fail(f"'xs_pb' missing for '{dataset}' in {xs_json_path}")
    if not n_original or n_original <= 0:
        return _fail(f"n_original for '{dataset}' is {n_original} "
                     "(need a positive cutflow[0] or --n-original)")

    xs_pb = float(xs[dataset]["xs_pb"])
    weight = lumi * xs_pb / n_original
    info.update(dataset=dataset, xs_pb=xs_pb, weight=weight)
    return weight, info


# ---------------------------------------------------------------------------
# Jet sub-branch access (nested vs flat layout) - shared with the evaluator
# ---------------------------------------------------------------------------

def jet_format(keys, branch=_JET_BRANCH):
    """Return "nested", "flat" or None for how a collection is stored in keys.

    uproot reports a tree's branches with dotted names, so the layouts are
    distinguishable directly: a nested "<branch>" record shows up as
    "<branch>.pt", the flat NanoAOD layout as "<branch>_pt".
    """
    keys = set(keys)
    if f"{branch}.pt" in keys:
        return "nested"
    if f"{branch}_pt" in keys:
        return "flat"
    return None


def jet_subarrays(chunk, branch=_JET_BRANCH):
    """Return (pt, eta, phi, m) jagged arrays for a collection, nested or flat."""
    fields = set(ak.fields(chunk))
    if branch in fields and ak.fields(chunk[branch]):
        jets = chunk[branch]
        return jets["pt"], jets["eta"], jets["phi"], jets["m"]
    if f"{branch}_pt" in fields:
        return (
            chunk[f"{branch}_pt"],
            chunk[f"{branch}_eta"],
            chunk[f"{branch}_phi"],
            chunk[f"{branch}_m"],
        )
    raise KeyError(
        f"Could not find jet branches '{branch}.pt' (nested) or "
        f"'{branch}_pt' (flat) in chunk fields: {sorted(fields)}"
    )


def _passthrough_branches(chunk, regroup_collections):
    """Build an out_record dict from a chunk, regrouping flat collection branches
    in ``regroup_collections`` into a single zipped record; everything else is
    passed through unchanged so the output is a faithful copy of the input.
    """
    chunk_fields = ak.fields(chunk)
    flat_by_collection = {
        b: [f for f in chunk_fields if f.startswith(f"{b}_")]
        for b in regroup_collections
    }
    all_flat       = {f for fs in flat_by_collection.values() for f in fs}
    skip_counters  = {f"n{b}" for b, fs in flat_by_collection.items() if fs}
    nested_targets = set(regroup_collections)

    out_record = {}
    for branch in chunk_fields:
        val = chunk[branch]
        if branch in nested_targets and ak.fields(val):
            out_record[branch] = ak.zip({f: val[f] for f in ak.fields(val)})
        elif branch in skip_counters or branch in all_flat:
            continue
        else:
            out_record[branch] = val

    for b, fs in flat_by_collection.items():
        if fs:
            out_record[b] = ak.zip({f[len(b) + 1:]: chunk[f] for f in fs})

    return out_record


# ---------------------------------------------------------------------------
# Transverse thrust axis
# ---------------------------------------------------------------------------

def _proj_sum(px, py, angle):
    """Sum_i |px_i cos(angle) + py_i sin(angle)| per event.

    ``angle`` is either a python float or a per-event numpy array; awkward
    broadcasts it across the jagged jet axis. Returns a (events,) numpy array.
    """
    return ak.to_numpy(ak.sum(np.abs(px * np.cos(angle) + py * np.sin(angle)), axis=1))


def transverse_thrust(px, py, sum_pt, n_scan):
    """Per-event transverse thrust axis angle and normalized thrust value.

    Maximizes  H(phi) = sum_i |p_T,i . n(phi)|  over the axis angle phi, where
    n(phi) = (cos phi, sin phi). H is pi-periodic (n and -n give the same value),
    so the axis lives on [0, pi). The maximum is found by an ``n_scan``-point
    grid scan on [0, pi) followed by a parabolic refinement of the best grid
    point (H is smooth away from the kinks where a jet's projection changes
    sign, so a 3-point parabola gives a good sub-grid estimate).

    px, py : jagged (events, var jets) awkward arrays of jet momentum components.
    sum_pt : (events,) numpy array, sum_i |p_T,i| (the thrust denominator).

    Returns (phi_T, T) as float32 numpy arrays; T = H(phi_T) / sum_pt in (0, 1].
    Events with no jets (sum_pt == 0) get phi_T = 0, T = 0.
    """
    n = len(sum_pt)
    step = np.pi / n_scan

    best_val = np.zeros(n)
    best_idx = np.zeros(n, dtype=np.int64)
    for k in range(n_scan):
        val = _proj_sum(px, py, k * step)
        upd = val > best_val
        best_val[upd] = val[upd]
        best_idx[upd] = k

    a0 = best_idx * step
    # Parabolic vertex from the three points a0-step, a0, a0+step. H is
    # pi-periodic so evaluating outside [0, pi) is well defined.
    h_minus = _proj_sum(px, py, a0 - step)
    h_zero  = best_val
    h_plus  = _proj_sum(px, py, a0 + step)
    denom = h_minus - 2.0 * h_zero + h_plus
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(np.abs(denom) > 1e-12, 0.5 * (h_minus - h_plus) / denom, 0.0)
    delta = np.clip(delta, -1.0, 1.0)
    phi_T = np.mod(a0 + delta * step, np.pi)

    t_num = _proj_sum(px, py, phi_T)
    with np.errstate(divide="ignore", invalid="ignore"):
        thrust = np.where(sum_pt > 0.0, t_num / sum_pt, 0.0)

    return phi_T.astype(np.float32), thrust.astype(np.float32)


# ---------------------------------------------------------------------------
# Hemisphere four-vectors
# ---------------------------------------------------------------------------

def hemisphere_fourvectors(pt, eta, phi, m, phi_T):
    """Split jets on the plane perpendicular to n_T and sum the four-vectors.

    A jet joins hemisphere index 0 (the +n_T side) when its transverse-momentum
    projection on the thrust axis is > 0, else hemisphere index 1 (the -n_T
    side). Each hemisphere's jet four-vectors are summed.

    pt, eta, phi, m : jagged (events, var jets) awkward arrays.
    phi_T           : (events,) numpy array, the thrust-axis angle from
                      transverse_thrust().

    Returns a dict of (events, 2) numpy arrays (index 0 = +n_T side): px, py,
    pz, energy, pt, eta, phi, mass, pt_par, pt_perp, partner_eta (eta of the
    event's other hemisphere), n_jets (int32), side (int32, +1/-1).
    """
    cos_t = np.cos(phi_T)          # (events,)
    sin_t = np.sin(phi_T)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)

    proj = px * cos_t + py * sin_t         # jagged: per-jet projection on n_T
    pos = proj > 0.0

    def _sum(arr, mask):
        return ak.to_numpy(ak.sum(ak.where(mask, arr, 0.0), axis=1))

    cols = {name: [] for name in
            ("px", "py", "pz", "energy", "pt", "eta", "phi", "mass",
             "pt_par", "pt_perp", "n_jets", "side")}

    for mask, side in ((pos, 1), (~pos, -1)):
        h_px = _sum(px, mask)
        h_py = _sum(py, mask)
        h_pz = _sum(pz, mask)
        h_e  = _sum(energy, mask)
        n_jets = ak.to_numpy(ak.sum(mask, axis=1)).astype(np.int32)

        h_pt = np.hypot(h_px, h_py)
        p2 = h_px ** 2 + h_py ** 2 + h_pz ** 2
        mass = np.sqrt(np.maximum(h_e ** 2 - p2, 0.0))
        h_phi = np.arctan2(h_py, h_px)
        with np.errstate(divide="ignore", invalid="ignore"):
            h_eta = np.arcsinh(np.where(h_pt > 0.0, h_pz / h_pt, 0.0))
        pt_par  = h_px * cos_t + h_py * sin_t
        pt_perp = -h_px * sin_t + h_py * cos_t

        cols["px"].append(h_px);         cols["py"].append(h_py)
        cols["pz"].append(h_pz);         cols["energy"].append(h_e)
        cols["pt"].append(h_pt);         cols["eta"].append(h_eta)
        cols["phi"].append(h_phi);       cols["mass"].append(mass)
        cols["pt_par"].append(pt_par);   cols["pt_perp"].append(pt_perp)
        cols["n_jets"].append(n_jets)
        cols["side"].append(np.full_like(n_jets, side))

    # Stack the two hemispheres into (events, 2) with index 0 = +n_T side.
    out = {}
    for name, (a, b) in cols.items():
        stacked = np.stack([a, b], axis=1)
        if name in ("n_jets", "side"):
            out[name] = stacked.astype(np.int32)
        else:
            out[name] = stacked.astype(np.float32)
    # eta of the event's other hemisphere: the eta column with the sides
    # swapped (an empty partner keeps the pt==0 -> eta=0 convention).
    out["partner_eta"] = out["eta"][:, ::-1].copy()
    return out


# ---------------------------------------------------------------------------
# Main mixing loop
# ---------------------------------------------------------------------------

def mix(input_path, output_path, config, config_path, in_tree_name, chunk_size,
        xs_weighting=True, xs_json=None, dataset=None, lumi=1.0,
        n_original=None, require_xs=False):
    version   = config["metadata"]["version"]
    n_scan    = int(config["mixer"]["thrust_scan_points"])
    hist_bins = int(config["mixer"]["hist_bins"])
    n_jets_req = int(config["mixer"]["n_jets"])

    print(f"Input:   {input_path}  (tree: {in_tree_name})")
    print(f"Output:  {output_path}  (tree: events)")
    print(f"Version: {version}  ({config_path})")
    print(f"Thrust:  transverse axis via {n_scan}-point scan on [0, pi) + parabolic refine")
    print(f"Cut:     exactly {n_jets_req} jets per event")

    with uproot.open(input_path) as in_file:
        tree_name = in_tree_name
        if tree_name not in in_file:
            # Accept the raw-NanoAOD tree name as a fallback for convenience.
            if in_tree_name == "events" and "Events" in in_file:
                tree_name = "Events"
            else:
                # The slimmer emits cutflow-only files (no events tree) for
                # slices where nothing passed its cuts. Mirror that: pass the
                # cutflow / version through and finish successfully so the
                # pipeline's bookkeeping stays intact.
                print(f"No '{in_tree_name}' tree in {input_path} (empty slice) -> "
                      "writing cutflow-only output.")
                with uproot.recreate(output_path) as out_file:
                    out_file["mixer_cutflow"] = _mixer_cutflow_hist(0, 0)
                    _write_passthrough_hists(in_file, out_file)
                    _write_version_and_meta(out_file, config)
                return

        tree = in_file[tree_name]
        tree_keys = set(tree.keys())
        fmt = jet_format(tree_keys)
        if fmt is None:
            sys.exit(
                f"Expected jet branch '{_JET_BRANCH}.pt' (nested) or "
                f"'{_JET_BRANCH}_pt' (flat) in tree '{tree_name}'. "
                f"Keys: {sorted(tree_keys)}"
            )
        print(f"Jet layout: {fmt} ('{_JET_BRANCH}{'.' if fmt == 'nested' else '_'}pt')")

        # Cross-section weight (per-event, constant per file). n_original defaults
        # to this file's cutflow[0] - the file's contribution to its slice's
        # xsec-normalisation denominator.
        cutflow_n = float(in_file["cutflow"].values()[0]) if "cutflow" in in_file else None
        n_orig = n_original if n_original is not None else cutflow_n
        if xs_weighting:
            xs_path = locate_xs_json(xs_json)
            weight, winfo = resolve_xs_weight(
                input_path, xs_path, dataset, lumi, n_orig, require_xs
            )
            print(f"Weight:  {weight:.6g} = {lumi:g} * xs_pb / n_original  "
                  f"(dataset={winfo['dataset']}, xs_pb={winfo['xs_pb']}, "
                  f"n_original={winfo['n_original']})")
        else:
            weight = 1.0
            winfo = {"dataset": dataset, "xs_pb": None, "n_original": n_orig,
                     "lumi": lumi, "weight": 1.0, "source": None}
            print("Weight:  disabled (--no-xs-weight) -> 1.0")

        thrust_hist = bh.Histogram(
            bh.axis.Regular(hist_bins, 0.0, np.pi), storage=bh.storage.Double()
        )
        njet_hist = bh.Histogram(
            bh.axis.Regular(20, 0.0, 20.0), storage=bh.storage.Double()
        )

        total_read = 0
        total = 0
        with uproot.recreate(output_path) as out_file:
            out_tree = None

            for chunk in tree.iterate(library="ak", step_size=chunk_size):
                total_read += len(chunk)

                # 0) exact-multiplicity cut
                pt, eta, phi, m = jet_subarrays(chunk)
                keep = ak.num(pt, axis=1) == n_jets_req
                chunk = chunk[keep]
                pt, eta, phi, m = pt[keep], eta[keep], phi[keep], m[keep]
                n_chunk = len(chunk)
                if n_chunk == 0:
                    continue
                total += n_chunk

                # 1) faithful copy of the slimmed branches
                out_record = _passthrough_branches(chunk, _REGROUP_COLLECTIONS)

                # 2) transverse thrust axis
                px = pt * np.cos(phi)
                py = pt * np.sin(phi)
                sum_pt = ak.to_numpy(ak.sum(pt, axis=1))
                phi_T, thrust = transverse_thrust(px, py, sum_pt, n_scan)

                # 3) split into two hemispheres + sum four-vectors
                hemi = hemisphere_fourvectors(pt, eta, phi, m, phi_T)

                # 4) xsec weight: every hemisphere carries its source event's
                # MC weight (both hemispheres of an event share it).
                hemi["weight"] = np.full((n_chunk, 2), weight, dtype=np.float32)

                out_record["thrust_axis_phi"] = ak.Array(phi_T)
                out_record["thrust"]          = ak.Array(thrust)
                out_record["xs_weight"]       = ak.Array(
                    np.full(n_chunk, weight, dtype=np.float32))
                # from_regular: turn the fixed (events, 2) hemisphere block into a
                # variable-length list so uproot writes it as a ScoutingPFJet-style
                # collection with an nHemisphere counter (it cannot write a fixed
                # "2 * {...}" record type directly).
                out_record["Hemisphere"]      = ak.from_regular(ak.zip(hemi))

                thrust_hist.fill(phi_T)
                njet_hist.fill(hemi["n_jets"].reshape(-1))

                if out_tree is None:
                    out_file.mktree(
                        "events",
                        {name: arr.type for name, arr in out_record.items()},
                    )
                    out_tree = out_file["events"]
                out_tree.extend(out_record)

                print(f"  {total:>10,} events processed", end="\r")

            out_file["thrust_axis_phi"]  = thrust_hist
            out_file["hemisphere_njets"] = njet_hist
            out_file["mixer_cutflow"]    = _mixer_cutflow_hist(total_read, total)
            _write_passthrough_hists(in_file, out_file)
            _write_version_and_meta(out_file, config, winfo)

    print(f"\nDone.   {total:,} / {total_read:,} events with exactly "
          f"{n_jets_req} jets  ->  {output_path}")


def _mixer_cutflow_hist(n_read, n_pass):
    """Two-bin cutflow: bin 1 = events read, bin 2 = events passing the
    exact-n_jets cut (the events written to the output tree)."""
    h = bh.Histogram(bh.axis.Regular(2, 0.0, 2.0), storage=bh.storage.Double())
    h.view()[:] = [n_read, n_pass]
    return h


def _write_passthrough_hists(in_file, out_file):
    """Copy the slimmer's cutflow (as 'cutflow') and version (as
    'slimmer_version') histograms through to the output if they are present."""
    if "cutflow" in in_file:
        out_file["cutflow"] = in_file["cutflow"].to_boost()
    if "version" in in_file:
        out_file["slimmer_version"] = in_file["version"].to_boost()


def _write_version_and_meta(out_file, config, winfo=None):
    """Write the mixer 'version' string histogram, a 'dataset' string histogram
    (when the xsec weight resolved a slice), and the numeric 'meta' tree
    (scan params + the xsec-weight bookkeeping)."""
    version = config["metadata"]["version"]
    version_hist = bh.Histogram(bh.axis.StrCategory([version]), storage=bh.storage.Double())
    version_hist.view()[0] = 1.0
    out_file["version"] = version_hist

    if winfo and winfo.get("dataset"):
        ds_hist = bh.Histogram(bh.axis.StrCategory([winfo["dataset"]]),
                               storage=bh.storage.Double())
        ds_hist.view()[0] = 1.0
        out_file["dataset"] = ds_hist

    def _f(x):  # None -> 0.0 for the numeric meta tree
        return 0.0 if x is None else float(x)

    winfo = winfo or {}
    meta_record = {
        "thrust_scan_points": np.array([config["mixer"]["thrust_scan_points"]], dtype=np.int32),
        "hist_bins":          np.array([config["mixer"]["hist_bins"]],          dtype=np.int32),
        "n_jets":             np.array([config["mixer"]["n_jets"]],             dtype=np.int32),
        "lumi":               np.array([_f(winfo.get("lumi"))],       dtype=np.float64),
        "xs_pb":              np.array([_f(winfo.get("xs_pb"))],      dtype=np.float64),
        "n_original":         np.array([_f(winfo.get("n_original"))], dtype=np.float64),
        "xs_weight":          np.array([_f(winfo.get("weight"))],     dtype=np.float64),
    }
    out_file.mktree("meta", {name: arr.dtype for name, arr in meta_record.items()})
    out_file["meta"].extend(meta_record)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hemisphere-mixing prototype: thrust axis + hemisphere "
                    "four-vectors for the 6-jet multijet analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input slimmed ROOT file")
    parser.add_argument("config", help="JSON file containing the mixer configuration")
    parser.add_argument(
        "--tree", default="events", metavar="NAME",
        help="Input tree name (the slimmer writes 'events')",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50_000, metavar="N",
        help="Events per processing chunk",
    )
    parser.add_argument(
        "--output-tag", type=str, default="",
        help="Optional tag added to the output file name",
    )
    parser.add_argument(
        "--xs-json", default=None, metavar="PATH",
        help="Cross-section JSON (mj_samples_xs.json). Default: auto-locate a "
             "run3-mj-pass-the-aux/ sibling of run3-mj-mixer (or ./ on a worker).",
    )
    parser.add_argument(
        "--dataset", default=None, metavar="NAME",
        help="Dataset/HT-slice name for the xsec lookup (default: infer from filename).",
    )
    parser.add_argument(
        "--lumi", type=float, default=1.0, metavar="X",
        help="Integrated luminosity factor in the weight lumi * xs_pb / n_original.",
    )
    parser.add_argument(
        "--n-original", type=float, default=None, metavar="N",
        help="Denominator for the xsec weight (default: this file's cutflow[0]). "
             "Pass the slice-summed n_original for absolute per-slice normalization.",
    )
    parser.add_argument(
        "--no-xs-weight", dest="xs_weighting", action="store_false",
        help="Disable cross-section weighting (store weight 1.0).",
    )
    parser.add_argument(
        "--require-xs", action="store_true",
        help="Fail if the cross section cannot be resolved (default: warn, weight 1.0).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.output_tag:
        output_path = "mixed_" + args.output_tag + "_" + os.path.basename(args.input)
    else:
        output_path = "mixed_" + os.path.basename(args.input)

    mix(
        input_path=args.input,
        output_path=output_path,
        config=cfg,
        config_path=args.config,
        in_tree_name=args.tree,
        chunk_size=args.chunk_size,
        xs_weighting=args.xs_weighting,
        xs_json=args.xs_json,
        dataset=args.dataset,
        lumi=args.lumi,
        n_original=args.n_original,
        require_xs=args.require_xs,
    )


if __name__ == "__main__":
    main()

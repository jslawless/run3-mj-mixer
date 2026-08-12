"""hemi_io.py - reading slimmed files, and reconstituting a hemisphere.

Everything the index deliberately does not store lives in the slimmed files.
This module is the one way to get it back: given an index row and its slimmed
file, ``load_hemisphere`` returns the hemisphere's jets and its derived summary.

It is also **the single API the analyzer imports.** Analyzer scripts must import
it directly and let a failure be loud - never

    try:    from run3_mj_mixer... import ...
    except ImportError:  sys.path.insert(...)

because that fallback only fires when the import *fails*, so a stale installed
copy always wins silently. That pattern cost real debugging time in July 2026.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from .hemisphere import hemisphere_summary, select_by_mask

JET_BRANCH = "ScoutingPFJet"
GEN_JET_BRANCH = "GenJet"
GEN_PART_BRANCH = "GenPart"

#: The four jet kinematics every stage needs. Anything beyond this is analysis
#: payload and is named explicitly in the stage-3a payload config.
JET_KINEMATICS = ("pt", "eta", "phi", "m")


def jet_format(keys, branch=JET_BRANCH):
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


def jet_subarrays(chunk, branch=JET_BRANCH):
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


def branch_names(keys, fields, branch=JET_BRANCH):
    """Map requested per-jet field names onto actual branch names in a tree.

    Accepts either layout, so a payload config can name ``pt_jerUp`` without
    knowing whether the file stores ``ScoutingPFJet_pt_jerUp`` or
    ``ScoutingPFJet.pt_jerUp``.
    """
    fmt = jet_format(keys, branch)
    sep = "." if fmt == "nested" else "_"
    return {f: f"{branch}{sep}{f}" for f in fields}


def missing_branches(keys, fields, branch=JET_BRANCH):
    """Which of ``fields`` are absent from a tree's ``keys``."""
    mapping = branch_names(keys, fields, branch)
    return sorted(f for f, b in mapping.items() if b not in set(keys))


def load_hemisphere(slimmed_path, entry, jet_mask, n_jets_event, *,
                    tree_name="events", fields=JET_KINEMATICS,
                    thrust_phi=None):
    """The jets of one hemisphere, plus its derived summary.

    ``jet_mask`` selects which of the event's ``n_jets_event`` stored jets belong
    to this hemisphere - the authoritative definition, recorded by stage 1a where
    the split was known exactly.

    Returns ``(jets, summary)``: ``jets`` a dict of 1-D arrays in stored-
    collection order, ``summary`` the output of ``hemisphere_summary`` for the
    summed hemisphere if ``thrust_phi`` was given, else ``None``.

    Convenience for diagnostics and displays; the production gather reads whole
    files sequentially rather than one entry at a time.
    """
    import uproot

    with uproot.open(slimmed_path) as f:
        tree = f[tree_name]
        mapping = branch_names(tree.keys(), fields)
        arrays = tree.arrays(
            list(mapping.values()), entry_start=entry, entry_stop=entry + 1,
            library="np",
        )
        jets = {
            field: select_by_mask(arrays[bname][0], jet_mask, n_jets_event)
            for field, bname in mapping.items()
        }

    summary = None
    if thrust_phi is not None and {"pt", "eta", "phi", "m"} <= set(jets):
        px = jets["pt"] * np.cos(jets["phi"])
        py = jets["pt"] * np.sin(jets["phi"])
        pz = jets["pt"] * np.sinh(jets["eta"])
        e = np.sqrt((jets["pt"] * np.cosh(jets["eta"])) ** 2 + jets["m"] ** 2)
        h_px, h_py, h_pz, h_e = px.sum(), py.sum(), pz.sum(), e.sum()
        h_pt = float(np.hypot(h_px, h_py))
        p2 = h_px ** 2 + h_py ** 2 + h_pz ** 2
        summary = hemisphere_summary(
            h_pt,
            float(np.arcsinh(h_pz / h_pt)) if h_pt > 0 else 0.0,
            float(np.arctan2(h_py, h_px)),
            float(np.sqrt(max(h_e ** 2 - p2, 0.0))),
            thrust_phi,
        )
    return jets, summary

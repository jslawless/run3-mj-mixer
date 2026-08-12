"""shard.py - stage 1a: one slimmed file -> one index shard.

Replaces the old ``mix.py``. Reads a slimmed file, applies the exact-jet-
multiplicity cut, finds the transverse thrust axis, splits each event into two
hemispheres, and writes the usable ones to a small index shard. It does not
write a copy of the event data: the old ``mixed_*.root`` intermediate was a
near-copy of its own input, and stage 3a looks payload up from the slimmed file
directly.

Two bugs the old arrangement had are unrepresentable here:

* ``entry`` is the entry in the **slimmed** file, so there is no post-cut
  renumbering to get wrong. The old scheme numbered against the mixed file and
  needed its own offset accumulator.
* ``jet_mask`` is recorded from the same boolean array the summary was summed
  from, so nothing downstream re-derives the split by comparing floats. The old
  stitcher recomputed it in float64 from float32-stored values and could
  silently mislabel a jet's parentage.

No weight is computed here - see ``filetable.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import awkward as ak
import numpy as np

from . import index as idx
from .hemi_io import JET_BRANCH, jet_format, jet_subarrays
from .hemisphere import (
    as_regular,
    hemisphere_fourvectors,
    hemisphere_split_mask,
    pack_jet_masks,
    popcount16,
    transverse_thrust,
)

_REQUIRED_METADATA_KEYS = {"version": str}
_REQUIRED_MIXER_KEYS = {
    "thrust_scan_points": int,
    "hist_bins": int,
    "n_jets": int,
}

#: Event-level branches copied into the index for provenance. Absent branches
#: are filled with zeros rather than failing - data and MC differ here.
_CMS_ID_BRANCHES = ("run", "luminosityBlock", "event")


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
                    f"Config '{section}.{key}' has wrong type: expected "
                    f"{expected.__name__}, got {type(cfg[section][key]).__name__}"
                )

    if cfg["mixer"]["thrust_scan_points"] < 2:
        sys.exit("Config 'mixer.thrust_scan_points' must be >= 2.")
    if cfg["mixer"]["hist_bins"] < 1:
        sys.exit("Config 'mixer.hist_bins' must be >= 1.")
    if cfg["mixer"]["n_jets"] < 1:
        sys.exit("Config 'mixer.n_jets' must be >= 1.")

    return cfg


def _id_column(chunk, name, n, dtype):
    """A CMS event-id column, zero-filled when the branch is absent."""
    if name in ak.fields(chunk):
        return np.asarray(ak.to_numpy(chunk[name]), dtype=dtype)
    return np.zeros(n, dtype=dtype)


def build_rows(pt, eta, phi, m, chunk, entries, n_jets_req, n_scan,
               index_n_jets, fhash):
    """Index rows for one chunk of already-cut events.

    ``entries`` are the entry numbers in the *slimmed* file of the kept events.
    Returns ``(cols, n_hemi_total)`` where ``cols`` holds only the hemispheres
    whose jet count equals ``index_n_jets``.
    """
    n_ev = len(entries)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    sum_pt = ak.to_numpy(ak.sum(pt, axis=1))
    phi_T, thrust = transverse_thrust(px, py, sum_pt, n_scan)

    # ONE split, three consumers: the summary, the mask, and the jet counts.
    pos = hemisphere_split_mask(pt, phi, phi_T)
    hemi = hemisphere_fourvectors(pt, eta, phi, m, phi_T, pos_mask=pos)
    masks = pack_jet_masks(as_regular(pos, n_jets_req))          # (n_ev, 2)

    # Broadcast the per-event quantities across the two hemisphere slots, then
    # keep only the usable slots.
    slot = np.tile(np.arange(2, dtype=np.uint8), (n_ev, 1))
    ev_entry = np.repeat(entries[:, None], 2, axis=1)
    ev_thrust = np.repeat(phi_T[:, None], 2, axis=1)
    ev_tval = np.repeat(thrust[:, None], 2, axis=1)

    ids = {name: np.repeat(
        _id_column(chunk, name, n_ev, dt)[:, None], 2, axis=1)
        for name, dt in (("run", np.uint32), ("luminosityBlock", np.uint32),
                         ("event", np.uint64))}

    n_jets_hemi = hemi["n_jets"]
    sel = n_jets_hemi == index_n_jets

    # popcount(jet_mask) must equal n_jets, or the mask and the summary disagree
    # about what this hemisphere is - which is the whole class of bug jet_mask
    # exists to remove. Check every slot, not just the selected ones.
    pc = popcount16(masks)
    if not np.array_equal(pc.astype(np.int32), n_jets_hemi.astype(np.int32)):
        bad = int(np.argmax(pc.astype(np.int32) != n_jets_hemi.astype(np.int32)))
        raise AssertionError(
            "jet_mask disagrees with n_jets at flat slot "
            f"{bad}: popcount={int(pc.ravel()[bad])} vs "
            f"n_jets={int(n_jets_hemi.ravel()[bad])}."
        )

    cols = {
        "file_hash":       np.full(int(sel.sum()), fhash, dtype=np.uint64),
        "entry":           ev_entry[sel].astype(np.int64),
        "side":            hemi["side"][sel].astype(np.int8),
        "hemi_slot":       slot[sel].astype(np.uint8),
        "run":             ids["run"][sel],
        "luminosityBlock": ids["luminosityBlock"][sel],
        "event":           ids["event"][sel],
        "pt":              hemi["pt"][sel],
        "eta":             hemi["eta"][sel],
        "partner_eta":     hemi["partner_eta"][sel],
        "thrust_phi":      ev_thrust[sel],
        "n_jets":          n_jets_hemi[sel].astype(np.uint8),
        "jet_mask":        masks[sel],
        "phi":             hemi["phi"][sel],
        "mass":            hemi["mass"][sel],
        "thrust":          ev_tval[sel],
    }
    return cols, int(n_jets_hemi.size)


def make_shard(input_path, output_path, config, config_path, in_tree_name,
               chunk_size, index_n_jets):
    """Stage 1a for one file. Returns the number of rows written."""
    import uproot

    version = config["metadata"]["version"]
    n_scan = int(config["mixer"]["thrust_scan_points"])
    n_jets_req = int(config["mixer"]["n_jets"])
    fhash = idx.file_hash(input_path)

    print(f"Input:   {input_path}  (tree: {in_tree_name})")
    print(f"Output:  {output_path}  (tree: {idx.SHARD_TREE})")
    print(f"Version: {version}  ({config_path})")
    print(f"Cut:     exactly {n_jets_req} jets/event; index {index_n_jets}-jet hemispheres")
    print(f"file_hash: {fhash:#018x}  ({idx.normalize_path(input_path)})")

    with uproot.open(input_path) as in_file:
        tree_name = in_tree_name
        if tree_name not in in_file:
            if in_tree_name == "events" and "Events" in in_file:
                tree_name = "Events"
            else:
                # The slimmer emits cutflow-only files for slices where nothing
                # passed. Emit a 0-row shard WITH the complete marker, so
                # "missing shard" stays unambiguously a job failure.
                print(f"No '{in_tree_name}' tree in {input_path} (empty slice) "
                      "-> writing a 0-row shard.")
                idx.write_shard(
                    output_path, idx.empty_columns(0),
                    dataset=None, source_path=idx.normalize_path(input_path),
                    version=version,
                    cutflow={"events read": 0.0, "events kept": 0.0,
                             "hemispheres total": 0.0, "hemispheres written": 0.0},
                )
                return 0

        tree = in_file[tree_name]
        if jet_format(tree.keys()) is None:
            sys.exit(
                f"Expected jet branch '{JET_BRANCH}.pt' (nested) or "
                f"'{JET_BRANCH}_pt' (flat) in tree '{tree_name}'."
            )

        parts = []
        n_read = n_kept = n_hemi_total = 0
        entry_start = 0
        for chunk in tree.iterate(library="ak", step_size=chunk_size):
            n_here = len(chunk)
            pt, eta, phi, m = jet_subarrays(chunk)
            keep = ak.to_numpy(ak.num(pt, axis=1) == n_jets_req)
            # entry numbers in the SLIMMED file - no renumbering anywhere.
            entries = np.arange(entry_start, entry_start + n_here)[keep]
            entry_start += n_here
            n_read += n_here

            if len(entries) == 0:
                continue
            n_kept += len(entries)
            sub = chunk[keep]
            cols, n_h = build_rows(
                pt[keep], eta[keep], phi[keep], m[keep], sub, entries,
                n_jets_req, n_scan, index_n_jets, fhash,
            )
            n_hemi_total += n_h
            parts.append(cols)

        if parts:
            cols = {name: np.concatenate([p[name] for p in parts])
                    for name in idx.SHARD_COLUMNS}
        else:
            cols = idx.empty_columns(0)
        n_rows = len(cols["entry"])

        dataset = None
        idx.write_shard(
            output_path, cols, dataset=dataset,
            source_path=idx.normalize_path(input_path), version=version,
            cutflow={"events read": float(n_read), "events kept": float(n_kept),
                     "hemispheres total": float(n_hemi_total),
                     "hemispheres written": float(n_rows)},
        )
        print(f"Done.   {n_kept:,} / {n_read:,} events with exactly {n_jets_req} "
              f"jets -> {n_rows:,} {index_n_jets}-jet hemispheres  ->  {output_path}")
        return n_rows


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="run3-mj-index",
        description="Stage 1a: slimmed file -> hemisphere index shard.",
    )
    p.add_argument("input", help="a slimmed ROOT file")
    p.add_argument("config", help="mixer config JSON")
    p.add_argument("--tree", default="events", help="input tree name")
    p.add_argument("-o", "--output", default=None,
                   help="output shard path (default: hindex_<input basename>)")
    p.add_argument("--outdir", default=".", metavar="DIR",
                   help="directory for the shard (created if missing)")
    p.add_argument("--output-tag", default=None,
                   help="inserted into the output name: hindex_<tag>_<basename>")
    p.add_argument("--index-n-jets", type=int, default=3,
                   help="hemisphere jet count to index (default 3)")
    p.add_argument("--chunk-size", type=int, default=50_000)
    args = p.parse_args(argv)

    config = load_config(args.config)

    if args.output:
        out = args.output
    else:
        base = os.path.basename(args.input)
        name = (f"hindex_{args.output_tag}_{base}" if args.output_tag
                else f"hindex_{base}")
        out = os.path.join(args.outdir, name)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    make_shard(args.input, out, config, args.config, args.tree,
               args.chunk_size, args.index_n_jets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

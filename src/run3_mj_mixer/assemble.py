"""assemble.py - stage 3b: pairs + legs -> pseudo-events.

Owns ``build_output``, lifted here from the deleted ``stitch.py`` so there is one
implementation of the pseudo-event schema.

**The join must be exact.** Every pair in the chunk needs both its legs; a
missing one means a stage-3a job failed, and the right response is to say which
pair and stop, not to write a shorter file. A silently dropped pseudo-event is
invisible downstream and would quietly bias the sample.

Two things are structurally impossible here that the old stitcher had to worry
about. Jets arrive as a fixed count per hemisphere, so the ragged ``np.stack``
that could fail at the very end of a run cannot happen. And parentage comes from
the leg's ``leg`` field rather than from re-deriving which side a jet fell on, so
a jet cannot be mislabelled.

**Weights are the product of both parents'** relative weights, resolved in-job
from ``seed_file_id`` and ``match_file_id`` through the file table. Note this is
not a cross section - it is pb^2/event^2 - so the sample needs a global rescale;
the sums it is derived from are recorded in the manifest rather than left to be
rediscovered.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import awkward as ak
import numpy as np

from . import filetable as ft
from . import gather as G
from . import match as M
from .hemisphere import hemisphere_fourvectors, transverse_thrust

N_SCAN = 180
#: transverse_thrust calls _proj_sum N_SCAN+2 times over the whole array, so a
#: whole chunk at once means ~N_SCAN awkward intermediates of the full size.
#: Blocking keeps the transient bounded regardless of chunk size.
THRUST_BLOCK = 50_000

OUT_TREE = "events"

CHUNK_CUTFLOW_BINS = ["pairs in", "pseudo-events", "legs joined"]


# ---------------------------------------------------------------------------
# the shared output schema
# ---------------------------------------------------------------------------

def _blocked_thrust(px, py, sum_pt, n_scan=N_SCAN, block=THRUST_BLOCK):
    """transverse_thrust in bounded-size blocks."""
    n = len(sum_pt)
    if n <= block:
        return transverse_thrust(px, py, sum_pt, n_scan)
    phi_t = np.empty(n, dtype=np.float32)
    thrust = np.empty(n, dtype=np.float32)
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        a, b = transverse_thrust(px[lo:hi], py[lo:hi], sum_pt[lo:hi], n_scan)
        phi_t[lo:hi], thrust[lo:hi] = a, b
    return phi_t, thrust


def build_output(jets, hemi_flag, distance, provenance, xs_weight,
                 payload=None):
    """The pseudo-event record.

    ``jets``      : dict of (n, n_out) float arrays, at least pt/eta/phi/m.
    ``hemi_flag`` : (n, n_out) int, 0 = the seed's jets, 1 = the partner's.
    ``provenance``: dict of per-event arrays written through as-is.
    ``payload``   : extra per-jet fields, sorted with the kinematics.

    Semantics are those of the old ``stitch.py::build_output``: pT-descending
    sort, thrust recomputed from the stitched jets, and the ``Hemisphere``
    collection summed by PARENTAGE rather than by re-splitting on the new axis -
    soft jets migrate across it, and matching QA needs the halves as stitched.
    """
    n, n_out = jets["pt"].shape
    counts = np.full(n, n_out)

    def jag(a):
        return ak.unflatten(ak.Array(np.asarray(a).reshape(-1)), counts)

    pt, eta, phi, m = (jag(jets[k]) for k in ("pt", "eta", "phi", "m"))
    flag = jag(hemi_flag)

    order = ak.argsort(pt, axis=1, ascending=False)
    pt, eta, phi, m, flag = pt[order], eta[order], phi[order], m[order], flag[order]
    extra = {}
    for name, vals in (payload or {}).items():
        extra[name] = jag(vals)[order]

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    sum_pt = ak.to_numpy(ak.sum(pt, axis=1))
    phi_t, thrust = _blocked_thrust(px, py, sum_pt)

    hemi = hemisphere_fourvectors(pt, eta, phi, m, phi_t, pos_mask=(flag == 0))
    hemi["side"] = np.where(hemi["pt_par"] >= 0.0, 1, -1).astype(np.int32)
    hemi["weight"] = np.ones((n, 2), dtype=np.float32)

    jet_rec = {"pt": pt, "eta": eta, "phi": phi, "m": m, "hemisphere": flag}
    jet_rec.update(extra)

    out = {
        "ScoutingPFJet": ak.from_regular(ak.zip(jet_rec)),
        "HT": ak.Array(sum_pt.astype(np.float32)),
        "thrust_axis_phi": ak.Array(phi_t),
        "thrust": ak.Array(thrust),
        "xs_weight": ak.Array(np.asarray(xs_weight, dtype=np.float32)),
        "Hemisphere": ak.from_regular(ak.zip(hemi)),
        "match_distance": ak.Array(np.asarray(distance, dtype=np.float32)),
    }
    for k, v in provenance.items():
        out[k] = ak.Array(v)
    return out


# ---------------------------------------------------------------------------
# the join
# ---------------------------------------------------------------------------

def load_legs(leg_paths, fields):
    """Every leg record, keyed by ``(pair_id, leg)``."""
    import uproot

    per_key = {}
    n_read = 0
    for p in leg_paths:
        with uproot.open(p) as f:
            keys = {k.split(";")[0] for k in f.keys()}
            if "complete" not in keys:
                raise ValueError(
                    f"{p} has no completion marker - the gather job that wrote "
                    "it died. Re-run it; joining against a truncated leg file "
                    "would silently drop pseudo-events."
                )
            if G.LEGS_TREE not in keys:
                continue
            t = f[G.LEGS_TREE]
            want = ["pair_id", "leg"] + [f"jet_{x}" for x in fields]
            a = t.arrays(want, library="ak")
        pid = ak.to_numpy(a["pair_id"])
        leg = ak.to_numpy(a["leg"])
        for k in range(len(pid)):
            per_key[(int(pid[k]), int(leg[k]))] = {
                fld: np.asarray(a[f"jet_{fld}"][k], dtype=np.float32)
                for fld in fields
            }
            n_read += 1
    return per_key, n_read


def join(pairs, legs, fields):
    """Stack the two legs of every pair. Raises on a missing leg."""
    pair_ids = pairs["pair_id"]
    n = len(pair_ids)
    missing = [(int(p), leg) for p in pair_ids for leg in (0, 1)
               if (int(p), leg) not in legs]
    if missing:
        shown = ", ".join(f"pair {p} leg {l}" for p, l in missing[:10])
        raise ValueError(
            f"{len(missing)} leg(s) missing from the join, e.g. {shown}. A "
            "stage-3a job failed for the file(s) those legs live in - find it "
            "with --check and re-run it. Refusing to write a short file."
        )

    n_seed = len(legs[(int(pair_ids[0]), 0)]["pt"])
    n_match = len(legs[(int(pair_ids[0]), 1)]["pt"])
    n_out = n_seed + n_match

    stacked = {fld: np.empty((n, n_out), dtype=np.float32) for fld in fields}
    for row, p in enumerate(pair_ids):
        a = legs[(int(p), 0)]
        b = legs[(int(p), 1)]
        if len(a["pt"]) != n_seed or len(b["pt"]) != n_match:
            raise ValueError(
                f"pair {int(p)} has {len(a['pt'])}+{len(b['pt'])} jets, expected "
                f"{n_seed}+{n_match}. Hemispheres must have a fixed jet count."
            )
        for fld in fields:
            stacked[fld][row, :n_seed] = a[fld]
            stacked[fld][row, n_seed:] = b[fld]

    flag = np.zeros((n, n_out), dtype=np.int32)
    flag[:, n_seed:] = 1
    return stacked, flag


def assemble_chunk(pair_path, leg_paths, table, out_path, *, fields=None,
                   index_id=None, write_chunk=100_000):
    """One pair chunk + its leg shards -> one stitched file."""
    import boost_histogram as bh
    import uproot

    fields = list(fields or G.DEFAULT_JET_BRANCHES)

    with uproot.open(pair_path) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        if "complete" not in keys:
            raise ValueError(f"{pair_path} has no completion marker.")
        chunk_id = int(f["chunk_id"].to_boost().axes[0][0]) if "chunk_id" in keys else 0
        pair_lo = int(f["pair_lo"].to_boost().axes[0][0]) if "pair_lo" in keys else 0
        pair_hi = int(f["pair_hi"].to_boost().axes[0][0]) if "pair_hi" in keys else 0
        pairs = f[M.PAIRS_TREE].arrays(library="np")

    legs, n_legs = load_legs(leg_paths, fields)
    stacked, flag = join(pairs, legs, fields)

    w = ft.w_rel_by_id(table)
    # The product of both parents' weights: symmetric in the two halves, as the
    # final 6-jet object is. Not a cross section - see the module docstring.
    xs_weight = w[pairs["seed_file_id"]] * w[pairs["match_file_id"]]

    prov = {}
    for tag in ("seed", "match"):
        for fld, dt in (("hemi_id", np.int64), ("file_id", np.int32),
                        ("entry", np.int32), ("side", np.int8),
                        ("slot", np.uint8), ("jet_mask", np.uint16),
                        ("run", np.uint32), ("luminosityBlock", np.uint32),
                        ("event", np.uint64)):
            src = f"{tag}_{fld}"
            if src in pairs:
                prov[src] = np.asarray(pairs[src], dtype=dt)
    prov["pair_id"] = np.asarray(pairs["pair_id"], dtype=np.int64)

    payload = {k: v for k, v in stacked.items()
               if k not in ("pt", "eta", "phi", "m")}
    out = build_output(stacked, flag, pairs["match_distance"], prov, xs_weight,
                       payload=payload)

    n = len(pairs["pair_id"])
    os.makedirs(os.path.dirname(str(out_path)) or ".", exist_ok=True)
    tmp = str(out_path) + ".tmp"
    with uproot.recreate(tmp) as f:
        # mktree + extend, NOT `f["events"] = out`: the latter writes an RNTuple
        # with dotted branch names (ScoutingPFJet.pt), and every consumer in this
        # pipeline reads the flat TTree layout (ScoutingPFJet_pt).
        f.mktree(OUT_TREE, {name: arr.type for name, arr in out.items()})
        for lo in range(0, n, write_chunk):
            hi = min(lo + write_chunk, n)
            f[OUT_TREE].extend({k: v[lo:hi] for k, v in out.items()})
        # Per-chunk counters only. Global counts live once, in the pair
        # manifest, so that hadd-ing these does not multiply them.
        cf = bh.Histogram(bh.axis.StrCategory(CHUNK_CUTFLOW_BINS),
                          storage=bh.storage.Double())
        cf.view()[:] = [float(n), float(n), float(n_legs)]
        f["chunk_cutflow"] = cf
        f["chunk_id"] = M._label(str(chunk_id))
        f["pair_lo"] = M._label(str(pair_lo))
        f["pair_hi"] = M._label(str(pair_hi))
        if index_id:
            f["index_id"] = M._label(str(index_id))
        f["sum_xs_weight"] = M._label("sum_xs_weight", float(xs_weight.sum()))
        f["complete"] = M._label("ok")
    os.replace(tmp, str(out_path))
    return {"n_pseudo_events": n, "n_legs": n_legs,
            "sum_xs_weight": float(xs_weight.sum()), "chunk_id": chunk_id}


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="run3-mj-assemble",
        description="Stage 3b: join pairs with their legs into pseudo-events.",
    )
    p.add_argument("pair_chunk")
    p.add_argument("legs", nargs="+", help="leg files for this chunk")
    p.add_argument("-t", "--file-table", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--expect-index-id", default=None)
    args = p.parse_args(argv)

    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    try:
        fields = G.payload_branches(config)
    except ValueError as exc:
        sys.exit(f"assemble: {exc}")

    table = ft.load_file_table(args.file_table)
    try:
        res = assemble_chunk(args.pair_chunk, args.legs, table, args.output,
                             fields=fields, index_id=args.expect_index_id)
    except ValueError as exc:
        sys.exit(f"assemble: {exc}")

    print(f"chunk {res['chunk_id']}: {res['n_pseudo_events']:,} pseudo-events "
          f"from {res['n_legs']:,} legs")
    print(f"sum(xs_weight) = {res['sum_xs_weight']:.6g}")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

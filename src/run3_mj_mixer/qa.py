"""qa.py - stage 4: check what the earlier stages produced.

Every check is either an invariant that must hold (a FAIL is a bug) or a profile
that has no right answer but must be looked at (reported as INFO). The split
matters: mixing the two produces a report nobody reads.

The subcommands mirror the pipeline: ``index`` / ``spot`` / ``pairs`` / ``legs``
/ ``stitched`` / ``unmatched``. ``spot`` is the one that closes the loop back to
the source - it re-reads slimmed files and checks that ``jet_mask`` really does
select the jets the index claims, which is what licenses stage 3a to trust it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

from . import assemble as A
from . import filetable as ft
from . import gather as G
from . import index as idx
from . import index_build as ib
from . import match as M
from .hemisphere import directed_phi_array, popcount16, unpack_jet_mask


class Report:
    """Checks with verdicts, profiles without."""

    def __init__(self, title):
        self.title = title
        self.rows = []
        self.failed = 0

    def check(self, name, ok, detail=""):
        ok = bool(ok)
        if not ok:
            self.failed += 1
        self.rows.append(("PASS" if ok else "FAIL", name, detail))
        return ok

    def info(self, name, detail=""):
        self.rows.append(("INFO", name, detail))

    def print(self):
        print(f"\n=== {self.title} ===")
        for verdict, name, detail in self.rows:
            tag = {"PASS": "  ok ", "FAIL": "FAIL ", "INFO": "  -- "}[verdict]
            line = f"{tag}{name}"
            if detail:
                line += f": {detail}"
            print(line)
        print(f"--- {self.failed} failure(s)" if self.failed else "--- all checks passed")
        return self.failed == 0


def _pct(a, b):
    return f"{(a / b * 100.0):.2f}%" if b else "n/a"


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

def qa_index(index_dir, table, shard_paths=None):
    r = Report("index")
    ix = ib.HemisphereIndex(index_dir, table)
    n = len(ix)
    r.info("rows", f"{n:,}")
    r.info("index_id", ix.manifest["index_id"])
    r.info("mode", ix.manifest["mode"])

    # Three independent counts of the same number. Agreement means the merge did
    # not silently drop or duplicate a shard.
    n_manifest = int(ix.manifest["n_rows"])
    r.check("manifest row count matches the columns", n_manifest == n,
            f"{n_manifest:,} vs {n:,}")
    if shard_paths:
        total = 0
        for p in shard_paths:
            _, meta = idx.read_shard(p)
            total += int(meta["cutflow"].get("hemispheres written", 0))
        r.check("sum of shard cutflows matches the index",
                total == n, f"{total:,} vs {n:,}")

    fid, ent, slot = ix["file_id"][:], ix["entry"][:], ix["hemi_slot"][:]
    key = fid.astype(np.int64) * (1 << 34) + ent.astype(np.int64) * 4 + slot
    r.check("(file_id, entry, hemi_slot) unique", len(np.unique(key)) == n)
    r.check("canonical sort order", bool(np.all(np.diff(key) >= 0)))

    nj = ix["n_jets"][:]
    r.check("popcount(jet_mask) == n_jets",
            bool(np.all(popcount16(ix["jet_mask"][:]) == nj)))
    r.check("slot 0 <=> side +1",
            bool(np.all((slot == 0) == (ix["side"][:] == 1))))
    tp = ix["thrust_phi"][:]
    r.check("thrust_phi in [0, pi)", bool(np.all((tp >= 0) & (tp < np.pi))))
    for name in ("pt", "eta", "phi", "mass", "partner_eta", "thrust"):
        r.check(f"{name} finite", bool(np.all(np.isfinite(ix[name][:]))))
    r.check("pt > 0", bool(np.all(ix["pt"][:] > 0)))

    # offsets must give every file a contiguous hemi_id range - the property
    # stage 3a depends on to look up a file's rows without scanning.
    off = ix.offsets
    contig = int(off[-1]) == n
    for f in range(ix.manifest["n_files"]):
        lo, hi = int(off[f]), int(off[f + 1])
        if hi > lo and not np.all(fid[lo:hi] == f):
            contig = False
            break
    r.check("hemi_id_offsets contiguous per file", contig)

    # A duplicate CMS event key means one physical event entered the index twice
    # - usually the same input indexed under two output tags.
    ev = np.stack([ix["run"][:].astype(np.int64),
                   ix["luminosityBlock"][:].astype(np.int64),
                   ix["event"][:].astype(np.int64)], axis=1)
    if np.any(ev != 0):
        uniq, counts = np.unique(ev, axis=0, return_counts=True)
        expect = int(ix.manifest.get("max_rows_per_event", 1))
        over = int((counts > expect).sum())
        detail = ""
        if over:
            bad = uniq[counts > expect][:3]
            detail = f"{over} key(s) over multiplicity {expect}, e.g. {bad.tolist()}"
        r.check(f"run:lumi:event multiplicity <= {expect}", over == 0, detail)
    else:
        r.info("run:lumi:event", "all zero (synthetic input) - dedup not checked")
    return r, ix


# ---------------------------------------------------------------------------
# spot: back to the source
# ---------------------------------------------------------------------------

def qa_spot(ix, table, n_spot=2000, seed=0, tree_name="events"):
    """Re-read slimmed files and verify jet_mask against the stored summary.

    The check that licenses everything downstream: if ``jet_mask`` does not
    select the jets the summary was summed from, stage 3a builds the wrong
    pseudo-events and nothing else would notice.
    """
    import uproot

    r = Report(f"spot check ({n_spot} rows)")
    paths = {int(x["file_id"]): x["path"] for x in table["files"]}
    n = len(ix)
    if n == 0:
        r.info("nothing to check", "empty index")
        return r

    rng = np.random.default_rng(seed)
    rows = rng.choice(n, size=min(n_spot, n), replace=False)
    by_file = {}
    for row in rows:
        by_file.setdefault(int(ix["file_id"][row]), []).append(int(row))

    worst = {k: 0.0 for k in ("pt", "eta", "phi", "mass")}
    n_bad_mask = n_checked = 0
    missing_files = []

    for fid, rws in sorted(by_file.items()):
        path = paths.get(fid)
        if path is None or not os.path.exists(path):
            missing_files.append(path or f"file_id {fid}")
            continue
        with uproot.open(path) as f:
            t = f[tree_name]
            a = t.arrays(["ScoutingPFJet_pt", "ScoutingPFJet_eta",
                          "ScoutingPFJet_phi", "ScoutingPFJet_m"], library="ak")
        for row in rws:
            e = int(ix["entry"][row])
            mask = int(ix["jet_mask"][row])
            n_ev = len(a["ScoutingPFJet_pt"][e])
            sel = unpack_jet_mask(mask, n_ev)
            if sel.sum() != int(ix["n_jets"][row]):
                n_bad_mask += 1
                continue
            jpt = np.asarray(a["ScoutingPFJet_pt"][e])[sel]
            jeta = np.asarray(a["ScoutingPFJet_eta"][e])[sel]
            jphi = np.asarray(a["ScoutingPFJet_phi"][e])[sel]
            jm = np.asarray(a["ScoutingPFJet_m"][e])[sel]
            px = float((jpt * np.cos(jphi)).sum())
            py = float((jpt * np.sin(jphi)).sum())
            pz = float((jpt * np.sinh(jeta)).sum())
            en = float(np.sqrt((jpt * np.cosh(jeta)) ** 2 + jm ** 2).sum())
            h_pt = float(np.hypot(px, py))
            got = {
                "pt": h_pt,
                "eta": float(np.arcsinh(pz / h_pt)) if h_pt > 0 else 0.0,
                "phi": float(np.arctan2(py, px)),
                "mass": float(np.sqrt(max(en ** 2 - (px**2 + py**2 + pz**2), 0.0))),
            }
            for k, v in got.items():
                ref = float(ix[k][row])
                worst[k] = max(worst[k], abs(v - ref) / max(abs(ref), 1.0))
            n_checked += 1

    if missing_files:
        r.info("slimmed files not reachable", f"{len(missing_files)}, e.g. "
               f"{missing_files[0]} - spot check skipped for those")
    r.check("rows checked", n_checked > 0, f"{n_checked:,}")
    r.check("jet_mask popcount always matches n_jets", n_bad_mask == 0,
            f"{n_bad_mask} mismatch(es)")
    for k, v in worst.items():
        r.check(f"{k} reproduced from the slimmed jets", v < 1e-4, f"max rel dev {v:.2e}")
    return r


# ---------------------------------------------------------------------------
# pairs
# ---------------------------------------------------------------------------

def qa_pairs(pairs_dir, table=None):
    import uproot

    r = Report("pairs")
    manifest = G.load_pair_manifest(pairs_dir)
    r.info("index_id", manifest["index_id"])
    r.info("cuts", f"pT +-{manifest['pt_tolerance']:g}, "
                   f"max_distance {manifest['max_distance']:g}, "
                   f"seed {manifest['rng_seed']}")

    cols = {}
    for c in manifest["chunks"]:
        with uproot.open(os.path.join(str(pairs_dir), c["file"])) as f:
            a = f[M.PAIRS_TREE].arrays(library="np")
        for k, v in a.items():
            cols.setdefault(k, []).append(v)
    if not cols:
        r.info("no pair chunks", "nothing to check")
        return r
    P = {k: np.concatenate(v) for k, v in cols.items()}
    n = len(P["pair_id"])

    r.check("chunk row counts sum to the manifest", n == manifest["n_pairs"],
            f"{n:,} vs {manifest['n_pairs']:,}")
    los = sorted(c["pair_lo"] for c in manifest["chunks"])
    his = sorted(c["pair_hi"] for c in manifest["chunks"])
    tiled = all(his[i] == los[i + 1] for i in range(len(los) - 1))
    r.check("[pair_lo, pair_hi) tile without gaps", tiled and los[0] == 0)
    r.check("pair_id contiguous from 0",
            bool(np.array_equal(np.sort(P["pair_id"]), np.arange(n))))

    # Two same-event checks: the first catches the matcher, the second catches a
    # same-PHYSICAL-event pair arising from an accidentally duplicated file,
    # which the first cannot see.
    same_prov = ((P["seed_file_id"] == P["match_file_id"])
                 & (P["seed_entry"] == P["match_entry"]))
    r.check("no same-event pairs by (file_id, entry)", not same_prov.any(),
            f"{int(same_prov.sum())} found")
    if np.any(P["seed_event"] != 0):
        same_cms = ((P["seed_run"] == P["match_run"])
                    & (P["seed_luminosityBlock"] == P["match_luminosityBlock"])
                    & (P["seed_event"] == P["match_event"]))
        r.check("no same-event pairs by run:lumi:event", not same_cms.any(),
                f"{int(same_cms.sum())} found")

    tol = manifest["pt_tolerance"]
    imbalance = np.abs(P["match_pt"] - P["seed_pt"])
    over = imbalance > tol * P["seed_pt"] + 1e-4
    r.check("pT window respected for every pair", not over.any(),
            f"{int(over.sum())} violation(s)")

    d = P["match_distance"]
    md = manifest["max_distance"]
    r.check("0 <= match_distance <= max_distance",
            bool(np.all((d >= 0) & (d <= md + 1e-6))))
    r.info("match_distance", f"median {np.median(d):.4f}, mean {d.mean():.4f}, "
                             f"p95 {np.percentile(d, 95):.4f} (cut {md:g})")
    top = float((d > 0.9 * md).mean())
    r.info("fraction in the top decile of the cut", f"{top:.2%}"
           + ("  <- a pile-up here means the cut is doing real work" if top > 0.1 else ""))

    # single use: no hemisphere may appear twice across either leg
    all_ids = np.concatenate([P["seed_hemi_id"], P["match_hemi_id"]])
    uniq = len(np.unique(all_ids))
    r.check("every hemi_id used at most once", uniq == len(all_ids),
            f"{len(all_ids) - uniq} reuse(s)")

    lo = np.minimum(P["seed_hemi_id"], P["match_hemi_id"])
    hi = np.maximum(P["seed_hemi_id"], P["match_hemi_id"])
    pair_key = lo * (1 + int(all_ids.max()) if len(all_ids) else 1) + hi
    r.check("no duplicate pairs", len(np.unique(pair_key)) == n)

    # geometry: the match must point back at the seed
    sd = directed_phi_array(P["seed_thrust_phi"], P["seed_side"])
    mdp = directed_phi_array(P["match_thrust_phi"], P["match_side"])
    opposite = np.cos(sd - mdp) < 0
    r.info("opening angle > 90 deg", f"{opposite.mean():.2%} of pairs")

    # accounting: nothing may vanish unexplained
    n_un = int(manifest["n_unmatched"])
    n_h = int(manifest["n_hemispheres"])
    r.check("2*pairs + unmatched == hemispheres", 2 * n + n_un == n_h,
            f"{2 * n} + {n_un} vs {n_h}")

    if table is not None:
        w = ft.w_rel_by_id(table)
        ds = np.array(ft.dataset_by_id(table))
        cross = ds[P["seed_file_id"]] != ds[P["match_file_id"]]
        xw = w[P["seed_file_id"]] * w[P["match_file_id"]]
        r.info("sum(xs_weight) over pairs", f"{xw.sum():.6g}")
        r.info("cross-slice pairs", f"{cross.mean():.2%}")
        if cross.any() and (~cross).any():
            r.info("mean xs_weight, same-slice vs cross-slice",
                   f"{xw[~cross].mean():.4g} vs {xw[cross].mean():.4g} "
                   "(the product suppresses cross-slice by the weight ratio)")
    return r


# ---------------------------------------------------------------------------
# legs
# ---------------------------------------------------------------------------

def qa_legs(pairs_dir, legs_dir):
    import uproot

    r = Report("legs")
    manifest = G.load_pair_manifest(pairs_dir)
    files = sorted(glob.glob(os.path.join(str(legs_dir), "legA_*.root")))
    r.info("leg files", f"{len(files)}")
    if not files:
        r.check("leg files present", False, "none found")
        return r

    seen = {}
    n_rows = 0
    for p in files:
        with uproot.open(p) as f:
            keys = {k.split(";")[0] for k in f.keys()}
            if "complete" not in keys:
                r.check(f"{os.path.basename(p)} complete", False,
                        "no completion marker - that gather job died")
                continue
            if G.LEGS_TREE not in keys:
                continue
            a = f[G.LEGS_TREE].arrays(["pair_id", "leg"], library="np")
        for pid, leg in zip(a["pair_id"], a["leg"]):
            k = (int(pid), int(leg))
            seen[k] = seen.get(k, 0) + 1
            n_rows += 1

    n_pairs = int(manifest["n_pairs"])
    r.check("total legs == 2 * pairs", n_rows == 2 * n_pairs,
            f"{n_rows:,} vs {2 * n_pairs:,}")
    dupes = [k for k, v in seen.items() if v > 1]
    r.check("no leg written twice", not dupes,
            f"{len(dupes)} duplicated, e.g. {dupes[:3]}")
    want = 2 * n_pairs
    r.check("every (pair, leg) present exactly once", len(seen) == want,
            f"{len(seen):,} distinct vs {want:,} expected")
    if len(seen) != want:
        missing = [(p, l) for p in range(min(n_pairs, 10_000)) for l in (0, 1)
                   if (p, l) not in seen][:5]
        if missing:
            r.info("first missing", str(missing))
    return r


# ---------------------------------------------------------------------------
# stitched
# ---------------------------------------------------------------------------

def qa_stitched(stitched_paths, pairs_dir=None, fields=None):
    import awkward as ak
    import uproot

    r = Report("stitched")
    fields = list(fields or G.DEFAULT_JET_BRANCHES)
    manifest = G.load_pair_manifest(pairs_dir) if pairs_dir else None
    total = 0
    for p in sorted(stitched_paths):
        with uproot.open(p) as f:
            keys = {k.split(";")[0] for k in f.keys()}
            if "complete" not in keys:
                r.check(f"{os.path.basename(p)} complete", False, "no marker")
                continue
            t = f["events"]
            tk = set(t.keys())
            lo = int(f["pair_lo"].to_boost().axes[0][0])
            hi = int(f["pair_hi"].to_boost().axes[0][0])
            n = t.num_entries
            pt = t["ScoutingPFJet_pt"].array()
            flag = t["ScoutingPFJet_hemisphere"].array()
            hn = t["Hemisphere_n_jets"].array()
            HT = t["HT"].array().to_numpy()
            th = t["thrust"].array().to_numpy()
            xw = t["xs_weight"].array().to_numpy()
        base = os.path.basename(p)
        total += n
        # A short file means the join dropped pseudo-events silently.
        r.check(f"{base}: n_events == pair_hi - pair_lo", n == hi - lo,
                f"{n:,} vs {hi - lo:,}")
        r.check(f"{base}: exactly 6 jets/event",
                bool(np.all(ak.to_numpy(ak.num(pt, axis=1)) == 6)))
        r.check(f"{base}: 3 jets flagged as the seed's",
                bool(np.all(ak.to_numpy(ak.sum(flag, axis=1)) == 3)))
        r.check(f"{base}: Hemisphere_n_jets == [3, 3]",
                bool(np.all(ak.to_numpy(ak.flatten(hn)) == 3)))
        r.check(f"{base}: jets pT-descending", bool(ak.all(pt[:, :-1] >= pt[:, 1:])))
        r.check(f"{base}: HT == sum(pt)",
                bool(np.allclose(HT, ak.to_numpy(ak.sum(pt, axis=1)), rtol=1e-5)))
        r.check(f"{base}: thrust in (0, 1]",
                bool(np.all((th > 0) & (th <= 1.0 + 1e-4))))
        r.check(f"{base}: branch names flat, not dotted",
                not any("." in k for k in tk))
        absent = [x for x in fields if f"ScoutingPFJet_{x}" not in tk]
        r.check(f"{base}: every payload branch present", not absent, str(absent))
        for x in fields:
            key = f"ScoutingPFJet_{x}"
            if key in tk:
                with uproot.open(p) as f:
                    a = f["events"][key].array()
                if not bool(np.all(ak.to_numpy(ak.num(a, axis=1)) == 6)):
                    r.check(f"{base}: {x} has 6 values/event", False)
        r.info(f"{base}: sum(xs_weight)", f"{xw.sum():.6g}")

    if manifest:
        r.check("total pseudo-events == manifest n_pairs",
                total == manifest["n_pairs"],
                f"{total:,} vs {manifest['n_pairs']:,}")
    return r


# ---------------------------------------------------------------------------
# unmatched
# ---------------------------------------------------------------------------

def qa_unmatched(pairs_dir, index_dir=None, table=None):
    import uproot

    r = Report("unmatched")
    manifest = G.load_pair_manifest(pairs_dir)
    path = os.path.join(str(pairs_dir), "unmatched.root")
    if not os.path.exists(path):
        r.check("unmatched.root present", False, path)
        return r
    with uproot.open(path) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        if M.UNMATCHED_TREE not in keys:
            r.info("no unmatched hemispheres", "every seed found a partner")
            return r
        U = f[M.UNMATCHED_TREE].arrays(library="np")

    n = len(U["hemi_id"])
    r.info("unmatched", f"{n:,} of {manifest['n_hemispheres']:,} hemispheres "
                        f"({_pct(n, manifest['n_hemispheres'])})")
    r.check("count matches the manifest", n == manifest["n_unmatched"],
            f"{n} vs {manifest['n_unmatched']}")
    r.check("hemi_id unique", len(np.unique(U["hemi_id"])) == n)

    # The reason breakdown IS the depletion-vs-geometry split. Note depletion
    # surfaces as TOO_FAR_DEPLETED, not ALL_CONSUMED: the pT window still has
    # candidates late in the run, they are just further away.
    for code, label in sorted(M.REASON_LABELS.items()):
        k = int((U["reason"] == code).sum())
        r.info(f"reason {label}", f"{k:,} ({_pct(k, n)})")
    dep = int((U["reason"] == M.REASON_TOO_FAR_DEPLETED).sum())
    intr = int((U["reason"] == M.REASON_TOO_FAR_INTRINSIC).sum())
    if dep + intr:
        r.info("of the too-far failures, supply vs geometry",
               f"{_pct(dep, dep + intr)} would have matched against a full pool")

    # draw_index: flat means intrinsic, end-loaded means depletion.
    di = U["draw_index"].astype(np.float64)
    ndraw = max(float(manifest["n_draws"]), 1.0)
    q = np.clip((di / ndraw * 4).astype(int), 0, 3)
    r.info("failures by quartile of the run",
           " | ".join(f"Q{k+1} {int((q == k).sum()):,}" for k in range(4)))
    late = float((di > 0.75 * ndraw).mean())
    r.info("fraction in the last quarter", f"{late:.1%}"
           + ("  <- end-loaded: depletion" if late > 0.5 else "  <- flat: intrinsic"))

    bd = U["best_distance"]
    fin = bd[bd >= 0]
    if len(fin):
        r.info("best_distance (TOO_FAR)",
               f"median {np.median(fin):.3f}, min {fin.min():.3f} "
               f"(cut {manifest['max_distance']:g})")
        near = float((fin < 1.2 * manifest["max_distance"]).mean())
        r.info("fraction just past the cut (<1.2x)", f"{near:.1%}"
               " - a loosened cut would recover these")

    # Are the rejects kinematically special, or just unlucky?
    if index_dir is not None:
        ix = ib.HemisphereIndex(index_dir, table)
        keep = np.ones(len(ix), dtype=bool)
        keep[U["hemi_id"]] = False
        for name in ("pt", "eta", "mass", "thrust"):
            allv = ix[name][:]
            r.info(f"{name}: matched vs unmatched (mean)",
                   f"{allv[keep].mean():.3f} vs {allv[~keep].mean():.3f}")
    return r


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        prog="run3-mj-mixqa",
        description="Stage 4: check the index, pairs, legs and pseudo-events.",
    )
    p.add_argument("stage", choices=["index", "spot", "pairs", "legs",
                                     "stitched", "unmatched", "all"])
    p.add_argument("-i", "--index-dir", default=None)
    p.add_argument("-t", "--file-table", default=None)
    p.add_argument("-p", "--pairs-dir", default=None)
    p.add_argument("-l", "--legs-dir", default=None)
    p.add_argument("-s", "--stitched", nargs="*", default=None)
    p.add_argument("--shards", nargs="*", default=None)
    p.add_argument("--n-spot", type=int, default=2000)
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)

    table = ft.load_file_table(args.file_table) if args.file_table else None
    fields = None
    if args.config:
        with open(args.config) as f:
            fields = G.payload_branches(json.load(f))

    want = ({"index", "spot", "pairs", "legs", "stitched", "unmatched"}
            if args.stage == "all" else {args.stage})
    reports = []

    ix = None
    if {"index", "spot"} & want:
        if not args.index_dir:
            sys.exit("--index-dir is required for the index and spot checks.")
        rep, ix = qa_index(args.index_dir, table, args.shards)
        if "index" in want:
            reports.append(rep)
        elif ix is None:
            reports.append(rep)
    if "spot" in want:
        if table is None:
            sys.exit("--file-table is required for the spot check.")
        reports.append(qa_spot(ix, table, n_spot=args.n_spot))
    if "pairs" in want:
        if not args.pairs_dir:
            sys.exit("--pairs-dir is required for the pairs check.")
        reports.append(qa_pairs(args.pairs_dir, table))
    if "legs" in want:
        if not (args.pairs_dir and args.legs_dir):
            sys.exit("--pairs-dir and --legs-dir are required for the legs check.")
        reports.append(qa_legs(args.pairs_dir, args.legs_dir))
    if "stitched" in want:
        paths = args.stitched or []
        if not paths:
            sys.exit("--stitched <files...> is required for the stitched check.")
        reports.append(qa_stitched(paths, args.pairs_dir, fields))
    if "unmatched" in want:
        if not args.pairs_dir:
            sys.exit("--pairs-dir is required for the unmatched report.")
        reports.append(qa_unmatched(args.pairs_dir, args.index_dir, table))

    ok = True
    for rep in reports:
        ok = rep.print() and ok
    print()
    print("QA PASSED" if ok else "QA FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""match.py - stage 2: pair hemispheres using only the index.

Reads the global index, draws hemispheres uniformly at random, finds each one's
partner, and streams out a list of pairs. Never touches event data.

**Single use.** A draw consumes the seed and a match consumes the partner, so
each hemisphere appears in at most one pair. There is no budget column, no
stochastic rounding, no re-use, and no flag to turn any of that back on. Two
consequences worth naming: a repeat pair is unrepresentable, so no prior-match
blacklist is needed; and since a 5-jet event yields at most one 3-jet
hemisphere, each *source event* lands in at most one pseudo-event, which is what
makes the output statistically independent.

**The search is exact, not approximate.** Candidates are bucketed on
(pT slab, directed-phi cell, eta cell) with every cell at least ``max_distance``
wide. A Euclidean distance of at most ``max_distance`` implies an L-infinity
distance of at most ``max_distance``, so every admissible candidate lies in the
3x3 neighbourhood of the query cell - the gather is exhaustive and the argmin
over it is the true argmin. Distances are then recomputed with the exact
``library.py`` expression, so the grid only ever decides *which* rows to look at.

**Rejections are recorded, not counted.** A seed with no partner produces a row
in ``unmatched.root`` carrying why it failed, how many candidates it had before
and after availability, the distance it would have achieved against a full pool,
and at which draw. That last pair of numbers is what separates depletion from
intrinsic sparseness as a measurement rather than an estimate - and the
separation is not the obvious one, because depletion surfaces as TOO_FAR (the
window still has candidates, they are just further away) rather than as
ALL_CONSUMED.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

from . import filetable as ft
from . import index_build as ib
from .hemisphere import directed_phi_array
from .library import _direction_delta

_TWO_PI = 2.0 * np.pi

PAIRS_TREE = "pairs"
UNMATCHED_TREE = "unmatched"
MANIFEST_NAME = "pairs_manifest.json"

# Why a seed produced no pair. Physically different causes wanting different
# responses, so they are recorded separately rather than summed.
#
# The split within TOO_FAR is the one that matters and is not obvious: as the
# pool drains, the pT window still contains candidates - they are just further
# away - so depletion shows up as TOO_FAR, not as ALL_CONSUMED. Asking what the
# best distance would have been against the *full* pool separates the two
# cleanly, and costs one extra query on the failure path only.
REASON_WINDOW_EMPTY = 0    # nothing in the sample has a compatible pT
REASON_ALL_CONSUMED = 1    # the window had candidates; every one was used up
REASON_TOO_FAR_DEPLETED = 2   # a full pool WOULD have matched it: supply
REASON_TOO_FAR_INTRINSIC = 3  # nothing near it even in a full pool: geometry
REASON_LABELS = {
    REASON_WINDOW_EMPTY: "WINDOW_EMPTY",
    REASON_ALL_CONSUMED: "ALL_CONSUMED",
    REASON_TOO_FAR_DEPLETED: "TOO_FAR_DEPLETED",
    REASON_TOO_FAR_INTRINSIC: "TOO_FAR_INTRINSIC",
}

PAIR_COLUMNS = {
    "pair_id":             np.dtype(np.int64),
    "seed_hemi_id":        np.dtype(np.int64),
    "match_hemi_id":       np.dtype(np.int64),
    "match_distance":      np.dtype(np.float32),
    "draw_index":          np.dtype(np.int64),
    "seed_file_id":        np.dtype(np.int32),
    "seed_entry":          np.dtype(np.int32),
    "seed_side":           np.dtype(np.int8),
    "seed_slot":           np.dtype(np.uint8),
    "seed_jet_mask":       np.dtype(np.uint16),
    "match_file_id":       np.dtype(np.int32),
    "match_entry":         np.dtype(np.int32),
    "match_side":          np.dtype(np.int8),
    "match_slot":          np.dtype(np.uint8),
    "match_jet_mask":      np.dtype(np.uint16),
    "seed_run":            np.dtype(np.uint32),
    "seed_luminosityBlock": np.dtype(np.uint32),
    "seed_event":          np.dtype(np.uint64),
    "match_run":           np.dtype(np.uint32),
    "match_luminosityBlock": np.dtype(np.uint32),
    "match_event":         np.dtype(np.uint64),
    # QA replay: lets stage 4 re-verify the pT window and the geometry without
    # opening the index at all.
    "seed_pt":             np.dtype(np.float32),
    "seed_eta":            np.dtype(np.float32),
    "seed_partner_eta":    np.dtype(np.float32),
    "seed_thrust_phi":     np.dtype(np.float32),
    "match_pt":            np.dtype(np.float32),
    "match_eta":           np.dtype(np.float32),
    "match_partner_eta":   np.dtype(np.float32),
    "match_thrust_phi":    np.dtype(np.float32),
}

UNMATCHED_COLUMNS = {
    "hemi_id":               np.dtype(np.int64),
    "reason":                np.dtype(np.uint8),
    "best_distance":         np.dtype(np.float32),
    "best_distance_full":    np.dtype(np.float32),
    "n_in_window":           np.dtype(np.int32),
    "n_available_in_window": np.dtype(np.int32),
    "draw_index":            np.dtype(np.int64),
    "file_id":               np.dtype(np.int32),
    "entry":                 np.dtype(np.int32),
    "side":                  np.dtype(np.int8),
    "hemi_slot":             np.dtype(np.uint8),
    "pt":                    np.dtype(np.float32),
    "eta":                   np.dtype(np.float32),
    "phi":                   np.dtype(np.float32),
    "mass":                  np.dtype(np.float32),
    "partner_eta":           np.dtype(np.float32),
    "thrust_phi":            np.dtype(np.float32),
    "dir_phi":               np.dtype(np.float64),
}


# ---------------------------------------------------------------------------
# The bucket grid
# ---------------------------------------------------------------------------

class Grid:
    """Bucketed (pT slab, directed-phi cell, eta cell) candidate lookup.

    Every cell is at least ``max_distance`` wide, which is what makes a 3x3
    neighbourhood search exhaustive rather than approximate.
    """

    def __init__(self, pt, dir_phi, eta, *, max_distance, pt_tolerance):
        self.max_distance = float(max_distance)
        self.pt_tolerance = float(pt_tolerance)
        n = len(pt)

        self.pt = np.asarray(pt, dtype=np.float64)
        self.dir_phi = np.asarray(dir_phi, dtype=np.float64)
        self.eta = np.asarray(eta, dtype=np.float64)

        # pT slabs: one tolerance unit wide, so a +-tol window spans a bounded
        # number of them regardless of scale.
        self.pt_min = float(self.pt[self.pt > 0].min()) if np.any(self.pt > 0) else 1.0
        self.log_step = math.log1p(self.pt_tolerance)
        slab = self._slab_of(self.pt)
        self.n_slab = int(slab.max()) + 1 if n else 1

        w = self.max_distance
        self.n_phi = max(3, int(math.floor(_TWO_PI / w)))
        self.phi_w = _TWO_PI / self.n_phi
        self.eta_lo = float(self.eta.min()) if n else 0.0
        eta_hi = float(self.eta.max()) if n else 0.0
        self.eta_w = w
        self.n_eta = max(1, int(math.floor((eta_hi - self.eta_lo) / w)) + 1)

        phi_c = self._phi_cell(self.dir_phi)
        eta_c = np.clip(self._eta_cell(self.eta), 0, self.n_eta - 1)
        bucket = (slab * self.n_phi + phi_c) * self.n_eta + eta_c
        self.n_bucket = self.n_slab * self.n_phi * self.n_eta

        self.rows = np.argsort(bucket, kind="stable").astype(np.int64)
        counts = np.bincount(bucket, minlength=self.n_bucket)
        self.starts = np.zeros(self.n_bucket + 1, dtype=np.int64)
        np.cumsum(counts, out=self.starts[1:])

        # pT-sorted order, for the exact window population count used when a
        # query fails (never on the success path - it is O(window)).
        self.pt_order = np.argsort(self.pt, kind="stable").astype(np.int64)
        self.pt_sorted = self.pt[self.pt_order]

    def _slab_of(self, pt):
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.floor(np.log(np.maximum(pt, 1e-9) / self.pt_min) / self.log_step)
        return np.maximum(s, 0).astype(np.int64)

    def _phi_cell(self, phi):
        return (np.floor((np.asarray(phi) % _TWO_PI) / self.phi_w)
                .astype(np.int64) % self.n_phi)

    def _eta_cell(self, eta):
        return np.floor((np.asarray(eta) - self.eta_lo) / self.eta_w).astype(np.int64)

    def window_bounds(self, pt_seed):
        """The hard pT window, exactly as ``library.py`` applies it."""
        return (pt_seed - self.pt_tolerance * pt_seed,
                pt_seed + self.pt_tolerance * pt_seed)

    def candidates(self, pt_seed, query_phi, target_eta):
        """Every row that could be within ``max_distance`` of the query.

        A superset: the exact pT cut and the distance cut are applied by the
        caller. Exhaustive by the L-infinity argument above.
        """
        lo, hi = self.window_bounds(pt_seed)
        s0 = int(self._slab_of(np.array([max(lo, 1e-9)]))[0])
        s1 = int(self._slab_of(np.array([max(hi, 1e-9)]))[0])
        s0 = max(s0 - 1, 0)
        s1 = min(s1 + 1, self.n_slab - 1)
        if s1 < s0:
            return np.empty(0, dtype=np.int64)

        pc = int(self._phi_cell(np.array([query_phi]))[0])
        ec = int(self._eta_cell(np.array([target_eta]))[0])
        phi_cells = sorted({(pc + d) % self.n_phi for d in (-1, 0, 1)})
        eta_cells = [e for e in (ec - 1, ec, ec + 1) if 0 <= e < self.n_eta]
        if not eta_cells:
            return np.empty(0, dtype=np.int64)

        chunks = []
        for s in range(s0, s1 + 1):
            base_s = s * self.n_phi
            for p in phi_cells:
                base = (base_s + p) * self.n_eta
                for e in eta_cells:
                    b = base + e
                    a, z = self.starts[b], self.starts[b + 1]
                    if z > a:
                        chunks.append(self.rows[a:z])
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(chunks)

    def window_population(self, pt_seed):
        """How many rows lie in the pT window, ignoring availability."""
        lo, hi = self.window_bounds(pt_seed)
        a = int(np.searchsorted(self.pt_sorted, lo, side="left"))
        z = int(np.searchsorted(self.pt_sorted, hi, side="right"))
        return a, z

    def window_rows(self, pt_seed):
        a, z = self.window_population(pt_seed)
        return self.pt_order[a:z]


# ---------------------------------------------------------------------------
# The matcher
# ---------------------------------------------------------------------------

class Matcher:
    """Single-use matching over an index.

    ``available`` replaces the old budget array. The draw pool reproduces
    ``HemisphereLibrary``'s swap-pop structure exactly, so a given rng seed
    yields the same draw sequence.
    """

    def __init__(self, pt, dir_phi, eta, partner_eta, event_key, *,
                 max_distance=0.5, pt_tolerance=0.10, seed=42):
        self.n = len(pt)
        self.pt = np.asarray(pt, dtype=np.float64)
        self.dir_phi = np.asarray(dir_phi, dtype=np.float64)
        self.eta = np.asarray(eta, dtype=np.float64)
        self.partner_eta = np.asarray(partner_eta, dtype=np.float64)
        self.event_key = np.asarray(event_key, dtype=np.int64)
        self.max_distance = float(max_distance)
        self.pt_tolerance = float(pt_tolerance)

        self.grid = Grid(self.pt, self.dir_phi, self.eta,
                         max_distance=max_distance, pt_tolerance=pt_tolerance)
        self.rng = np.random.default_rng(seed)

        self.available = np.ones(self.n, dtype=bool)
        self._avail = list(range(self.n))
        self._avail_pos = list(range(self.n))
        self.draw_index = 0

    # -- pool ---------------------------------------------------------------

    @property
    def n_available(self):
        return len(self._avail)

    def _consume(self, i):
        if not self.available[i]:
            return
        self.available[i] = False
        pos = self._avail_pos[i]
        last = self._avail[-1]
        self._avail[pos] = last
        self._avail_pos[last] = pos
        self._avail.pop()
        self._avail_pos[i] = -1

    def draw(self):
        """A uniformly random available hemisphere, consumed. None when dry."""
        if not self._avail:
            return None
        k = int(self.rng.integers(len(self._avail)))
        i = self._avail[k]
        self._consume(i)
        self.draw_index += 1
        return i

    # -- geometry -----------------------------------------------------------

    def query_direction(self, i):
        """Where seed ``i``'s lost partner actually pointed."""
        return (self.dir_phi[i] + np.pi) % _TWO_PI

    def distances(self, cand, query_phi, target_eta):
        """The exact library.py metric, never the grid's approximation."""
        return np.sqrt(
            _direction_delta(self.dir_phi[cand], query_phi) ** 2
            + (self.eta[cand] - target_eta) ** 2
        )

    def find_partner(self, i, *, respect_availability=True):
        """The best partner for seed ``i``.

        Pure - mutates nothing. Returns ``(match, distance)`` with ``match=None``
        on failure and ``distance`` the nearest allowed candidate's distance
        (``inf`` when there was none at all).
        """
        qphi = self.query_direction(i)
        tgt = self.partner_eta[i]
        cand = self.grid.candidates(self.pt[i], qphi, tgt)
        if len(cand) == 0:
            return None, np.inf

        lo, hi = self.grid.window_bounds(self.pt[i])
        ok = (self.pt[cand] >= lo) & (self.pt[cand] <= hi)
        ok &= self.event_key[cand] != self.event_key[i]
        if respect_availability:
            ok &= self.available[cand]
        cand = cand[ok]
        if len(cand) == 0:
            return None, np.inf

        d = self.distances(cand, qphi, tgt)
        # Ties resolve to the lowest row index, matching library.py's chunked
        # argmin with a strict <.
        best = cand[np.lexsort((cand, d))[0]]
        bd = float(d.min())
        if bd > self.max_distance:
            return None, bd
        return int(best), bd

    def classify_failure(self, i, best_distance):
        """Why seed ``i`` failed, and the numbers that make it interpretable.

        Only ever called on the failure path: ``window_rows`` is O(window) and
        the second ``find_partner`` doubles the query, neither of which is
        affordable on every success and neither of which is needed there.

        Returns ``(reason, n_in_window, n_available_in_window, best_full)``,
        where ``best_full`` is the distance to the nearest candidate ignoring
        availability - the counterfactual that separates a supply failure from a
        geometric one.
        """
        rows = self.grid.window_rows(self.pt[i])
        rows = rows[self.event_key[rows] != self.event_key[i]]
        n_in = int(len(rows))
        n_avail = int(self.available[rows].sum()) if n_in else 0

        if n_in == 0:
            return REASON_WINDOW_EMPTY, n_in, n_avail, np.inf
        if n_avail == 0:
            return REASON_ALL_CONSUMED, n_in, n_avail, np.inf

        _, best_full = self.find_partner(i, respect_availability=False)
        reason = (REASON_TOO_FAR_DEPLETED if best_full <= self.max_distance
                  else REASON_TOO_FAR_INTRINSIC)
        return reason, n_in, n_avail, best_full

    def run(self, *, on_pair=None, on_unmatched=None, progress=False,
            min_fill_rate=None, check_every=10_000):
        """Draw until the pool is dry. Returns counters."""
        n_pairs = n_fail = 0
        bar = None
        if progress:
            try:
                from tqdm import tqdm
                bar = tqdm(total=self.n, unit="hemi", desc="matching")
            except ImportError:
                pass

        while True:
            i = self.draw()
            if i is None:
                break
            if bar is not None:
                bar.update(1)
            j, d = self.find_partner(i)
            if j is None:
                reason, n_in, n_avail, best_full = self.classify_failure(i, d)
                n_fail += 1
                if on_unmatched is not None:
                    on_unmatched(i, reason, d, n_in, n_avail, self.draw_index,
                                 best_full)
            else:
                self._consume(j)
                if bar is not None:
                    bar.update(1)
                if on_pair is not None:
                    on_pair(i, j, d, self.draw_index)
                n_pairs += 1

            if (min_fill_rate is not None and n_pairs + n_fail >= check_every
                    and (n_pairs + n_fail) % check_every == 0):
                rate = n_pairs / float(n_pairs + n_fail)
                if rate < min_fill_rate:
                    print(f"[stop] fill rate {rate:.3f} < --min-fill-rate "
                          f"{min_fill_rate}; {self.n_available:,} hemispheres left")
                    break

        if bar is not None:
            bar.close()
        return {"pairs": n_pairs, "unmatched": n_fail,
                "draws": self.draw_index, "left": self.n_available}


# ---------------------------------------------------------------------------
# Building a matcher from an index
# ---------------------------------------------------------------------------

def matcher_from_index(ix, *, max_distance=0.5, pt_tolerance=0.10, seed=42,
                       rows=None):
    """A Matcher over a whole index, or over a subset of its rows."""
    sel = slice(None) if rows is None else rows
    dir_phi = directed_phi_array(ix["thrust_phi"][:], ix["side"][:])
    key = (ix["file_id"][:].astype(np.int64) << np.int64(32)) | \
          ix["entry"][:].astype(np.int64)
    m = Matcher(
        ix["pt"][:][sel], dir_phi[sel], ix["eta"][:][sel],
        ix["partner_eta"][:][sel], key[sel],
        max_distance=max_distance, pt_tolerance=pt_tolerance, seed=seed,
    )
    m.hemi_ids = (np.arange(len(ix), dtype=np.int64) if rows is None
                  else np.asarray(rows, dtype=np.int64))
    return m


class PairWriter:
    """Streams pair chunks, so a crash leaves k complete usable chunks."""

    def __init__(self, out_dir, ix, *, pairs_per_chunk=100_000, hemi_ids=None):
        self.dir = str(out_dir)
        os.makedirs(self.dir, exist_ok=True)
        self.ix = ix
        self.per_chunk = int(pairs_per_chunk)
        self.hemi_ids = hemi_ids
        self.buf = {k: [] for k in PAIR_COLUMNS}
        self.chunks = []
        self.n_pairs = 0

    def add(self, i, j, d, draw_index):
        ix, b = self.ix, self.buf
        si = int(self.hemi_ids[i]) if self.hemi_ids is not None else int(i)
        mj = int(self.hemi_ids[j]) if self.hemi_ids is not None else int(j)
        b["pair_id"].append(self.n_pairs)
        b["seed_hemi_id"].append(si)
        b["match_hemi_id"].append(mj)
        b["match_distance"].append(d)
        b["draw_index"].append(draw_index)
        for tag, h in (("seed", si), ("match", mj)):
            b[f"{tag}_file_id"].append(int(ix["file_id"][h]))
            b[f"{tag}_entry"].append(int(ix["entry"][h]))
            b[f"{tag}_side"].append(int(ix["side"][h]))
            b[f"{tag}_slot"].append(int(ix["hemi_slot"][h]))
            b[f"{tag}_jet_mask"].append(int(ix["jet_mask"][h]))
            b[f"{tag}_run"].append(int(ix["run"][h]))
            b[f"{tag}_luminosityBlock"].append(int(ix["luminosityBlock"][h]))
            b[f"{tag}_event"].append(int(ix["event"][h]))
            b[f"{tag}_pt"].append(float(ix["pt"][h]))
            b[f"{tag}_eta"].append(float(ix["eta"][h]))
            b[f"{tag}_partner_eta"].append(float(ix["partner_eta"][h]))
            b[f"{tag}_thrust_phi"].append(float(ix["thrust_phi"][h]))
        self.n_pairs += 1
        if len(b["pair_id"]) >= self.per_chunk:
            self.flush()

    def flush(self):
        if not self.buf["pair_id"]:
            return
        import uproot
        cid = len(self.chunks)
        name = f"pairs_{cid:05d}.root"
        tmp = os.path.join(self.dir, name + ".tmp")
        cols = {k: np.asarray(self.buf[k], dtype=dt)
                for k, dt in PAIR_COLUMNS.items()}
        lo = int(cols["pair_id"][0])
        hi = int(cols["pair_id"][-1]) + 1
        with uproot.recreate(tmp) as f:
            f.mktree(PAIRS_TREE, {k: v.dtype for k, v in cols.items()})
            f[PAIRS_TREE].extend(cols)
            f["chunk_id"] = _label(str(cid))
            f["pair_lo"] = _label(str(lo))
            f["pair_hi"] = _label(str(hi))
            f["complete"] = _label("ok")
        os.replace(tmp, os.path.join(self.dir, name))
        self.chunks.append({"file": name, "chunk_id": cid,
                            "pair_lo": lo, "pair_hi": hi,
                            "n_pairs": len(cols["pair_id"])})
        for v in self.buf.values():
            v.clear()


class UnmatchedWriter:
    """Every rejection, with why - see the module docstring."""

    def __init__(self, ix, hemi_ids=None):
        self.ix = ix
        self.hemi_ids = hemi_ids
        self.buf = {k: [] for k in UNMATCHED_COLUMNS}

    def add(self, i, reason, best_d, n_in, n_avail, draw_index,
            best_full=np.inf):
        ix, b = self.ix, self.buf
        h = int(self.hemi_ids[i]) if self.hemi_ids is not None else int(i)
        b["hemi_id"].append(h)
        b["reason"].append(reason)
        # inf does not survive float32 meaningfully; -1 marks "no candidate at all"
        b["best_distance"].append(-1.0 if not np.isfinite(best_d) else best_d)
        b["best_distance_full"].append(
            -1.0 if not np.isfinite(best_full) else best_full)
        b["n_in_window"].append(n_in)
        b["n_available_in_window"].append(n_avail)
        b["draw_index"].append(draw_index)
        b["file_id"].append(int(ix["file_id"][h]))
        b["entry"].append(int(ix["entry"][h]))
        b["side"].append(int(ix["side"][h]))
        b["hemi_slot"].append(int(ix["hemi_slot"][h]))
        for k in ("pt", "eta", "phi", "mass", "partner_eta", "thrust_phi"):
            b[k].append(float(ix[k][h]))
        b["dir_phi"].append(float(
            directed_phi_array(ix["thrust_phi"][h], ix["side"][h])))

    def write(self, path):
        import uproot
        cols = {k: np.asarray(self.buf[k], dtype=dt)
                for k, dt in UNMATCHED_COLUMNS.items()}
        with uproot.recreate(path) as f:
            if len(cols["hemi_id"]):
                f.mktree(UNMATCHED_TREE, {k: v.dtype for k, v in cols.items()})
                f[UNMATCHED_TREE].extend(cols)
            f["complete"] = _label("ok")
        return len(cols["hemi_id"])


def _label(text, value=1.0):
    import boost_histogram as bh
    h = bh.Histogram(bh.axis.StrCategory([text]), storage=bh.storage.Double())
    h.view()[0] = float(value)
    return h


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        prog="run3-mj-match",
        description="Stage 2: pair hemispheres using only the index.",
    )
    p.add_argument("index_dir")
    p.add_argument("-t", "--file-table", default=None)
    p.add_argument("-o", "--out-dir", default="pairs")
    p.add_argument("--max-distance", type=float, default=0.5)
    p.add_argument("--pt-tolerance", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pairs-per-chunk", type=int, default=100_000)
    p.add_argument("--min-fill-rate", type=float, default=None,
                   help="stop early when the running success rate drops below "
                        "this, rather than draining the pool")
    p.add_argument("--rematch", default=None, metavar="UNMATCHED",
                   help="re-run matching over ONLY the hemispheres in an "
                        "unmatched.root, pairing them among themselves. "
                        "Answers 'is a second pass worth running' - the recovery "
                        "fraction is what a second pass would add. It is NOT the "
                        "depletion measurement: the rejects are precisely the "
                        "hemispheres that are far from each other, so a low "
                        "recovery here is expected even when depletion caused the "
                        "rejections. For that, use the TOO_FAR_DEPLETED vs "
                        "TOO_FAR_INTRINSIC split, which compares against the FULL "
                        "pool.")
    p.add_argument("--progress", action="store_true")
    args = p.parse_args(argv)

    table = ft.load_file_table(args.file_table) if args.file_table else None
    ix = ib.load_index(args.index_dir, args.file_table)

    rows = None
    if args.rematch:
        import uproot
        with uproot.open(args.rematch) as f:
            if UNMATCHED_TREE not in {k.split(";")[0] for k in f.keys()}:
                sys.exit(f"{args.rematch} holds no unmatched hemispheres.")
            rows = np.sort(f[UNMATCHED_TREE]["hemi_id"].array(library="np"))
        print(f"rematch: {len(rows):,} previously-unmatched hemispheres")

    m = matcher_from_index(ix, max_distance=args.max_distance,
                           pt_tolerance=args.pt_tolerance, seed=args.seed,
                           rows=rows)
    os.makedirs(args.out_dir, exist_ok=True)
    pw = PairWriter(args.out_dir, ix, pairs_per_chunk=args.pairs_per_chunk,
                    hemi_ids=m.hemi_ids)
    uw = UnmatchedWriter(ix, hemi_ids=m.hemi_ids)

    print(f"index:   {len(ix):,} hemispheres  (id {ix.manifest['index_id']})")
    print(f"cuts:    pT window +-{args.pt_tolerance:g}, max_distance "
          f"{args.max_distance:g}, seed {args.seed}")
    counters = m.run(on_pair=pw.add, on_unmatched=uw.add,
                     progress=args.progress, min_fill_rate=args.min_fill_rate)
    pw.flush()
    n_unmatched = uw.write(os.path.join(args.out_dir, "unmatched.root"))

    reasons = np.asarray(uw.buf["reason"], dtype=np.int64)
    breakdown = {REASON_LABELS[r]: int((reasons == r).sum())
                 for r in sorted(REASON_LABELS)}

    manifest = {
        "schema": "run3-mj-pairs",
        "schema_version": 1,
        "index_id": ix.manifest["index_id"],
        "index_dir": os.path.abspath(args.index_dir),
        "mode": ix.manifest["mode"],
        "max_distance": args.max_distance,
        "pt_tolerance": args.pt_tolerance,
        "rng_seed": args.seed,
        "rematch_of": args.rematch,
        "n_hemispheres": int(m.n),
        "n_pairs": int(counters["pairs"]),
        "n_unmatched": int(counters["unmatched"]),
        "n_draws": int(counters["draws"]),
        "n_left_undrawn": int(counters["left"]),
        "unmatched_reasons": breakdown,
        "chunks": pw.chunks,
    }
    # Because the pseudo-event weight is a product of two per-hemisphere
    # weights, it is not a cross section and the sample needs a global rescale.
    # Record the sum the rescale is derived from rather than leaving it to be
    # rediscovered later. The pair-weight sum belongs to stage 3b, which is where
    # xs_weight is actually formed.
    if table is not None:
        w_rel = ft.w_rel_by_id(table)[ix["file_id"][:]]
        manifest["sum_w_rel_hemispheres"] = float(w_rel.sum())
    with open(os.path.join(args.out_dir, MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    total = counters["pairs"] + counters["unmatched"]
    rate = counters["pairs"] / total if total else 0.0
    print(f"pairs:      {counters['pairs']:,}")
    print(f"unmatched:  {counters['unmatched']:,}  ({1 - rate:.2%} of draws)")
    for label, k in breakdown.items():
        print(f"    {label:<14} {k:,}")
    print(f"chunks:     {len(pw.chunks)}")
    print(f"-> {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""library.py - the hemisphere library and its nearest-neighbor matching.

Step 3 of the hemisphere-mixing method: collect the hemispheres of many
5-jet events into a ``HemisphereLibrary``, then, for a seed hemisphere, find
the library hemisphere that best "continues" its event. The matching axes are
the three we settled on:

  * hemisphere pT   - the summed transverse momentum, normalized by the
                      largest hemisphere pT in the library (encodes the
                      pT-balance condition);
  * directed phi    - the thrust-axis angle SIGNED by which end of the axis
                      the hemisphere points along (axis phi for the +n_T
                      side, phi + pi for the -n_T side; 2pi-periodic). The
                      query is the phi -> -phi mirror of where the seed's
                      LOST partner pointed, so the reflection at stitch time
                      lands the match opposite the seed - with the correct
                      side built into the coordinate, no rotation is ever
                      applied (phi is physical in the detector);
  * eta, CROSS-matched (two coordinates) - the candidate's OWN eta must sit
                      where the seed's discarded partner was
                      (candidate.eta ~ seed.partner_eta), so the pseudo-event
                      keeps the seed event's longitudinal boost; and the
                      candidate's partner eta must sit where the seed is
                      (candidate.partner_eta ~ seed.eta), so the seed is a
                      drop-in replacement for the candidate's lost partner.

"Closest" is the Euclidean distance in these four coordinates. pT enters
normalized to [0, 1]; the etas are used raw; the phi difference is folded
on the 2pi period of the directed angle.

This module is deliberately standalone (numpy only) so the analyzer can
``from run3_mj_mixer.library import HemisphereLibrary`` without pulling in
uproot/awkward.
"""

import math
from dataclasses import dataclass, field

import numpy as np

_PI = math.pi
_TWO_PI = 2.0 * math.pi


@dataclass
class Hemisphere:
    """One hemisphere: its jets plus the per-event context it was cut from."""

    jet_pt: np.ndarray
    jet_eta: np.ndarray
    jet_phi: np.ndarray
    jet_mass: np.ndarray
    pt: float           # summed transverse momentum (matching variable)
    thrust_phi: float   # event's transverse thrust axis, in [0, pi)
    partner_eta: float  # eta of the event's OTHER hemisphere (matching variable)
    eta: float = None   # eta of THIS hemisphere's summed four-vector
                        # (matching variable; computed from the jets if None)
    side: int = 0       # +1 = +n_T side, -1 = -n_T side (0 = unknown)
    weight: float = 1.0
    event_id: object = None  # opaque tag (e.g. (filename, entry)) of the source event

    def __post_init__(self):
        if self.eta is None:
            pt = np.asarray(self.jet_pt, dtype=float)
            phi = np.asarray(self.jet_phi, dtype=float)
            pz = float((pt * np.sinh(np.asarray(self.jet_eta, dtype=float))).sum())
            ptv = math.hypot(float((pt * np.cos(phi)).sum()),
                             float((pt * np.sin(phi)).sum()))
            self.eta = math.asinh(pz / ptv) if ptv > 0.0 else 0.0

    @property
    def n_jets(self):
        return len(self.jet_pt)


def reflect_phi(phi):
    """The reflection phi -> -phi on the [0, pi) thrust-axis range."""
    return (-phi) % _PI


def directed_phi(thrust_phi, side):
    """The transverse direction a hemisphere points along, in [0, 2pi):
    the thrust-axis angle for the +n_T side, + pi for the -n_T side.
    ``side == 0`` (unknown) is treated as the +n_T side."""
    alpha = thrust_phi % _TWO_PI
    return alpha if side >= 0 else (alpha + _PI) % _TWO_PI


def query_direction(seed):
    """Where the seed's ideal new partner points, in the library frame: the
    phi -> -phi mirror of the seed's lost partner's direction,
    (-(alpha_seed + pi)) mod 2pi = pi - alpha_seed."""
    return (_PI - directed_phi(seed.thrust_phi, seed.side)) % _TWO_PI


def _direction_delta(a, b):
    """|a - b| folded onto the 2pi-periodic directed-angle topology."""
    d = np.abs(a - b) % _TWO_PI
    return np.minimum(d, _TWO_PI - d)


class HemisphereLibrary:
    """All hemispheres handed to it, searchable by nearest-neighbor matching.

    ``addHemisphere`` stores a hemisphere; ``findPartnerHemisphere`` returns
    the stored hemisphere closest to a seed in the 4-coordinate matching
    space; ``returnRandomHemisphere`` draws a random still-available
    hemisphere to seed a pseudo-event.

    Usage budgets: every hemisphere gets an integer budget from its event
    weight by stochastic rounding, floor(w) + Bernoulli(frac(w)), with the
    library's own RNG (``seed`` parameter, default 42) - statistically an
    unweighted representation of the weighted sample (weight 1.0 -> exactly
    one use). Every time a hemisphere is handed out (random draw or returned
    match) its budget drops by one; at zero it is no longer drawn nor
    matchable. Pseudo-events built this way all carry weight 1.
    """

    def __init__(self, seed=42):
        self._hemispheres = []
        self._pt = []
        self._dir_phi = []  # directed_phi(thrust_phi, side) per hemisphere
        self._eta = []
        self._partner_eta = []
        self._pt_max = 0.0  # largest hemisphere pT seen: the pT normalization
        self._rng = np.random.default_rng(seed)
        self._budget = []        # remaining uses per hemisphere
        self._avail = []         # indices with budget > 0 (for random draws)
        self._avail_pos = []     # index -> position in _avail (-1 if absent)
        self._total_budget = 0
        self._key_to_index = {}  # (event_id, side) -> stored index
        # Prior matches: seed key -> set of stored-hemisphere indices already
        # paired with it, in EITHER direction (if A matched B, B also can't
        # match A). Hemispheres MAY be reused by different seeds, but the
        # same pair never repeats - that would produce duplicate events.
        self._prior_matches = {}

    def __len__(self):
        return len(self._hemispheres)

    @property
    def pt_max(self):
        return self._pt_max

    @property
    def n_available(self):
        """Hemispheres with budget remaining."""
        return len(self._avail)

    @property
    def total_budget(self):
        """Sum of remaining budgets (drops as hemispheres are used)."""
        return self._total_budget

    @staticmethod
    def _seed_key(seed):
        """Identity of a seed for prior-match bookkeeping: (event_id, side)
        when the seed carries an event_id, else the object itself (queries
        with keyless, distinct seed objects are not deduplicated)."""
        return (seed.event_id, seed.side) if seed.event_id is not None else id(seed)

    def resetMatches(self):
        """Forget all recorded prior matches (budgets are NOT restored)."""
        self._prior_matches.clear()

    def _consume(self, index, amount=1):
        """Take ``amount`` uses (or everything, if fewer remain) from a
        hemisphere's budget, maintaining the random-draw pool."""
        take = min(amount, self._budget[index])
        if take <= 0:
            return
        self._budget[index] -= take
        self._total_budget -= take
        if self._budget[index] == 0 and self._avail_pos[index] >= 0:
            pos = self._avail_pos[index]
            last = self._avail[-1]
            self._avail[pos] = last
            self._avail_pos[last] = pos
            self._avail.pop()
            self._avail_pos[index] = -1

    def returnRandomHemisphere(self):
        """A uniformly random hemisphere among those with budget left, with
        its budget decremented by one. Returns ``None`` when the library is
        exhausted. Deterministic for a given construction ``seed``."""
        if not self._avail:
            return None
        k = int(self._rng.integers(len(self._avail)))
        index = self._avail[k]
        self._consume(index)
        return self._hemispheres[index]

    def addHemisphere(self, jet_pt, jet_eta, jet_phi, jet_mass, thrust_phi,
                      partner_eta, pt=None, eta=None, side=0, weight=1.0,
                      event_id=None):
        """Store one hemisphere.

        ``pt`` (summed transverse momentum) and ``eta`` (of the summed
        four-vector) are computed from the jets when omitted.
        Returns the stored ``Hemisphere``.
        """
        jet_pt = np.asarray(jet_pt, dtype=float)
        jet_eta = np.asarray(jet_eta, dtype=float)
        jet_phi = np.asarray(jet_phi, dtype=float)
        jet_mass = np.asarray(jet_mass, dtype=float)
        if pt is None:
            pt = float(np.hypot((jet_pt * np.cos(jet_phi)).sum(),
                                (jet_pt * np.sin(jet_phi)).sum()))
        hemi = Hemisphere(jet_pt, jet_eta, jet_phi, jet_mass,
                          pt=float(pt), thrust_phi=float(thrust_phi) % _PI,
                          partner_eta=float(partner_eta),
                          eta=None if eta is None else float(eta),
                          side=int(side), weight=float(weight),
                          event_id=event_id)
        index = len(self._hemispheres)
        self._hemispheres.append(hemi)
        self._pt.append(hemi.pt)
        self._dir_phi.append(directed_phi(hemi.thrust_phi, hemi.side))
        self._eta.append(hemi.eta)
        self._partner_eta.append(hemi.partner_eta)
        self._pt_max = max(self._pt_max, hemi.pt)
        # Usage budget: stochastic rounding of the event weight.
        frac = hemi.weight - math.floor(hemi.weight)
        budget = int(math.floor(hemi.weight)) + int(self._rng.random() < frac)
        self._budget.append(budget)
        self._total_budget += budget
        if budget > 0:
            self._avail_pos.append(len(self._avail))
            self._avail.append(index)
        else:
            self._avail_pos.append(-1)
        if hemi.event_id is not None:
            self._key_to_index[(hemi.event_id, hemi.side)] = index
        return hemi

    def findPartnerHemisphere(self, seed, exclude_event_id=None,
                              max_distance=None, record=True,
                              with_distance=False, progress=False,
                              chunk_size=100_000):
        """The stored hemisphere closest to ``seed`` in the matching space.

        Four coordinates: candidate pT vs seed pT (both / max pT); candidate
        directed phi vs ``query_direction(seed)``; candidate eta vs seed
        PARTNER eta and candidate partner eta vs seed eta (the cross-matched
        pair: the returned hemisphere sits where the seed's discarded partner
        was, and its own lost partner sat where the seed is - preserving the
        event's longitudinal boost). The directed-phi coordinate builds the
        correct side into the match: reflecting the returned hemisphere's
        jets phi -> -phi puts it opposite the seed, so no rotation is ever
        needed. ``seed`` is a ``Hemisphere`` (its ``pt``, ``thrust_phi``,
        ``side``, ``eta`` and ``partner_eta`` are used).

        ``exclude_event_id`` skips stored hemispheres with that ``event_id``
        (pass the seed's own to forbid same-event pairing, seed itself
        included).

        Prior matches are rejected SYMMETRICALLY: pairs already produced (in
        either direction - if A matched B, a later B-seeded search excludes
        A) are skipped, so repeated queries return the next-nearest partner
        instead of duplicating a pseudo-event; different seeds may still
        reuse the same hemisphere while it has budget left.

        With ``record=True`` (production): exhausted candidates (budget 0)
        are excluded, the returned partner's budget is decremented, and the
        pair is blacklisted both ways. With ``record=False`` (diagnostics):
        a pure geometry query - no budgets, no recording, no side effects.

        ``max_distance``: if no candidate lies within it, returns ``None``
        instead of a hemisphere - and (record mode, seed in the library) the
        SEED's remaining budget is zeroed: the best match is deterministic,
        so a seed that failed once would only ever fail again.

        ``with_distance=True`` returns ``(hemisphere, distance)`` (the
        distance of the best candidate even when rejected as too far).
        ``progress=True`` shows a tqdm bar over the scan (``chunk_size``
        hemispheres per step; no-op if tqdm is unavailable).
        Raises ``ValueError`` if the library is empty, or fully excluded
        when no ``max_distance`` is set.
        """
        n = len(self._hemispheres)
        if n == 0:
            raise ValueError("HemisphereLibrary is empty.")

        pt = np.asarray(self._pt)
        dir_phi = np.asarray(self._dir_phi)
        eta = np.asarray(self._eta)
        partner_eta = np.asarray(self._partner_eta)
        norm = self._pt_max if self._pt_max > 0.0 else 1.0
        query_phi = query_direction(seed)

        excluded = np.zeros(n, dtype=bool)
        if exclude_event_id is not None:
            excluded |= np.fromiter(
                (h.event_id == exclude_event_id for h in self._hemispheres),
                dtype=bool, count=n,
            )
        prior = self._prior_matches.get(self._seed_key(seed))
        if prior:
            excluded[list(prior)] = True
        if record:  # production: exhausted hemispheres are not matchable
            excluded |= np.asarray(self._budget) <= 0

        bar = None
        if progress:
            try:
                from tqdm import tqdm
                bar = tqdm(total=n, unit="hemi", desc="matching")
            except ImportError:
                pass

        best, best_d2 = -1, np.inf
        for lo in range(0, n, chunk_size):
            hi = min(lo + chunk_size, n)
            d2 = (
                ((pt[lo:hi] - seed.pt) / norm) ** 2
                + _direction_delta(dir_phi[lo:hi], query_phi) ** 2
                + (eta[lo:hi] - seed.partner_eta) ** 2
                + (partner_eta[lo:hi] - seed.eta) ** 2
            )
            d2[excluded[lo:hi]] = np.inf
            k = int(np.argmin(d2))
            if d2[k] < best_d2:
                best, best_d2 = lo + k, float(d2[k])
            if bar is not None:
                bar.update(hi - lo)
        if bar is not None:
            bar.close()

        best_d = math.sqrt(best_d2) if np.isfinite(best_d2) else np.inf
        viable = np.isfinite(best_d2) and (
            max_distance is None or best_d <= max_distance
        )

        if not viable:
            if max_distance is None:
                raise ValueError(
                    "Every hemisphere in the library is excluded "
                    f"(event_id={exclude_event_id!r}, prior matches for this "
                    f"seed: {len(prior) if prior else 0})."
                )
            if record:
                # A failed seed would fail identically forever: retire it.
                seed_index = self._key_to_index.get(self._seed_key(seed))
                if seed_index is not None:
                    self._consume(seed_index, amount=self._budget[seed_index])
            return (None, float(best_d)) if with_distance else None

        if record:
            self._consume(best)
            self._prior_matches.setdefault(self._seed_key(seed), set()).add(best)
            # Symmetric: the partner may never match back onto this seed.
            seed_index = self._key_to_index.get(self._seed_key(seed))
            if seed_index is not None:
                match_key = self._seed_key(self._hemispheres[best])
                self._prior_matches.setdefault(match_key, set()).add(seed_index)
        if with_distance:
            return self._hemispheres[best], float(best_d)
        return self._hemispheres[best]

"""hemisphere.py - the hemisphere physics, shared by every stage that needs it.

Lifted out of the old ``mix.py`` so that stage 1a (which records the split) and
stage 3b (which recomputes hemisphere four-vectors by parentage) cannot disagree
about what a hemisphere is.

The split rule lives in exactly one place, ``hemisphere_split_mask``. Stage 1a
computes it once and feeds the *same* boolean array to
``hemisphere_fourvectors`` and ``pack_jet_masks``, so the recorded ``jet_mask``
and the recorded summary are guaranteed consistent by construction rather than
by two implementations agreeing. This replaces the old arrangement, where
``stitch.py`` re-derived the mask in float64 from float32-stored values and
could silently disagree with the mixer that wrote them.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

# jet_mask is a uint16, so a hemisphere can describe at most 16 jets.
MAX_MASK_JETS = 16


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
    point.

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
    h_zero = best_val
    h_plus = _proj_sum(px, py, a0 + step)
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
# The split
# ---------------------------------------------------------------------------

def hemisphere_split_mask(pt, phi, phi_T):
    """Which jets fall on the +n_T side of the splitting plane.

    THE single definition of the split. A jet joins hemisphere index 0 (the
    +n_T side) when its transverse-momentum projection on the thrust axis is
    > 0, else hemisphere index 1. Returns a jagged boolean array shaped like
    ``pt``.

    Everything downstream - the summed four-vectors, the recorded ``jet_mask``,
    the ``n_jets`` count - must be derived from *this* array, not recomputed
    from a rounded copy of ``phi_T``.
    """
    cos_t = np.cos(phi_T)
    sin_t = np.sin(phi_T)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    return (px * cos_t + py * sin_t) > 0.0


def as_regular(arr, n_jets):
    """A jagged array whose rows all have length ``n_jets`` as an (n, n_jets)
    numpy array. Raises if any row differs - the caller is expected to have
    applied an exact-multiplicity cut already.
    """
    counts = ak.to_numpy(ak.num(arr, axis=1))
    if len(counts) and not np.all(counts == n_jets):
        bad = int(counts[counts != n_jets][0])
        raise ValueError(
            f"as_regular expected every row to have {n_jets} entries, found a "
            f"row with {bad}. Apply the exact-multiplicity cut first."
        )
    return ak.to_numpy(ak.to_regular(arr))


def pack_jet_masks(pos_regular):
    """Pack a per-jet membership mask into one uint16 per hemisphere.

    ``pos_regular`` is an (events, n_jets) numpy bool array as returned by
    ``as_regular(hemisphere_split_mask(...), n_jets)``.

    Bit *i* of the result is set when jet *i* of the event's stored jet
    collection belongs to that hemisphere - bit 0 is jet 0 (least significant).
    Returns an (events, 2) uint16 array, column 0 = the +n_T side, matching the
    hemisphere ordering of ``hemisphere_fourvectors``.
    """
    n_jets = pos_regular.shape[1] if pos_regular.ndim == 2 else 0
    if n_jets > MAX_MASK_JETS:
        raise ValueError(
            f"jet_mask is uint16 and cannot describe {n_jets} jets "
            f"(max {MAX_MASK_JETS})."
        )
    bit = (1 << np.arange(n_jets, dtype=np.uint32))
    m0 = (pos_regular * bit).sum(axis=1).astype(np.uint16)
    m1 = ((~pos_regular) * bit).sum(axis=1).astype(np.uint16)
    return np.stack([m0, m1], axis=1)


def popcount16(x):
    """Number of set bits, elementwise, for uint16 input."""
    x = np.asarray(x, dtype=np.uint16).astype(np.uint32)
    x = x - ((x >> 1) & 0x5555)
    x = (x & 0x3333) + ((x >> 2) & 0x3333)
    x = (x + (x >> 4)) & 0x0F0F
    return ((x * 0x0101) >> 8).astype(np.uint8) & 0x1F


def unpack_jet_mask(mask, n_jets):
    """The boolean membership of one jet_mask over ``n_jets`` slots."""
    bit = (1 << np.arange(n_jets, dtype=np.uint32))
    return (np.uint32(mask) & bit) != 0


def select_by_mask(values, mask, n_jets):
    """The entries of a length-``n_jets`` per-jet array selected by ``mask``.

    ``values`` is one event's jet array; ``mask`` its uint16 membership. Returns
    them in stored-collection order (ascending jet index).
    """
    return np.asarray(values)[unpack_jet_mask(mask, n_jets)]


# ---------------------------------------------------------------------------
# Hemisphere four-vectors
# ---------------------------------------------------------------------------

def hemisphere_fourvectors(pt, eta, phi, m, phi_T, pos_mask=None):
    """Split jets on the plane perpendicular to n_T and sum the four-vectors.

    ``pos_mask`` overrides the geometric assignment with an explicit per-jet
    membership (True -> hemisphere index 0). Stage 1a passes the mask from
    ``hemisphere_split_mask`` so the summary and the recorded ``jet_mask`` come
    from one computation; stage 3b passes a parentage mask to sum the two
    stitched halves by origin rather than by geometry, and the ``side`` slots
    are then labels, not a geometric guarantee.

    pt, eta, phi, m : jagged (events, var jets) awkward arrays.
    phi_T           : (events,) numpy array, the thrust-axis angle.

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

    if pos_mask is None:
        pos = (px * cos_t + py * sin_t) > 0.0
    else:
        pos = pos_mask

    def _sum(arr, mask):
        return ak.to_numpy(ak.sum(ak.where(mask, arr, 0.0), axis=1))

    cols = {name: [] for name in
            ("px", "py", "pz", "energy", "pt", "eta", "phi", "mass",
             "pt_par", "pt_perp", "n_jets", "side")}

    for mask, side in ((pos, 1), (~pos, -1)):
        h_px = _sum(px, mask)
        h_py = _sum(py, mask)
        h_pz = _sum(pz, mask)
        h_e = _sum(energy, mask)
        n_jets = ak.to_numpy(ak.sum(mask, axis=1)).astype(np.int32)

        h_pt = np.hypot(h_px, h_py)
        p2 = h_px ** 2 + h_py ** 2 + h_pz ** 2
        mass = np.sqrt(np.maximum(h_e ** 2 - p2, 0.0))
        h_phi = np.arctan2(h_py, h_px)
        with np.errstate(divide="ignore", invalid="ignore"):
            h_eta = np.arcsinh(np.where(h_pt > 0.0, h_pz / h_pt, 0.0))
        pt_par = h_px * cos_t + h_py * sin_t
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
# Matching coordinates
# ---------------------------------------------------------------------------

def directed_phi_array(thrust_phi, side):
    """Vectorized ``library.directed_phi``: the direction a hemisphere points.

    The thrust-axis angle for the +n_T side, + pi for the -n_T side, folded onto
    [0, 2pi). ``side == 0`` (unknown) counts as +n_T, matching the scalar
    version, which ``tests/test_index.py`` pins this against elementwise.

    Computed in **float64 from the stored float32 ``thrust_phi``**, and
    deliberately not stored as a column: rounding the result back to float32
    would make the matcher's coordinate differ in the last bits from what
    ``library.py`` computes, and those bits decide boundary cases and tie-breaks.
    Recomputing is a modulo over the index and costs milliseconds.
    """
    alpha = np.asarray(thrust_phi, dtype=np.float64) % (2.0 * np.pi)
    return np.where(np.asarray(side) >= 0, alpha, (alpha + np.pi) % (2.0 * np.pi))


# ---------------------------------------------------------------------------
# Read-side derivations
# ---------------------------------------------------------------------------

def hemisphere_summary(pt, eta, phi, mass, thrust_phi):
    """Recover the quantities the index deliberately does NOT store.

    ``px, py, pz, energy, pt_par, pt_perp`` are exact functions of
    ``(pt, eta, phi, mass, thrust_phi)``, so storing them would be six
    redundant columns on every row. This is the one place they are derived, so
    every consumer agrees.

    Inputs are numpy arrays (any shape, broadcast together). Returns a dict of
    arrays of that shape.
    """
    pt = np.asarray(pt, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    thrust_phi = np.asarray(thrust_phi, dtype=np.float64)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    p2 = px ** 2 + py ** 2 + pz ** 2
    return {
        "px": px,
        "py": py,
        "pz": pz,
        "energy": np.sqrt(p2 + mass ** 2),
        "pt_par": px * np.cos(thrust_phi) + py * np.sin(thrust_phi),
        "pt_perp": -px * np.sin(thrust_phi) + py * np.cos(thrust_phi),
    }

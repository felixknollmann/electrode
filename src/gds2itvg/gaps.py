"""Gap-profile correction strips (milestone 2, GUIDELINES §5.2).

With the active electrode e at 1 V and everything else at 0 V, the midline baseline puts
1 V on e's half of every adjacent gap and 0 V elsewhere. The cross-section model replaces
this with the surface profile q_g(u), u being the distance from e's edge into the gap.

The difference Δ(u) = q_g(u) − [u < g/2] is laid along every drawn edge of e as thin strips
of constant value. Their potential is added to the baseline.

- **Facing distance.** For each edge, the outward-normal probe finds the facing distance
  g(s) to the nearest other metal. It is piecewise constant along the edge, exactly so for
  Manhattan and parallel geometry.
- **Classification:**
  - g ≤ ``max_gap``: a gap, with profile q_g. If the facing metal is e itself (a slot inside
    e), both sides contribute and the sum is q(u) + q(g − u) − 1, as it should be.
  - otherwise: an opening, with the single-edge fringe q_∞(u), cut off at
    min(``opening_extent``, g).
- **Chip outline.** Edges lying on the chip outline get no strips; the finite-chip model
  (§5.3) covers the surface beyond it.
- **Lumping.** Strips farther than ``lump_distance`` × width from the ion ROI are merged
  into one strip with the same integral (the same far-field monopole).
- **Corners.** Corner wedges, O(g²) each, are left uncovered or doubly covered. That is a
  second-order error.
"""

from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box

from .crosssection import gap_profile, opening_profile
from .geometry import as_multipolygon, polygon_rings

__all__ = ["GapModel", "edge_intervals", "correction_strips", "merge_strip_edges"]


@dataclass(frozen=True)
class GapModel:
    """Device-stack parameters for the cross-section gap profiles (µm)."""

    metal_thickness: float = 1.0
    ground_depth: float | None = 10.0
    epsilon_r: float = 3.9
    strips_per_gap: int = 8
    opening_extent: float | None = None   # default: 10 x ground depth (or 100 µm)
    lump_distance: float = 20.0           # in units of the strip width

    def extent(self):
        if self.opening_extent is not None:
            return self.opening_extent
        return 10.0 * self.ground_depth if self.ground_depth else 100.0

    def profile(self, g):
        return gap_profile(g, self.metal_thickness, self.ground_depth, self.epsilon_r)

    def opening(self):
        return opening_profile(self.metal_thickness, self.ground_depth, self.epsilon_r,
                               extent=self.extent())


def _local_frame(p0, p1):
    t = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([t[1], -t[0]])  # outward for CCW exteriors and CW holes (material on left)
    return t, n


def edge_intervals(p0, p1, others, own, probe):
    """Facing distances along edge p0 -> p1.

    ``others`` and ``own`` are geometries (other metal, the electrode's own metal); ``probe``
    is how far to look. Returns ``[(s_a, s_b, d, is_self)]`` with d = inf if nothing is hit.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    t, n = _local_frame(p0, p1)
    R = Polygon([p0, p1, p1 + probe * n, p0 + probe * n])
    eps = 1e-7
    hits = []
    for geom, is_self in ((others, False), (own, True)):
        if geom is None or geom.is_empty:
            continue
        cut = geom.intersection(R)
        if is_self:
            # drop the electrode's own body touching the edge itself
            cut = cut.difference(Polygon([p0, p1, p1 + eps * 10 * n, p0 + eps * 10 * n]).buffer(0))
        cut = as_multipolygon(cut)
        if not cut.is_empty and cut.area > 1e-12:
            hits.append((cut, is_self))

    def to_s(xy):
        return (np.asarray(xy) - p0) @ t

    brk = {0.0, L}
    for cut, _ in hits:
        for s in to_s(shapely.get_coordinates(cut)):
            if eps < s < L - eps:
                brk.add(float(s))
    brk = sorted(brk)
    out = []
    for a, b in zip(brk[:-1], brk[1:]):
        if b - a < 1e-9:
            continue
        sm = 0.5 * (a + b)
        base = p0 + sm * t
        ray = LineString([base + eps * n, base + probe * n])
        best, best_self = np.inf, False
        for cut, is_self in hits:
            x = ray.intersection(cut)
            if not x.is_empty:
                d = float(np.min((shapely.get_coordinates(x) - base) @ n))
                if d < best:
                    best, best_self = d, is_self
        if out and out[-1][2] == best and out[-1][3] == best_self and abs(out[-1][1] - a) < 1e-9:
            out[-1] = (out[-1][0], b, best, best_self)
        else:
            out.append((a, b, best, best_self))
    return out


def _quad(p0, t, n, sa, sb, ua, ub):
    a, b = p0 + sa * t, p0 + sb * t
    return np.array([a + ua * n, b + ua * n, b + ub * n, a + ub * n])


def _delta_strips(model, d, is_self, max_gap, far):
    """[(u_a, u_b, value)] for one interval with facing distance d."""
    n = max(2, model.strips_per_gap // 2 * 2)
    if d <= max_gap:
        q = model.profile(round(d, 6))
        if far:
            mean = (q.integral() - d / 2) / d
            return [(0.0, d, mean)]
        half = n // 2
        e1 = np.linspace(0, d / 2, half + 1)
        e2 = np.linspace(d / 2, d, half + 1)
        m1 = q.strip_means(e1) - 1.0
        m2 = q.strip_means(e2)
        out = [(a, b, v) for a, b, v in zip(e1[:-1], e1[1:], m1)]
        out += [(a, b, v) for a, b, v in zip(e2[:-1], e2[1:], m2)]
        return out
    if model.ground_depth is None:
        return []  # without a buried ground an opening's surface potential is not local
    o = model.opening()
    U = min(model.extent(), d)
    if far:
        uu = o.u[o.u <= U]
        return [(0.0, U, float(np.trapezoid(o(uu), uu) / U))]
    # finer near the edge, where the fringe is steep
    edges = np.unique(np.concatenate([np.linspace(0, min(U, 4 * (model.ground_depth or 10)), n + 1),
                                      np.geomspace(min(U, 4 * (model.ground_depth or 10)), U, 5)]))
    return [(a, b, v) for a, b, v in zip(edges[:-1], edges[1:], o.strip_means(edges))]


def correction_strips(own_conductors, other_conductors, model: GapModel, max_gap,
                      chip_outline=None, ion_region=None):
    """Δ strips for one electrode. Returns a list of ``(quad (4, 2) array, value)`` in the
    input (GDS) coordinates."""
    own = as_multipolygon(shapely.union_all(list(own_conductors)))
    tree = shapely.STRtree(list(other_conductors)) if len(other_conductors) else None
    probe = max(max_gap, model.extent()) + 1.0
    outline_b = chip_outline.boundary if chip_outline is not None else None
    strips = []
    for ring in polygon_rings(own):
        for k in range(len(ring)):
            p0, p1 = ring[k], ring[(k + 1) % len(ring)]
            L = np.linalg.norm(p1 - p0)
            if L < 1e-9:
                continue
            t, n = _local_frame(p0, p1)
            if outline_b is not None:
                mid = 0.5 * (p0 + p1)
                if outline_b.distance(shapely.Point(mid)) < 1e-3 and \
                        not chip_outline.contains(shapely.Point(mid + 1e-2 * n)):
                    continue
            R = box(*Polygon([p0, p1, p1 + probe * n, p0 + probe * n]).bounds)
            near = [other_conductors[i] for i in tree.query(R)] if tree is not None else []
            others = shapely.union_all(near) if near else None
            for sa, sb, d, is_self in edge_intervals(p0, p1, others, own, probe):
                width = d if d <= max_gap else min(model.extent(), d)
                far = False
                if ion_region is not None:
                    seg = LineString([p0 + sa * t, p0 + sb * t])
                    far = seg.distance(ion_region) > model.lump_distance * width
                for ua, ub, v in _delta_strips(model, d, is_self, max_gap, far):
                    if v != 0.0 and ub > ua:
                        strips.append((_quad(p0, t, n, sa, sb, ua, ub), float(v)))
    return strips


def merge_strip_edges(strips, decimals=9):
    """Merge correction strips into one weighted edge set.

    Each strip ``(quad, value)`` contributes its four counter-clockwise edges with weight
    ``value``. Neighbouring strips share edges traversed in opposite directions, e.g. the
    long edges between the sub-strips across a gap, or the end edges between consecutive
    intervals. Such pairs collapse into one edge carrying the difference of the two
    weights, and pairs that cancel are dropped. The potential is unchanged, because the
    edge sum is linear in the weights.

    Returns ``(p1, p2, w)`` arrays.
    """
    acc = {}
    for quad, value in strips:
        ring = polygon_rings(Polygon(quad))
        if not ring:
            continue
        r = ring[0]
        for a, b in zip(r, np.roll(r, -1, axis=0)):
            ka = tuple(np.round(a, decimals))
            kb = tuple(np.round(b, decimals))
            if ka == kb:
                continue
            if ka < kb:
                key, sign = (ka, kb), 1.0
            else:
                key, sign = (kb, ka), -1.0
            acc[key] = acc.get(key, 0.0) + sign * value
    keys = [k for k, v in acc.items() if abs(v) > 1e-15]
    if not keys:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    p1 = np.array([k[0] for k in keys])
    p2 = np.array([k[1] for k in keys])
    w = np.array([acc[k] for k in keys])
    return p1, p2, w

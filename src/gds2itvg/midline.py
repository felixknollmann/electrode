"""Gapless baseline: split every inter-conductor gap along its midline.

Each conductor is grown into the adjacent gaps up to the gap centre line, so that the
in-plane potential steps from one electrode's value to the other's at the midline.

Algorithm (exact for gaps between parallel edges, whatever their width):

1. ``gap = closing(U, max_gap/2) - U`` with ``U`` the union of all conductors. The
   morphological closing fills gaps narrower than ``max_gap`` and leaves wider openings and
   open surface alone. Those stay at 0 V, consistent with a grounded plane.
2. For distance levels t_k, ``B_i(t) = buffer(C_i, t)``. The set
   ``B_i(t) \\ ∪_{j≠i} B_j(t)`` contains only points with d_i ≤ t < min_j d_j, i.e. points
   strictly closer to conductor i. Its union over the levels is an exact subset of the
   nearest-conductor (Voronoi) cell. The levels include half of every pairwise gap width
   (the exact midline of a uniform gap) plus a uniform ladder with step ``step``.
3. Leftover gap pieces, which only occur where gaps meet or taper, go to the conductor
   with the longest shared boundary.
"""

import numpy as np
import shapely

from .geometry import as_multipolygon

__all__ = ["gap_region", "split_gaps"]

_QUAD = 16


def gap_region(conductors, max_gap):
    union = shapely.union_all(conductors)
    r = max_gap / 2.0
    closed = union.buffer(r, quad_segs=_QUAD).buffer(-r, quad_segs=_QUAD)
    return as_multipolygon(shapely.make_valid(closed.difference(union)))


def _levels(conductors, tree, max_gap, step):
    half = set()
    for i, c in enumerate(conductors):
        for j in tree.query(c, predicate="dwithin", distance=max_gap):
            if j > i:
                d = c.distance(conductors[j])
                if 0 < d <= max_gap:
                    half.add(round(d / 2, 9))
    ladder = np.arange(step, max_gap / 2 + step, step)
    return sorted(half | {round(float(t), 9) for t in ladder})


def split_gaps(conductors, max_gap=30.0, step=1.0, return_info=False):
    """Grow each conductor to the midlines of its gaps (width <= ``max_gap``, µm).

    Returns a list of MultiPolygons in the order of ``conductors`` (and, with
    ``return_info``, a dict with the gap and leftover areas).
    """
    conductors = [as_multipolygon(c) for c in conductors]
    n = len(conductors)
    gap = gap_region(conductors, max_gap)
    info = {"gap_area": gap.area, "leftover_area": 0.0, "levels": []}
    if gap.is_empty or n < 2:
        return (conductors, info) if return_info else conductors
    tree = shapely.STRtree(conductors)
    levels = _levels(conductors, tree, max_gap, step)
    info["levels"] = levels
    gap_tree_geom = gap.buffer(0)

    cells = [[] for _ in range(n)]
    for t in levels:
        buf = [c.buffer(t, quad_segs=_QUAD) for c in conductors]
        btree = shapely.STRtree(buf)
        for i in range(n):
            others = [buf[j] for j in btree.query(buf[i], predicate="intersects") if j != i]
            own = buf[i].intersection(gap_tree_geom)
            if others:
                own = own.difference(shapely.union_all(others))
            if not own.is_empty:
                cells[i].append(own)
    claimed = [as_multipolygon(shapely.make_valid(shapely.union_all(c))) if c else None
               for c in cells]

    # Leftovers: assign each piece to the neighbour with the longest shared boundary.
    allclaimed = shapely.union_all([c for c in claimed if c is not None])
    leftover = as_multipolygon(shapely.make_valid(gap.difference(allclaimed)))
    extra = [[] for _ in range(n)]
    for piece in leftover.geoms:
        if piece.area <= 0:
            continue
        info["leftover_area"] += piece.area
        best, best_len = None, -1.0
        ring = piece.buffer(1e-6)
        for i in range(n):
            g = conductors[i] if claimed[i] is None else shapely.union_all([conductors[i], claimed[i]])
            ln = ring.intersection(g).area
            if ln > best_len:
                best, best_len = i, ln
        extra[best].append(piece)

    grown = []
    for i in range(n):
        parts = [conductors[i]] + ([claimed[i]] if claimed[i] is not None else []) + extra[i]
        g = shapely.union_all(parts)
        grown.append(as_multipolygon(shapely.make_valid(g)))
    return (grown, info) if return_info else grown

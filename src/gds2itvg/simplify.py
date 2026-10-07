"""Distance-graded simplification of electrode geometry far from the ions.

The midline split puts round arcs at convex corners and gap junctions (GUIDELINES §3.2),
about 7× more edges than the drawn polygons, and the potential cost scales with the edge
count. The arcs are tiny (radius ≤ max_gap/2) and almost all of them are millimetres from
the ions, where they do not matter.

Geometry inside the ion region grown by ``keep`` µm is left exact. Beyond that, nested
boxes double in size, and each band between them is simplified (Douglas–Peucker, topology
preserving) with tolerance min(``max_tol``, ``rel_tol`` × band distance). Band boundaries
are straight box edges, so the pieces re-join without seams.

``max_tol`` (1 µm) stays well below the narrowest gap (typically ≥ 4 µm), so gap
junctions are never removed, while the arcs, with segments of about 1.5 µm, collapse to a
few segments.

A vertex moved by δ at distance d changes the potential at height z by about
δ·ℓ·z/(2π d³) for an edge piece of length ℓ, so tolerances that grow with d keep the
error at the ions roughly uniform. The default's effect is measured in RESULTS.md.
"""

import shapely
from shapely.geometry import box

from .geometry import as_multipolygon

__all__ = ["simplify_far"]


def simplify_far(geom, ion_region, keep=200.0, rel_tol=0.01, max_tol=1.0):
    """Simplify ``geom`` away from ``ion_region`` (both in the same in-plane frame)."""
    geom = as_multipolygon(geom)
    if geom.is_empty:
        return geom
    x0, y0, x1, y1 = ion_region.bounds
    gx0, gy0, gx1, gy1 = geom.bounds
    reach = max(x0 - gx0, y0 - gy0, gx1 - x1, gy1 - y1)
    pieces = []
    inner = box(x0 - keep, y0 - keep, x1 + keep, y1 + keep)
    pieces.append(geom.intersection(inner))
    d = keep
    while d < reach:
        outer = box(x0 - 2 * d, y0 - 2 * d, x1 + 2 * d, y1 + 2 * d)
        band = geom.intersection(outer).difference(inner)
        if not band.is_empty:
            pieces.append(band.simplify(min(max_tol, rel_tol * d), preserve_topology=True))
        inner, d = outer, 2 * d
    rest = geom.difference(inner)
    if not rest.is_empty:
        pieces.append(rest.simplify(min(max_tol, rel_tol * d), preserve_topology=True))
    return as_multipolygon(shapely.make_valid(shapely.union_all(pieces)))

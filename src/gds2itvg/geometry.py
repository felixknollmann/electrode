"""Electrode geometry: named shapely (Multi)Polygons in µm, exported as oriented rings."""

from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient

__all__ = ["Electrode", "polygon_rings", "as_multipolygon"]

_VERTEX_TOL = 1e-6  # µm; duplicate / collinear vertex removal


def as_multipolygon(geom):
    """Return the polygonal part of ``geom`` as a MultiPolygon (drops lines/points)."""
    if geom.is_empty:
        return MultiPolygon()
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    if isinstance(geom, MultiPolygon):
        return geom
    polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, (Polygon, MultiPolygon))]
    out = []
    for g in polys:
        out.extend(as_multipolygon(g).geoms)
    return MultiPolygon(out)


def _clean_ring(coords):
    """Drop the closing vertex, duplicate vertices and vertices on the chord of their neighbours."""
    ring = np.asarray(coords, dtype=np.float64)[:-1]
    changed = True
    while changed and len(ring) > 3:
        prev = np.roll(ring, 1, axis=0)
        nxt = np.roll(ring, -1, axis=0)
        chord = nxt - prev
        clen = np.hypot(chord[:, 0], chord[:, 1])
        dist = np.abs(chord[:, 0] * (ring[:, 1] - prev[:, 1]) - chord[:, 1] * (ring[:, 0] - prev[:, 0]))
        dup = np.hypot(*(ring - prev).T) < _VERTEX_TOL
        # Collinear: vertex within tolerance of the chord and between its neighbours.
        between = np.einsum("ij,ij->i", ring - prev, nxt - ring) > 0
        drop = dup | ((dist <= _VERTEX_TOL * clen) & between)
        # Never drop two neighbours in the same pass.
        drop &= ~np.roll(drop, 1)
        changed = bool(drop.any())
        ring = ring[~drop]
    return ring


def polygon_rings(geom):
    """Oriented rings (exterior CCW, holes CW), each an (M, 2) float64 array without closure."""
    rings = []
    for poly in as_multipolygon(geom).geoms:
        poly = orient(poly, sign=1.0)
        for ring in [poly.exterior, *poly.interiors]:
            r = _clean_ring(ring.coords)
            if len(r) >= 3:
                rings.append(r)
    return rings


@dataclass
class Electrode:
    """A named electrode at unit potential.

    ``geometry`` is its in-plane area (µm, the gapless baseline shape). ``corrections`` are
    the gap-profile strips ``(quad (4, 2), value)``, kept for inspection and plotting.
    ``fields`` are additive terms with ``evaluate(points) -> (N,)``: the merged strips
    (:class:`gds2itvg.fields.EdgeField`) and the finite-chip correction.
    """

    name: str
    geometry: MultiPolygon
    corrections: list = field(default_factory=list)
    fields: list = field(default_factory=list)

    def __post_init__(self):
        self.geometry = as_multipolygon(shapely.make_valid(self.geometry))

    def rings(self):
        return polygon_rings(self.geometry)

    def potential(self, points, kernel):
        """Baseline polygon potential plus all additive fields."""
        v = kernel(points, self.rings())
        for f in self.fields:
            v = v + f.evaluate(points)
        return v

    @property
    def area(self):
        return self.geometry.area

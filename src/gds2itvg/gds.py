"""GDSII import: one layer/datatype → connected conductors (shapely Polygons, µm)."""

import logging

import gdstk
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Point, Polygon, box

from .geometry import as_multipolygon

__all__ = ["read_layer_polygons", "conductors_from_polygons", "load_conductors",
           "select_in_box", "assign_names"]

log = logging.getLogger(__name__)

_DEDUP_DECIMALS = 6  # µm; coordinates are rounded to 1 pm for duplicate detection


def _pick_cell(lib, cell):
    if cell is None:
        tops = lib.top_level()
        if len(tops) != 1:
            raise ValueError(f"GDS has {len(tops)} top-level cells; pass cell=<name>")
        return tops[0]
    for c in lib.cells:
        if c.name == cell:
            return c
    raise KeyError(f"cell {cell!r} not found")


def read_layer_polygons(path, layer, datatype=0, cell=None):
    """All polygons on (layer, datatype) of ``cell``, flattened, in µm.

    Database units are converted by gdstk (``unit=1e-6``), so layouts written with any
    database unit give the same geometry. Paths are converted to polygons honouring width,
    ends and bends, and cell references/arrays are flattened.
    """
    lib = gdstk.read_gds(str(path), unit=1e-6, filter={(layer, datatype)})
    c = _pick_cell(lib, cell)
    polys = c.get_polygons(apply_repetitions=True, include_paths=True,
                           layer=layer, datatype=datatype)
    return [np.array(p.points, dtype=np.float64) for p in polys]


def _dedupe(polygons):
    """Remove geometrically identical polygons (same vertex set up to start/orientation)."""
    seen, out, ndup = set(), [], 0
    for pts in polygons:
        poly = Polygon(np.round(pts, _DEDUP_DECIMALS))
        key = shapely.normalize(poly).wkb
        if key in seen:
            ndup += 1
            continue
        seen.add(key)
        out.append(poly)
    return out, ndup


def conductors_from_polygons(polygons):
    """De-duplicate, repair and union polygons into connected conductors.

    Returns ``(conductors, n_duplicates)``; conductors are sorted by centroid (y, then x)
    for a deterministic order.
    """
    polys, ndup = _dedupe(polygons)
    if ndup:
        log.warning("removed %d duplicate polygons", ndup)
    polys = [shapely.make_valid(p) for p in polys]
    merged = as_multipolygon(shapely.union_all(polys)) if polys else MultiPolygon()
    cond = sorted(merged.geoms, key=lambda g: (round(g.centroid.y, 3), round(g.centroid.x, 3)))
    return cond, ndup


def load_conductors(path, layer, datatype=0, cell=None):
    polys = read_layer_polygons(path, layer, datatype, cell)
    return conductors_from_polygons(polys)


def select_in_box(conductors, roi_box):
    """Indices of conductors intersecting ``roi_box`` = (xmin, ymin, xmax, ymax).

    The ROI selects conductors; it never clips them.
    """
    b = box(*roi_box)
    return [i for i, c in enumerate(conductors) if c.intersects(b)]


def assign_names(conductors, seeds):
    """Map names to conductor indices via seed points.

    ``seeds`` is ``{name: [[x, y], ...]}``; every seed must lie inside exactly one conductor
    and no conductor may be named twice. Several seeds per name merge several conductors
    into one electrode. Returns ``{name: [conductor indices]}``.
    """
    tree = shapely.STRtree(conductors)
    named, used = {}, set()
    for name, pts in seeds.items():
        idx = []
        for xy in pts:
            hits = [int(i) for i in tree.query(Point(xy), predicate="intersects")]
            if len(hits) != 1:
                raise ValueError(f"seed {xy} of electrode {name!r} hits {len(hits)} conductors")
            if hits[0] in used:
                raise ValueError(f"conductor of seed {xy} ({name!r}) is already named")
            used.add(hits[0])
            idx.append(hits[0])
        named[str(name)] = idx
    return named

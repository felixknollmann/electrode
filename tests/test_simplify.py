"""Distance-graded simplification far from the ions."""

import numpy as np
import shapely
from shapely.geometry import Point, box

from gds2itvg.geometry import polygon_rings
from gds2itvg.kernels import get_kernel
from gds2itvg.midline import split_gaps
from gds2itvg.simplify import simplify_far


def test_near_geometry_is_untouched_and_far_arcs_are_reduced():
    ion = box(-10, -100, 10, 100)
    disk_near = Point(0, 150).buffer(20, quad_segs=32)      # inside keep zone
    disk_far = Point(3000, 0).buffer(20, quad_segs=32)      # far away
    g = shapely.union_all([disk_near, disk_far])
    s = simplify_far(g, ion, keep=200.0, rel_tol=0.01)
    near_part = s.intersection(box(-300, -300, 300, 300))
    assert near_part.symmetric_difference(disk_near).area < 1e-6
    far_part = s.intersection(box(2500, -500, 3500, 500))
    assert len(shapely.get_coordinates(far_part)) < len(shapely.get_coordinates(disk_far)) / 2
    assert abs(far_part.area - disk_far.area) < 0.05 * disk_far.area  # tol <= 1 µm on r = 20


def test_simplified_split_layout_potential_unchanged_at_ions():
    # comb of electrodes with 4-µm gaps reaching 5 mm away
    cond = [box(i * 104.0, 30, i * 104.0 + 100, 5000) for i in range(-10, 10)]
    cond += [box(-1100, -26, 1100, 26)]
    grown = split_gaps(cond, max_gap=10)
    ion = box(-10, -10, 10, 10)
    k = get_kernel("numpy")
    pts = np.array([[x, 0.0, z] for x in (-10.0, 0.0, 10.0) for z in (40.0, 60.0)])
    worst, n0, n1 = 0.0, 0, 0
    for g in grown:
        s = simplify_far(g, ion, keep=200.0, rel_tol=0.01)
        n0 += sum(len(r) for r in polygon_rings(g))
        n1 += sum(len(r) for r in polygon_rings(s))
        worst = max(worst, np.abs(k(pts, polygon_rings(s)) - k(pts, polygon_rings(g))).max())
    # harsh case: gap junctions 30 µm off-axis along 1 mm, outside the tiny ion box
    assert n1 < n0 and worst < 3e-5

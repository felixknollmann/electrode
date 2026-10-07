"""Merged gap strips and grid-interpolated correction fields."""

import numpy as np
from shapely.geometry import Polygon, box

from gds2itvg.fields import EdgeField, coarse_axes
from gds2itvg.gaps import GapModel, correction_strips, merge_strip_edges
from gds2itvg.geometry import polygon_rings
from gds2itvg.kernels import get_kernel


def _strips():
    a = box(-800, -300, 800, 0)
    b = box(-800, 4, 800, 300)
    c = box(-800, -310, 800, -304)  # second, 4-µm gap below a
    return correction_strips([a], [b, c], GapModel(strips_per_gap=8), max_gap=30)


def test_merging_preserves_the_potential_and_drops_shared_edges():
    strips = _strips()
    k = get_kernel("numpy")
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-900, 900, 400), rng.uniform(-400, 400, 400),
                           rng.uniform(1, 100, 400)])
    rings = [polygon_rings(Polygon(q))[0] for q, _ in strips]
    direct = k(pts, rings, [v for _, v in strips])
    p1, p2, w = merge_strip_edges(strips)
    merged = EdgeField(p1, p2, w).direct(pts)
    assert np.max(np.abs(merged - direct)) < 1e-12  # vertices are snapped to 1e-9 µm
    assert len(w) < 0.8 * 4 * len(strips)


def test_interpolated_field_matches_direct_evaluation():
    strips = _strips()
    p1, p2, w = merge_strip_edges(strips)
    ion_box, ion_z = (-10.0, 10.0, -150.0, 150.0), (40.0, 60.0)
    f = EdgeField(p1, p2, w, axes=coarse_axes(ion_box, ion_z))
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(-10, 10, 2000), rng.uniform(-150, 150, 2000),
                           rng.uniform(40, 60, 2000)])
    d = f.direct(pts)
    assert np.max(np.abs(f.evaluate(pts) - d)) < 1e-3 * np.max(np.abs(d))
    # points outside the coarse grid fall back to direct evaluation
    out = np.array([[0.0, 500.0, 50.0]])
    assert np.allclose(f.evaluate(out), f.direct(out), rtol=0, atol=0)


def test_coarse_axes_spacing():
    ax, ay, az = coarse_axes((-10, 10, -800, 800), (40, 60))
    assert len(ax) >= 4 and len(az) >= 4
    assert np.max(np.diff(ay)) <= 5.0 + 1e-9 and ay[0] == -800 and ay[-1] == 800

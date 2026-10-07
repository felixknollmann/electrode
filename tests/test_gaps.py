"""Milestone 2: gap-profile correction strips."""

import numpy as np
import pytest
from shapely.geometry import box

from gds2itvg.gaps import GapModel, correction_strips, edge_intervals
from gds2itvg.geometry import polygon_rings
from gds2itvg.kernels import get_kernel
from gds2itvg.midline import split_gaps

K = get_kernel("numpy")


def strips_potential(strips, pts):
    out = np.zeros(len(pts))
    for quad, v in strips:
        out += v * K(pts, polygon_rings(box(*quad.min(0), *quad.max(0))))
    return out


def test_edge_intervals_piecewise_facing_distance():
    other = box(0, 4, 50, 20).union(box(50, 10, 100, 20))
    iv = edge_intervals(np.array([100.0, 0.0]), np.array([0.0, 0.0]), None, None, 30)
    assert iv == [(0.0, 100.0, np.inf, False)]
    # edge along y = 0 from x=100 to 0 has outward normal +y (material below)
    iv = edge_intervals(np.array([100.0, 0.0]), np.array([0.0, 0.0]), other, None, 30)
    d = {round(a): dd for a, b, dd, _ in iv}
    assert d[0] == pytest.approx(10) and d[50] == pytest.approx(4)


def test_long_straight_gap_matches_2d_integral():
    """For a long straight gap the strips' 3-D potential equals the 2-D Poisson integral
    (1/pi) ∫ Δ(u) z / ((x-u)^2 + z^2) du of the profile correction."""
    g, L = 4.0, 40000.0
    left = box(-L / 2, -600, L / 2, 0)        # active electrode, edge along y = 0
    right = box(-L / 2, g, L / 2, 600)
    model = GapModel(strips_per_gap=16)
    strips = correction_strips([left], [right], model, max_gap=30)
    # keep only the gap strips (drop the far edges' opening fringes)
    strips = [(q, v) for q, v in strips if q[:, 1].min() >= -1e-9 and q[:, 1].max() <= g + 1e-9]
    pts = np.array([[0, y, z] for y in (-20.0, 0.0, 2.0, 6.0, 30.0) for z in (20.0, 50.0)])
    got = strips_potential(strips, pts)
    q = model.profile(g)
    u = np.linspace(0, g, 4001)
    delta = q(u) - (u < g / 2)
    want = np.array([np.trapezoid(delta * z / ((y - u) ** 2 + z ** 2), u) / np.pi
                     for _, y, z in pts])
    assert np.max(np.abs(got - want)) < 1e-2 * np.max(np.abs(want))  # 16 constant strips


def test_sag_lowers_potential_and_correction_is_small():
    left, right = box(-500, -300, 500, 0), box(-500, 4, 500, 300)
    model = GapModel()
    grown = split_gaps([left, right], max_gap=30)
    strips = correction_strips([left], [right], model, max_gap=30)
    pts = np.array([[0.0, 2.0, 50.0]])
    base = K(pts, polygon_rings(grown[0]))
    in_gap = [bool(q[:, 1].min() >= -1e-9 and q[:, 1].max() <= 4 + 1e-9) for q, _ in strips]
    gap_part = strips_potential([x for x, f in zip(strips, in_gap) if f], pts)
    fringe = strips_potential([x for x, f in zip(strips, in_gap) if not f], pts)
    assert 0.4 < base[0] < 0.5          # finite 300-µm electrode seen from above its edge
    assert -0.01 < gap_part[0] < 0      # the buried ground makes the gap sag
    assert 0 < fringe[0] < 0.01         # open outer edges fringe over the dielectric


def test_lumping_preserves_far_field():
    left, right = box(-500, -300, 500, 0), box(-500, 4, 500, 300)
    model = GapModel()
    fine = correction_strips([left], [right], model, max_gap=30)
    lumped = correction_strips([left], [right], model, max_gap=30,
                               ion_region=box(1e5, 1e5, 1e5 + 1, 1e5 + 1))
    assert len(lumped) < len(fine)
    pts = np.array([[3000.0, 2000.0, 400.0], [-4000.0, 0.0, 1000.0]])
    a, b = strips_potential(fine, pts), strips_potential(lumped, pts)
    assert np.allclose(a, b, rtol=2e-2, atol=1e-12)


def test_no_ground_vacuum_gap_correction_is_odd():
    """Without a buried ground the gap profile is odd about the midline (Schmied Eq. 6),
    so the net strip value integrates to ~0."""
    left, right = box(-500, -300, 500, 0), box(-500, 4, 500, 300)
    model = GapModel(metal_thickness=1e-3, ground_depth=None, epsilon_r=1.0)
    strips = correction_strips([left], [right], model, max_gap=30)
    total = sum(v * (q[:, 1].max() - q[:, 1].min()) for q, v in strips if np.ptp(q[:, 0]) > 900)
    assert abs(total) < 1e-3 * 1000

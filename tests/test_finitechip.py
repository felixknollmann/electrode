"""Milestone 2: finite-chip surface-potential solver."""

from math import gamma, pi

import numpy as np
import pytest
from shapely.geometry import Point, box

from gds2itvg.finitechip import FiniteChip, FiniteChipSolver, image_kernel, quadtree_panels
from gds2itvg.frame import Frame


def schmied_eq29(z, R1, R2, S, N=60):
    """Correction term of Schmied 2010 Eq. (29): ring electrode on a finite disk (vacuum)."""
    tot = 0.0
    for i in range(N):
        poch = gamma(i + 2) / gamma(i + 1.5)  # (i + 3/2)_{1/2}
        for k in range(N):
            tot += ((-1) ** k * ((R2 / S) ** (2 + 2 * i) - (R1 / S) ** (2 + 2 * i))
                    / ((1.5 + i + k) * pi ** 1.5 * poch) * (z / S) ** (1 + 2 * k))
    return tot


def test_image_kernel_limits():
    assert np.all(image_kernel(np.array([0.0, 10.0]), np.inf) == 0)
    D = 1000.0
    big = image_kernel(np.array([1e6]), D, nmax=200000)
    assert abs(big[0]) < 1e-15
    # direct sum agrees with the default truncation + tail
    r = np.array([0.0, 800.0, 3000.0])
    assert np.allclose(image_kernel(r, D), image_kernel(r, D, nmax=20000), rtol=1e-6)


def test_quadtree_covers_outer_region_without_overlap():
    chip = box(0, 0, 1000, 1000)
    panels, centres = quadtree_panels(chip, 2000, 50, 1.0)
    total = sum(p.area for p in panels)
    assert total == pytest.approx(5000 ** 2 - 1000 ** 2, rel=1e-9)
    assert not any(chip.contains(Point(c)) for c in centres)


def test_finite_disk_matches_schmied_eq29():
    """Disk electrode (R = 300) on a finite thin disk (S = 1000) in vacuum: on-axis
    correction vs Schmied Eq. (29)."""
    S, R2 = 1000.0, 300.0
    chip = Point(0, 0).buffer(S, quad_segs=128)
    disk = Point(0, 0).buffer(R2, quad_segs=128)
    fc = FiniteChip(chip, np.inf, Frame(), (-1, 1, -1, 1), (40, 60), alpha=0.5, h_min=40,
                    margin=128000, under_chip="mirror")
    z = np.array([50.0, 200.0])
    pts = np.column_stack([np.zeros(2), np.zeros(2), z])
    got = fc.correction(disk).evaluate(pts)
    want = np.array([schmied_eq29(zz, 0, R2, S) for zz in z])
    assert np.allclose(got, want, rtol=0.02)


@pytest.fixture(scope="module")
def square_chip():
    chip = box(-5000, -5000, 5000, 5000)
    electrode = box(-5000, -400, 5000, 400)  # strip reaching the chip edge on both sides
    return chip, electrode


def test_outer_ground_depth_trend(square_chip):
    chip, electrode = square_chip
    pts = np.array([[0.0, 0.0, 50.0]])
    vals = []
    for D in (10.0, 1000.0, 3000.0, np.inf):
        fc = FiniteChipSolver(chip, D, Frame(), (-10, 10, -10, 10), (40, 60), h_min=200, alpha=1.0)
        vals.append(fc.correction(electrode)._direct(pts)[0])
    assert 0 < vals[1] < vals[2] < vals[3]       # deeper outer ground -> larger effect
    assert abs(vals[0]) < 0.1 * vals[1]          # ground right below -> grounded plane


def test_correction_is_uniform_field_and_interpolates(square_chip):
    chip, electrode = square_chip
    fc = FiniteChipSolver(chip, 1000.0, Frame(), (-10, 10, -50, 50), (40, 60), h_min=200)
    c = fc.correction(electrode)
    pts = np.array([[0.0, 0.0, 40.0], [0.0, 0.0, 60.0], [3.3, 17.0, 47.5]])
    direct = c._direct(pts)
    assert direct[1] / direct[0] == pytest.approx(60 / 40, rel=0.02)   # ~ a * z
    assert np.allclose(c.evaluate(pts), direct, rtol=1e-6, atol=1e-12)

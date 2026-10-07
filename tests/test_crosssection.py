"""Milestone 2: 2-D gap cross-section solver."""

import numpy as np
import pytest

from gds2itvg.crosssection import gap_profile, opening_profile


def schmied_eq6(u, g):
    """Schmied 2010 Eq. (6) for a thin gap in vacuum, left electrode at 1 V."""
    return 0.5 - np.arcsin((u - g / 2) / (g / 2)) / np.pi


@pytest.mark.parametrize("g", [2.0, 4.0, 10.0])
def test_vacuum_thin_limit_is_schmied_eq6(g):
    p = gap_profile(g, metal_thickness=1e-3, ground_depth=None, epsilon_r=1.0)
    u = np.linspace(0, g, 401)
    err = np.abs(p(u) - schmied_eq6(u, g))
    # sqrt-type singularity at the metal edges: tight away from them, looser next to them
    inner = (u > 0.05 * g) & (u < 0.95 * g)
    assert err[inner].max() < 4e-3 and err.max() < 1.5e-2
    assert np.trapezoid(err, u) / g < 2e-3
    assert p.integral() == pytest.approx(g / 2, abs=1e-6)  # odd about the midline


def test_vacuum_limit_converges_with_grid():
    g = 4.0
    u = np.linspace(0, g, 41)
    err = [np.max(np.abs(gap_profile(g, 1e-3, None, 1.0, fine=f)(u) - schmied_eq6(u, g)))
           for f in (0.1, 0.05, 0.025)]
    assert err[2] < err[1] < err[0]


def test_dielectric_without_ground_stays_antisymmetric():
    # With no ground, both sides of the sheet are linear media and the gap stays odd.
    p = gap_profile(4.0, metal_thickness=1e-3, ground_depth=None, epsilon_r=3.9)
    u = np.linspace(0, 4, 21)
    assert np.allclose(p(u) + p(4 - u), 1.0, atol=5e-3)


@pytest.mark.parametrize("g", [4.0, 6.0, 20.0])
def test_buried_ground_makes_gap_sag(g):
    p = gap_profile(g, metal_thickness=1.0, ground_depth=10.0, epsilon_r=3.9)
    u = np.linspace(0, g, 21)
    s = p(u) + p(g - u)
    assert p(0) == pytest.approx(1) and p(g) == pytest.approx(0, abs=1e-12)
    assert np.all(np.diff(p(u)) <= 1e-12)          # monotonic from 1 V to 0 V
    assert np.all(s[1:-1] < 1) and p.integral() < g / 2


def test_sag_grows_with_gap_and_shrinks_with_deeper_ground():
    sag = lambda g, h: g / 2 - gap_profile(g, 1.0, h, 3.9).integral()
    assert sag(4, 10) < sag(6, 10) < sag(20, 10)
    assert sag(6, 40) < sag(6, 10)


def test_opening_fringe_decays():
    o = opening_profile(1.0, 10.0, 3.9, extent=200)
    assert o(0) == pytest.approx(1)
    assert 0 < o(20) < o(5) < 1
    # algebraic tail q ~ (h / eps) / (pi u) far from the edge
    for u in (50.0, 100.0):
        assert 0.7 < o(u) * u * np.pi * 3.9 / 10.0 < 1.4


def test_strip_means():
    p = gap_profile(4.0)
    m = p.strip_means(np.linspace(0, 4, 5))
    assert len(m) == 4 and np.all(np.diff(m) < 0)
    assert np.mean(m) * 4 == pytest.approx(p.integral(), rel=1e-9)


def test_sag_integral_converged():
    """The integrated sag (what the 3-D correction uses) is grid-converged to < 1 %."""
    sag = [4.0 / 2 - gap_profile(4.0, 1.0, 10.0, 3.9, fine=f).integral() for f in (0.02, 0.01, 0.005)]
    assert abs(sag[1] - sag[2]) < 0.01 * abs(sag[2])

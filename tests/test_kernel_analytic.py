"""Milestone 1 (a): edge kernel vs the closed-form rectangle potential and physics limits."""

import numpy as np
import pytest

from gds2itvg.kernels.numpy_kernel import polygon_potential_dz0


def rect_ring(a1, a2, b1, b2):
    return np.array([[a1, b1], [a2, b1], [a2, b2], [a1, b2]], dtype=float)


def rect_analytic(p, a1, a2, b1, b2):
    """phi = (1/2pi) sum_{i,j} (-1)^{i+j} atan((x-a_i)(y-b_j) / (z R_ij))."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    out = np.zeros(len(p))
    for i, a in enumerate((a1, a2)):
        for j, b in enumerate((b1, b2)):
            dx, dy = x - a, y - b
            r = np.sqrt(dx * dx + dy * dy + z * z)
            out += (-1) ** (i + j) * np.arctan(dx * dy / (z * r))
    return out / (2 * np.pi)


def random_points(rng, n, a1, a2, b1, b2):
    w, h = a2 - a1, b2 - b1
    s = max(w, h)
    pts = []
    # near field, generic
    pts.append(np.column_stack([rng.uniform(a1 - s, a2 + s, n), rng.uniform(b1 - s, b2 + s, n),
                                rng.uniform(1e-3, 0.1, n) * s]))
    # above edges and corners
    for cx, cy in [(a1, (b1 + b2) / 2), (a2, b1), ((a1 + a2) / 2, b2)]:
        pts.append(np.column_stack([cx + rng.normal(0, 1e-3 * s, n // 4),
                                    cy + rng.normal(0, 1e-3 * s, n // 4),
                                    rng.uniform(1e-3, 1, n // 4) * s]))
    # far field
    pts.append(np.column_stack([rng.normal(0, 50 * s, n // 4), rng.normal(0, 50 * s, n // 4),
                                rng.uniform(10, 100, n // 4) * s]))
    return np.concatenate(pts)


@pytest.mark.parametrize("box", [(0, 1, 0, 1), (-120.0, 380.0, 15.0, 40.0), (-5e3, 5e3, -2.5, 2.5)])
def test_rectangle_closed_form(kernel, box):
    rng = np.random.default_rng(1)
    pts = random_points(rng, 4000, *box)
    got = kernel(pts, [rect_ring(*box)])
    want = rect_analytic(pts, *box)
    assert np.max(np.abs(got - want)) < 1e-12


def test_odd_in_z(kernel):
    rng = np.random.default_rng(2)
    pts = random_points(rng, 1000, 0, 3, 0, 1)
    neg = pts * np.array([1, 1, -1])
    assert np.allclose(kernel(neg, [rect_ring(0, 3, 0, 1)]),
                       -kernel(pts, [rect_ring(0, 3, 0, 1)]), atol=1e-14)


def test_surface_limits(kernel):
    ring = [rect_ring(0, 10, 0, 4)]
    z = 1e-7
    p = np.array([[5, 2, z], [20, 2, z], [5, 0, z], [0, 0, z]])  # inside, outside, edge, corner
    v = kernel(p, ring)
    assert v[0] == pytest.approx(1, abs=1e-6)
    assert v[1] == pytest.approx(0, abs=1e-6)
    assert v[2] == pytest.approx(0.5, abs=1e-6)
    assert v[3] == pytest.approx(0.25, abs=1e-6)


def test_far_field_monopole(kernel):
    a = 2.0 * 3.0
    p = np.array([[1e4, -3e3, 7e3]])
    r = np.linalg.norm(p - [1, 1.5, 0])
    want = a * p[0, 2] / (2 * np.pi * r ** 3)
    assert kernel(p, [rect_ring(0, 2, 0, 3)])[0] == pytest.approx(want, rel=1e-6)


def test_tiling_superposition(kernel):
    rng = np.random.default_rng(3)
    pts = random_points(rng, 500, 0, 9, 0, 6)
    whole = kernel(pts, [rect_ring(0, 9, 0, 6)])
    tiles = [rect_ring(i * 3, i * 3 + 3, j * 2, j * 2 + 2) for i in range(3) for j in range(3)]
    assert np.max(np.abs(kernel(pts, tiles) - whole)) < 1e-12


def test_rotation_invariance(kernel):
    rng = np.random.default_rng(4)
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    ring = rect_ring(-1, 3, 0.5, 2)
    pts = random_points(rng, 500, -1, 3, 0.5, 2)
    prot = pts.copy()
    prot[:, :2] = pts[:, :2] @ R.T
    assert np.max(np.abs(kernel(prot, [ring @ R.T]) - kernel(pts, [ring]))) < 1e-12


def test_hole_is_outer_minus_inner(kernel):
    rng = np.random.default_rng(5)
    pts = random_points(rng, 500, 0, 10, 0, 10)
    outer = rect_ring(0, 10, 0, 10)
    inner = rect_ring(3, 6, 2, 5)[::-1]  # clockwise hole
    got = kernel(pts, [outer, inner])
    want = rect_analytic(pts, 0, 10, 0, 10) - rect_analytic(pts, 3, 6, 2, 5)
    assert np.max(np.abs(got - want)) < 1e-12


def test_backends_agree():
    from gds2itvg.kernels import get_kernel
    pytest.importorskip("numba")
    rng = np.random.default_rng(6)
    pts = random_points(rng, 2000, 0, 7, 0, 3)
    ring = [np.array([[0, 0], [7, 0], [7, 3], [4, 1.5], [0, 3]], dtype=float)]
    a = get_kernel("numpy")(pts, ring)
    b = get_kernel("numba")(pts, ring)
    assert np.max(np.abs(a - b)) < 1e-14


def test_dz0_matches_finite_difference():
    from gds2itvg.kernels.numpy_kernel import polygon_potential
    ring = [np.array([[0, 0], [7, 0], [7, 3], [4, 1.5], [0, 3]], dtype=float)]
    xy = np.array([[8.0, 1.0], [-2.0, 5.0], [4.0, 2.5], [3.5, -0.4]])  # off the polygon
    h = 1e-5
    fd = polygon_potential(np.column_stack([xy, np.full(4, h)]), ring) / h
    assert np.allclose(polygon_potential_dz0(xy, ring), fd, rtol=1e-4, atol=1e-9)

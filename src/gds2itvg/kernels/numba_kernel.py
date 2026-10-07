"""Parallel Numba version of :mod:`gds2itvg.kernels.numpy_kernel` (same formula)."""

import math

import numba
import numpy as np

from .numpy_kernel import _edges

__all__ = ["polygon_potential", "edge_potential", "edge_matrix"]


@numba.njit(parallel=True, fastmath=False, cache=True)
def _potential(points, p1, p2, w, out):
    n = points.shape[0]
    ne = p1.shape[0]
    for i in numba.prange(n):
        x = points[i, 0]
        y = points[i, 1]
        z = points[i, 2]
        zs = abs(z)
        acc = 0.0
        for e in range(ne):
            x1 = x - p1[e, 0]
            y1 = y - p1[e, 1]
            x2 = x - p2[e, 0]
            y2 = y - p2[e, 1]
            r1 = math.sqrt(x1 * x1 + y1 * y1 + z * z)
            r2 = math.sqrt(x2 * x2 + y2 * y2 + z * z)
            num = z * (x1 * y2 - y1 * x2)
            den = zs * (r1 * r2 + x1 * x2 + y1 * y2 + zs * (zs + r1 + r2))
            acc += w[e] * math.atan2(num, den)
        out[i] = acc / math.pi


@numba.njit(parallel=True, fastmath=False, cache=True)
def _matrix(points, p1, p2, col, out):
    n = points.shape[0]
    ne = p1.shape[0]
    for i in numba.prange(n):
        x = points[i, 0]
        y = points[i, 1]
        z = points[i, 2]
        zs = abs(z)
        for e in range(ne):
            x1 = x - p1[e, 0]
            y1 = y - p1[e, 1]
            x2 = x - p2[e, 0]
            y2 = y - p2[e, 1]
            r1 = math.sqrt(x1 * x1 + y1 * y1 + z * z)
            r2 = math.sqrt(x2 * x2 + y2 * y2 + z * z)
            num = z * (x1 * y2 - y1 * x2)
            den = zs * (r1 * r2 + x1 * x2 + y1 * y2 + zs * (zs + r1 + r2))
            out[i, col[e]] += math.atan2(num, den) / math.pi


def edge_matrix(points, p1, p2, col, ncols):
    """Response matrix M (N, ncols): M[i, c] = potential at point i of the edges with
    ``col[e] == c`` at unit weight, so that edge_potential(points, p1, p2, v[col]) == M @ v."""
    points = np.ascontiguousarray(points, dtype=np.float64)
    out = np.zeros((points.shape[0], int(ncols)))
    if len(p1):
        _matrix(points, np.ascontiguousarray(p1, dtype=np.float64),
                np.ascontiguousarray(p2, dtype=np.float64),
                np.ascontiguousarray(col, dtype=np.int64), out)
    return out


def polygon_potential(points, rings, weights=None):
    """Potential at ``points`` (N, 3) of the rings held at 1 V (or ``weights[k]`` V)."""
    p1, p2, w = _edges(rings, weights)
    return edge_potential(points, p1, p2, w)


def edge_potential(points, p1, p2, w):
    """Weighted sum over directed edges (see :func:`numpy_kernel.edge_potential`)."""
    points = np.ascontiguousarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    out = np.zeros(points.shape[0])
    if len(p1):
        _potential(points, np.ascontiguousarray(p1, dtype=np.float64),
                   np.ascontiguousarray(p2, dtype=np.float64),
                   np.ascontiguousarray(w, dtype=np.float64), out)
    return out

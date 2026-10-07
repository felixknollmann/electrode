"""Vectorised NumPy evaluation of the gapless-plane polygon potential.

For a polygon ring at unit potential in the grounded plane z = 0 the potential is the
solid angle it subtends divided by 2*pi (Wesenberg 2008, Eq. 4). Summed over directed edges
(p1 -> p2) seen from r = (x, y, z) this is

    phi = (1/pi) * sum atan( z (x1 y2 - y1 x2)
                             / (|z| (r1 r2 + x1 x2 + y1 y2 + |z| (|z| + r1 + r2))) )

with x_i = x - p_i,x, y_i = y - p_i,y, r_i = |r - p_i|. Exterior rings are counter-clockwise,
holes clockwise, so holes subtract automatically.

The denominator is non-negative (r1 r2 >= |d1||d2| >= -d1.d2), so arctan2(num, den) equals
atan(num / den) and stays well defined when den underflows.
"""

import numpy as np

__all__ = ["polygon_potential", "edge_potential", "polygon_potential_dz0"]

_CHUNK = 1 << 16  # points per block; keeps (points x edges) temporaries bounded


def _edges(rings, weights=None):
    """Concatenate rings into start/end vertex arrays (E, 2) and per-edge weights (E,)."""
    p1, p2, w = [], [], []
    if weights is None:
        weights = np.ones(len(rings))
    for ring, wt in zip(rings, weights):
        ring = np.asarray(ring, dtype=np.float64)
        if ring.ndim != 2 or ring.shape[1] != 2 or ring.shape[0] < 3:
            raise ValueError("each ring must be an (M>=3, 2) array")
        p1.append(ring)
        p2.append(np.roll(ring, -1, axis=0))
        w.append(np.full(len(ring), float(wt)))
    if not p1:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    return np.concatenate(p1), np.concatenate(p2), np.concatenate(w)


def polygon_potential(points, rings, weights=None):
    """Potential at ``points`` (N, 3) of the rings held at 1 V (or at ``weights[k]`` V per
    ring). Returns shape (N,)."""
    p1, p2, w = _edges(rings, weights)
    return edge_potential(points, p1, p2, w)


def edge_potential(points, p1, p2, w):
    """Weighted sum over directed edges ``p1[e] -> p2[e]`` with weights ``w[e]``.

    Closed rings with a common weight give that ring's potential; arbitrary weighted edge
    sets (e.g. merged correction strips, see :mod:`gds2itvg.gaps`) are also allowed.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    p1, p2, w = (np.asarray(a, dtype=np.float64) for a in (p1, p2, w))
    out = np.zeros(points.shape[0])
    if p1.shape[0] == 0:
        return out
    # Edge blocks bound memory for polygons with many vertices.
    eblock = max(1, (1 << 22) // max(1, min(_CHUNK, points.shape[0])))
    for s in range(0, points.shape[0], _CHUNK):
        pt = points[s:s + _CHUNK]
        x, y, z = pt[:, 0:1], pt[:, 1:2], pt[:, 2:3]
        zs = np.abs(z)
        acc = np.zeros(pt.shape[0])
        for e in range(0, p1.shape[0], eblock):
            a, b, wb = p1[e:e + eblock], p2[e:e + eblock], w[e:e + eblock]
            x1 = x - a[:, 0]
            y1 = y - a[:, 1]
            x2 = x - b[:, 0]
            y2 = y - b[:, 1]
            r1 = np.sqrt(x1 * x1 + y1 * y1 + z * z)
            r2 = np.sqrt(x2 * x2 + y2 * y2 + z * z)
            num = z * (x1 * y2 - y1 * x2)
            den = zs * (r1 * r2 + x1 * x2 + y1 * y2 + zs * (zs + r1 + r2))
            acc += np.arctan2(num, den) @ wb
        out[s:s + _CHUNK] = acc / np.pi
    return out


def polygon_potential_dz0(points, rings, weights=None):
    """d(phi)/dz at z -> 0+ for in-plane ``points`` (N, 2) lying off the rings' area.

    Differentiating the edge formula at z = 0+ (with D = r1 r2 + d1.d2, c = d1 x d2, and the
    identity D^2 + c^2 = 2 r1 r2 D) gives per edge

        d/dz atan(c / D(z)) |_{z=0+} = -c (r1 + r2) / (2 r1 r2 D),

    finite for points off the edges. Used for the gap charge-density collocation (M2).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    p1, p2, w = _edges(rings, weights)
    out = np.zeros(pts.shape[0])
    if p1.shape[0] == 0:
        return out
    for s in range(0, pts.shape[0], _CHUNK):
        x, y = pts[s:s + _CHUNK, 0:1], pts[s:s + _CHUNK, 1:2]
        x1 = x - p1[:, 0]
        y1 = y - p1[:, 1]
        x2 = x - p2[:, 0]
        y2 = y - p2[:, 1]
        r1 = np.hypot(x1, y1)
        r2 = np.hypot(x2, y2)
        c = x1 * y2 - y1 * x2
        d = r1 * r2 + x1 * x2 + y1 * y2
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(d > 0, c * (r1 + r2) / (r1 * r2 * d), 0.0)
        out[s:s + _CHUNK] = -(term @ w) / (2 * np.pi)
    return out

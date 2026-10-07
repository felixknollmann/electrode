"""Additive correction fields evaluated on a coarse grid and interpolated.

The gap-correction strips of one electrode can number in the thousands, but at ion heights
z ≥ z_min their combined potential varies only on length scales ≳ z_min. It is therefore
evaluated exactly (edge-sum kernel) on a coarse grid covering the ion ROI, with spacing
z_min / 8, and interpolated tricubically. Points outside that grid are evaluated directly.
The interpolation error is tested in tests/test_fields.py and reported in RESULTS.md.
"""

import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator

from .kernels import get_edge_kernel

__all__ = ["coarse_axes", "EdgeField", "tensor_cubic"]


def coarse_axes(ion_box, ion_z, spacing=None):
    """Axes (x, y, z) covering ``ion_box`` = (xmin, xmax, ymin, ymax) and ``ion_z``.

    Each axis has at least 4 points (cubic interpolation) and a spacing of at most
    ``spacing`` (default z_min / 8, clipped to [0.25, 10] µm).
    """
    if spacing is None:
        spacing = float(np.clip(ion_z[0] / 8.0, 0.25, 10.0))
    axes = []
    for lo, hi in ((ion_box[0], ion_box[1]), (ion_box[2], ion_box[3]), ion_z):
        n = max(4, int(np.ceil((hi - lo) / spacing - 1e-9)) + 1)
        axes.append(np.linspace(lo, hi, n))
    return tuple(axes)


class EdgeField:
    """Potential of a weighted edge set (ITVG frame), optionally grid-interpolated."""

    def __init__(self, p1, p2, w, axes=None, backend="auto"):
        self.p1, self.p2, self.w = (np.asarray(a, dtype=np.float64) for a in (p1, p2, w))
        self.axes = axes
        self.kernel = get_edge_kernel(backend)
        self._interp = None

    @property
    def n_edges(self):
        return len(self.w)

    def direct(self, points):
        return self.kernel(points, self.p1, self.p2, self.w)

    def evaluate(self, points):
        pts = np.asarray(points, dtype=np.float64)
        if self.axes is None or self.n_edges == 0:
            return self.direct(pts)
        if self._interp is None:
            ax, ay, az = self.axes
            zz, yy, xx = np.meshgrid(az, ay, ax, indexing="ij")
            vals = self.direct(np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]))
            vals = vals.reshape(len(az), len(ay), len(ax)).transpose(2, 1, 0)
            self._interp = RegularGridInterpolator(self.axes, vals, method="cubic")
        lo = np.array([a[0] for a in self.axes])
        hi = np.array([a[-1] for a in self.axes])
        inside = np.all((pts >= lo - 1e-9) & (pts <= hi + 1e-9), axis=1)
        out = np.empty(len(pts))
        if inside.any():
            out[inside] = self._interp(np.clip(pts[inside], lo, hi))
        if (~inside).any():
            out[~inside] = self.direct(pts[~inside])
        return out


def tensor_cubic(values, coarse, fine):
    """Interpolate ``values`` (nz, ny, nx) on ``coarse`` axes (x, y, z) to the regular grid
    ``fine`` (x, y, z) with separable cubic splines, one axis at a time.

    Returns (nz', ny', nx'). This is much faster than point-wise interpolation when the
    output is a full grid (always the case for ITVG files).
    """
    v = np.asarray(values, dtype=float)
    for axis, (c, f) in ((2, (coarse[0], fine[0])), (1, (coarse[1], fine[1])),
                         (0, (coarse[2], fine[2]))):
        if len(c) == len(f) and np.allclose(c, f):
            continue
        if len(c) < 2:
            raise ValueError("coarse axis needs at least 2 points")
        v = CubicSpline(c, v, axis=axis)(np.clip(f, c[0], c[-1]))
    return v

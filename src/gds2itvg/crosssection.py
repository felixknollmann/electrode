"""2-D cross-section of an inter-electrode gap → in-plane surface potential profile.

Geometry (x lateral, z vertical, top surface at z = 0):

- left electrode: metal for x <= -g/2, -t <= z <= 0, at 1 V;
- right electrode: metal for x >= +g/2, -t <= z <= 0, at 0 V (absent for an *opening*);
- dielectric eps_r for z < 0 outside the metal (fills the gap to the top surface);
- buried ground at z = -h (``ground_depth``), or none (``None``: the region below extends
  far down and its far boundary carries the gapless solution);
- vacuum above z = 0.

Far boundaries carry the gapless (midline-step) half-plane solution. Above the surface this
is phi = 1/2 - atan((x - x_step)/z)/pi. Below, it is a parallel-plate profile under the
metal with a buried ground, or the mirror image without one. The domain is many gap widths
wide, so the profile near the gap does not depend on these values.

The equation div(eps grad phi) = 0 is solved by node-centred finite volumes on a graded
tensor grid, with grid lines on every material boundary.

Returns q(u), the surface potential at z = 0 across the gap, u = x + g/2 in [0, g].
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["GapProfile", "gap_profile", "opening_profile"]


def _graded(a, b, h0, ratio, hmax):
    """Points from a to b (b > a) starting with spacing h0 at a, growing by ``ratio``."""
    pts, x, h = [a], a, h0
    while x + h < b - 1e-12:
        x += h
        pts.append(x)
        h = min(h * ratio, hmax)
    pts.append(b)
    return np.array(pts)


def _axis(breaks, fine, ratio, hmax, extent_lo, extent_hi):
    """Graded axis with nodes on all ``breaks``; spacing ``fine`` at breaks."""
    breaks = sorted(set(breaks))
    segs = [np.array([breaks[0]])]
    for a, b in zip(breaks[:-1], breaks[1:]):
        n = max(2, int(np.ceil((b - a) / fine)))
        if (b - a) <= 40 * fine:
            seg = np.linspace(a, b, n + 1)
        else:
            hcore = min(hmax, (b - a) / 60)  # resolve the interior of gaps and layers
            left = _graded(a, (a + b) / 2, fine, ratio, hcore)
            right = (a + b) - _graded(a, (a + b) / 2, fine, ratio, hcore)[::-1]
            seg = np.concatenate([left, right[1:]])
        segs.append(seg[1:])
    core = np.concatenate(segs)
    lo = breaks[0] - _graded(0, breaks[0] - extent_lo, fine, ratio, hmax)[::-1]
    hi = breaks[-1] + _graded(0, extent_hi - breaks[-1], fine, ratio, hmax)
    return np.unique(np.concatenate([lo, core, hi]))


@dataclass(frozen=True)
class GapProfile:
    """Surface potential across a gap (left electrode 1 V, right 0 V)."""

    g: float            # gap width (µm); np.inf for an opening
    u: np.ndarray       # sample positions from the left metal edge (µm)
    q: np.ndarray       # surface potential at u

    def __call__(self, u):
        return np.interp(u, self.u, self.q)

    def strip_means(self, edges):
        """Mean of q over consecutive intervals [edges[k], edges[k+1]] (trapezoidal)."""
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            uu = np.unique(np.concatenate([[a, b], self.u[(self.u > a) & (self.u < b)]]))
            out.append(np.trapezoid(self(uu), uu) / (b - a))
        return np.array(out)

    def integral(self):
        """Integral of q over the gap (µm·V); g/2 for the gapless midline step."""
        return float(np.trapezoid(self.q, self.u))


def _solve(g, t, h, eps_r, opening, fine=None, span=None):
    fine = fine or min(g if np.isfinite(g) else 4.0, max(t, 1.0)) / 100.0
    gg = g if np.isfinite(g) else 0.0
    span = span or max(400.0, 40 * max(gg, h or 0.0, 10.0))
    xl, xr = -gg / 2, gg / 2
    xbreaks = [xl, xr] if not opening else [xl]
    zbreaks = [0.0, -t] + ([-h] if h is not None else [])
    zlo = -h if h is not None else -span
    X = _axis(xbreaks, fine, 1.15, span / 20, -span, span)
    Z = _axis([b for b in zbreaks if b >= zlo], fine, 1.15, span / 20, zlo, span)
    Z = Z[Z >= zlo]
    nx, nz = len(X), len(Z)
    XX, ZZ = np.meshgrid(X, Z, indexing="ij")

    # Cell permittivity (cells between grid lines); metal cells get eps 1 (not used).
    xc = 0.5 * (X[:-1] + X[1:])
    zc = 0.5 * (Z[:-1] + Z[1:])
    XC, ZC = np.meshgrid(xc, zc, indexing="ij")
    eps = np.where(ZC < 0, eps_r, 1.0)

    # Dirichlet nodes and values.
    fixed = np.zeros((nx, nz), dtype=bool)
    val = np.zeros((nx, nz))
    tol = 1e-9
    in_metal_z = (ZZ <= tol) & (ZZ >= -t - tol)
    left = (XX <= xl + tol) & in_metal_z
    fixed |= left
    val[left] = 1.0
    if not opening:
        right = (XX >= xr - tol) & in_metal_z
        fixed |= right
        val[right] = 0.0
    step_x = 0.0 if not opening else xl

    def far(x, z):
        above = 0.5 - np.arctan2(x - step_x, np.abs(z)) / np.pi
        if h is None:
            return above  # mirror-symmetric without a buried ground
        under = np.where(x - step_x < 0, (z + h) / (h - t), 0.0)
        return np.where(z >= 0, above, np.clip(under, 0, 1))

    bnd = np.zeros((nx, nz), dtype=bool)
    bnd[0, :] = bnd[-1, :] = True
    bnd[:, -1] = True
    if h is None:
        bnd[:, 0] = True
    newb = bnd & ~fixed
    fixed |= newb
    val[newb] = far(XX[newb], ZZ[newb])
    if h is not None:
        g0 = (np.abs(ZZ + h) < tol) & ~fixed
        fixed |= g0
        val[g0] = 0.0

    # Finite-volume assembly: neighbour couplings from the half-cells around each face.
    idx = -np.ones((nx, nz), dtype=np.int64)
    free = ~fixed
    idx[free] = np.arange(free.sum())
    dx = np.diff(X)
    dz = np.diff(Z)
    rows, cols, data = [], [], []
    rhs = np.zeros(free.sum())
    I, J = np.nonzero(free)
    diag = np.zeros(len(I))

    def cell_eps(ci, cj):
        ok = (ci >= 0) & (ci < nx - 1) & (cj >= 0) & (cj < nz - 1)
        out = np.zeros(ci.shape)
        out[ok] = eps[ci[ok], cj[ok]]
        return out

    def half(arr, k, n):
        ok = (k >= 0) & (k < n)
        out = np.zeros(k.shape)
        out[ok] = arr[k[ok]] / 2
        return out

    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = I + di, J + dj
        if di:
            ci = I if di > 0 else I - 1
            dist = dx[np.clip(ci, 0, nx - 2)]
            coef = (cell_eps(ci, J) * half(dz, J, nz - 1) + cell_eps(ci, J - 1) * half(dz, J - 1, nz - 1)) / dist
        else:
            cj = J if dj > 0 else J - 1
            dist = dz[np.clip(cj, 0, nz - 2)]
            coef = (cell_eps(I, cj) * half(dx, I, nx - 1) + cell_eps(I - 1, cj) * half(dx, I - 1, nx - 1)) / dist
        valid = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < nz) & (coef > 0)
        diag[valid] += coef[valid]
        nb_free = valid & free[np.clip(ni, 0, nx - 1), np.clip(nj, 0, nz - 1)]
        nb_fix = valid & ~nb_free
        rows.append(np.flatnonzero(nb_free))
        cols.append(idx[ni[nb_free], nj[nb_free]])
        data.append(-coef[nb_free])
        np.add.at(rhs, np.flatnonzero(nb_fix), coef[nb_fix] * val[ni[nb_fix], nj[nb_fix]])
    n = len(I)
    A = sp.csr_matrix((np.concatenate(data + [diag]),
                       (np.concatenate(rows + [np.arange(n)]), np.concatenate(cols + [np.arange(n)]))),
                      shape=(n, n))
    sol = val.copy()
    sol[free] = spla.spsolve(A.tocsc(), rhs)
    j0 = int(np.argmin(np.abs(Z)))
    return X, sol[:, j0]


@lru_cache(maxsize=64)
def _cached(g, t, h, eps_r, opening, fine):
    return _solve(g, t, h, eps_r, opening, fine)


def gap_profile(g, metal_thickness=1.0, ground_depth=10.0, epsilon_r=3.9, fine=None):
    """Surface potential across a gap of width ``g`` (µm); left electrode 1 V, right 0 V."""
    t = max(float(metal_thickness), 1e-6)
    X, s = _cached(float(g), t, None if ground_depth is None else float(ground_depth),
                   float(epsilon_r), False, fine)
    m = (X >= -g / 2 - 1e-12) & (X <= g / 2 + 1e-12)
    return GapProfile(g=float(g), u=X[m] + g / 2, q=s[m])


def opening_profile(metal_thickness=1.0, ground_depth=10.0, epsilon_r=3.9, extent=None,
                    fine=None):
    """Surface potential beyond a single electrode edge facing an opening (no metal).

    Over a dielectric with a buried ground the fringe decays only algebraically,
    q(u) ~ (h / eps_r) / (pi u), because the vacuum field of the metal keeps charging the
    surface. ``extent`` truncates it (callers use 10 x ground depth).
    """
    t = max(float(metal_thickness), 1e-6)
    X, s = _cached(np.inf, t, None if ground_depth is None else float(ground_depth),
                   float(epsilon_r), True, fine)
    m = X >= 0
    u, q = X[m], s[m]
    if extent is not None:
        keep = u <= extent
        u, q = u[keep], q[keep]
    return GapProfile(g=np.inf, u=u, q=q)

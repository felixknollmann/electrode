"""Finite chip: the surface potential outside the chip outline (GUIDELINES §5.3).

Inside ``chip_outline`` the plane z = 0 is metal, or dielectric over the buried ground,
which is effectively 0 V from below. Outside it the plane is empty: vacuum above, and
vacuum below down to an outer ground at depth D. The unknown surface potential ψ there
must carry no surface charge, so the normal field is continuous through z = 0:

    A[f_up] = -(A + S_D)[f_low]    on the outer surface,

with:

- f_up = ψ + 1_e, the active electrode at 1 V seen from above;
- f_low = ψ (+ 1_e for ``under_chip="mirror"``, a thin sheet as in Schmied §III; 0 under
  the chip for the device's buried ground);
- A the half-space normal-derivative operator, i.e. ``polygon_potential_dz0``;
- S_D the smooth part of the slab image series,
  K(ρ) = (1/π) Σ_{n≥1} (ρ² − 8n²D²)/(ρ² + 4n²D²)^{5/2} (zero for D = ∞).

ψ is piecewise constant on a graded quadtree mesh of the outer surface, finest at the chip
edge and truncated at ``margin`` µm. One LU factorisation serves all electrodes.

The correction above the plane is the Poisson integral of ψ. It is smooth over the ion
ROI, which is millimetres from the edge, so it is evaluated on a coarse grid and
interpolated cubically. Points outside that grid are evaluated directly.
"""

import numpy as np
import scipy.linalg as sla
import shapely
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Polygon, box

from .geometry import as_multipolygon, polygon_rings
from .kernels import get_kernel
from .kernels.numpy_kernel import polygon_potential_dz0

__all__ = ["FiniteChip", "FiniteChipSolver", "quadtree_panels", "image_kernel"]


def image_kernel(rho, D, nmax=None):
    """Smooth slab image part of d/ds of the Poisson kernel at s = 0 (see module doc).

    Terms behave like -1/(4 n^3 D^3) once 2nD >> rho; the sum runs to ``nmax`` (default
    from the largest rho) and adds that tail analytically.
    """
    rho = np.asarray(rho, dtype=float)
    if not np.isfinite(D):
        return np.zeros_like(rho)
    if nmax is None:
        nmax = max(32, int(8 * rho.max() / (2 * D)) + 1) if rho.size else 32
    r2 = rho * rho
    s = np.zeros_like(rho)
    for n in range(1, nmax + 1):
        a2 = 4.0 * n * n * D * D
        s += (r2 - 2.0 * a2) / (r2 + a2) ** 2.5
    s += -1.0 / (4.0 * D ** 3) * (1.0 / (2.0 * nmax ** 2) - 1.0 / (2.0 * nmax ** 3))
    return s / np.pi


def quadtree_panels(chip, margin, h_min, alpha=1.0):
    """Panels covering (bbox(chip) grown by ``margin``) minus ``chip``.

    A cell is split while its size exceeds max(h_min, alpha * distance to the chip edge).
    Returns ``(panels, centres)``: shapely polygons and collocation points inside them.
    """
    x0, y0, x1, y1 = chip.bounds
    root = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    size = max(root[2] - root[0], root[3] - root[1])
    stack = [(root[0], root[1], size)]
    edge = chip.boundary
    panels = []
    while stack:
        cx, cy, s = stack.pop()
        cell = box(cx, cy, cx + s, cy + s)
        if chip.contains(cell):
            continue
        d = max(0.0, edge.distance(cell))
        if s > max(h_min, alpha * d) and s > h_min * 1.0001:
            h = s / 2
            stack += [(cx, cy, h), (cx + h, cy, h), (cx, cy + h, h), (cx + h, cy + h, h)]
            continue
        part = cell if not cell.intersects(chip) else cell.difference(chip)
        for g in as_multipolygon(part).geoms:
            if g.area > 1e-6 * s * s:
                panels.append(g)
    centres = np.array([(p.centroid if p.contains(p.centroid) else p.representative_point()).coords[0]
                        for p in panels])
    return panels, centres


class _Smooth:
    """Additive correction phi(points) = Poisson integral of ψ (ITVG-frame points)."""

    def __init__(self, chip_model, psi):
        self.m = chip_model
        self.psi = psi
        self._interp = None
        self.axes = chip_model.coarse_axes  # varies on mm scales: a coarse grid suffices

    def direct(self, pts_itvg):
        return self._direct(np.asarray(pts_itvg, dtype=float))

    def _direct(self, pts_itvg):
        return self.m.response(pts_itvg) @ self.psi

    def evaluate(self, points):
        pts = np.asarray(points, dtype=float)
        if self._interp is None:
            ax = self.m.coarse_axes
            zz, yy, xx = np.meshgrid(ax[2], ax[1], ax[0], indexing="ij")
            vals = self._direct(np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]))
            self._interp = RegularGridInterpolator(
                (ax[0], ax[1], ax[2]), vals.reshape(len(ax[2]), len(ax[1]), len(ax[0])).transpose(2, 1, 0),
                method="cubic")
        lo = np.array([a[0] for a in self.m.coarse_axes])
        hi = np.array([a[-1] for a in self.m.coarse_axes])
        inside = np.all((pts >= lo - 1e-9) & (pts <= hi + 1e-9), axis=1)
        out = np.empty(len(pts))
        if inside.any():
            out[inside] = self._interp(np.clip(pts[inside], lo, hi))
        if (~inside).any():
            out[~inside] = self._direct(pts[~inside])
        return out


class _Richardson:
    """2 c(alpha/2) - c(alpha): removes the leading O(alpha) mesh error (GUIDELINES §5.3)."""

    def __init__(self, fine, coarse):
        self.fine, self.coarse = fine, coarse
        self.axes = fine.axes

    def evaluate(self, points):
        return 2.0 * self.fine.evaluate(points) - self.coarse.evaluate(points)

    def direct(self, points):
        return 2.0 * self.fine.direct(points) - self.coarse.direct(points)


class FiniteChip:
    """Finite-chip correction with Richardson extrapolation over two quadtree meshes
    (``alpha`` and ``alpha / 2``). Arguments as for :class:`FiniteChipSolver`."""

    def __init__(self, chip, outer_ground_depth, frame, ion_box, ion_z, alpha=1.0,
                 extrapolate=True, **kw):
        self.fine = FiniteChipSolver(chip, outer_ground_depth, frame, ion_box, ion_z,
                                     alpha=alpha / 2 if extrapolate else alpha, **kw)
        self.coarse = (FiniteChipSolver(chip, outer_ground_depth, frame, ion_box, ion_z,
                                        alpha=alpha, **kw) if extrapolate else None)

    def correction(self, geom):
        if self.coarse is None:
            return self.fine.correction(geom)
        return _Richardson(self.fine.correction(geom), self.coarse.correction(geom))


class FiniteChipSolver:
    """Solver for the surface potential outside the chip on one quadtree mesh.

    ``chip`` and electrode geometries are in GDS µm; ``frame`` maps GDS → ITVG.
    ``ion_box`` = (xmin, xmax, ymin, ymax) and ``ion_z`` = (zmin, zmax) in ITVG µm define
    the interpolation grid.
    """

    def __init__(self, chip, outer_ground_depth, frame, ion_box, ion_z, h_min=100.0,
                 alpha=1.0, margin=None, under_chip="ground", coarse=(5, 41, 5)):
        self.chip = chip
        self.D = float(outer_ground_depth) if outer_ground_depth is not None else np.inf
        self.frame = frame
        if under_chip not in ("ground", "mirror"):
            raise ValueError("under_chip must be 'ground' or 'mirror'")
        if under_chip == "mirror" and np.isfinite(self.D):
            raise ValueError("under_chip='mirror' (thin sheet in vacuum) needs outer_ground_depth=inf")
        self.under_chip = under_chip
        size = max(chip.bounds[2] - chip.bounds[0], chip.bounds[3] - chip.bounds[1])
        if margin is None:
            margin = 10 * self.D if np.isfinite(self.D) else 10 * size
        self.panels, self.centres = quadtree_panels(chip, margin, h_min, alpha)
        self.kernel = get_kernel("auto")
        # rings of all panels, with the ring -> panel index for weights
        self.rings, self.owner = [], []
        for j, p in enumerate(self.panels):
            for r in polygon_rings(p):
                self.rings.append(r)
                self.owner.append(j)
        self.owner = np.array(self.owner)
        # Edge arrays built once; each correction only changes the per-edge weights.
        from .kernels import get_edge_kernel
        from .kernels.numpy_kernel import _edges
        self.p1, self.p2, ring_of_edge = _edges(self.rings, np.arange(len(self.rings)))
        self.edge_owner = self.owner[ring_of_edge.astype(int)]
        self.edge_kernel = get_edge_kernel("auto")
        self._responses = {}
        self.coarse_axes = (np.linspace(ion_box[0], ion_box[1], coarse[0]),
                            np.linspace(ion_box[2], ion_box[3], coarse[1]),
                            np.linspace(ion_z[0], ion_z[1], coarse[2]))
        self._lu = None

    def response(self, pts_itvg):
        """Matrix (N, n_panels): potential at ITVG points of each panel at 1 V. Cached per
        point set, so all electrodes share one evaluation (their corrections are M @ ψ)."""
        pts = np.ascontiguousarray(pts_itvg, dtype=float)
        key = (pts.shape, hash(pts.tobytes()))
        if key not in self._responses:
            g = pts.copy()
            g[:, :2] = self.frame.to_gds(pts[:, :2])
            try:
                from .kernels.numba_kernel import edge_matrix
                M = edge_matrix(g, self.p1, self.p2, self.edge_owner, len(self.panels))
            except ImportError:  # pragma: no cover - NumPy fallback, one panel at a time
                M = np.column_stack([self.kernel(g, polygon_rings(p)) for p in self.panels])
            if len(self._responses) > 8:
                self._responses.clear()
            self._responses[key] = M
        return self._responses[key]

    def ring_weights(self, psi):
        return np.asarray(psi)[self.owner]

    def _matrix(self):
        n = len(self.panels)
        M = np.empty((n, n))
        for j, p in enumerate(self.panels):
            M[:, j] = 2.0 * polygon_potential_dz0(self.centres, polygon_rings(p))
        if np.isfinite(self.D):
            areas = np.array([p.area for p in self.panels])
            diff = self.centres[:, None, :] - self.centres[None, :, :]
            rho = np.hypot(diff[..., 0], diff[..., 1])
            del diff
            # K depends on rho only and is smooth on the scale D: tabulate and interpolate.
            grid = np.concatenate([np.linspace(0, 4 * self.D, 4001)[:-1],
                                   np.geomspace(4 * self.D, max(rho.max(), 4 * self.D) * 1.01, 2000)])
            table = image_kernel(grid, self.D)
            M += np.interp(rho, grid, table) * areas[None, :]
        return M

    def _rhs(self, geom):
        rings = polygon_rings(geom)
        a = polygon_potential_dz0(self.centres, rings)
        if self.under_chip == "mirror":
            return -2.0 * a  # thin sheet: the electrode is seen from both sides
        return -a

    def solve(self, geom):
        """Surface potential ψ on the outer panels for electrode ``geom`` at 1 V."""
        if self._lu is None:
            self._lu = sla.lu_factor(self._matrix())
        return sla.lu_solve(self._lu, self._rhs(geom))

    def correction(self, geom):
        """Smooth additive correction for electrode ``geom`` (GDS µm) at 1 V."""
        return _Smooth(self, self.solve(geom))

"""Regular evaluation grids in the ITVG frame, ordered as ITVG expects (x fastest, z slowest)."""

from dataclasses import dataclass
from decimal import Decimal

import numpy as np

__all__ = ["GridSpec", "axis"]


def axis(start, stop, step):
    """Uniform axis from ``start`` to ``stop`` inclusive, computed in exact decimal."""
    d0, d1, ds = Decimal(str(start)), Decimal(str(stop)), Decimal(str(step))
    if ds <= 0:
        raise ValueError("step must be positive")
    n = (d1 - d0) / ds
    if n != n.to_integral_value() or n < 0:
        raise ValueError(f"({stop} - {start}) is not a non-negative multiple of {step}")
    return np.array([float(d0 + i * ds) for i in range(int(n) + 1)])


@dataclass(frozen=True)
class GridSpec:
    """Axis triples (start, stop, step) in µm in the ITVG frame."""

    x: tuple
    y: tuple
    z: tuple

    def axes(self):
        return axis(*self.x), axis(*self.y), axis(*self.z)

    @property
    def shape(self):
        """(nz, ny, nx), the C-order shape of the value array."""
        ax, ay, az = self.axes()
        return len(az), len(ay), len(ax)

    def points(self):
        """(N, 3) points with x fastest, then y, then z (ITVG order, as in COMSOL exports)."""
        ax, ay, az = self.axes()
        zz, yy, xx = np.meshgrid(az, ay, ax, indexing="ij")
        return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

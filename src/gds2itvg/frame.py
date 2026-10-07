"""In-plane transform from GDS coordinates (µm) to the ITVG frame (µm).

    p_itvg = R(quarter_turns * 90°) · M · (p_gds - origin)

with M = diag(-1, 1) when ``mirror`` is set. z is unchanged (height above the surface).
"""

from dataclasses import asdict, dataclass

import numpy as np
import shapely

__all__ = ["Frame"]


@dataclass(frozen=True)
class Frame:
    origin: tuple = (0.0, 0.0)
    quarter_turns: int = 0
    mirror: bool = False

    def matrix(self):
        k = self.quarter_turns % 4
        c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k]
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        m = np.diag([-1.0 if self.mirror else 1.0, 1.0])
        return rot @ m

    def to_itvg(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        return (xy - np.asarray(self.origin, dtype=np.float64)) @ self.matrix().T

    def to_gds(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        return xy @ self.matrix() + np.asarray(self.origin, dtype=np.float64)

    def transform_geometry(self, geom):
        """Apply the transform to a shapely geometry (ring orientation is fixed later)."""
        return shapely.transform(geom, self.to_itvg)

    def to_dict(self):
        d = asdict(self)
        d["origin"] = [float(v) for v in self.origin]
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(origin=tuple(d.get("origin", (0.0, 0.0))),
                   quarter_turns=int(d.get("quarter_turns", 0)),
                   mirror=bool(d.get("mirror", False)))

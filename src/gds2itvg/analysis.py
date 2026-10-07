"""Small analysis helpers on built electrodes."""

import numpy as np
from scipy.optimize import minimize

from .kernels import get_kernel

__all__ = ["rf_null"]


def rf_null(electrodes, rf, y=0.0, x0=0.0, z_range=(2.0, 1000.0), backend="auto", h=1e-3):
    """Position of the RF field null (pseudopotential minimum) above the trap.

    ``rf`` is the RF electrode name or a list of names (all at the same RF voltage). The null
    is searched in the (x, z) plane at axial position ``y`` (ITVG frame, µm): a scan in z
    above ``x0``, refined by Nelder–Mead in (x, z) on |∇φ_RF|².

    Returns ``dict(x, y, z, field, phi)``: the null position, the residual |∇φ| per volt at
    the null (V/µm; ≈ 0 for a true null) and the unit RF potential there.
    """
    names = [rf] if isinstance(rf, str) else list(rf)
    kernel = get_kernel(backend)

    def phi(p):
        p = np.atleast_2d(np.asarray(p, dtype=float))
        return sum(electrodes[n].potential(p, kernel) for n in names)

    def grad2(xz):
        x, z = xz
        if z <= 0:
            return np.inf
        pts = np.array([[x + h, y, z], [x - h, y, z], [x, y + h, z], [x, y - h, z],
                        [x, y, z + h], [x, y, z - h]])
        v = phi(pts)
        g = (v[0::2] - v[1::2]) / (2 * h)
        return float(g @ g)

    zs = np.geomspace(z_range[0], z_range[1], 300)
    pts = np.column_stack([np.full_like(zs, x0), np.full_like(zs, y), zs])
    v = phi(pts)
    ez = np.gradient(v, zs)
    k = int(np.argmin(np.abs(ez)))
    res = minimize(grad2, [x0, zs[k]], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-20, "maxiter": 2000})
    x, z = res.x
    return {"x": float(x), "y": float(y), "z": float(z), "field": float(np.sqrt(grad2(res.x))),
            "phi": float(phi([[x, y, z]])[0])}

"""ITVG grid files: tab-separated ``x y z V`` rows (µm, volts), x fastest, then y, then z.

These are the rules ``VoltageGenerator.import_unit_potentials`` relies on: one batch of
``<stub><sep><n>.<ext>`` files per directory, identical uniform grids, and a pickled cache
``savefields.dat`` that must not be stale.
"""

import os
import re

import numpy as np

from .grid import GridSpec

__all__ = ["write_grid_file", "read_grid_file", "write_batch", "coordinate_prefixes"]


def _fmt_coord(v):
    s = repr(float(v))
    return s[:-2] if s.endswith(".0") else s


def coordinate_prefixes(grid: GridSpec):
    """Per-row ``"x\\ty\\tz\\t"`` strings, shared by all files of a batch."""
    ax, ay, az = grid.axes()
    sx = [_fmt_coord(v) for v in ax]
    sy = [_fmt_coord(v) for v in ay]
    sz = [_fmt_coord(v) for v in az]
    return [f"{x}\t{y}\t{z}\t" for z in sz for y in sy for x in sx]


def write_grid_file(path, grid: GridSpec, values, prefixes=None):
    values = np.asarray(values, dtype=np.float64).ravel()
    if prefixes is None:
        prefixes = coordinate_prefixes(grid)
    if values.shape[0] != len(prefixes):
        raise ValueError("values do not match the grid size")
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite potential values")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(p + repr(v) for p, v in zip(prefixes, values.tolist())))
        f.write("\n")


def read_grid_file(path):
    """Return ``(x_axis, y_axis, z_axis, V)`` with V shaped (nz, ny, nx)."""
    data = np.loadtxt(path, usecols=(0, 1, 2, 3))
    axes = [np.unique(np.round(data[:, i], 9)) for i in range(3)]
    nx, ny, nz = (len(a) for a in axes)
    if nx * ny * nz != data.shape[0]:
        raise ValueError(f"{path}: not a full regular grid")
    v = data[:, 3].reshape(nz, ny, nx)
    # Verify the ordering (x fastest, then y, then z).
    if not (np.allclose(data[:nx, 0], axes[0]) and np.allclose(data[::nx, 1][:ny], axes[1])):
        raise ValueError(f"{path}: rows are not ordered x fastest, then y, then z")
    return axes[0], axes[1], axes[2], v


_BATCH_RE = re.compile(r"^(?P<stub>.*\D)(?P<num>\d+)\.(?P<ext>\w+)$")


def write_batch(outdir, grid: GridSpec, potentials, stub="Electrode-", ext="txt", force=False):
    """Write ``{electrode_number: values}`` as one ITVG batch into ``outdir``.

    Refuses to mix with an existing batch or a stale ``savefields.dat`` unless ``force``.
    Returns the list of written paths.

    ``stub`` must not contain digits: ITVG recovers the electrode number with
    ``filename.lstrip(stub)``, which strips a *character set*, so a digit in the stub would
    eat leading digits of the number.
    """
    if not stub or any(c.isdigit() for c in stub) or "." in stub:
        raise ValueError("stub must be non-empty and contain no digits or '.'")
    os.makedirs(outdir, exist_ok=True)
    existing = os.listdir(outdir)
    if not force:
        if "savefields.dat" in existing:
            raise FileExistsError(f"{outdir} contains savefields.dat (stale ITVG cache); use force")
        others = [f for f in existing if _BATCH_RE.match(f)]
        if others:
            raise FileExistsError(f"{outdir} already contains grid files, e.g. {others[0]}; use force")
    else:
        cache = os.path.join(outdir, "savefields.dat")
        if os.path.exists(cache):
            os.remove(cache)
    prefixes = coordinate_prefixes(grid)
    paths = []
    for num in sorted(potentials, key=int):
        path = os.path.join(outdir, f"{stub}{int(num)}.{ext}")
        write_grid_file(path, grid, potentials[num], prefixes)
        paths.append(path)
    return paths

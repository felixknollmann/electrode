"""Julia kernel runner for the milestone-3 benchmark (not part of the package).

Julia was benchmarked against the Numba kernel and was not > 2x faster, so it was left out
of gds2itvg (RESULTS.md). This runner and ``Gds2ItvgKernel.jl`` stay here so the benchmark
can be reproduced.

The executable comes from $GDS2ITVG_JULIA or ``julia`` on PATH. The kernel runs in a
subprocess, with raw little-endian float64 files for input and output, so no Julia packages
are needed. The kernel is timed inside Julia after a warm-up call.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np

__all__ = ["julia_executable", "kernel_script", "polygon_potential", "run_kernel"]

_SCRIPT = os.path.join(os.path.dirname(__file__), "Gds2ItvgKernel.jl")


def julia_executable():
    exe = os.environ.get("GDS2ITVG_JULIA") or shutil.which("julia")
    return exe if exe and os.path.exists(exe) else None


def kernel_script():
    path = os.environ.get("GDS2ITVG_JULIA_KERNEL") or os.path.abspath(_SCRIPT)
    if not os.path.exists(path):
        raise FileNotFoundError("Gds2ItvgKernel.jl not found; set GDS2ITVG_JULIA_KERNEL")
    return path


def run_kernel(points, p1, p2, w, threads=None, repeats=0):
    """Run the Julia kernel; returns ``(phi, kernel_times_seconds)``."""
    exe = julia_executable()
    if exe is None:
        raise RuntimeError("Julia not found; set GDS2ITVG_JULIA")
    points = np.ascontiguousarray(points, dtype="<f8")
    threads = threads or os.cpu_count()
    with tempfile.TemporaryDirectory() as d:
        inp, out = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
        with open(inp, "wb") as f:
            np.array([len(points), len(p1)], dtype="<i8").tofile(f)
            points.tofile(f)  # row-major (N, 3) == column-major (3, N)
            np.ascontiguousarray(p1, dtype="<f8").tofile(f)
            np.ascontiguousarray(p2, dtype="<f8").tofile(f)
            np.ascontiguousarray(w, dtype="<f8").tofile(f)
        res = subprocess.run([exe, "--startup-file=no", "-t", str(threads), kernel_script(),
                              inp, out, str(repeats)],
                             check=True, capture_output=True, text=True)
        phi = np.fromfile(out, dtype="<f8")
    times = [float(t) for t in res.stdout.split()] if repeats else []
    return phi, times


def polygon_potential(points, rings, weights=None):
    """Same contract as the NumPy/Numba kernels."""
    from gds2itvg.kernels.numpy_kernel import _edges
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    p1, p2, w = _edges(rings, weights)
    if len(p1) == 0:
        return np.zeros(len(points))
    return run_kernel(points, p1, p2, w)[0]

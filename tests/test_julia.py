"""Milestone 3: the benchmark's Julia kernel agrees with the Python kernels.

Julia is not part of the package (it was not > 2x faster than Numba); this checks the
benchmark harness in benchmarks/julia/. Skipped without Julia.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "julia"))
import julia_runner as jl  # noqa: E402

from gds2itvg.kernels import get_kernel  # noqa: E402

pytestmark = pytest.mark.skipif(jl.julia_executable() is None, reason="Julia not available")


def test_julia_matches_numpy_and_numba():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-60, 60, 3000), rng.uniform(-60, 60, 3000),
                           rng.uniform(0.5, 90, 3000)])
    rings = [np.array([[0, 0], [70, 0], [70, 30], [40, 15], [0, 30.0]]),
             np.array([[-30, -30], [-10, -30], [-10, -10], [-30, -10.0]]),
             np.array([[-50, 20], [-20, 20], [-20, 50], [-50, 50.0]])[::-1]]
    w = [1.0, -0.37, 2.5]
    ref = get_kernel("numpy")(pts, rings, w)
    got = jl.polygon_potential(pts, rings, w)
    assert np.max(np.abs(got - ref)) < 1e-13
    try:
        assert np.max(np.abs(got - get_kernel("numba")(pts, rings, w))) < 1e-13
    except ImportError:  # pragma: no cover
        pass

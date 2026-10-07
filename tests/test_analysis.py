"""RF null finder on the bundled five-wire example."""

import os

import pytest

from gds2itvg.analysis import rf_null
from gds2itvg.core import Layout, build_electrodes

EX = os.path.join(os.path.dirname(__file__), "..", "examples", "fivewire")


def test_rf_null_of_example_trap():
    layout = Layout.from_json(os.path.join(EX, "fivewire.json"))
    layout.gap_model, layout.outer_ground_depth = "midline", None
    e = build_electrodes(os.path.join(EX, "fivewire.gds"), layout,
                         ion_box=(-10, 10, -400, 400), ion_z=(57, 77))
    r = rf_null(e, "16", y=0.0)
    assert abs(r["x"]) < 1e-3                 # on the symmetry plane
    assert 66.5 < r["z"] < 67.5               # ~67 µm above the surface
    assert r["field"] < 1e-8                  # a true null
    # the null height is uniform along the trap axis
    assert abs(rf_null(e, "16", y=200.0)["z"] - r["z"]) < 0.5

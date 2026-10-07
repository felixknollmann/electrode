"""End-to-end run on the bundled five-wire example (examples/fivewire)."""

import importlib.util
import json
import os
import shutil

import numpy as np
import pytest

from gds2itvg.cli import main
from gds2itvg.itvg_io import read_grid_file

EX = os.path.join(os.path.dirname(__file__), "..", "examples", "fivewire")


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    """Regenerate the example GDS + layout with the bundled script into a temp dir."""
    d = tmp_path_factory.mktemp("fivewire")
    shutil.copy(os.path.join(EX, "make_fivewire.py"), d)
    spec = importlib.util.spec_from_file_location("make_fivewire", d / "make_fivewire.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
    return d


def test_generated_example_matches_committed_layout(example):
    with open(os.path.join(EX, "fivewire.json")) as f:
        committed = json.load(f)
    with open(example / "fivewire.json") as f:
        assert json.load(f) == committed


@pytest.mark.parametrize("model, tol", [
    # exact geometry: the mirror symmetry holds to rounding
    (["--gap-model", "midline", "--no-finite-chip", "--no-simplify"], 1e-12),
    # default device-stack model: far-field simplification and the finite-chip quadtree
    # are not exactly mirror-symmetric (µV level, see RESULTS.md)
    ([], 2e-5),
])
def test_example_grid_files_and_mirror_symmetry(example, tmp_path, model, tol):
    out = tmp_path / "out"
    args = [str(example / "fivewire.gds"), "--layout", str(example / "fivewire.json"),
            "--x=-6:6:2", "--y=-300:300:20", "--z=57:77:5", "--out", str(out),
            "--stub", "FiveWire-"] + model
    assert main(args) == 0
    files = sorted(p.name for p in out.iterdir())
    assert files == sorted(f"FiveWire-{n}.txt" for n in range(1, 17))
    v = {n: read_grid_file(out / f"FiveWire-{n}.txt")[3] for n in (3, 8, 15, 16)}
    # Fingers 3 and 8 face each other across the axis: mirror images in ITVG x.
    assert np.max(np.abs(v[3] - v[8][:, :, ::-1])) < tol
    # Centre DC rail is symmetric in x; RF dominates at the ion height.
    assert np.max(np.abs(v[15] - v[15][:, :, ::-1])) < tol
    assert np.all(v[16][:, 15, 3] > 0.3)


def test_label_png(example, tmp_path):
    pytest.importorskip("matplotlib")
    png = tmp_path / "labels.png"
    assert main([str(example / "fivewire.gds"), "--layout", str(example / "fivewire.json"),
                 "--x=-10:10:1", "--y=-400:400:1", "--z=57:77:1",
                 "--label-png", str(png), "--label-only"]) == 0
    assert png.stat().st_size > 10000

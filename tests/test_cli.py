"""End-to-end CLI run on a small synthetic layout."""

import json

import gdstk
import numpy as np

from gds2itvg.cli import main
from gds2itvg.itvg_io import read_grid_file


def test_cli_end_to_end(tmp_path):
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    c = lib.new_cell("TOP")
    c.add(gdstk.rectangle((-300, -12), (300, 12), layer=30))      # centre strip
    c.add(gdstk.rectangle((-300, 16), (300, 200), layer=30))      # upper, 4 µm gap
    c.add(gdstk.rectangle((-300, -200), (300, -16), layer=30))    # lower, 4 µm gap
    c.add(gdstk.rectangle((-300, 204), (-250, 260), layer=30))    # ground pad
    c.add(gdstk.rectangle((1000, 1000), (1100, 1100), layer=30))  # far away, not in ROI
    lib.write_gds(str(tmp_path / "t.gds"))
    layout = {
        "layer": 30, "roi_box": [-310, -210, 310, 210],
        "frame": {"origin": [0, 0], "quarter_turns": 1, "mirror": False},
        "electrodes": {"1": [[0, 0]], "2": [[0, 100]], "3": [[0, -100]]},
        "ground": {"16": [[-270, 230]]},
        "gap_model": "midline", "outer_ground_depth": None,
    }
    (tmp_path / "layout.json").write_text(json.dumps(layout))
    out = tmp_path / "out"
    rc = main([str(tmp_path / "t.gds"), "--layout", str(tmp_path / "layout.json"),
               "--x=-2:2:1", "--y=-5:5:5", "--z=40:50:10", "--out", str(out),
               "--stub", "T-", "--write-ground", "--backend", "numpy"])
    assert rc == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["T-1.txt", "T-16.txt", "T-2.txt", "T-3.txt"]
    _, _, _, v1 = read_grid_file(out / "T-1.txt")
    _, _, _, v16 = read_grid_file(out / "T-16.txt")
    assert v1.shape == (2, 3, 5) and np.all(v16 == 0)
    # Midline-split centre strip spans GDS y in [-14, 14]; long-strip limit (2/pi) atan(14/z).
    assert abs(v1[0, 1, 2] - 2 / np.pi * np.arctan(14 / 40)) < 2e-3
    # ITVG x = -(GDS y): electrode 2 (GDS +y) is stronger at negative ITVG x.
    _, _, _, v2 = read_grid_file(out / "T-2.txt")
    assert v2[0, 1, 0] > v2[0, 1, -1]


def _fill_layout(tmp_path, fill):
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    c = lib.new_cell("TOP")
    # Rail with a notch (x 40..60 cut back to y 0..2), and a neighbour across a 4 µm gap.
    c.add(gdstk.Polygon([(0, 0), (100, 0), (100, 10), (60, 10), (60, 2), (40, 2), (40, 10),
                         (0, 10)], layer=30))
    c.add(gdstk.rectangle((0, 14), (100, 30), layer=30))
    lib.write_gds(str(tmp_path / "n.gds"))
    from gds2itvg.core import Layout
    return Layout(layer=30, roi_box=(0, 0, 100, 30),
                  electrodes={"1": [[10, 5]], "2": [[10, 20]]}, fill=fill,
                  gap_model="midline", outer_ground_depth=None)


def test_fill_restores_notched_rail(tmp_path):
    from gds2itvg.core import build_electrodes
    layout = _fill_layout(tmp_path, {"1": [[[40, 2], [60, 2], [60, 10], [40, 10]]]})
    e = build_electrodes(tmp_path / "n.gds", layout)
    from shapely.geometry import box
    # Uniform rail up to the gap midline (y = 12); the gap ends at x = 0, 100 meet open
    # surface, so check the interior, including the former notch at x 40..60.
    inner = box(5, 0, 95, 30)
    assert abs(e["1"].geometry.intersection(inner).area - 90 * 12) < 1e-9
    assert e["1"].geometry.contains(box(40, 0, 60, 11.999))
    assert abs(e["2"].geometry.intersection(inner).bounds[1] - 12) < 1e-9


def test_fill_into_wrong_electrode_is_rejected(tmp_path):
    import pytest
    from gds2itvg.core import build_electrodes
    layout = _fill_layout(tmp_path, {"2": [[[40, 2], [60, 2], [60, 10], [40, 10]]]})
    with pytest.raises(ValueError):
        build_electrodes(tmp_path / "n.gds", layout)


def test_cli_physical_model_runs(tmp_path):
    """Default physical model (gap profiles + finite chip) on a small layout."""
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    c = lib.new_cell("TOP")
    c.add(gdstk.rectangle((-600, -12), (600, 12), layer=30))
    c.add(gdstk.rectangle((-600, 16), (600, 600), layer=30))
    c.add(gdstk.rectangle((-600, -600), (600, -16), layer=30))
    lib.write_gds(str(tmp_path / "p.gds"))
    layout = {"layer": 30, "roi_box": [-610, -610, 610, 610],
              "frame": {"origin": [0, 0], "quarter_turns": 1},
              "electrodes": {"1": [[0, 0]], "2": [[0, 300]], "3": [[0, -300]]}}
    (tmp_path / "l.json").write_text(json.dumps(layout))
    out = tmp_path / "o"
    args = [str(tmp_path / "p.gds"), "--layout", str(tmp_path / "l.json"), "--x=-2:2:2",
            "--y=-4:4:4", "--z=40:50:10", "--out", str(out), "--stub", "P-"]
    assert main(args) == 0
    assert main(args + ["--gap-model", "midline", "--no-finite-chip", "--force",
                        "--out", str(tmp_path / "b")]) == 0
    _, _, _, phys = read_grid_file(out / "P-1.txt")
    _, _, _, base = read_grid_file(tmp_path / "b" / "P-1.txt")
    # the buried ground lowers the narrow centre strip by a few percent at most
    assert np.all(phys < base) and np.all(phys > 0.9 * base)


def test_no_ground_with_finite_outer_ground_is_rejected(tmp_path):
    import pytest
    from gds2itvg.core import build_electrodes
    layout = _fill_layout(tmp_path, {})
    layout.gap_model, layout.ground_depth, layout.outer_ground_depth = "profile", None, 1000.0
    with pytest.raises(ValueError, match="vacuum outside"):
        build_electrodes(tmp_path / "n.gds", layout)

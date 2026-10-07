"""Milestone 1 (b): agreement with nist-ionstorage/electrode on the five-wire layout.

The reference package is only imported here, from $GDS2ITVG_REFERENCE (default
/tmp/reference). The test is skipped when it is not available.
"""

import json
import os
import sys

import numpy as np
import pytest
from shapely.geometry import Polygon

from conftest import REFERENCE
from gds2itvg.geometry import polygon_rings
from gds2itvg.midline import split_gaps

SCALE = 50.0  # µm per tutorial length unit (ion height ~ 1 unit)


def five_wire(edge, width, top, mid, bot):
    """Electrode layout of the reference tutorial's ``five_wire`` (examples/tutorial.py)."""
    e, r, t, m, b = edge, width / 2, top + mid / 2, mid / 2, -bot - mid / 2
    return [
        ("tl", [[(-e, e), (-e, t), (-r, t), (-r, e)]]),
        ("tm", [[(-r, e), (-r, t), (r, t), (r, e)]]),
        ("tr", [[(r, e), (r, t), (e, t), (e, e)]]),
        ("bl", [[(-e, -e), (-r, -e), (-r, b), (-e, b)]]),
        ("bm", [[(-r, -e), (r, -e), (r, b), (-r, b)]]),
        ("br", [[(r, -e), (e, -e), (e, b), (r, b)]]),
        ("r", [[(-e, t), (-e, m), (e, m), (e, t)], [(-e, b), (e, b), (e, -m), (-e, -m)]]),
        ("c", [[(-e, m), (-e, -m), (e, -m), (e, m)]]),
    ]


def reference_system():
    if not os.path.isdir(os.path.join(REFERENCE, "electrode")):
        pytest.skip("reference package not available")
    sys.path.insert(0, REFERENCE)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from electrode import PolygonPixelElectrode
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"reference package does not import: {exc}")
    finally:
        sys.path.remove(REFERENCE)
    return PolygonPixelElectrode


def eval_grid():
    x = np.linspace(-1.5, 1.5, 13) * SCALE
    y = np.linspace(-0.6, 0.6, 9) * SCALE
    z = np.linspace(0.4, 2.0, 9) * SCALE
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def compare(layout_rings, kernel):
    PPE = reference_system()
    pts = eval_grid()
    rows = {}
    for name, rings in layout_rings.items():
        ref = PPE(paths=[np.array(r, dtype=float) for r in rings]).potential(pts, 0)[:, 0]
        got = kernel(pts, rings)
        err = np.abs(got - ref)
        rows[name] = (float(err.max()), float(err.max() / np.abs(ref).max()))
    return rows


def _record(tag, rows):
    path = os.environ.get("GDS2ITVG_RESULTS")
    if path:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"m1_fivewire_{tag}.json"), "w") as f:
            json.dump(rows, f, indent=1)


def test_five_wire_matches_reference(kernel):
    layout = {n: [np.array(p, dtype=float) * SCALE for p in polys]
              for n, polys in five_wire(5, 2.0, 1.0, 1.0, 1.0)}
    # Use our ring orientation/cleaning so both packages see the same geometry.
    rings = {n: polygon_rings(_union(polys)) for n, polys in layout.items()}
    rows = compare(rings, kernel)
    _record("gapless", rows)
    for name, (abs_err, rel_err) in rows.items():
        assert abs_err < 1e-12, (name, abs_err)
        assert rel_err < 1e-10, (name, rel_err)


def test_gapped_five_wire_midline_matches_reference(kernel):
    """Same layout with 0.1-unit gaps, midline-split by us, evaluated by both packages."""
    g = 0.1
    layout = []
    for n, polys in five_wire(5, 2.0, 1.0, 1.0, 1.0):
        geom = _union([np.array(p, dtype=float) * SCALE for p in polys]).buffer(
            -g / 2 * SCALE, join_style="mitre")
        layout.append((n, geom))
    grown = split_gaps([gm for _, gm in layout], max_gap=2 * g * SCALE)
    rings = {n: polygon_rings(gm) for (n, _), gm in zip(layout, grown)}
    rows = compare(rings, kernel)
    _record("gapped_midline", rows)
    for name, (abs_err, _) in rows.items():
        assert abs_err < 1e-12, (name, abs_err)


def _union(polys):
    from shapely import union_all
    return union_all([Polygon(p) for p in polys])

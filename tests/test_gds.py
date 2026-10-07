"""Milestone 1: GDS import (database units, paths, duplicates, holes, references, naming)."""

import gdstk
import numpy as np
import pytest

from gds2itvg import gds


def write_lib(path, unit, precision, shapes):
    lib = gdstk.Library(unit=unit, precision=precision)
    cell = lib.new_cell("TOP")
    for s in shapes:
        cell.add(s)
    lib.write_gds(str(path))
    return lib


def test_database_unit_invariance(tmp_path):
    shapes = lambda: [gdstk.rectangle((0, 0), (10.5, 4.25), layer=30)]
    write_lib(tmp_path / "um.gds", 1e-6, 1e-9, shapes())
    # Same geometry written in mm user units with 1 nm database units.
    lib = gdstk.Library(unit=1e-3, precision=1e-9)
    c = lib.new_cell("TOP")
    c.add(gdstk.rectangle((0, 0), (0.0105, 0.00425), layer=30))
    lib.write_gds(str(tmp_path / "mm.gds"))
    a, _ = gds.load_conductors(tmp_path / "um.gds", 30)
    b, _ = gds.load_conductors(tmp_path / "mm.gds", 30)
    assert len(a) == len(b) == 1
    assert a[0].bounds == pytest.approx((0, 0, 10.5, 4.25), abs=1e-9)
    assert b[0].symmetric_difference(a[0]).area < 1e-9


def test_duplicates_removed(tmp_path):
    r = gdstk.rectangle((0, 0), (5, 5), layer=30)
    rev = gdstk.Polygon(r.points[::-1], layer=30)  # same polygon, other orientation/start
    write_lib(tmp_path / "d.gds", 1e-6, 1e-9, [r, r.copy(), rev])
    cond, ndup = gds.load_conductors(tmp_path / "d.gds", 30)
    assert ndup == 2
    assert len(cond) == 1 and cond[0].area == pytest.approx(25)


def test_touching_polygons_merge_and_layer_filter(tmp_path):
    write_lib(tmp_path / "t.gds", 1e-6, 1e-9, [
        gdstk.rectangle((0, 0), (5, 5), layer=30),
        gdstk.rectangle((5, 0), (9, 5), layer=30),
        gdstk.rectangle((20, 0), (25, 5), layer=30),
        gdstk.rectangle((0, 0), (100, 100), layer=31),
    ])
    cond, _ = gds.load_conductors(tmp_path / "t.gds", 30)
    assert sorted(round(c.area, 6) for c in cond) == [25.0, 45.0]


def test_paths_become_polygons(tmp_path):
    path = gdstk.FlexPath([(0, 0), (100, 0)], 4, layer=30)
    write_lib(tmp_path / "p.gds", 1e-6, 1e-9, [path])
    cond, _ = gds.load_conductors(tmp_path / "p.gds", 30)
    assert len(cond) == 1
    assert cond[0].bounds == pytest.approx((0, -2, 100, 2), abs=1e-6)


def test_polygon_with_hole(tmp_path):
    ring = gdstk.boolean(gdstk.rectangle((0, 0), (10, 10)), gdstk.rectangle((3, 3), (6, 6)),
                         "not", layer=30)
    write_lib(tmp_path / "h.gds", 1e-6, 1e-9, ring)
    cond, _ = gds.load_conductors(tmp_path / "h.gds", 30)
    assert len(cond) == 1
    assert cond[0].area == pytest.approx(100 - 9)
    from gds2itvg.geometry import polygon_rings
    rings = polygon_rings(cond[0])
    assert len(rings) == 2  # exterior + hole


def test_references_are_flattened(tmp_path):
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    sub = lib.new_cell("SUB")
    sub.add(gdstk.rectangle((0, 0), (1, 2), layer=30))
    top = lib.new_cell("TOP")
    top.add(gdstk.Reference(sub, (10, 0), columns=3, rows=1, spacing=(5, 0)))
    lib.write_gds(str(tmp_path / "r.gds"))
    cond, _ = gds.load_conductors(tmp_path / "r.gds", 30, cell="TOP")
    assert len(cond) == 3


def test_select_and_assign_names():
    from shapely.geometry import box
    cond = [box(0, 0, 10, 10), box(20, 0, 30, 10), box(100, 0, 110, 10)]
    assert gds.select_in_box(cond, (5, 5, 25, 6)) == [0, 1]
    named = gds.assign_names(cond, {"1": [[5, 5]], "2": [[25, 5], [105, 5]]})
    assert named == {"1": [0], "2": [1, 2]}
    with pytest.raises(ValueError):
        gds.assign_names(cond, {"1": [[50, 50]]})
    with pytest.raises(ValueError):
        gds.assign_names(cond, {"1": [[5, 5]], "2": [[6, 6]]})

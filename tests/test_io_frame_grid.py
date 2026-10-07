"""Milestone 1: frame transform, grid ordering, ITVG file format and ITVG round-trip."""

import importlib
import os
import sys

import numpy as np
import pytest
from shapely.geometry import Polygon

from conftest import ITVG
from gds2itvg.frame import Frame
from gds2itvg.geometry import Electrode
from gds2itvg.grid import GridSpec, axis
from gds2itvg.itvg_io import read_grid_file, write_batch, write_grid_file


@pytest.mark.parametrize("k", range(4))
@pytest.mark.parametrize("mirror", [False, True])
def test_frame_roundtrip(k, mirror):
    f = Frame(origin=(-1234.0, 5678.0), quarter_turns=k, mirror=mirror)
    xy = np.random.default_rng(k).normal(size=(10, 2)) * 1e3
    assert np.allclose(f.to_gds(f.to_itvg(xy)), xy)


def test_quarter_turn_frame_convention():
    f = Frame(origin=(-1234.0, 5678.0), quarter_turns=1)
    # GDS trap axis (along x) maps to ITVG y; GDS +y maps to ITVG -x.
    assert np.allclose(f.to_itvg([[-1234 + 100, 5678]]), [[0, 100]])
    assert np.allclose(f.to_itvg([[-1234, 5678 + 5]]), [[-5, 0]])


def test_mirrored_frame_keeps_rings_valid():
    e = Electrode("a", Frame(mirror=True).transform_geometry(Polygon([(0, 0), (2, 0), (2, 1)])))
    (ring,) = e.rings()
    x, y = ring[:, 0], ring[:, 1]
    assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0  # CCW after transform


def test_axis_is_exact_decimal():
    a = axis(-800, 800, 1)
    assert len(a) == 1601 and a[0] == -800.0 and a[-1] == 800.0 and a[1] == -799.0
    assert axis(0, 1, 0.1)[3] == 0.3
    with pytest.raises(ValueError):
        axis(0, 1, 0.3)


def test_grid_order_x_fastest():
    g = GridSpec((0, 2, 1), (10, 11, 1), (40, 41, 1))
    p = g.points()
    assert g.shape == (2, 2, 3)
    assert np.array_equal(p[:3, 0], [0, 1, 2]) and np.all(p[:3, 1] == 10) and np.all(p[:6, 2] == 40)
    assert np.array_equal(p[3, :], [0, 11, 40]) and np.array_equal(p[6, :], [0, 10, 41])


def test_write_read_roundtrip(tmp_path):
    g = GridSpec((-1, 1, 0.5), (-2, 2, 1), (40, 42, 1))
    v = np.random.default_rng(0).normal(size=g.points().shape[0])
    write_grid_file(tmp_path / "a-1.txt", g, v)
    x, y, z, V = read_grid_file(tmp_path / "a-1.txt")
    ax, ay, az = g.axes()
    assert np.array_equal(x, ax) and np.array_equal(y, ay) and np.array_equal(z, az)
    assert np.array_equal(V.ravel(), v)  # repr() round-trips float64 exactly
    first = (tmp_path / "a-1.txt").read_text().splitlines()[0].split("\t")
    assert first[:3] == ["-1", "-2", "40"]


def test_batch_rules(tmp_path):
    g = GridSpec((0, 1, 1), (0, 1, 1), (40, 40, 1))
    v = {"1": np.zeros(4), "12": np.ones(4)}
    paths = write_batch(tmp_path, g, v, stub="Trap-")
    assert [os.path.basename(p) for p in paths] == ["Trap-1.txt", "Trap-12.txt"]
    with pytest.raises(FileExistsError):
        write_batch(tmp_path, g, v, stub="Trap-")
    (tmp_path / "savefields.dat").write_bytes(b"x")
    write_batch(tmp_path, g, v, stub="Trap-", force=True)
    assert not (tmp_path / "savefields.dat").exists()
    with pytest.raises(ValueError):
        write_batch(tmp_path / "b", g, v, stub="GDS2ITVG-")  # digit in stub breaks ITVG


def test_itvg_voltage_generator_roundtrip(tmp_path):
    """Files load in ITVG's VoltageGenerator and interpolate the written potential."""
    if not os.path.exists(os.path.join(ITVG, "VoltageGenerator.py")):
        pytest.skip("ITVG checkout not available")
    sys.path.insert(0, ITVG)
    try:
        vg = importlib.import_module("VoltageGenerator")
    finally:
        sys.path.remove(ITVG)
    from gds2itvg.kernels import get_kernel
    k = get_kernel("numpy")
    rings = {"1": [np.array([[-30, -5], [30, -5], [30, 5], [-30, 5.0]])],
             "2": [np.array([[-30, 8], [30, 8], [30, 40], [-30, 40.0]])],
             "13": [np.array([[-30, -40], [30, -40], [30, -8], [-30, -8.0]])]}
    g = GridSpec((-4, 4, 1), (-10, 10, 1), (40, 48, 1))
    pts = g.points()
    write_batch(tmp_path, g, {n: k(pts, r) for n, r in rings.items()}, stub="Unit-")
    gen = vg.VoltageGenerator(str(tmp_path), fit_ranges=[2, 2, 2], order=2)
    assert gen.electrodes == [1, 2, 13]
    q = np.array([[0.5, 1.5, 44.5], [-2.25, -3.75, 41.0]])
    for n, r in rings.items():
        got = gen.unit_potentials[int(n)](q)
        want = k(q, r)
        # ITVG stores float32 and interpolates linearly on a 1 µm grid.
        assert np.allclose(got, want, rtol=2e-3, atol=1e-6)

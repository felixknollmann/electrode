"""Milestone 1: midline gap split (the gapless baseline)."""

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box

from gds2itvg.geometry import polygon_rings
from gds2itvg.kernels import get_kernel
from gds2itvg.midline import gap_region, split_gaps


def test_two_rectangles_meet_at_midline():
    a, b = box(0, 0, 100, 50), box(104, 0, 200, 50)
    ga, gb = split_gaps([a, b], max_gap=10)
    assert ga.bounds == pytest.approx((0, 0, 102, 50), abs=1e-9)
    assert gb.bounds == pytest.approx((102, 0, 200, 50), abs=1e-9)
    assert ga.intersection(gb).area == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("g", [0.5, 4.0, 6.0, 20.0, 26.0])
def test_uniform_gap_widths(g):
    a, b = box(0, 0, 300, 50), box(0, 50 + g, 300, 100)
    ga, gb = split_gaps([a, b], max_gap=30)
    # Midline across the gap interior (gap ends meet open surface and are not tested here).
    cut = box(20, -1, 280, 101)
    assert ga.intersection(cut).bounds[3] == pytest.approx(50 + g / 2, abs=1e-9)
    assert gb.intersection(cut).bounds[1] == pytest.approx(50 + g / 2, abs=1e-9)
    assert ga.intersection(gb).area < 1e-9


def test_wider_gaps_are_not_split():
    a, b = box(0, 0, 100, 50), box(140, 0, 200, 50)
    ga, gb = split_gaps([a, b], max_gap=30)
    assert ga.equals(a) and gb.equals(b)


def test_layout_without_gaps_is_unchanged():
    tiles = [box(i * 10, 0, i * 10 + 10, 10) for i in range(4)]
    for t, g in zip(tiles, split_gaps(tiles, max_gap=5)):
        assert g.symmetric_difference(t).area < 1e-9


def test_partition_is_exact_on_l_shaped_gaps():
    # Three conductors around an L-shaped gap and a T-junction.
    a = box(0, 0, 50, 50)
    b = Polygon([(54, 0), (100, 0), (100, 100), (0, 100), (0, 54), (54, 54)])
    c = box(54 - 2, -60, 100, -4)
    cond = [a, b, c]
    grown, info = split_gaps(cond, max_gap=10, return_info=True)
    gap = gap_region(cond, 10)
    union = shapely.union_all(cond)
    total = sum(g.area for g in grown)
    assert total == pytest.approx(union.area + gap.area, rel=1e-12)
    for i in range(3):
        for j in range(i + 1, 3):
            assert grown[i].intersection(grown[j]).area < 1e-9
    # Along the straight part of the a|b gap the boundary is the midline x = 52.
    probe = box(51.9, 10, 52.1, 40)
    assert grown[0].intersection(probe).area == pytest.approx(0.1 * 30, rel=1e-9)


def test_partition_of_unity():
    """Sum of grown unit potentials == potential of the closed union."""
    rng = np.random.default_rng(0)
    cond = [box(0, 0, 40, 40), box(44, 0, 80, 40), box(0, 46, 80, 70)]
    grown = split_gaps(cond, max_gap=10)
    k = get_kernel("numpy")
    pts = np.column_stack([rng.uniform(-20, 100, 300), rng.uniform(-20, 90, 300),
                           rng.uniform(5, 60, 300)])
    total = sum(k(pts, polygon_rings(g)) for g in grown)
    whole = k(pts, polygon_rings(shapely.union_all(grown)))
    assert np.max(np.abs(total - whole)) < 1e-12


def test_gapped_five_wire_restores_gapless_layout():
    """Shrinking abutting electrodes by g/2 and splitting the gaps gives back the original
    shared boundaries."""
    a = box(-100, -10, 100, 10)
    b = box(-100, 10, 100, 60)
    c = box(-100, -60, 100, -10)
    g = 2.0
    shrunk = [p.buffer(-g / 2, join_style="mitre") for p in (a, b, c)]
    grown = split_gaps(shrunk, max_gap=5)
    assert grown[0].bounds[1] == pytest.approx(-10, abs=1e-9)
    assert grown[0].bounds[3] == pytest.approx(10, abs=1e-9)

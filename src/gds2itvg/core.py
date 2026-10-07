"""High-level pipeline: GDS layout → named electrodes (ITVG frame) → unit potentials."""

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import Polygon, box

from . import gds
from .frame import Frame
from .geometry import Electrode, as_multipolygon
from .grid import GridSpec
from .kernels import get_kernel
from .midline import split_gaps

__all__ = ["Layout", "build_electrodes", "unit_potentials"]

log = logging.getLogger(__name__)


@dataclass
class Layout:
    """Everything needed to turn a GDS file into named electrodes.

    ``roi_box`` (GDS µm) selects the trap's conductors; conductors within ``max_gap`` of
    them also take part (they border the trap across a gap). Names come from ``electrodes``
    seeds ``{name: [[x, y], ...]}``; ``ground`` seeds name conductors held at 0 V (no
    output file). Without seeds, ROI conductors are numbered 1..N (sorted by centroid y, x).

    ``fill`` adds metal that is not in the GDS: ``{name: [polygon, ...]}`` with polygons as
    ``[[x, y], ...]`` in GDS µm. The polygons are merged into the drawn metal before the gaps
    are split, and each must end up in the conductor(s) of electrode ``name``, e.g. to
    run a rail uniformly through a notch cut for an in-plane structure.

    Physical model (milestone 2, GUIDELINES §5):

    - ``gap_model``:
      - ``"midline"``: the gapless baseline;
      - ``"profile"``: the baseline plus cross-section gap profiles from the device stack
        ``metal_thickness``, ``ground_depth`` and ``epsilon_r``;
      - ``"drawn"``: gaps at 0 V.
    - ``outer_ground_depth``: if not None, the surface outside ``chip_outline`` floats over
      a ground at that depth (finite chip). None keeps the surface grounded to infinity.
      ``chip_outline`` is ``"auto"`` (the bounding box of the trap's metal) or a polygon
      ``[[x, y], ...]`` in GDS µm.
    - ``simplify_keep``: geometry farther than this (µm) from the ion region is simplified
      with tolerance ``simplify_rel_tol`` × distance (``simplify.simplify_far``). None keeps
      everything exact.
    - ``strip_interpolation``: the merged gap strips are evaluated on a coarse grid over
      the ion ROI (spacing z_min / 8) and interpolated (``fields.EdgeField``); False
      evaluates them exactly at every point.
    """

    layer: int = 30
    datatype: int = 0
    cell: str | None = None
    roi_box: tuple = None
    frame: Frame = field(default_factory=Frame)
    electrodes: dict = field(default_factory=dict)
    ground: dict = field(default_factory=dict)
    max_gap: float = 30.0
    gap_model: str = "profile"
    fill: dict = field(default_factory=dict)
    metal_thickness: float = 1.0
    ground_depth: float | None = 10.0
    epsilon_r: float = 3.9
    strips_per_gap: int = 8
    chip_outline: object = "auto"
    outer_ground_depth: float | None = 1000.0
    simplify_keep: float | None = 200.0
    simplify_rel_tol: float = 0.01
    strip_interpolation: bool = True

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        d["frame"] = Frame.from_dict(d.get("frame", {}))
        if d.get("roi_box") is not None:
            d["roi_box"] = tuple(d["roi_box"])
        return cls(**{k: v for k, v in d.items() if not k.startswith("_")})

    def to_json(self, path, comment=None):
        d = {
            "layer": self.layer, "datatype": self.datatype, "cell": self.cell,
            "roi_box": list(self.roi_box) if self.roi_box else None,
            "frame": self.frame.to_dict(), "max_gap": self.max_gap,
            "gap_model": self.gap_model,
            "metal_thickness": self.metal_thickness, "ground_depth": self.ground_depth,
            "epsilon_r": self.epsilon_r, "strips_per_gap": self.strips_per_gap,
            "chip_outline": self.chip_outline, "outer_ground_depth": self.outer_ground_depth,
            "simplify_keep": self.simplify_keep, "simplify_rel_tol": self.simplify_rel_tol,
            "strip_interpolation": self.strip_interpolation,
            "electrodes": self.electrodes, "ground": self.ground,
        }
        if self.fill:
            d["fill"] = self.fill
        if comment:
            d = {"_comment": comment, **d}
        with open(path, "w") as f:
            json.dump(d, f, indent=1)
            f.write("\n")


def _participants(conductors, roi_box, max_gap):
    sel = gds.select_in_box(conductors, roi_box) if roi_box else list(range(len(conductors)))
    core = shapely.union_all([conductors[i] for i in sel])
    near = [i for i, c in enumerate(conductors) if i in sel or c.distance(core) <= max_gap]
    return sel, near


def chip_polygon(layout, local):
    if layout.chip_outline is None:
        return None
    if isinstance(layout.chip_outline, str):
        if layout.chip_outline != "auto":
            raise ValueError("chip_outline must be 'auto', None or a polygon")
        return box(*shapely.union_all(local).bounds)
    return Polygon(layout.chip_outline)


def ion_region_gds(layout, ion_box):
    """ITVG (xmin, xmax, ymin, ymax) → polygon in GDS coordinates."""
    x0, x1, y0, y1 = ion_box
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    return Polygon(layout.frame.to_gds(corners)).buffer(0)


def build_electrodes(gds_path, layout: Layout, return_conductors=False,
                     ion_box=(-10.0, 10.0, -800.0, 800.0), ion_z=(40.0, 60.0),
                     return_ground=False):
    """Load, select, split gaps, name, apply the physical model and transform.

    Returns ``{name: Electrode}`` in the ITVG frame. ``ion_box``/``ion_z`` (ITVG µm) tell
    the corrections where accuracy matters (lumping, finite-chip interpolation grid).
    With ``return_ground`` also returns ``{name: geometry}`` of the ground electrodes
    (ITVG frame), e.g. for :func:`gds2itvg.plot.label_map`.
    """
    polys = gds.read_layer_polygons(gds_path, layout.layer, layout.datatype, layout.cell)
    fills = [(str(n), np.asarray(p, dtype=np.float64)) for n, ps in layout.fill.items() for p in ps]
    if fills and not layout.electrodes:
        raise ValueError("fill requires named electrodes")
    conductors, _ = gds.conductors_from_polygons(polys + [p for _, p in fills])
    sel, near = _participants(conductors, layout.roi_box, layout.max_gap)
    local = [conductors[i] for i in near]
    if layout.gap_model in ("midline", "profile"):
        grown = split_gaps(local, max_gap=layout.max_gap)
    elif layout.gap_model == "drawn":
        grown = local
    else:
        raise ValueError(f"unknown gap_model {layout.gap_model!r}")

    if layout.electrodes:
        # Seeds lie in the drawn metal; the grown shapes of those conductors are used.
        named = gds.assign_names(local, {**layout.electrodes, **layout.ground})
        for name, p in fills:
            owner = shapely.union_all([local[i] for i in named.get(name, [])])
            if owner.is_empty or not owner.buffer(1e-6).contains(Polygon(p)):
                raise ValueError(f"fill polygon {p.tolist()} did not merge into electrode {name!r}")
        out = {n: shapely.union_all([grown[i] for i in named[n]]) for n in layout.electrodes}
        members = {n: named[n] for n in layout.electrodes}
        used = {i for idx in named.values() for i in idx}
        unnamed = [near[i] for i in range(len(local)) if i not in used and near[i] in sel]
        if unnamed:
            log.warning("ROI conductors %s are not named; they are treated as ground", unnamed)
    else:
        # Unnamed layouts: number the ROI conductors 1..N in deterministic order.
        roi = [k for k in range(len(local)) if near[k] in sel]
        out = {str(i + 1): grown[k] for i, k in enumerate(roi)}
        members = {str(i + 1): [k] for i, k in enumerate(roi)}

    chip = chip_polygon(layout, local)
    ion_gds = ion_region_gds(layout, ion_box)
    if layout.simplify_keep is not None:
        from .simplify import simplify_far
        out = {n: simplify_far(g, ion_gds, layout.simplify_keep, layout.simplify_rel_tol)
               for n, g in out.items()}
    electrodes = {n: Electrode(n, as_multipolygon(layout.frame.transform_geometry(g)))
                  for n, g in out.items()}

    if layout.gap_model == "profile":
        from .fields import EdgeField, coarse_axes
        from .gaps import GapModel, correction_strips, merge_strip_edges
        model = GapModel(layout.metal_thickness, layout.ground_depth, layout.epsilon_r,
                         layout.strips_per_gap)
        axes = coarse_axes(ion_box, ion_z) if layout.strip_interpolation else None
        for n, idx in members.items():
            others = [local[i] for i in range(len(local)) if i not in idx]
            strips = correction_strips([local[i] for i in idx], others, model, layout.max_gap,
                                       chip_outline=chip, ion_region=ion_gds)
            strips = [(layout.frame.to_itvg(q), v) for q, v in strips]
            electrodes[n].corrections = strips
            if strips:
                electrodes[n].fields.append(EdgeField(*merge_strip_edges(strips), axes=axes))

    if layout.outer_ground_depth is not None:
        from .finitechip import FiniteChip
        # With a buried ground the chip's underside is 0 V; without one the chip is a thin
        # sheet seen from both sides (Schmied §III), which needs vacuum outside.
        if layout.ground_depth is None:
            if np.isfinite(layout.outer_ground_depth):
                raise ValueError("without a buried ground the finite chip needs vacuum outside: "
                                 "use outer_ground_depth=inf (--outer-ground-depth inf) or "
                                 "no finite chip (--no-finite-chip)")
            under = "mirror"
        else:
            under = "ground"
        fc = FiniteChip(chip, layout.outer_ground_depth, layout.frame, ion_box, ion_z,
                        under_chip=under)
        for n, idx in members.items():
            electrodes[n].fields.append(fc.correction(out[n]))
    if return_ground:
        gnd = {}
        if layout.electrodes and layout.ground:
            gnd = {n: as_multipolygon(layout.frame.transform_geometry(
                shapely.union_all([grown[i] for i in named[n]]))) for n in layout.ground}
        return electrodes, gnd
    if return_conductors:
        return electrodes, local
    return electrodes


def unit_potentials(electrodes, points, backend="auto"):
    """``{name: phi(points)}`` with each electrode at 1 V and all else grounded."""
    kernel = get_kernel(backend)
    pts = np.asarray(points, dtype=np.float64)
    if np.any(pts[:, 2] <= 0):
        raise ValueError("evaluation points must lie above the electrode plane (z > 0)")
    return {n: e.potential(pts, kernel) for n, e in electrodes.items()}


def grid_unit_potentials(electrodes, grid: GridSpec, backend="auto"):
    """Unit potentials on a full grid (x fastest). Returns ``{name: (N,) array}``.

    Fast path for the smooth additive fields (gap strips, finite chip): each is evaluated
    exactly on its own coarse grid and interpolated with separable cubic splines
    (``fields.tensor_cubic``). The strips use spacing z_min / 8; the finite chip, which
    varies on mm scales, uses its interpolation grid. The baseline polygons are evaluated
    exactly at every point.
    """
    from .fields import coarse_axes, tensor_cubic
    kernel = get_kernel(backend)
    pts = grid.points()
    if np.any(pts[:, 2] <= 0):
        raise ValueError("evaluation points must lie above the electrode plane (z > 0)")
    fine = grid.axes()
    default = coarse_axes((fine[0][0], fine[0][-1], fine[1][0], fine[1][-1]),
                          (fine[2][0], fine[2][-1]))

    def field_on_grid(f):
        ax = getattr(f, "axes", None) or default
        covers = all(a[0] <= g[0] + 1e-9 and a[-1] >= g[-1] - 1e-9 for a, g in zip(ax, fine))
        if not covers:
            ax = default
        ax = tuple(a if len(a) < len(g) else g for a, g in zip(ax, fine))
        zz, yy, xx = np.meshgrid(ax[2], ax[1], ax[0], indexing="ij")
        c = f.direct(np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]))
        return tensor_cubic(c.reshape(len(ax[2]), len(ax[1]), len(ax[0])), ax, fine).ravel()

    out = {}
    for n, e in electrodes.items():
        v = kernel(pts, e.rings())
        for f in e.fields:
            v = v + field_on_grid(f)
        out[n] = v
    return out

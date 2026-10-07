"""Command line: ``gds2itvg TRAP.gds --layout TRAP.json --out DIR``."""

import argparse
import logging
import sys
import time

import numpy as np

from .core import Layout, build_electrodes, grid_unit_potentials
from .frame import Frame
from .grid import GridSpec
from .itvg_io import write_batch


def _axis(text):
    try:
        a, b, c = (float(v) for v in text.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError("axis must be START:STOP:STEP (µm)")
    return a, b, c


def list_conductors(gds_path, layout):
    """Print conductors as a starting point for a layout file's seeds."""
    from . import gds
    from .core import _participants
    cond, ndup = gds.load_conductors(gds_path, layout.layer, layout.datatype, layout.cell)
    if layout.roi_box:
        sel, near = _participants(cond, layout.roi_box, layout.max_gap)
    else:
        sel, near = list(range(len(cond))), list(range(len(cond)))
    print(f"# layer {layout.layer}/{layout.datatype}: {len(cond)} conductors "
          f"({ndup} duplicate polygons removed); showing {len(near)}")
    print("# index  in_roi  xmin ymin xmax ymax (µm)  area (µm²)  seed [x, y]")
    for i in near:
        c = cond[i]
        seed = [round(v, 3) for v in c.representative_point().coords[0]]
        b = " ".join(f"{v:.1f}" for v in c.bounds)
        print(f"{i:6d}  {'yes' if i in sel else 'no ':>6}  {b}  {c.area:.0f}  {seed}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gds2itvg", description=__doc__)
    ap.add_argument("gds", help="input GDSII file")
    ap.add_argument("--layout", help="layout JSON (layer, ROI, frame, electrode names)")
    ap.add_argument("--layer", type=int, help="override/define the electrode layer")
    ap.add_argument("--datatype", type=int, default=None)
    ap.add_argument("--cell", default=None)
    ap.add_argument("--roi", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                    help="electrode ROI box in GDS µm (selects conductors, never clips)")
    ap.add_argument("--x", type=_axis, default=(-10, 10, 1), help="ion ROI x axis START:STOP:STEP (use --x=-10:10:1 for negative starts)")
    ap.add_argument("--y", type=_axis, default=(-800, 800, 1), help="ion ROI y axis")
    ap.add_argument("--z", type=_axis, default=(40, 60, 1), help="ion ROI z axis (height)")
    ap.add_argument("--gap-model", choices=["profile", "midline", "drawn"], default=None,
                    help="profile: cross-section gap profiles (default); midline: gapless "
                         "baseline; drawn: gaps at 0 V")
    ap.add_argument("--ground-depth", default=None,
                    help="buried ground depth below the surface (µm), or 'none'")
    ap.add_argument("--epsilon-r", type=float, default=None, help="dielectric constant")
    ap.add_argument("--metal-thickness", type=float, default=None, help="electrode thickness (µm)")
    ap.add_argument("--outer-ground-depth", type=float, default=None,
                    help="ground depth outside the chip outline (µm)")
    ap.add_argument("--simplify-keep", type=float, default=None,
                    help="keep geometry exact within this distance (µm) of the ion ROI and "
                         "simplify beyond it (default 200)")
    ap.add_argument("--no-simplify", action="store_true", help="keep all geometry exact")
    ap.add_argument("--no-finite-chip", action="store_true",
                    help="ground the surface outside the chip (M1 behaviour)")
    ap.add_argument("--max-gap", type=float, default=None, help="largest gap to split (µm)")
    ap.add_argument("--backend", default="auto", choices=["auto", "numpy", "numba"])
    ap.add_argument("--out", help="output directory (one ITVG batch)")
    ap.add_argument("--label-png", default=None,
                    help="write a PNG map of the electrodes and their grid-file numbers "
                         "(needs matplotlib); with --label-only nothing else is computed")
    ap.add_argument("--label-only", action="store_true", help="only write --label-png")
    ap.add_argument("--rf-null", metavar="NAME", nargs="+", default=None,
                    help="report the RF null (pseudopotential minimum) for these RF electrode "
                         "numbers at the axial positions of --y (start, centre, end) and exit")
    ap.add_argument("--list-conductors", action="store_true",
                    help="print the conductors on the layer (index, bbox, seed point) and exit; "
                         "with --roi only those it selects or that border them")
    ap.add_argument("--stub", default="Electrode-", help="file name stub (no digits)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing batch/cache")
    ap.add_argument("--write-ground", action="store_true",
                    help="also write all-zero files for the layout's ground electrodes (e.g. 16)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    layout = Layout.from_json(a.layout) if a.layout else Layout(frame=Frame())
    if a.layer is not None:
        layout.layer = a.layer
    if a.datatype is not None:
        layout.datatype = a.datatype
    if a.cell is not None:
        layout.cell = a.cell
    if a.roi is not None:
        layout.roi_box = tuple(a.roi)
    if a.gap_model:
        layout.gap_model = a.gap_model
    if a.max_gap is not None:
        layout.max_gap = a.max_gap
    if a.ground_depth is not None:
        layout.ground_depth = None if a.ground_depth.lower() == "none" else float(a.ground_depth)
    if a.epsilon_r is not None:
        layout.epsilon_r = a.epsilon_r
    if a.metal_thickness is not None:
        layout.metal_thickness = a.metal_thickness
    if a.outer_ground_depth is not None:
        layout.outer_ground_depth = a.outer_ground_depth
    if a.no_finite_chip:
        layout.outer_ground_depth = None
    if a.simplify_keep is not None:
        layout.simplify_keep = a.simplify_keep
    if a.no_simplify:
        layout.simplify_keep = None

    if a.list_conductors:
        return list_conductors(a.gds, layout)
    if a.label_only and not a.label_png:
        ap.error("--label-only needs --label-png")
    if not a.out and not a.label_only and not a.rf_null:
        ap.error("--out is required")

    t0 = time.perf_counter()
    grid = GridSpec(a.x, a.y, a.z)
    ion_box = (a.x[0], a.x[1], a.y[0], a.y[1])
    if a.label_only:
        layout.gap_model, layout.outer_ground_depth = "midline", None  # geometry only
    electrodes, ground = build_electrodes(a.gds, layout, ion_box=ion_box,
                                          ion_z=(a.z[0], a.z[1]), return_ground=True)
    if a.rf_null:
        from .analysis import rf_null
        ys = sorted({a.y[0], 0.5 * (a.y[0] + a.y[1]), a.y[1]})
        print(f"# RF null of electrode(s) {' '.join(a.rf_null)} (ITVG µm; field = |grad phi| per V)")
        for yv in ys:
            r = rf_null(electrodes, a.rf_null, y=yv, backend=a.backend)
            print(f"y = {yv:9.2f}:  x = {r['x']:8.3f}  z = {r['z']:8.3f}  "
                  f"field = {r['field']:.2e} V/µm  phi_RF = {r['phi']:.4f}")
        return 0
    if a.label_png:
        from .plot import label_map
        label_map(electrodes, a.label_png, ground=ground, ion_box=ion_box,
                  title=f"{a.gds}: electrode numbers = grid-file numbers")
        print(f"label map -> {a.label_png}", file=sys.stderr)
        if a.label_only:
            return 0
    pts = grid.points()
    t1 = time.perf_counter()
    pot = grid_unit_potentials(electrodes, grid, a.backend)
    if a.write_ground:
        for name in layout.ground:
            pot[name] = np.zeros(len(pts))
    t2 = time.perf_counter()
    paths = write_batch(a.out, grid, pot, stub=a.stub, force=a.force)
    t3 = time.perf_counter()
    print(f"{len(paths)} electrodes x {len(pts)} points -> {a.out}  "
          f"(geometry {t1 - t0:.1f} s, potentials {t2 - t1:.1f} s, write {t3 - t2:.1f} s)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

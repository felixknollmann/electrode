"""Electrode label map: which grid-file number belongs to which electrode.

Draws the electrodes in the ITVG frame (the coordinates of the grid files), with:

- each electrode labelled by its grid-file number;
- ground electrodes hatched;
- the ion grid region outlined.

The left panel shows the whole trap and the right panel the region around the ion grid.
Useful for checking a layout file before generating, or for testing an existing ITVG batch
against the drawing. Needs matplotlib (``pip install gds2itvg[plot]``).
"""

import numpy as np
import shapely
from shapely.geometry import box

from .geometry import as_multipolygon

__all__ = ["label_map"]


def _patches(ax, geom, **kw):
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    for poly in as_multipolygon(geom).geoms:
        verts, codes = [], []
        for ring in [poly.exterior, *poly.interiors]:
            c = np.asarray(ring.coords)
            verts.extend(c)
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(c) - 2) + [Path.CLOSEPOLY])
        ax.add_patch(PathPatch(Path(verts, codes), **kw))


def _label(ax, geom, text, window, **kw):
    part = geom.intersection(window)
    if part.is_empty:
        return
    # put the label in the largest visible piece
    pieces = sorted(as_multipolygon(part).geoms, key=lambda g: -g.area)
    if not pieces:
        return
    p = pieces[0].representative_point()
    ax.text(p.x, p.y, text, ha="center", va="center", **kw)


def label_map(electrodes, path, ground=None, ion_box=None, title=None, dpi=150):
    """Write a PNG label map.

    ``electrodes``: ``{name: Electrode}`` (ITVG frame). ``ground``: ``{name: geometry}`` of
    ground conductors (ITVG frame), optional. ``ion_box``: (xmin, xmax, ymin, ymax).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ground = ground or {}
    names = sorted(electrodes, key=lambda n: (len(n), n))
    cmap = plt.get_cmap("tab20")
    # tab20 comes in dark/light pairs of one hue: take all dark hues first, then the light
    # ones, so neighbouring numbers differ in hue. Greys (14, 15) are kept for ground.
    palette = [cmap(i) for i in list(range(0, 20, 2)) + list(range(1, 20, 2)) if i not in (14, 15)]
    colours = {n: palette[i % len(palette)] for i, n in enumerate(names)}
    allgeom = shapely.union_all([e.geometry for e in electrodes.values()] + list(ground.values()))
    bx = allgeom.bounds

    if ion_box is not None:
        x0, x1, y0, y1 = ion_box
        padx = max(150.0, 5 * (x1 - x0))
        pady = max(100.0, 0.15 * (y1 - y0))
        zoom = (x0 - padx, x1 + padx, y0 - pady, y1 + pady)
    else:
        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        zoom = (cx - 200, cx + 200, cy - 1000, cy + 1000)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, (vx0, vx1, vy0, vy1), fs, equal in (
            (axes[0], (bx[0], bx[2], bx[1], bx[3]), 7, True),
            (axes[1], zoom, 8, False)):
        window = box(vx0, vy0, vx1, vy1)
        for n in names:
            _patches(ax, electrodes[n].geometry, facecolor=colours[n], edgecolor="k",
                     linewidth=0.3, alpha=0.75)
        for n, g in ground.items():
            _patches(ax, g, facecolor="0.85", edgecolor="0.3", hatch="///", linewidth=0.3)
        for n in names:
            _label(ax, electrodes[n].geometry, n, window, fontsize=fs, weight="bold",
                   bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        for n, g in ground.items():
            _label(ax, g, f"{n} (gnd)", window, fontsize=fs - 1, color="0.2")
        if ion_box is not None:
            x0, x1, y0, y1 = ion_box
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red",
                                       lw=1.2, ls="--", zorder=5))
        ax.set_xlim(vx0, vx1)
        ax.set_ylim(vy0, vy1)
        if equal:
            ax.set_aspect("equal")
        ax.set_xlabel("ITVG x (µm)")
        ax.set_ylabel("ITVG y (µm)")
    axes[0].set_title("whole trap (ITVG frame)")
    axes[1].set_title("around the ion grid (red dashed); x stretched")
    fig.suptitle(title or "Electrode numbers = grid-file numbers")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path

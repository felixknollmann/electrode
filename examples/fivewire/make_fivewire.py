"""Generate the example five-wire surface trap (``fivewire.gds`` + ``fivewire.json``).

Five-wire geometry along the trap axis (GDS x), all gaps 4 µm:

    y (µm)
     +620 ┌──────┬────┬────┬────┬────┬────┬──────┐
          │  11  │ 1  │ 2  │ 3  │ 4  │ 5  │  12  │   DC: end caps + 5 fingers (120 µm)
     +139 ├──────┴────┴────┴────┴────┴────┴──────┤
          │                 16 (RF)              │   RF rail, 100 µm
      +35 ├──────────────────────────────────────┤
          │                 15                   │   centre DC rail, 62 µm
      -35 ├──────────────────────────────────────┤
          │                 16 (RF)              │   RF rail, 100 µm
     -139 ├──────┬────┬────┬────┬────┬────┬──────┤
          │  13  │ 6  │ 7  │ 8  │ 9  │ 10 │  14  │
     -620 └──────┴────┴────┴────┴────┴────┴──────┘
          -1500   x: fingers centred at 0, ±124, ±248              +1500

Electrode 16 (RF) is the two RF rails, joined by two seed points in the layout file. A
grounded guard ring (17, 0 V) surrounds the trap across a 4 µm gap.

Run: ``python examples/fivewire/make_fivewire.py`` (writes next to this script).
"""

import json
import os

import gdstk

HERE = os.path.dirname(os.path.abspath(__file__))
G = 4.0          # gap
F = 120.0        # finger width (along the axis)
L = 1500.0       # half length of the trap
C = 62.0         # centre DC rail width
R = 100.0        # RF rail width
D = 481.0        # depth of the outer DC electrodes (perpendicular to the axis)
GUARD = 200.0    # width of the grounded guard ring


def rect(x0, y0, x1, y1):
    return gdstk.rectangle((x0, y0), (x1, y1), layer=1, datatype=0)


def build():
    cells, seeds, ground = [], {}, {}
    yc = C / 2                        # centre rail: |y| <= 31
    yr0, yr1 = yc + G, yc + G + R     # RF rail: 35 .. 135
    yd0, yd1 = yr1 + G, yr1 + G + D   # outer DC: 139 .. 620
    cells.append(rect(-L, -yc, L, yc))
    seeds["15"] = [[0.0, 0.0]]
    for s in (1, -1):
        cells.append(rect(-L, s * yr0, L, s * yr1) if s > 0 else rect(-L, -yr1, L, -yr0))
    seeds["16"] = [[0.0, (yr0 + yr1) / 2], [0.0, -(yr0 + yr1) / 2]]
    centres = [-2 * (F + G), -(F + G), 0.0, F + G, 2 * (F + G)]
    for side, (y0, y1), first, (left, right) in ((1, (yd0, yd1), 1, ("11", "12")),
                                                 (-1, (-yd1, -yd0), 6, ("13", "14"))):
        for k, xc in enumerate(centres):
            cells.append(rect(xc - F / 2, y0, xc + F / 2, y1))
            seeds[str(first + k)] = [[xc, (y0 + y1) / 2]]
        xe = centres[-1] + F / 2 + G
        cells.append(rect(-L, y0, -xe, y1))
        cells.append(rect(xe, y0, L, y1))
        seeds[left] = [[-(xe + L) / 2, (y0 + y1) / 2]]
        seeds[right] = [[(xe + L) / 2, (y0 + y1) / 2]]
    outer = gdstk.rectangle((-L - G - GUARD, -yd1 - G - GUARD), (L + G + GUARD, yd1 + G + GUARD))
    inner = gdstk.rectangle((-L - G, -yd1 - G), (L + G, yd1 + G))
    cells += gdstk.boolean(outer, inner, "not", layer=1, datatype=0)
    ground["17"] = [[0.0, yd1 + G + GUARD / 2]]
    return cells, seeds, ground


def main():
    cells, seeds, ground = build()
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    top = lib.new_cell("FIVEWIRE")
    for c in cells:
        top.add(c)
    lib.write_gds(os.path.join(HERE, "fivewire.gds"))
    order = sorted(seeds, key=int)
    layout = {
        "_comment": "Example five-wire trap (examples/fivewire/make_fivewire.py). Trap axis along "
                    "GDS x -> ITVG y; ITVG x = -GDS y. Device stack defaults: 1 um metal, SiO2 "
                    "(eps 3.9), ground 10 um below; outside the chip, ground 1 mm down.",
        "layer": 1, "datatype": 0,
        "roi_box": [-L, -D - 200, L, D + 200],
        "frame": {"origin": [0.0, 0.0], "quarter_turns": 1, "mirror": False},
        "max_gap": 30.0,
        "electrodes": {k: seeds[k] for k in order},
        "ground": ground,
    }
    with open(os.path.join(HERE, "fivewire.json"), "w") as f:
        json.dump(layout, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()

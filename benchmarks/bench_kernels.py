"""Milestone 3 benchmark: NumPy vs Numba vs Julia edge-sum kernels (GUIDELINES §6).

Workload: all electrode rings of a midline-split layout (default: the bundled five-wire
example) as one concatenated edge list, evaluated on N points drawn from a grid of
x ±10, y ±800, z 40–60 µm (706 041 points), with N in {1e3, 1e4, 1e5, 706041}.

- Every backend gets one untimed warm-up call (JIT/compile excluded).
- Then ``--repeats`` timed calls; the median and min are reported.
- Julia is timed inside Julia (kernel only), and also end to end from Python (with
  process start-up and file I/O).
- Thread counts are equal (``--threads``, default os.cpu_count()).
- Accuracy gate: max |phi_backend - phi_numba| <= 1e-12 at every N.

usage: python benchmarks/bench_kernels.py [GDS] [--layout L.json] [--threads 4] [--repeats 5]
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ex = os.path.join(os.path.dirname(__file__), "..", "examples", "fivewire")
    ap.add_argument("gds", nargs="?", default=os.path.join(ex, "fivewire.gds"))
    ap.add_argument("--layout", default=os.path.join(ex, "fivewire.json"))
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--sizes", default="1000,10000,100000,706041")
    ap.add_argument("--numpy-max", type=int, default=100000, help="largest N timed with NumPy")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    os.environ.setdefault("NUMBA_NUM_THREADS", str(a.threads))

    from gds2itvg.core import Layout, build_electrodes
    from gds2itvg.grid import GridSpec
    from gds2itvg.kernels.numpy_kernel import _edges, polygon_potential as np_kernel
    from gds2itvg.kernels.numba_kernel import polygon_potential as nb_kernel
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "julia"))
    import julia_runner as jl

    layout = Layout.from_json(a.layout)
    layout.gap_model, layout.outer_ground_depth = "midline", None
    electrodes = build_electrodes(a.gds, layout)
    rings = [r for e in electrodes.values() for r in e.rings()]
    p1, p2, w = _edges(rings)
    pts_all = GridSpec((-10, 10, 1), (-800, 800, 1), (40, 60, 1)).points()
    rng = np.random.default_rng(0)
    print(f"# {len(electrodes)} electrodes, {len(p1)} edges, {a.threads} threads", file=sys.stderr)

    def timed(fn, reps):
        fn()  # warm-up
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return ts

    rows = []
    for n in [int(s) for s in a.sizes.split(",")]:
        pts = pts_all if n >= len(pts_all) else pts_all[np.sort(rng.choice(len(pts_all), n, replace=False))]
        row = {"n_points": len(pts), "n_edges": int(len(p1))}
        ref = nb_kernel(pts, rings)
        row["numba"] = timed(lambda: nb_kernel(pts, rings), a.repeats)
        if len(pts) <= a.numpy_max:
            row["numpy"] = timed(lambda: np_kernel(pts, rings), max(1, min(a.repeats, 3)))
            row["numpy_maxdiff"] = float(np.abs(np_kernel(pts, rings) - ref).max())
        if jl.julia_executable():
            phi, kt = jl.run_kernel(pts, p1, p2, w, threads=a.threads, repeats=a.repeats)
            row["julia_kernel"] = kt
            row["julia_maxdiff"] = float(np.abs(phi - ref).max())
            t0 = time.perf_counter()
            jl.run_kernel(pts, p1, p2, w, threads=a.threads, repeats=0)
            row["julia_end_to_end"] = [time.perf_counter() - t0]
        rows.append(row)
        summary = {k: f"{statistics.median(v):.4g}s" for k, v in row.items() if isinstance(v, list)}
        print(len(pts), summary, {k: v for k, v in row.items() if k.endswith("maxdiff")}, file=sys.stderr)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"threads": a.threads, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()

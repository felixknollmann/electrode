# GUIDELINES: design of gds2itvg

This document covers the package's design: physics, architecture, conventions, validation
approach, assumptions and limitations. Measured results are in [RESULTS.md](RESULTS.md).

## 1. Problem

The inputs are a GDSII layout of a surface-electrode trap, a selection of its electrodes,
and an ion grid. The output is one unit-potential grid file per electrode: that electrode
at 1 V, everything else at 0 V, in the format ITVG's `VoltageGenerator` reads.

## 2. Physics

### 2.1 Kernel

For a potential V(x′, y′) prescribed on the plane z = 0, the potential above is the
Poisson integral

  φ(r) = (1/2π) ∫ V(r′) z / |r − r′|³ dA′.

For a polygon at unit potential this is Ω/2π, its solid angle divided by 2π (Wesenberg
2008, Eqs. 3–5). As a sum over directed edges p₁ → p₂:

  Δφ = (1/π) atan( z (x₁y₂ − y₁x₂) / ( |z| (r₁r₂ + x₁x₂ + y₁y₂ + |z|(|z| + r₁ + r₂)) ) ),

with x_i = x − p_i,x, y_i = y − p_i,y and r_i = |r − p_i|. Exterior rings are
counter-clockwise and holes clockwise. The kernel accepts a weight per edge, so any
piecewise-constant surface potential is a weighted edge set.

### 2.2 Gapless model (`--gap-model midline --no-finite-chip`)

- Every gap narrower than `max_gap` is split along its midline. Each electrode is grown to
  the gap centre lines by an exact distance-level partition (`midline.py`).
- The surface outside the electrodes is at 0 V, out to infinity.
- Openings wider than `max_gap` stay at 0 V.

### 2.3 Device-stack model (default)

The midline geometry is the skeleton. Two corrections are added.

**(a) Gap profiles** (`crosssection.py`, `gaps.py`).

- A 2-D finite-volume solve of the gap cross-section gives the surface potential q_g(u)
  across a gap of width g. The cross-section has metal of thickness t, a dielectric ε_r
  filling the gap to the surface and continuing under the metal, a buried ground at depth
  h (optional) and vacuum above.
- Without a ground and with thin metal, q_g reduces to Schmied 2010, Eq. 6.
- With a ground, the gap sags toward 0 V. A ground 10 µm below screens sideways over about
  10 µm, so the profile is a local property of the cross-section, and Schmied's long-range
  polarization term is not used.
- Edges facing an opening get the single-edge fringe q_∞(u), which has a 1/u tail
  truncated at 10 h.
- The correction Δ = q − (midline step) is laid along each electrode edge as strips. The
  facing distance comes from an outward-normal probe. Strips far from the ions are lumped
  with the same integral.
- Strips are merged into one weighted edge set: shared edges are combined, with their
  weights subtracted.

**(b) Finite chip** (`finitechip.py`).

- Outside `chip_outline` the surface carries an unknown potential ψ. Above it is vacuum;
  below it is vacuum down to an outer ground at depth D, or no ground at all.
- No surface charge is allowed there:

  2·A[ψ] + S_D[ψ] = −A[1_e],

  with A the half-space normal-derivative operator and S_D the slab image series.
- Under the chip the buried ground makes the lower side 0 V. Without a buried ground, the
  chip is a thin sheet seen from both sides (Schmied §III), which needs D = ∞.
- Discretisation: piecewise-constant panels on a graded quadtree, solved with one LU, with
  Richardson extrapolation over two meshes.

### 2.4 Evaluation

- Baseline polygons are evaluated exactly at every grid point.
- The smooth additive fields (merged strips, finite chip) are evaluated exactly on coarse
  grids and interpolated with separable cubic splines:
  - strips: spacing z_min/8;
  - finite chip: 5 × 41 × 5, with a panel response matrix shared by all electrodes.
- Geometry farther than 200 µm from the ion grid is simplified (Douglas–Peucker) with
  tolerance min(1 µm, 1 % of the distance). That stays below typical gap widths, so gap
  junctions are kept.

## 3. Architecture

```
src/gds2itvg/
  gds.py           GDS read (gdstk): DB units -> µm, paths -> polygons, flatten references,
                   de-duplicate, union into conductors, ROI selection, naming by seed points
  midline.py       exact midline split of gaps
  crosssection.py  2-D gap cross-section solver -> gap / opening profiles
  gaps.py          profile-correction strips along electrode edges; strip merging
  finitechip.py    finite-chip surface-potential solver
  simplify.py      distance-graded simplification far from the ions
  fields.py        coarse-grid fields and separable cubic interpolation
  kernels/         NumPy and Numba edge-sum kernels (weighted edges, response matrices)
  frame.py         GDS <-> ITVG frame (quarter turns, mirror, origin)
  grid.py          regular grids in ITVG order
  core.py          Layout (layout file) and the pipeline
  itvg_io.py       ITVG grid-file writer/reader
  plot.py          electrode / grid-file-number label maps
  analysis.py      RF null finder
  cli.py           command line
examples/fivewire/ generated example trap (GDS, layout, label map)
benchmarks/        kernel benchmark; julia/ = Julia comparison kernel (benchmark only)
tests/             pytest
```

## 4. Conventions

- **Units:** lengths in µm (float64), potentials in volts per volt applied. Any GDS database
  unit is converted.
- **Frames:** the ITVG frame is `R(quarter_turns·90°)·M·(p_gds − origin)`, with z the
  height. The trap axis is normally ITVG y.
- **Selection:** the ROI box selects conductors and never clips them. Conductors within
  `max_gap` of the selection also take part.
- **Polygons:** identical duplicate polygons are removed, and the polygons of one conductor
  are merged. `fill` adds metal that is not in the GDS.
- **Names:** electrode numbers come from seed points. Ground electrodes are 0 V and get no
  file unless `--write-ground` is given.
- **Output:**
  - tab-separated `x y z V`, with x fastest, then y, then z;
  - a uniform grid, with full float64 precision;
  - files named `<stub><n>.txt`, where the stub has no digits, because ITVG recovers n
    with `str.lstrip(stub)`;
  - one batch per directory, and a stale `savefields.dat` is refused unless `--force`.

## 5. Validation approach

The tests are in `tests/`, and the numbers in RESULTS.md.

- **Kernel:**
  - against the closed-form rectangle potential;
  - against physics limits: surface values, far field, superposition, rotation, holes,
    z-parity and dφ/dz;
  - against nist-ionstorage/electrode on its five-wire layout, gapless and gapped (optional
    test).
- **Midline split:** exact midlines for uniform gaps, a partition without overlap,
  partition of unity, and the `max_gap` cut-off.
- **Cross-section solver:** the vacuum, thin-metal limit equals Schmied Eq. 6; grid
  convergence of the integrated sag; trends with gap width and ground depth.
- **Strips:** a long straight gap must equal the 2-D Poisson integral of the profile, and
  merging must not change the potential.
- **Finite chip:** Schmied Eq. 29 (disk electrode on a finite disk, vacuum), and the limit
  D → 0.
- **Speed settings:** simplification, merging and interpolation against exact evaluation.
- **End to end:** the bundled five-wire example. Its grid files must be mirror-symmetric
  between facing fingers, and its RF null must be correct. There is also an ITVG
  `VoltageGenerator` round-trip (optional test).

## 6. Assumptions and limitations

- **Thin, coplanar electrodes:** metal thickness enters only through the gap
  cross-section.
- **The gap model is local and 2-D.** Corners (O(g²) wedges) are not treated. Long-range
  gap polarization is neglected; it is screened when there is a buried ground.
- **The chip outline defaults to the bounding box of the trap metal.** Other metal outside
  it is treated as part of the outer surface.
- **No cover electrode**, and the evaluation points must have z > 0.

## 7. Julia backend (decision)

A Julia version of the kernel, with the same loop and `Base.Threads` only, was benchmarked
against Numba at equal accuracy. It was not more than 2× faster (0.8–1.2×), so it is not
part of the package. The kernel and its runner stay in `benchmarks/julia/`; see
RESULTS.md.

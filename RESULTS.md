# RESULTS

These are validation and performance results for gds2itvg 0.1.0. The test suite
reproduces them (`pytest -q`), together with `benchmarks/bench_kernels.py`.

Machine: 4-core Linux container, Python 3.13, NumPy 2.5, Numba 0.68.

## Kernel

**Closed-form rectangle.**

- 24 000 points across three rectangles (aspect ratios 1–2000): near field, above edges
  and corners, far field, negative z.
- The max absolute error is 1.9 × 10⁻¹⁵, and ≤ 2 × 10⁻¹⁵ relative to each set's max |φ|.
  NumPy and Numba agree to ≤ 10⁻¹⁴.
- The physics limits pass:
  - surface values 1, ½, ¼, 0;
  - the far-field monopole;
  - superposition, rotation, and holes;
  - odd parity in z;
  - dφ/dz at z → 0⁺.

**Against nist-ionstorage/electrode** on the five-wire tutorial layout, scaled to 50 µm
ion height, evaluated on 1053 points:

| layout | max abs | max rel |
|---|---|---|
| gapless | 2.8 × 10⁻¹⁶ | 8.7 × 10⁻¹⁶ |
| 5 µm gaps, midline-split by gds2itvg, evaluated by both packages | 9.4 × 10⁻¹⁶ | 2.1 × 10⁻¹⁵ |

## Device-stack model

| test | result |
|---|---|
| Cross-section solver, vacuum, no ground, thin metal vs Schmied Eq. 6 | max error 1.7 × 10⁻³ inside a 4 µm gap, converging with the grid; the profile's integral is exact |
| Integrated gap sag with the buried ground, grid convergence | < 0.1 % change on a 4× finer grid |
| 3-D strips vs the 2-D Poisson integral of the profile (long straight gap) | 0.5 % of the correction (16 strips) |
| Strip merging (shared edges combined) | potential unchanged to ≤ 10⁻¹²; about 20 % fewer edges |
| Finite chip vs Schmied Eq. 29 (disk electrode on a finite disk, vacuum, on axis) | within 2 % (Richardson-extrapolated mesh) |
| Finite chip, outer ground depth → 0 | recovers the grounded plane |

**Size of the corrections** for the default stack (1 µm metal, SiO₂ ε_r = 3.9, ground 10 µm
below):

- **Gap sag:** the gap's integrated potential, compared with the midline step, changes by
  −0.10 µm·V (4 µm gap), −0.31 µm·V (6 µm) and −4.25 µm·V (20 µm). Electrodes with long
  gaps close to the ions (DC rails, centre electrodes) drop by a few 10⁻³ of their
  potential.
- **Opening fringe:** an edge facing an opening fringes out like q ≈ (h/ε_r)/(πu). This is
  equivalent to about 7.5 µm of extra metal; it is truncated at 10 h.
- **Finite chip** (≈10 mm chip, ground 1 mm down): a nearly uniform vertical field of a
  few 10⁻⁶ /µm per volt for large electrodes. It affects the field terms much more than
  the curvatures.

## Example trap (`examples/fivewire`)

The example is a five-wire trap: five 120 µm fingers per side, 4 µm gaps, a 62 µm centre
rail, 100 µm RF rails, and a grounded guard ring.

- **RF null** (`--rf-null 16`): x = 0, z = 67.2 µm (device stack) and 67.0 µm (gapless).
  It varies by < 0.1 µm over ±400 µm along the axis.
- **Mirror symmetry** between the facing fingers 3 and 8: 10⁻¹⁶ with exact geometry, and
  6 × 10⁻⁶ V per V with the default far-field simplification.
- **Run time**, 16 electrodes × 353 241 points, end to end: 12 s (device stack), 8.5 s
  (gapless).

## Performance settings

The following were measured on a 30-electrode, 10 mm production trap (6074 midline edges)
on a 706 041-point grid.

| setting | effect on the potentials | max change vs exact |
|---|---|---|
| Far-field simplification (keep 200 µm, tol ≤ 1 µm) | gapless 30.5 s → 9.5 s | 4.7 × 10⁻⁶ V per V |
| Strip merging + coarse-grid fields with separable cubic interpolation + shared finite-chip response | device stack 77 s → 14 s | 1.6 × 10⁻⁷ V per V |

End to end, the same trap takes 75 s with the device stack and 40 s gapless. The
geometry setup (about 40 s: finite-chip solve and strip construction) and file writing
(about 20 s) now dominate.

## Kernel benchmark: NumPy vs Numba vs Julia

The same 6074-edge layout was evaluated in one call on N points, with 4 threads per
backend. Each backend had one untimed warm-up call (JIT excluded). Times are medians of 5
runs (3 for NumPy). The Julia kernel time is measured inside Julia.

| points | NumPy | Numba | Julia (kernel) | Julia (end to end) | Julia / Numba speed | max \|Julia − Numba\| |
|---|---|---|---|---|---|---|
| 1 000 | 0.568 s | 0.0358 s | 0.0440 s | 0.68 s | 0.81× | 6.7 × 10⁻¹⁶ |
| 10 000 | 4.61 s | 0.454 s | 0.382 s | 0.98 s | 1.19× | 8.9 × 10⁻¹⁶ |
| 100 000 | 53.0 s | 3.78 s | 4.02 s | 4.47 s | 0.94× | 8.9 × 10⁻¹⁶ |
| 706 041 | — | 26.0 s | 26.9 s | 28.2 s | 0.97× | 1.1 × 10⁻¹⁵ |

Julia is not more than 2× faster than Numba, so it is not part of the package. Numba is
about 14× faster than vectorised NumPy and is the default.

## Open issues

- Gap corners, O(g²), are not treated, and long-range gap polarization is not modelled
  without a buried ground.
- Geometry setup is now the slowest stage of the device-stack model. The finite-chip
  matrix assembly and the strip construction could be parallelised.
- Writing the text files takes about 20 s for 30 × 706 041 points.

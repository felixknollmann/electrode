# Changelog

## 0.1.0

First release.

**GDS import**

- Database-unit conversion; paths become polygons and references are flattened.
- Duplicate polygons are removed, and holes are supported.
- An ROI box selects conductors without clipping them; the conductors that border the
  trap across a gap are included too.
- Electrodes are named via seed points.
- `fill` adds metal that is not in the GDS.
- `--list-conductors` lists a layer's conductors.

**Models**

- Gapless: an exact midline split of the gaps, with the surface grounded to infinity.
- Device stack (default): gap profiles from a 2-D cross-section (metal thickness,
  dielectric, optional buried ground) and a finite-chip surface potential.

**Performance**

- NumPy and Numba edge-sum kernels with per-edge weights.
- Far-field simplification.
- Merged correction strips.
- Coarse-grid fields with separable cubic interpolation.

**Output and tools**

- An ITVG grid-file writer that enforces `VoltageGenerator`'s rules.
- `--label-png`: electrode / grid-file-number maps.
- `--rf-null`: the RF null finder.

**Example and benchmark**

- `examples/fivewire`: a generated five-wire trap with five 120 µm fingers per side.
- Julia kernel benchmark: not > 2× faster than Numba, so not included.

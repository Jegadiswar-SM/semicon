# Wafer Pattern Localization Report

## Methodology

The generator defines an illustrative FinFET-like intensity function in
continuous nanometer coordinates. It contains periodic vertical fins and one
or two horizontal gate bars. The default 34 nm fin pitch is an illustrative
public-domain 10 nm-class value, not proprietary process or PDK data.

Reference and search images are sampled independently from the same function:
the reference is 1000×1000 pixels at 1 nm/px and the search is 1000×1000 pixels
at 10 nm/px. The search image is never made by resizing the reference. Each
capture receives independent signal-dependent Poisson shot noise and
signal-independent Gaussian read noise; the search capture has the lower
photon budget and higher read noise. A gradient-magnitude map of the clean
pattern is added after noise to approximate SEM edge brightening.

The localizer in `localize.py` is classical normalized cross-correlation only:

1. Convert inputs to normalized grayscale.
2. Downsample the reference to 100×100 with `cv2.INTER_AREA`.
3. Upsample search and template by the configurable factor (default 4) using
   cubic interpolation and calculate `TM_CCOEFF_NORMED`.
4. Find distinct 3×3 local maxima at or above the configurable fraction of
   the global maximum (default 0.98).
5. Select the candidate whose footprint center is closest to the search-image
   center, as required by the problem rule.
6. Refine the selected upsampled peak with 3×3 parabolic interpolation and map
   it back to native search-pixel coordinates.

Image loading is outside the `time.perf_counter` region. The CLI emits a JSON
object containing coordinates, confidence, timing, and candidates.

## Evaluation

`evaluate.py` generated 30 reproducible pairs with varying asymmetric noise and
periodic stress cases. The primary configuration was `upsample=4`,
`threshold_ratio=0.98`, and `TOLERANCE_PX=1.0`.

- Hits: 14/30
- Hit rate: 46.67%
- Mean computation time: 855.020 ms
- Standard deviation: 278.330 ms
- Results: `evaluation_results/results.json`
- Failure plot: `evaluation_results/failure_case.png`

The evaluator also sweeps four threshold ratios (`0.90`, `0.95`, `0.98`,
`0.995`) and four upsample factors (`1`, `2`, `4`, `8`). Correlation surfaces
are computed once per pair/factor and reused across thresholds, so threshold
comparisons do not repeat the expensive NCC operation.

The best measured accuracy was 17/30 (56.67%) at 4×/0.995. The 8×/0.995
configuration also reached 17/30 but averaged about 3038 ms, versus about 703
ms for the cached 4× sweep. The primary threshold remains 0.98 because it is
the specified near-tie rule; the sweep exposes the accuracy/speed tradeoff.

The current Phase 1 generator places valid footprints across the search field,
so the report records `placement_offset_nm` as a drift proxy rather than
claiming an unimplemented Gaussian stage-drift model. The bucketed results are
stored in `results.json`.

## Failure case

The recorded first failure has:

- Ground truth: `(472.800, 232.923)`
- Prediction: `(533.999, 232.939)`
- Error: `61.199 px`

The correlation surface contains many nearly tied peaks because the fin
pattern repeats. Appearance-only template matching cannot identify which
repeated period is the physically correct one. The mandated center tiebreak
therefore selects the candidate closest to the search center, which can be a
different period. This is a structural limitation of the measurement and
matching architecture, not an implementation bug.

## Citations

### Noise model

- Foi et al., *Practical Poissonian-Gaussian noise modeling and fitting for
  single-image raw-data*, IEEE TIP (2008).
- Kockentiedt et al., *Poisson shot noise parameter estimation from a single
  scanning electron microscopy image*, SPIE proceedings.

### SEM edge effect

- JEOL, *edge effect | Glossary of SEM Terms*.
- ETH Zurich ScopeM, *SEM - Imaging with Secondary Electrons*.
- Nanoscience Instruments, *Secondary Electrons in SEM: Unlocking Surface
  Insights at the Nanoscale*.

### Geometry rationale

- Cadence Breakfast Bytes, *Intel 10nm*.
- Semiconductor Engineering, *The Race To 10/7nm*.
- Tsu-Jae King Liu, *FinFET History, Fundamentals and Future*, UC Berkeley
  VLSI short-course slides.

Full URLs are listed in [docs/citations.md](citations.md).

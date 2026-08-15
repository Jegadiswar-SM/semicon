# Wafer Pattern Localization Report

## Methodology

The dataset generator creates a synthetic FinFET-style pattern as a continuous function of physical coordinates in nanometers. The pattern contains repeated vertical fins, plus one or two horizontal gate bars. Fin pitch is set to an illustrative `34 nm`, a public 10 nm-class FinFET value reported in multiple public sources; this is not proprietary PDK data and is used only for a synthetic benchmark.

`reference.png` and `search.png` are rasterized independently from the same continuous pattern function. The reference uses `1 nm/px` over `1000x1000 px`; the search image uses `10 nm/px` over `1000x1000 px`. The generator never creates the search image by resizing the reference.

Noise is added independently after rasterization. The SEM-mode model combines signal-dependent Poisson shot noise with signal-independent Gaussian read/electronic noise. The search image is deliberately noisier than the reference image. Edge brightening is added from the clean-pattern gradient magnitude to mimic increased secondary-electron yield near edges and sidewalls.

The localizer is classical normalized cross-correlation only:

1. Downsample the `1000x1000` reference to a `100x100` template with `cv2.INTER_AREA`.
2. Run `cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)`.
3. Find local maxima at `>= 0.98 * global_max`.
4. Choose the candidate closest to the center of the search image.
5. Convert the selected top-left match location to a footprint center.
6. Refine the selected peak with parabolic interpolation in the 3x3 neighborhood.

No deep learning model is used.

## Metrics

Latest command:

```bash
python3 evaluate.py
```

Results:

- Evaluation pairs: `36`
- Hit tolerance: `TOLERANCE_PX = 1.0`
- Hits: `14/36`
- Hit rate: `38.89%`
- Mean computation time: `11.738 ms`
- Std computation time: `0.616 ms`
- Results file: `evaluation_results/results.json`
- Failure plot: `evaluation_results/failure_case.png`

Timing uses `time.perf_counter` around the matching algorithm and excludes image loading/saving.

## Failure Case

Concrete failure from `evaluation_results/results.json`:

- Ground truth center: `(472.800, 232.923)`
- Predicted center: `(500.001, 232.944)`
- Error: `27.201 px`
- Confidence: `0.959246`

The failure is structural, not an implementation bug. Periodic fin patterns produce multiple near-identical normalized-correlation peaks separated by the fin pitch in search pixels. The required center-tiebreak heuristic has no image evidence to identify which repeated period is physically correct, so it selects the near-tied candidate closest to the search image center.

## Citation Notes

### FinFET Pitch Rationale

- Intel 10 nm public process discussions report `34 nm` fin pitch.
- Cadence and Semiconductor Engineering articles cite the same public 10 nm-class value.
- The Berkeley FinFET short-course provides public background on FinFET and spacer-defined fin concepts.

### Noise Model

- Foi et al. describe a practical Poissonian-Gaussian raw-image sensor noise model with a signal-dependent Poisson component and Gaussian stationary component.
- PubMed indexes the same IEEE Transactions on Image Processing paper.
- Kockentiedt et al. discuss SEM shot-noise parameter estimation and note signal-dependent Poisson behavior in SEM imagery.

### Edge Effect

- JEOL's SEM glossary describes edge effect as brightening at protrusions and step edges due to increased secondary-electron emission.
- ETH Zurich ScopeM educational material describes SEM contrast dominated by edge effect in secondary-electron imaging.
- Nanoscience Instruments describes brighter SEM edges from secondary-electron behavior near edges.

### Rotation, Distortion, and Variable Scaling

No rotation, geometric distortion, or scale variation augmentation is implemented. The problem statement fixes the search footprint scale at exactly `10x`, so those knobs are intentionally absent.


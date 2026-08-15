# SEM Wafer Pattern Localization Handover

This repository contains a classical, non-deep-learning Python solution for the wafer-navigation pattern-localization problem. It generates synthetic SEM-like FinFET pattern pairs, localizes the high-resolution reference footprint inside the low-resolution search image using normalized cross-correlation, and evaluates the method across randomized cases.

## What Is Done

- `dataset_generator.py`
  - Defines `pattern(x_nm, y_nm)` as a continuous physical-coordinate FinFET-style intensity function.
  - Uses illustrative public-domain dimensions only: the default fin pitch is `34 nm`, based on public 10 nm-class FinFET references, not proprietary process data.
  - Rasterizes `reference.png` directly at `1 nm/px` and `search.png` directly at `10 nm/px`.
  - Does not create the search image by resizing the reference.
  - Adds independent mixed Poisson shot noise plus Gaussian read/electronic noise to each capture.
  - Makes search images noisier than reference images.
  - Adds SEM-like edge brightening from the clean-pattern gradient magnitude.
  - Provides the requested CLI:

```bash
python3 dataset_generator.py --seed 1 --out_dir phase1_sample --sanity_check
```

- `localize.py`
  - Loads grayscale or color inputs and converts to grayscale/luminance for matching.
  - Downsamples the `1000x1000` reference to a `100x100` template with `cv2.INTER_AREA`.
  - Runs `cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)`.
  - Collects local maxima at `>= 0.98 * global_max`.
  - Applies the required tiebreak rule: choose the candidate closest to the search image center.
  - Converts top-left match coordinates to search-image center coordinates.
  - Refines the selected peak with 3x3 parabolic subpixel interpolation.
  - Reports `(x, y, confidence, time_ms)`.

```bash
python3 localize.py --reference phase1_sample/reference.png --search phase1_sample/search.png
```

- `evaluate.py`
  - Generates `36` reproducible randomized pairs with varied asymmetric noise.
  - Includes deliberately ambiguous periodic cases.
  - Uses `TOLERANCE_PX = 1.0`.
  - Writes `evaluation_results/results.json`.
  - Saves `evaluation_results/failure_case.png`.
  - Prints the structural failure explanation for periodic near-tied peaks.

```bash
python3 evaluate.py
```

- Documentation and references
  - `docs/report.md` summarizes methodology, metrics, and the failure case.
  - `docs/citations.md` lists the public sources referenced in the report.
  - `README.md` is this handover document.

## Current Verified Results

From the latest `python3 evaluate.py` run:

- Pairs: `36`
- Tolerance: `1.0 px`
- Hits: `14/36`
- Hit rate: `38.89%`
- Mean localization time: `11.738 ms`
- Std localization time: `0.616 ms`
- Concrete failure case:
  - Ground truth: `(472.800, 232.923)`
  - Predicted: `(500.001, 232.944)`
  - Error: `27.201 px`
  - Plot: `evaluation_results/failure_case.png`

The low hit rate is expected for this deliberately periodic benchmark and the specified center-tiebreak rule. When repeated fin periods produce near-identical NCC peaks, the localizer has no unique visual evidence to distinguish the true repeated period from another one closer to the search center.

## How It Works

1. The generator chooses a reference center in continuous nanometer world coordinates.
2. It chooses a valid random footprint position inside the larger `10 um x 10 um` search field.
3. The same continuous pattern function is sampled independently for:
   - reference: `1000x1000`, `1 nm/px`
   - search: `1000x1000`, `10 nm/px`
4. SEM-style augmentation is applied after clean rasterization:
   - Poisson shot noise
   - Gaussian read noise
   - edge brightening from gradient magnitude
5. The localizer area-averages the reference to `100x100`.
6. NCC finds all near-tied local maxima.
7. The required center-distance heuristic picks one candidate.
8. Parabolic interpolation estimates subpixel peak location.

## What Is Remaining

- RGB bonus mode from Phase 5 is not implemented.
- No rotation, distortion, or scale-variation knobs are implemented. The spec fixes scale at exactly `10x`, and the core method assumes that exact geometry.
- A stronger production localizer would need extra disambiguating information for repeated-period patterns, such as stage prior, larger non-periodic context, multiple templates, or a different acquisition plan. That is outside the requested NCC-only architecture.

## Reproducibility Notes

- All generation randomness is seed-based and reproducible.
- The environment here uses `python3`; if your shell has a `python` alias, the same commands also work with `python`.
- Install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```


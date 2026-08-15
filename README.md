# SEM Wafer Pattern Localization

This repository contains a classical, non-deep-learning solution for locating
a 1000×1000 high-resolution wafer reference inside a 1000×1000 search image at
exactly 10× lower spatial resolution.

## Components

- `dataset_generator.py`: independently samples a continuous physical-coordinate
  FinFET-like pattern at 1 nm/px and 10 nm/px, adds independent SEM-like noise,
  and writes ground truth.
- `localize.py`: area-downsamples the reference, performs configurable
  upsampled normalized cross-correlation, applies the search-center tiebreak,
  and refines the result subpixel-wise.
- `evaluate.py`: runs 30 randomized cases, records primary metrics, creates a
  periodic failure plot, and sweeps threshold/upsample settings.
- `docs/report.md`: methodology, metrics, failure analysis, and citations.

## Commands

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Generate a sample:

```bash
python3 dataset_generator.py --seed 1 --out_dir phase1_sample --sanity_check
```

Run localization; output is JSON:

```bash
python3 localize.py --reference phase1_sample/reference.png --search phase1_sample/search.png
```

Run the evaluation:

```bash
python3 evaluate.py
```

## Current evaluation

The primary configuration (`upsample=4`, threshold `0.98`) achieved 14/30
within 1 px, with 855.020 ms mean measured computation time. The best sweep
setting was 4× with threshold `0.995`: 17/30, or 56.67%. Periodic fins create
near-identical candidate peaks, so the required center tiebreak can select a
wrong repeated period; this is an inherent ambiguity of appearance-only NCC.

## Geometric / chamfer contour matching: research finding

Chamfer matching is a plausible geometric pre-filter, but it has not been
implemented. The original Barrow et al. paper introduced matching collections
of curve fragments by distance rather than comparing every image intensity
value. Borgefors later developed hierarchical chamfer matching with distance
transforms and multiresolution processing, reporting robustness to noise and
other disturbances. Oriented chamfer variants add an edge-direction penalty to
reject geometrically incompatible clutter.

Relevant papers:

- [Barrow et al., *Parametric Correspondence and Chamfer Matching* (1977)](https://people.eecs.berkeley.edu/~malik/cs294/chamfer77.pdf)
- [Borgefors, *Hierarchical Chamfer Matching: A Parametric Edge Matching Algorithm* (1988)](https://people.eecs.berkeley.edu/~malik/cs294/borgefors88.pdf)
- [Shotton et al., *Contour-Based Learning for Object Detection* (2005), oriented chamfer background](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/iccv05.pdf)

### Proposed pre-NCC design

If a geometric stage is tested later, it should remain a candidate-generation
stage before the existing NCC pipeline:

1. Convert the already area-matched reference and search images to edge maps,
   preferably with gradient magnitude and orientation rather than a brittle
   single intensity threshold.
2. Build a truncated Euclidean distance transform of the search edge map.
3. Slide the reference edge template over the search image and score each
   location by the mean distance-transform value at template edge pixels.
4. Optionally add an orientation mismatch penalty and use a coarse-to-fine
   distance-transform pyramid for speed.
5. Keep the best geometric locations as a small candidate set, then run the
   existing intensity NCC and subpixel refinement on those candidates. The
   final reported answer and mandated search-center tiebreak must still come
   from the NCC architecture.

This could improve tolerance to independent SEM brightness/noise and partially
missing edges. It will not solve the central failure mode: a perfectly
periodic fin grating produces the same edge geometry at every fin period, so
the chamfer surface also has repeated near-ties. Gate-bar context or a stage
prior is still required to disambiguate those cases. At 10 nm/px, aggressive
edge extraction can also erase narrow fins, so thresholds and orientation
penalties would require a separate ablation and reproducible benchmark.

The core solution intentionally remains NCC-only. The geometric stage is a
research option, not a silent algorithm substitution.

## Status, remaining work, and final goal

Completed:

- Phase 1: continuous-coordinate independent rasterization, asymmetric
  Poisson/Gaussian SEM noise, edge effect, seeded generation, and sanity plot.
- Phase 2: named classical-NCC pipeline with area downsampling, upsampled
  correlation, local-maxima collection, center tiebreak, subpixel refinement,
  confidence, timing, and JSON CLI.
- Phase 3: 30-pair evaluation, threshold/upsample sweep, drift-proxy buckets,
  and periodic failure plot.
- Phase 4: methodology, metrics, failure analysis, and verified citations in
  `docs/report.md` and `docs/citations.md`.

Remaining:

- Optional gradient-profile subpixel ablation is research-only and is not part
  of the core result; its interrupted run should be rerun only if that ablation
  is needed for the presentation.
- RGB, rotation, distortion, and variable-scale modes remain intentionally
  unimplemented.
- The current validated Phase 1 generator uses random valid placement; its
  evaluation reports placement offset as a drift proxy. A bounded Gaussian
  drift generator would be a separate Phase 1 change and is not silently mixed
  into this submission.
- If organizers resolve the Slide 4 versus Slide 6 tiebreak wording conflict,
  update `select_by_center_tiebreak` and rerun the evaluation.

The final goal is an explainable, reproducible classical solution that accepts
independently captured reference/search images and reports the footprint center
in search pixels, equivalent nanometers, confidence, and runtime. It should be
presented as a successful NCC localization pipeline with a measured speed/
accuracy tradeoff and an explicit structural limitation for periodic patterns,
not as a claim that appearance-only matching can recover information absent
from both images.

## Suggested final presentation

1. **Problem and geometry:** 1 nm/px reference, 10 nm/px search, exact 100×100
   search footprint.
2. **Phase 1 data validity:** continuous physical sampling, independent noise,
   edge effect, and sanity-check image.
3. **NCC pipeline:** six named steps, 4× upsampling, candidate threshold, and
   search-center tiebreak.
4. **Measured results:** 30-pair accuracy, timing, threshold/upsample table,
   and placement-offset/error plot.
5. **Failure analysis:** correlation surface with ground truth/prediction;
   explain repeated fin-period ambiguity as structural.
6. **Research direction and close:** geometric/chamfer matching as a possible
   pre-NCC candidate filter, why it may improve noise robustness, and why it
   cannot remove periodic ambiguity without additional context.

RGB mode, rotation, distortion, and scale variation are not implemented.

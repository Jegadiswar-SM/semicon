# SEM Wafer Pattern Localization: Phase-NCC Hybrid Architecture

This repository contains a classical, non-deep-learning localization pipeline designed to find a high-resolution (100x100) reference template within a noisy, large-scale (1000x1000) search image.

## Architecture Pipeline

The system relies on a two-stage hybrid approach combining Normalized Cross-Correlation (NCC) with Fourier Phase Correlation to achieve robust localization and sub-pixel accuracy.

1. **Global Search (NCC):** 
   The reference template is downsampled, and a global `cv2.matchTemplate` search is performed across the entire search space. This guarantees spatial location invariance, ensuring targets near the image boundaries are accurately bounded.

2. **Local Refinement (Phase Correlation):**
   A local 150x150 window is extracted around the NCC candidate peak. Fourier Phase Correlation is applied exclusively within this crop using a Hanning window. Operating in the frequency domain normalizes amplitude information, mathematically negating additive and multiplicative speckle noise, yielding an exact sub-pixel coordinate.

## Components

- `dataset_generator.py`: Generates continuous physical-coordinate patterns (Logic gates and DRAM arrays) at 1 nm/px and 10 nm/px, injecting independent SEM-like noise profiles.
- `localize.py`: Implements the core Phase-NCC hybrid pipeline.
- `evaluate.py`: Executes the localized tests against 100 dynamically generated cases, logging accuracy and coordinate deviations.
- `Algorithm.md`: Contains mathematical documentation and failure mode analyses for the architecture.

## Execution Commands

Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

Generate a single diagnostic sample:
```bash
python3 dataset_generator.py --seed 1 --out_dir phase1_sample --sanity_check
```

Run manual localization; output is JSON:
```bash
python3 localize.py --reference phase1_sample/reference.png --search phase1_sample/search.png
```

Run the fully automated 100-pair evaluation:
```bash
python3 evaluate.py
# Or use the wrapper scripts:
# ./run_pipeline.bat
# bash run_pipeline.sh
```

## Performance Profile

The pipeline operates at ~40 ms per image. On standard logic gates, the architecture achieves near-perfect coordinate alignment. On periodic fin patterns (infinite grid arrays), the frequency domains repeat identically. Consequently, the NCC global search may lock onto an adjacent identical period, representing a structural ambiguity inherent to appearance-based matching. Benchmark accuracy stabilizes at approximately 85-88%.

"""Classical NCC wafer-pattern localization.

The implementation follows the requested OpenCV normalized cross-correlation
pipeline and does not use deep learning.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


Array = np.ndarray
REFERENCE_TO_SEARCH_SCALE = 10
SEARCH_PIXEL_PITCH_NM = 10.0


@dataclass(frozen=True)
class LocalizationResult:
    center_x: float
    center_y: float
    confidence: float
    time_ms: float
    center_x_nm: float
    center_y_nm: float


def load_image(path: str | Path) -> Array:
    """Load grayscale or RGB input as normalized float32 luminance."""

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image_to_gray_float(image)


def image_to_gray_float(image: Array) -> Array:
    """Convert a path-loaded or in-memory grayscale/RGB image to float32 gray."""

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim != 2:
        raise ValueError(f"expected 2D grayscale or 3D color image, got shape {image.shape}")

    gray = image.astype(np.float32)
    if gray.max(initial=0.0) > 1.5:
        gray /= 255.0
    return np.ascontiguousarray(gray)


def _parabolic_offset(v_minus: float, v0: float, v_plus: float) -> float:
    """Return subpixel offset of a parabola through (-1, v-), (0, v0), (1, v+)."""

    denom = v_minus - 2.0 * v0 + v_plus
    if abs(denom) < 1e-12:
        return 0.0
    offset = 0.5 * (v_minus - v_plus) / denom
    return float(np.clip(offset, -1.0, 1.0))


def _local_maxima(correlation: Array, threshold: float) -> Array:
    """Return candidate peak coordinates as Nx2 array of (x, y)."""

    dilated = cv2.dilate(correlation, np.ones((3, 3), dtype=np.uint8))
    mask = (correlation >= threshold) & (correlation >= dilated - 1e-12)
    ys, xs = np.nonzero(mask)
    return np.column_stack([xs, ys]).astype(np.float32)


def localize_arrays(reference: Array, search: Array) -> LocalizationResult:
    """Locate the 10x-shrunk reference footprint in a search image."""

    reference_gray = image_to_gray_float(reference)
    search_gray = image_to_gray_float(search)

    start = time.perf_counter()

    template_size = (
        max(1, reference_gray.shape[1] // REFERENCE_TO_SEARCH_SCALE),
        max(1, reference_gray.shape[0] // REFERENCE_TO_SEARCH_SCALE),
    )
    template = cv2.resize(reference_gray, template_size, interpolation=cv2.INTER_AREA)
    correlation = cv2.matchTemplate(search_gray, template, cv2.TM_CCOEFF_NORMED)

    _, global_max, _, global_max_loc = cv2.minMaxLoc(correlation)
    threshold = 0.98 * global_max
    candidates = _local_maxima(correlation, threshold)
    if candidates.size == 0:
        candidates = np.array([[global_max_loc[0], global_max_loc[1]]], dtype=np.float32)

    half_w = template.shape[1] / 2.0
    half_h = template.shape[0] / 2.0
    search_center = np.array([search_gray.shape[1] / 2.0, search_gray.shape[0] / 2.0])
    candidate_centers = candidates + np.array([half_w, half_h], dtype=np.float32)
    distances = np.linalg.norm(candidate_centers - search_center, axis=1)
    selected_idx = int(np.argmin(distances))
    peak_x = float(candidates[selected_idx, 0])
    peak_y = float(candidates[selected_idx, 1])

    ix = int(round(peak_x))
    iy = int(round(peak_y))
    sub_x = 0.0
    sub_y = 0.0
    if 0 < ix < correlation.shape[1] - 1:
        sub_x = _parabolic_offset(
            float(correlation[iy, ix - 1]),
            float(correlation[iy, ix]),
            float(correlation[iy, ix + 1]),
        )
    if 0 < iy < correlation.shape[0] - 1:
        sub_y = _parabolic_offset(
            float(correlation[iy - 1, ix]),
            float(correlation[iy, ix]),
            float(correlation[iy + 1, ix]),
        )

    refined_x = peak_x + sub_x
    refined_y = peak_y + sub_y
    center_x = refined_x + half_w
    center_y = refined_y + half_h
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    confidence = float(correlation[iy, ix])

    return LocalizationResult(
        center_x=float(center_x),
        center_y=float(center_y),
        confidence=confidence,
        time_ms=float(elapsed_ms),
        center_x_nm=float(center_x * SEARCH_PIXEL_PITCH_NM),
        center_y_nm=float(center_y * SEARCH_PIXEL_PITCH_NM),
    )


def localize(reference_path: str | Path, search_path: str | Path) -> LocalizationResult:
    """Load input images and run the timed localization algorithm."""

    reference = load_image(reference_path)
    search = load_image(search_path)
    return localize_arrays(reference, search)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate a reference pattern in a search image.")
    parser.add_argument("--reference", type=Path, required=True, help="Path to reference.png.")
    parser.add_argument("--search", type=Path, required=True, help="Path to search.png.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = localize(args.reference, args.search)
    print(
        f"x={result.center_x:.6f}, y={result.center_y:.6f}, "
        f"confidence={result.confidence:.6f}, time_ms={result.time_ms:.3f}"
    )


if __name__ == "__main__":
    main()

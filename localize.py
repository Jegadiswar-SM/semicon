"""Fourier Phase Correlation localizer for wafer-navigation images.

The reference is reduced to the search image's effective resolution, padded to 
the center, and windowed before frequency-domain localization.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


Array = np.ndarray
REFERENCE_TO_SEARCH_SCALE = 10
SEARCH_PIXEL_PITCH_NM = 10.0


def image_to_gray_float(image: Array) -> Array:
    """Convert grayscale, BGR/RGB, or BGRA input to contiguous float32 gray."""

    image = np.asarray(image)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.shape[2] != 3:
            raise ValueError(f"expected 3 or 4 channels, got shape {image.shape}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim != 2:
        raise ValueError(f"expected 2D grayscale or 3D color image, got shape {image.shape}")

    gray = image.astype(np.float32, copy=False)
    if gray.size and float(gray.max()) > 1.5:
        gray = gray / 255.0
    return np.ascontiguousarray(gray)


def load_image(path: str | Path) -> Array:
    """Load an image from disk and convert it to normalized grayscale."""

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image_to_gray_float(image)


def load_and_downsample_template(reference: str | Path | Array) -> Array:
    """Load or accept a reference and area-average it to a 100x100 template."""

    reference_gray = load_image(reference) if isinstance(reference, (str, Path)) else image_to_gray_float(reference)
    if reference_gray.shape[0] < REFERENCE_TO_SEARCH_SCALE or reference_gray.shape[1] < REFERENCE_TO_SEARCH_SCALE:
        raise ValueError("reference image is too small to downsample by the required factor")
    template_size = (
        reference_gray.shape[1] // REFERENCE_TO_SEARCH_SCALE,
        reference_gray.shape[0] // REFERENCE_TO_SEARCH_SCALE,
    )
    return cv2.resize(reference_gray, template_size, interpolation=cv2.INTER_AREA)


def hybrid_localize(
    reference: Array,
    search: Array,
) -> dict[str, Any]:
    """Run Hybrid NCC Global Search + Phase Correlation Local Refinement."""

    reference_gray = image_to_gray_float(reference)
    search_gray = image_to_gray_float(search)
    start = time.perf_counter()
    template = load_and_downsample_template(reference_gray)
    h, w = template.shape

    # 1. Global Search: Normalized Cross-Correlation
    ncc_surface = cv2.matchTemplate(search_gray, template, cv2.TM_CCOEFF_NORMED)
    _, global_max, _, _ = cv2.minMaxLoc(ncc_surface)
    
    # Find all peaks that are mathematically tied with the maximum (within precision)
    threshold = global_max - 1e-4
    y_locs, x_locs = np.where(ncc_surface >= threshold)
    
    # Problem Statement MANDATORY Requirement:
    # "If more than one matching region is found, return the one closest to the center of the Search Image."
    search_center_x = search_gray.shape[1] / 2.0
    search_center_y = search_gray.shape[0] / 2.0
    
    best_dist = float('inf')
    best_loc = (0, 0)
    
    for px, py in zip(x_locs, y_locs):
        # Coordinate of the template center if placed at (px, py)
        cx = px + w / 2.0
        cy = py + h / 2.0
        dist = (cx - search_center_x)**2 + (cy - search_center_y)**2
        if dist < best_dist:
            best_dist = dist
            best_loc = (int(px), int(py))
            
    ncc_x, ncc_y = best_loc
    
    # 2. Local Refinement: Extract Search Crop
    pad_h, pad_w = h * 2, w * 2
    center_x_int = ncc_x + w // 2
    center_y_int = ncc_y + h // 2
    
    y1 = max(0, center_y_int - pad_h // 2)
    y2 = min(search_gray.shape[0], center_y_int + pad_h // 2)
    x1 = max(0, center_x_int - pad_w // 2)
    x2 = min(search_gray.shape[1], center_x_int + pad_w // 2)
    
    search_crop = search_gray[y1:y2, x1:x2]
    
    # Pad crop safely if it hits image edges
    padded_search_crop = np.zeros((pad_h, pad_w), dtype=np.float32)
    dy = pad_h // 2 - (center_y_int - y1)
    dx = pad_w // 2 - (center_x_int - x1)
    padded_search_crop[dy:dy+(y2-y1), dx:dx+(x2-x1)] = search_crop
    
    # Pad template to match crop size for Phase Correlation
    padded_template = np.zeros((pad_h, pad_w), dtype=np.float32)
    ty = (pad_h - h) // 2
    tx = (pad_w - w) // 2
    padded_template[ty:ty+h, tx:tx+w] = template
    
    # 3. Phase Correlation on the local crop
    hann = cv2.createHanningWindow((pad_w, pad_h), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(padded_template, padded_search_crop, hann)
    x_shift, y_shift = shift
    
    final_center_x = center_x_int + x_shift
    final_center_y = center_y_int + y_shift
    
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Pad NCC surface back to original image shape for visualization
    padded_ncc = np.zeros_like(search_gray)
    padded_ncc[h//2 : h//2 + ncc_surface.shape[0], w//2 : w//2 + ncc_surface.shape[1]] = ncc_surface

    return {
        "center_x": float(final_center_x),
        "center_y": float(final_center_y),
        "confidence": float(response),
        "time_ms": float(elapsed_ms),
        "center_x_nm": float(final_center_x * SEARCH_PIXEL_PITCH_NM),
        "center_y_nm": float(final_center_y * SEARCH_PIXEL_PITCH_NM),
        "surface": padded_ncc,
    }


def localize(
    reference_path: str | Path,
    search_path: str | Path,
) -> dict[str, Any]:
    """Load images outside the timed region and localize them."""
    return hybrid_localize(
        load_image(reference_path),
        load_image(search_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate a reference pattern in a search image with Phase Correlation.")
    parser.add_argument("--reference", "--ref", dest="reference", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = localize(args.reference, args.search)
    # Exclude the large surface array from json printing
    result.pop("surface", None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

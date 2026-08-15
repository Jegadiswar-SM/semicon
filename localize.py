"""Classical, explainable NCC localizer for wafer-navigation images.

The reference is reduced to the search image's effective resolution, then the
search/template pair is optionally upsampled before correlation.  Upsampling
does not change the physical scale; it makes the continuous correlation peak
less sensitive to subpixel placement before candidate selection.
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
DEFAULT_UPSAMPLE = 4
DEFAULT_THRESHOLD_RATIO = 0.98


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


def correlate(search: Array, template: Array, upsample: int = DEFAULT_UPSAMPLE) -> tuple[Array, int]:
    """Return an NCC surface after optional cubic upsampling of both inputs."""

    if int(upsample) != upsample or upsample < 1:
        raise ValueError("upsample must be a positive integer")
    upsample = int(upsample)
    search_gray = image_to_gray_float(search)
    template_gray = image_to_gray_float(template)
    if template_gray.shape[0] > search_gray.shape[0] or template_gray.shape[1] > search_gray.shape[1]:
        raise ValueError("template must not be larger than search image")

    if upsample == 1:
        search_work = search_gray
        template_work = template_gray
    else:
        search_work = cv2.resize(
            search_gray,
            (search_gray.shape[1] * upsample, search_gray.shape[0] * upsample),
            interpolation=cv2.INTER_CUBIC,
        )
        template_work = cv2.resize(
            template_gray,
            (template_gray.shape[1] * upsample, template_gray.shape[0] * upsample),
            interpolation=cv2.INTER_CUBIC,
        )
    surface = cv2.matchTemplate(search_work, template_work, cv2.TM_CCOEFF_NORMED)
    return surface, upsample


def find_candidates(
    surface: Array,
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
) -> list[tuple[float, float, float]]:
    """Find distinct 3x3 local maxima above a fraction of the global max."""

    if not 0.0 < threshold_ratio <= 1.0:
        raise ValueError("threshold_ratio must be in (0, 1]")
    _, global_max, _, global_loc = cv2.minMaxLoc(surface)
    threshold = threshold_ratio * global_max
    neighborhood_max = cv2.dilate(surface, np.ones((3, 3), dtype=np.uint8))
    mask = (surface >= threshold) & (surface >= neighborhood_max - 1e-7)
    ys, xs = np.nonzero(mask)
    candidates = [(float(x), float(y), float(surface[y, x])) for x, y in zip(xs, ys)]
    if not candidates:
        x, y = global_loc
        candidates = [(float(x), float(y), float(surface[y, x]))]
    return candidates


def select_by_center_tiebreak(
    candidates: list[tuple[float, float, float]],
    search_shape: tuple[int, ...],
    upsample: int = 1,
    template_shape: tuple[int, int] | None = None,
) -> tuple[float, float, float]:
    """Select the candidate whose footprint center is nearest search center."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    scale = float(upsample)
    half_w = (template_shape[1] / 2.0) if template_shape else 0.0
    half_h = (template_shape[0] / 2.0) if template_shape else 0.0
    search_center = np.array([search_shape[1] * scale / 2.0, search_shape[0] * scale / 2.0])

    def key(candidate: tuple[float, float, float]) -> tuple[float, float]:
        x, y, score = candidate
        center = np.array([x + half_w, y + half_h])
        return float(np.linalg.norm(center - search_center)), -score

    return min(candidates, key=key)


def _parabolic_offset(v_minus: float, v0: float, v_plus: float) -> float:
    denominator = v_minus - 2.0 * v0 + v_plus
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (v_minus - v_plus) / denominator, -1.0, 1.0))


def refine_subpixel(surface: Array, peak_xy: tuple[float, float]) -> tuple[float, float]:
    """Refine a correlation-surface peak with separable parabolic fits."""

    x, y = peak_xy
    ix, iy = int(round(x)), int(round(y))
    if not (0 < ix < surface.shape[1] - 1 and 0 < iy < surface.shape[0] - 1):
        return float(x), float(y)
    dx = _parabolic_offset(float(surface[iy, ix - 1]), float(surface[iy, ix]), float(surface[iy, ix + 1]))
    dy = _parabolic_offset(float(surface[iy - 1, ix]), float(surface[iy, ix]), float(surface[iy + 1, ix]))
    return float(x + dx), float(y + dy)


def gradient_profile_refine(
    search: Array,
    template: Array,
    peak_xy: tuple[float, float],
    upsample: int = DEFAULT_UPSAMPLE,
) -> tuple[float, float, float]:
    """Optional edge-profile ablation around an already selected NCC peak.

    This does not replace intensity NCC. It correlates Sobel gradient magnitude
    images only to estimate a second subpixel offset in the local neighborhood
    of the NCC-selected candidate. The returned coordinates are in the same
    upsampled-surface coordinate system as ``peak_xy``.
    """

    search_gray = image_to_gray_float(search)
    template_gray = image_to_gray_float(template)
    factor = int(upsample)
    if factor < 1:
        raise ValueError("upsample must be a positive integer")
    if factor > 1:
        search_gray = cv2.resize(
            search_gray, (search_gray.shape[1] * factor, search_gray.shape[0] * factor), interpolation=cv2.INTER_CUBIC
        )
        template_gray = cv2.resize(
            template_gray, (template_gray.shape[1] * factor, template_gray.shape[0] * factor), interpolation=cv2.INTER_CUBIC
        )

    def gradient_magnitude(image: Array) -> Array:
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    search_edges = gradient_magnitude(search_gray)
    template_edges = gradient_magnitude(template_gray)
    edge_surface = cv2.matchTemplate(search_edges, template_edges, cv2.TM_CCOEFF_NORMED)
    refined_x, refined_y = refine_subpixel(edge_surface, peak_xy)
    ix, iy = int(round(peak_xy[0])), int(round(peak_xy[1]))
    score = float(edge_surface[iy, ix]) if 0 <= iy < edge_surface.shape[0] and 0 <= ix < edge_surface.shape[1] else float("nan")
    return refined_x, refined_y, score


def localize_arrays(
    reference: Array,
    search: Array,
    upsample: int = DEFAULT_UPSAMPLE,
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
) -> dict[str, Any]:
    """Run the complete timed localization pipeline on in-memory arrays."""

    reference_gray = image_to_gray_float(reference)
    search_gray = image_to_gray_float(search)
    start = time.perf_counter()
    template = load_and_downsample_template(reference_gray)
    surface, factor = correlate(search_gray, template, upsample=upsample)
    candidates = find_candidates(surface, threshold_ratio=threshold_ratio)
    selected = select_by_center_tiebreak(
        candidates,
        search_gray.shape,
        upsample=factor,
        template_shape=template.shape,
    )
    refined_x, refined_y = refine_subpixel(surface, (selected[0], selected[1]))
    native_top_left = np.array([refined_x, refined_y]) / factor
    half_template = np.array([template.shape[1], template.shape[0]], dtype=np.float64) / 2.0
    center = native_top_left + half_template
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "confidence": float(selected[2]),
        "time_ms": float(elapsed_ms),
        "center_x_nm": float(center[0] * SEARCH_PIXEL_PITCH_NM),
        "center_y_nm": float(center[1] * SEARCH_PIXEL_PITCH_NM),
        "upsample": factor,
        "threshold_ratio": float(threshold_ratio),
        "candidate_count": len(candidates),
        "candidates": [
            {"x_surface": x, "y_surface": y, "score": score}
            for x, y, score in candidates
        ],
    }


def localize(
    reference_path: str | Path,
    search_path: str | Path,
    upsample: int = DEFAULT_UPSAMPLE,
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
) -> dict[str, Any]:
    """Load images outside the timed region and localize them."""

    return localize_arrays(
        load_image(reference_path),
        load_image(search_path),
        upsample=upsample,
        threshold_ratio=threshold_ratio,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate a reference pattern in a search image with classical NCC.")
    parser.add_argument("--reference", "--ref", dest="reference", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--upsample", type=int, default=DEFAULT_UPSAMPLE)
    parser.add_argument("--threshold-ratio", type=float, default=DEFAULT_THRESHOLD_RATIO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = localize(args.reference, args.search, args.upsample, args.threshold_ratio)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Synthetic wafer-navigation dataset generator.

Phase 1 implements a continuous-coordinate FinFET-style pattern and renders
the two captures independently from that pattern at their own pixel pitches.
It intentionally does not produce the search image by resizing the reference.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


Array = np.ndarray
PatternFn = Callable[[Array, Array], Array]

IMAGE_SIZE_PX = 1000
REFERENCE_PIXEL_PITCH_NM = 1.0
SEARCH_PIXEL_PITCH_NM = 10.0
REFERENCE_FOV_NM = IMAGE_SIZE_PX * REFERENCE_PIXEL_PITCH_NM
SEARCH_FOV_NM = IMAGE_SIZE_PX * SEARCH_PIXEL_PITCH_NM
REFERENCE_FOOTPRINT_IN_SEARCH_PX = int(
    REFERENCE_FOV_NM / SEARCH_PIXEL_PITCH_NM
)


@dataclass(frozen=True)
class PatternParams:
    """Parameters for an illustrative, non-proprietary FinFET-like pattern."""

    # Illustrative public-domain value: 10 nm-class FinFET discussions commonly
    # cite fin pitch near 34 nm. This is a synthetic benchmark value, not PDK or
    # proprietary process data.
    fin_pitch_nm: float = 34.0
    fin_width_nm: float = 8.0
    fin_phase_nm: float = 0.0
    gate_center_x_nm: float = 0.0
    gate_length_nm: float = 650.0
    gate_centers_y_nm: Tuple[float, ...] = (0.0,)
    gate_width_nm: float = 42.0
    edge_softness_nm: float = 1.6
    background: float = 0.22
    fin_contrast: float = 0.32
    gate_contrast: float = 0.46
    intersection_contrast: float = 0.16


@dataclass(frozen=True)
class SemNoiseParams:
    """SEM-like mixed Poisson shot noise plus Gaussian read/electronic noise."""

    peak_electrons: float
    read_noise_sigma: float
    edge_gain: float


REFERENCE_SEM_NOISE = SemNoiseParams(
    peak_electrons=2200.0,
    read_noise_sigma=0.010,
    edge_gain=0.10,
)
SEARCH_SEM_NOISE = SemNoiseParams(
    peak_electrons=420.0,
    read_noise_sigma=0.030,
    edge_gain=0.16,
)


def _smooth_box_1d(coord_nm: Array, center_nm: float, width_nm: float, softness_nm: float) -> Array:
    """Continuous soft rectangular indicator in [0, 1]."""

    half_width = width_nm / 2.0
    left = center_nm - half_width
    right = center_nm + half_width
    softness = max(float(softness_nm), 1e-6)
    return 0.5 * (
        np.tanh((coord_nm - left) / softness)
        - np.tanh((coord_nm - right) / softness)
    )


def _periodic_fin_mask(x_nm: Array, params: PatternParams) -> Array:
    """Soft vertical fins repeated with the configured fin pitch."""

    centered = (
        (x_nm - params.fin_phase_nm + params.fin_pitch_nm / 2.0)
        % params.fin_pitch_nm
    ) - params.fin_pitch_nm / 2.0
    return _smooth_box_1d(
        centered,
        center_nm=0.0,
        width_nm=params.fin_width_nm,
        softness_nm=params.edge_softness_nm,
    )


def make_pattern(params: PatternParams) -> PatternFn:
    """Create a continuous physical-coordinate pattern function.

    The returned function accepts x/y coordinates in nanometers, either scalars
    or NumPy arrays, and returns normalized intensity in [0, 1].
    """

    def pattern(x_nm: Array, y_nm: Array) -> Array:
        fins = _periodic_fin_mask(x_nm, params)
        gate_x_envelope = _smooth_box_1d(
            x_nm,
            center_nm=params.gate_center_x_nm,
            width_nm=params.gate_length_nm,
            softness_nm=params.edge_softness_nm * 2.0,
        )
        gates = np.zeros_like(y_nm, dtype=np.float32)
        for gate_center in params.gate_centers_y_nm:
            gates = np.maximum(
                gates,
                _smooth_box_1d(
                    y_nm,
                    center_nm=gate_center,
                    width_nm=params.gate_width_nm,
                    softness_nm=params.edge_softness_nm * 1.6,
                )
                * gate_x_envelope,
            )

        intensity = (
            params.background
            + params.fin_contrast * fins
            + params.gate_contrast * gates
            + params.intersection_contrast * fins * gates
        )
        return np.clip(intensity, 0.0, 1.0).astype(np.float32)

    return pattern


def random_pattern_params(
    seed: int,
    reference_center_nm: Tuple[float, float],
    ambiguous_region: bool = False,
) -> PatternParams:
    """Generate reproducible illustrative pattern parameters for one pair."""

    rng = np.random.default_rng(seed)
    _, ref_center_y_nm = reference_center_nm
    gate_count = int(rng.integers(1, 3))

    if gate_count == 1:
        gate_offsets = [float(rng.uniform(-250.0, 250.0))]
    else:
        first = float(rng.uniform(-280.0, -80.0))
        second = float(rng.uniform(80.0, 280.0))
        gate_offsets = [first, second]

    return PatternParams(
        fin_pitch_nm=34.0,
        fin_width_nm=float(rng.uniform(7.0, 10.0)),
        fin_phase_nm=float(rng.uniform(0.0, 34.0)),
        gate_center_x_nm=float(
            reference_center_nm[0]
            + (0.0 if ambiguous_region else rng.uniform(-160.0, 160.0))
        ),
        gate_length_nm=20_000.0 if ambiguous_region else float(rng.uniform(360.0, 760.0)),
        gate_centers_y_nm=tuple(ref_center_y_nm + offset for offset in gate_offsets),
        gate_width_nm=float(rng.uniform(36.0, 52.0)),
        edge_softness_nm=float(rng.uniform(1.2, 2.2)),
    )


def render(
    pattern_fn: PatternFn,
    center_nm: Tuple[float, float],
    size_px: int = IMAGE_SIZE_PX,
    pixel_pitch_nm: float = REFERENCE_PIXEL_PITCH_NM,
    samples_per_pixel: int = 3,
) -> Array:
    """Rasterize a continuous pattern around a physical center.

    Each image is sampled directly from the continuous function at its own
    pitch. Small supersampling approximates pixel-area integration without ever
    scaling one raster to create the other.
    """

    if samples_per_pixel < 1:
        raise ValueError("samples_per_pixel must be >= 1")

    center_x_nm, center_y_nm = center_nm
    cols = np.arange(size_px, dtype=np.float32)
    rows = np.arange(size_px, dtype=np.float32)
    x_base = center_x_nm + (cols + 0.5 - size_px / 2.0) * pixel_pitch_nm
    y_base = center_y_nm + (rows + 0.5 - size_px / 2.0) * pixel_pitch_nm

    offsets = (
        np.array([0.0], dtype=np.float32)
        if samples_per_pixel == 1
        else (np.arange(samples_per_pixel, dtype=np.float32) + 0.5) / samples_per_pixel - 0.5
    )
    offsets *= pixel_pitch_nm

    accum = np.zeros((size_px, size_px), dtype=np.float32)
    for dy in offsets:
        y = y_base + dy
        for dx in offsets:
            x = x_base + dx
            xx, yy = np.meshgrid(x, y)
            accum += pattern_fn(xx, yy)

    accum /= float(samples_per_pixel * samples_per_pixel)
    return np.clip(accum, 0.0, 1.0).astype(np.float32)


def add_sem_noise_and_edges(clean: Array, rng: np.random.Generator, params: SemNoiseParams) -> Array:
    """Add independent SEM-like noise and clean-pattern edge brightening."""

    signal = np.clip(clean, 0.0, 1.0)
    expected_electrons = signal * params.peak_electrons
    shot = rng.poisson(expected_electrons).astype(np.float32) / params.peak_electrons
    read = rng.normal(0.0, params.read_noise_sigma, size=signal.shape).astype(np.float32)
    noisy = shot + read

    grad_x = cv2.Sobel(signal, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(signal, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(grad_x, grad_y)
    edge = cv2.GaussianBlur(edge, (0, 0), sigmaX=0.8)
    edge_scale = float(np.percentile(edge, 99.5))
    if edge_scale > 0.0:
        edge = edge / edge_scale

    image = noisy + params.edge_gain * edge
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def to_uint8(image: Array) -> Array:
    """Convert a normalized image to 8-bit grayscale."""

    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def save_png(path: Path, image: Array) -> None:
    """Save a normalized grayscale image as PNG."""

    ok = cv2.imwrite(str(path), to_uint8(image))
    if not ok:
        raise OSError(f"failed to write image: {path}")


def generate_pair(
    seed: int,
    reference_noise: SemNoiseParams = REFERENCE_SEM_NOISE,
    search_noise: SemNoiseParams = SEARCH_SEM_NOISE,
    ambiguous_region: bool = False,
) -> tuple[Array, Array, tuple[float, float]]:
    """Generate one reproducible reference/search pair and ground truth.

    Returns:
        reference_img: 1000x1000 normalized float32 image at 1 nm/px.
        search_img: 1000x1000 normalized float32 image at 10 nm/px.
        ground_truth_center_xy_in_search_px: (x, y) in search-image coords.
    """

    rng = np.random.default_rng(seed)
    ref_center_nm = (
        float(rng.uniform(20_000.0, 80_000.0)),
        float(rng.uniform(20_000.0, 80_000.0)),
    )

    half_footprint_px = REFERENCE_FOOTPRINT_IN_SEARCH_PX / 2.0
    if ambiguous_region:
        # Deliberately choose a location near a periodic fin phase. With the
        # horizontal gate bar extending through the search image, the x direction
        # can produce many nearly identical NCC peaks separated by the fin pitch.
        search_center_x_nm = ref_center_nm[0]
        relative_center_x_px = (
            ref_center_nm[0] - search_center_x_nm
        ) / SEARCH_PIXEL_PITCH_NM + IMAGE_SIZE_PX / 2.0
        fin_pitch_search_px = PatternParams.fin_pitch_nm / SEARCH_PIXEL_PITCH_NM
        target_x = float(rng.uniform(half_footprint_px, IMAGE_SIZE_PX - half_footprint_px))
        k = round((target_x - relative_center_x_px) / fin_pitch_search_px)
        gt_x = float(
            np.clip(
                relative_center_x_px + k * fin_pitch_search_px,
                half_footprint_px,
                IMAGE_SIZE_PX - half_footprint_px,
            )
        )
    else:
        gt_x = float(rng.uniform(half_footprint_px, IMAGE_SIZE_PX - half_footprint_px))

    gt_y = float(rng.uniform(half_footprint_px, IMAGE_SIZE_PX - half_footprint_px))
    search_center_nm = (
        ref_center_nm[0] - (gt_x - IMAGE_SIZE_PX / 2.0) * SEARCH_PIXEL_PITCH_NM,
        ref_center_nm[1] - (gt_y - IMAGE_SIZE_PX / 2.0) * SEARCH_PIXEL_PITCH_NM,
    )

    pattern_params = random_pattern_params(seed + 10_000, ref_center_nm, ambiguous_region)
    pattern_fn = make_pattern(pattern_params)

    clean_reference = render(
        pattern_fn,
        ref_center_nm,
        size_px=IMAGE_SIZE_PX,
        pixel_pitch_nm=REFERENCE_PIXEL_PITCH_NM,
        samples_per_pixel=3,
    )
    clean_search = render(
        pattern_fn,
        search_center_nm,
        size_px=IMAGE_SIZE_PX,
        pixel_pitch_nm=SEARCH_PIXEL_PITCH_NM,
        samples_per_pixel=3,
    )

    reference_img = add_sem_noise_and_edges(
        clean_reference,
        np.random.default_rng(seed + 20_000),
        reference_noise,
    )
    search_img = add_sem_noise_and_edges(
        clean_search,
        np.random.default_rng(seed + 30_000),
        search_noise,
    )

    return reference_img, search_img, (gt_x, gt_y)


def save_sanity_check(
    path: Path,
    reference_img: Array,
    search_img: Array,
    ground_truth_xy: tuple[float, float],
) -> None:
    """Save side-by-side visual check with the search footprint boxed."""

    gt_x, gt_y = ground_truth_xy
    footprint = REFERENCE_FOOTPRINT_IN_SEARCH_PX
    half = footprint / 2.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes[0].imshow(reference_img, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("reference.png, 1 nm/px")
    axes[0].axis("off")

    axes[1].imshow(search_img, cmap="gray", vmin=0.0, vmax=1.0)
    rect = plt.Rectangle(
        (gt_x - half, gt_y - half),
        footprint,
        footprint,
        fill=False,
        edgecolor="red",
        linewidth=2.0,
    )
    axes[1].add_patch(rect)
    axes[1].plot([gt_x], [gt_y], marker="+", color="yellow", markersize=12, markeredgewidth=2)
    axes[1].set_title("search.png, 10 nm/px with GT footprint")
    axes[1].axis("off")

    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_metadata(path: Path, seed: int, ground_truth_xy: tuple[float, float]) -> None:
    """Write exact ground truth center for evaluation only."""

    center_x, center_y = ground_truth_xy
    with path.open("w", encoding="utf-8") as f:
        json.dump({"center_x": center_x, "center_y": center_y}, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one synthetic wafer-navigation pair.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducible output.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory for output files.")
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        help="Save a side-by-side PNG with the ground-truth search footprint boxed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reference_img, search_img, ground_truth_xy = generate_pair(args.seed)

    reference_path = args.out_dir / "reference.png"
    search_path = args.out_dir / "search.png"
    truth_path = args.out_dir / "ground_truth.json"
    save_png(reference_path, reference_img)
    save_png(search_path, search_img)
    write_metadata(truth_path, args.seed, ground_truth_xy)

    print(f"wrote {reference_path}")
    print(f"wrote {search_path}")
    print(f"wrote {truth_path}")
    print(f"ground_truth center_x={ground_truth_xy[0]:.6f}, center_y={ground_truth_xy[1]:.6f}")

    if args.sanity_check:
        sanity_path = args.out_dir / "sanity_check.png"
        save_sanity_check(sanity_path, reference_img, search_img, ground_truth_xy)
        print(f"wrote {sanity_path}")


if __name__ == "__main__":
    main()

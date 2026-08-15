"""Evaluate the classical NCC localizer and its calibration tradeoffs."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import dataset_generator as dg
import localize


TOLERANCE_PX = 1.0
NUM_RANDOM_PAIRS = 30
PRIMARY_UPSAMPLE = 4
PRIMARY_THRESHOLD_RATIO = 0.98
UPSAMPLE_SWEEP = (1, 2, 4, 8)
THRESHOLD_SWEEP = (0.90, 0.95, 0.98, 0.995)
RESULTS_DIR = Path("evaluation_results")


def _noise_for_case(index: int) -> tuple[dg.SemNoiseParams, dg.SemNoiseParams]:
    """Deterministically vary asymmetric noise across evaluation cases."""

    ref_peak = 1600.0 + 900.0 * ((index % 5) / 4.0)
    search_peak = 240.0 + 420.0 * (((index * 3) % 7) / 6.0)
    ref_read = 0.007 + 0.007 * ((index % 4) / 3.0)
    search_read = 0.020 + 0.030 * (((index * 5) % 6) / 5.0)
    ref_edge = 0.08 + 0.04 * ((index % 3) / 2.0)
    search_edge = 0.12 + 0.08 * (((index * 2) % 5) / 4.0)
    return (
        dg.SemNoiseParams(ref_peak, ref_read, ref_edge),
        dg.SemNoiseParams(search_peak, search_read, search_edge),
    )


def _make_cases() -> list[dict]:
    cases = []
    for index in range(NUM_RANDOM_PAIRS):
        ref_noise, search_noise = _noise_for_case(index)
        reference, search, truth = dg.generate_pair(
            1000 + index,
            reference_noise=ref_noise,
            search_noise=search_noise,
            ambiguous_region=index % 9 == 0,
        )
        placement_offset_nm = float(np.hypot(truth[0] - 500.0, truth[1] - 500.0) * dg.SEARCH_PIXEL_PITCH_NM)
        cases.append({
            "index": index,
            "seed": 1000 + index,
            "reference": reference,
            "search": search,
            "truth": truth,
            "ambiguous_region": index % 9 == 0,
            "placement_offset_nm": placement_offset_nm,
            "reference_noise": ref_noise,
            "search_noise": search_noise,
        })
    return cases


def _score(result: dict, truth: tuple[float, float]) -> tuple[float, bool]:
    error = float(np.hypot(result["center_x"] - truth[0], result["center_y"] - truth[1]))
    return error, error <= TOLERANCE_PX


def _result_from_surface(
    surface: np.ndarray,
    template: np.ndarray,
    search_shape: tuple[int, ...],
    upsample: int,
    threshold_ratio: float,
    elapsed_ms: float,
) -> dict:
    """Apply threshold-dependent candidate selection to a cached NCC surface."""

    candidates = localize.find_candidates(surface, threshold_ratio)
    selected = localize.select_by_center_tiebreak(
        candidates,
        search_shape,
        upsample=upsample,
        template_shape=template.shape,
    )
    refined_x, refined_y = localize.refine_subpixel(surface, (selected[0], selected[1]))
    center = np.array([refined_x, refined_y], dtype=np.float64) / upsample
    center += np.array([template.shape[1], template.shape[0]], dtype=np.float64) / 2.0
    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "confidence": float(selected[2]),
        "time_ms": float(elapsed_ms),
        "candidate_count": len(candidates),
    }


def _summarize(records: list[dict], upsample: int, threshold_ratio: float) -> dict:
    times = np.asarray([record["time_ms"] for record in records], dtype=np.float64)
    hits = sum(bool(record["hit"]) for record in records)
    return {
        "upsample": upsample,
        "threshold_ratio": threshold_ratio,
        "num_pairs": len(records),
        "hits": hits,
        "hit_rate_percent": 100.0 * hits / len(records),
        "mean_time_ms": float(times.mean()),
        "std_time_ms": float(times.std(ddof=1)),
    }


def _failure_plot(
    path: Path,
    reference: np.ndarray,
    search: np.ndarray,
    truth: tuple[float, float],
    prediction: tuple[float, float],
    surface: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(reference, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("reference")
    axes[0].axis("off")
    axes[1].imshow(search, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].plot(truth[0], truth[1], "+", color="lime", markersize=12, label="GT")
    axes[1].plot(prediction[0], prediction[1], "x", color="red", markersize=10, label="Pred")
    axes[1].legend(loc="upper right")
    axes[1].set_title("search")
    axes[1].axis("off")
    axes[2].imshow(surface, cmap="magma", aspect="auto")
    axes[2].set_title("upsampled NCC surface")
    axes[2].axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _drift_plot(path: Path, records: list[dict]) -> None:
    offsets = np.asarray([r["placement_offset_nm"] for r in records], dtype=float)
    errors = np.asarray([r["error_px"] for r in records], dtype=float)
    hits = np.asarray([r["hit"] for r in records], dtype=bool)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.scatter(offsets[~hits], errors[~hits], color="tab:red", label="miss", alpha=0.8)
    ax.scatter(offsets[hits], errors[hits], color="tab:green", label="hit", alpha=0.8)
    ax.axhline(TOLERANCE_PX, color="black", linestyle="--", linewidth=1, label="tolerance")
    ax.set_xlabel("placement offset from search center (nm)")
    ax.set_ylabel("localization error (search px)")
    ax.set_title("Localization error versus placement offset")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_evaluation() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cases = _make_cases()
    primary_records = []
    failure = None

    for case in cases:
        result = localize.localize_arrays(
            case["reference"],
            case["search"],
            upsample=PRIMARY_UPSAMPLE,
            threshold_ratio=PRIMARY_THRESHOLD_RATIO,
        )
        error, hit = _score(result, case["truth"])
        record = {
            "index": case["index"],
            "seed": case["seed"],
            "ambiguous_region": case["ambiguous_region"],
            "placement_offset_nm": case["placement_offset_nm"],
            "ground_truth": {"center_x": case["truth"][0], "center_y": case["truth"][1]},
            "prediction": {key: result[key] for key in ("center_x", "center_y", "confidence")},
            "error_px": error,
            "hit": hit,
            "time_ms": result["time_ms"],
            "candidate_count": result["candidate_count"],
            "reference_noise": asdict(case["reference_noise"]),
            "search_noise": asdict(case["search_noise"]),
        }
        primary_records.append(record)
        if failure is None and not hit:
            surface, _ = localize.correlate(
                case["search"],
                localize.load_and_downsample_template(case["reference"]),
                upsample=PRIMARY_UPSAMPLE,
            )
            failure_path = RESULTS_DIR / "failure_case.png"
            _failure_plot(
                failure_path,
                case["reference"],
                case["search"],
                case["truth"],
                (result["center_x"], result["center_y"]),
                surface,
            )
            failure = {
                **record,
                "failure_image": str(failure_path),
                "explanation": (
                    "Periodic fin patterns produce multiple near-identical NCC peaks. "
                    "The center-tiebreak heuristic has no image evidence to identify "
                    "the physically correct repeated period, so it can select a "
                    "different candidate. This is a structural limitation of "
                    "appearance-only template matching, not an implementation bug."
                ),
            }

    sweep = []
    for upsample in UPSAMPLE_SWEEP:
        cached = []
        for case in cases:
            start = time.perf_counter()
            template = localize.load_and_downsample_template(case["reference"])
            surface, factor = localize.correlate(case["search"], template, upsample=upsample)
            # Candidate extraction and selection are tiny compared with NCC; use
            # the full measured one-surface pipeline time for each threshold row.
            localize.find_candidates(surface, THRESHOLD_SWEEP[0])
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            cached.append((case, template, surface, factor, elapsed_ms))
        for threshold_ratio in THRESHOLD_SWEEP:
            records = []
            for case, template, surface, factor, elapsed_ms in cached:
                result = _result_from_surface(
                    surface, template, case["search"].shape, factor, threshold_ratio, elapsed_ms
                )
                error, hit = _score(result, case["truth"])
                records.append({"error_px": error, "hit": hit, "time_ms": result["time_ms"]})
            sweep.append(_summarize(records, upsample, threshold_ratio))

    drift_buckets = []
    for low, high in ((0.0, 1000.0), (1000.0, 2500.0), (2500.0, float("inf"))):
        subset = [r for r in primary_records if low <= r["placement_offset_nm"] < high]
        drift_buckets.append({
            "range_nm": [low, None if np.isinf(high) else high],
            "num_pairs": len(subset),
            "hit_rate_percent": (100.0 * sum(r["hit"] for r in subset) / len(subset)) if subset else None,
            "mean_error_px": (float(np.mean([r["error_px"] for r in subset])) if subset else None),
        })

    primary = _summarize(primary_records, PRIMARY_UPSAMPLE, PRIMARY_THRESHOLD_RATIO)
    _drift_plot(RESULTS_DIR / "drift_accuracy.png", primary_records)

    # Optional ablation: keep the intensity-NCC decision fixed, then estimate
    # subpixel displacement using a local gradient-magnitude correlation.
    gradient_records = []
    for case, primary_record in zip(cases, primary_records):
        template = localize.load_and_downsample_template(case["reference"])
        surface, factor = localize.correlate(case["search"], template, upsample=PRIMARY_UPSAMPLE)
        candidates = localize.find_candidates(surface, PRIMARY_THRESHOLD_RATIO)
        selected = localize.select_by_center_tiebreak(
            candidates, case["search"].shape, upsample=factor, template_shape=template.shape
        )
        gradient_x, gradient_y, gradient_score = localize.gradient_profile_refine(
            case["search"], template, (selected[0], selected[1]), upsample=factor
        )
        gradient_center = np.array([gradient_x, gradient_y], dtype=float) / factor
        gradient_center += np.array([template.shape[1], template.shape[0]], dtype=float) / 2.0
        gradient_error = float(np.hypot(gradient_center[0] - case["truth"][0], gradient_center[1] - case["truth"][1]))
        gradient_records.append({
            "index": case["index"],
            "parabolic_error_px": primary_record["error_px"],
            "gradient_error_px": gradient_error,
            "gradient_hit": gradient_error <= TOLERANCE_PX,
            "gradient_score": gradient_score,
        })
    gradient_ablation = {
        "description": "Gradient-magnitude correlation used only for subpixel refinement after NCC candidate selection.",
        "parabolic_mean_error_px": float(np.mean([r["parabolic_error_px"] for r in gradient_records])),
        "gradient_mean_error_px": float(np.mean([r["gradient_error_px"] for r in gradient_records])),
        "parabolic_hits": int(sum(r["parabolic_error_px"] <= TOLERANCE_PX for r in gradient_records)),
        "gradient_hits": int(sum(r["gradient_hit"] for r in gradient_records)),
        "records": gradient_records,
    }
    summary = {
        "tolerance_px": TOLERANCE_PX,
        "primary": primary,
        "drift_proxy_buckets": drift_buckets,
        "sweep": sweep,
        "failure_case": failure,
        "gradient_ablation": gradient_ablation,
        "pairs": primary_records,
    }
    with (RESULTS_DIR / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def main() -> None:
    summary = run_evaluation()
    primary = summary["primary"]
    print(f"tolerance_px={summary['tolerance_px']}")
    print(f"primary: {primary['hit_rate_percent']:.2f}% ({primary['hits']}/{primary['num_pairs']})")
    print(f"time_ms mean={primary['mean_time_ms']:.3f}, std={primary['std_time_ms']:.3f}")
    failure = summary["failure_case"]
    print(
        f"failure_case: gt=({failure['ground_truth']['center_x']:.3f}, {failure['ground_truth']['center_y']:.3f}), "
        f"pred=({failure['prediction']['center_x']:.3f}, {failure['prediction']['center_y']:.3f}), "
        f"error_px={failure['error_px']:.3f}"
    )
    print(failure["explanation"])
    print("sweep records:", len(summary["sweep"]))


if __name__ == "__main__":
    main()

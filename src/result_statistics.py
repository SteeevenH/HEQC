"""
Output serialization, prediction statistics, and reproducibility utilities.

Provides atomic file writing, environment metadata, label-level statistics,
and aggregated inference summaries.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import openslide
import PIL
import torch
import torchvision

from constants import LABELS


def _json_default(value: Any) -> Any:
    """Convert supported non-native types to JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json_atomic(payload: dict[str, Any], path: Path) -> None:
    """Write JSON through a temporary file before replacing the destination."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            default=_json_default,
        )
        handle.write("\n")
    temporary.replace(path)


def save_csv_atomic(
    rows: Iterable[dict[str, Any]],
    path: Path,
    fieldnames: list[str],
) -> None:
    """Write a UTF-8 CSV file using atomic replacement."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def stable_signature(payload: dict[str, Any]) -> str:
    """Generate a deterministic SHA-256 signature from serialized metadata."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def relative_project_path(path: Path, project_root: Path) -> str:
    """Return a project-relative path or fall back to the filename."""
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return path.name


def file_identity(path: Path) -> dict[str, Any]:
    """Collect filesystem metadata used to identify an input file."""
    stat = path.stat()
    return {
        "filename": path.name,
        "size_bytes": int(stat.st_size),
        "modified_time_ns": int(stat.st_mtime_ns),
    }


def environment_versions() -> dict[str, Any]:
    """Record software versions for inference reproducibility."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "matplotlib": matplotlib.__version__,
        "openslide_python": openslide.__version__,
        "openslide_library": getattr(openslide, "__library_version__", None),
    }


LABEL_STATISTICS_FIELDS = [
    "model_name",
    "slide_id",
    "threshold_profile",
    "label_index",
    "label",
    "threshold",
    "positive_patch_count",
    "positive_patch_fraction",
    "probability_mean",
    "probability_max",
    "probability_p95",
    "positive_heatmap_pixel_fraction",
]


def build_label_statistics(
    probabilities: np.ndarray,
    thresholds: dict[str, float],
    heatmap_positive_fractions: dict[str, float],
    model_name: str,
    slide_id: str,
    threshold_profile: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Summarize per-label predictions as tabular and nested statistics."""
    if probabilities.shape[1] != len(LABELS):
        raise ValueError("Probability matrix does not contain five labels")

    rows: list[dict[str, Any]] = []
    nested: dict[str, dict[str, Any]] = {}

    for index, label in enumerate(LABELS):
        values = probabilities[:, index]
        threshold = float(thresholds[label])
        positive_count = int((values >= threshold).sum())

        item = {
            "model_name": model_name,
            "slide_id": slide_id,
            "threshold_profile": threshold_profile,
            "label_index": index + 1,
            "label": label,
            "threshold": threshold,
            "positive_patch_count": positive_count,
            "positive_patch_fraction": float(positive_count / len(values)),
            "probability_mean": float(values.mean()),
            "probability_max": float(values.max()),
            "probability_p95": float(np.quantile(values, 0.95)),
            "positive_heatmap_pixel_fraction": float(heatmap_positive_fractions[label]),
        }

        rows.append(item)
        nested[label] = {
            key: value
            for key, value in item.items()
            if key
            not in {
                "model_name",
                "slide_id",
                "threshold_profile",
                "label_index",
                "label",
            }
        }

    return rows, nested


RUN_SUMMARY_FIELDS = [
    "model_name",
    "slide_id",
    "status",
    "threshold_profile",
    "actual_magnification",
    "selected_mpp",
    "total_patches",
    "abnormal_patch_count",
    "abnormal_patch_fraction",
    "overall_positive_heatmap_pixel_fraction",
    "model_prediction_seconds",
    "other_processing_seconds",
    "total_seconds",
    "model_only_patches_per_second",
    "end_to_end_patches_per_second",
    "warning_count",
    "error",
]


def _flatten_run_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested WSI statistics into a single summary record."""
    status = str(payload.get("status", "unknown"))
    wsi = payload.get("wsi") if isinstance(payload.get("wsi"), dict) else {}
    patch = (
        payload.get("patch_statistics")
        if isinstance(payload.get("patch_statistics"), dict)
        else {}
    )
    heatmap = (
        payload.get("heatmap_statistics")
        if isinstance(payload.get("heatmap_statistics"), dict)
        else {}
    )
    timing = payload.get("timing") if isinstance(payload.get("timing"), dict) else {}
    warnings = payload.get("warnings")

    return {
        "model_name": payload.get("model_name"),
        "slide_id": payload.get("slide_id"),
        "status": status,
        "threshold_profile": payload.get("threshold_profile"),
        "actual_magnification": wsi.get("actual_magnification"),
        "selected_mpp": wsi.get("selected_mpp"),
        "total_patches": patch.get("total_patches"),
        "abnormal_patch_count": patch.get("abnormal_patch_count"),
        "abnormal_patch_fraction": patch.get("abnormal_patch_fraction"),
        "overall_positive_heatmap_pixel_fraction": heatmap.get(
            "overall_positive_pixel_fraction"
        ),
        "model_prediction_seconds": timing.get("model_prediction_seconds"),
        "other_processing_seconds": timing.get("other_processing_seconds"),
        "total_seconds": timing.get("total_seconds"),
        "model_only_patches_per_second": timing.get("model_only_patches_per_second"),
        "end_to_end_patches_per_second": timing.get("end_to_end_patches_per_second"),
        "warning_count": len(warnings) if isinstance(warnings, list) else 0,
        "error": payload.get("error"),
    }


def rebuild_run_summary(model_output_dir: Path) -> None:
    """Rebuild the model-level CSV summary from individual slide statistics."""
    rows: list[dict[str, Any]] = []

    if model_output_dir.is_dir():
        for slide_dir in sorted(
            (path for path in model_output_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name.lower(),
        ):
            statistics_path = slide_dir / "statistics.json"
            if not statistics_path.is_file():
                continue

            try:
                with statistics_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue

            if isinstance(payload, dict):
                rows.append(_flatten_run_summary(payload))

    save_csv_atomic(rows, model_output_dir / "run_summary.csv", RUN_SUMMARY_FIELDS)


def save_patch_predictions_npz(
    path: Path,
    coordinates: dict[str, np.ndarray],
    probabilities: np.ndarray,
) -> None:
    """Save patch coordinates and prediction probabilities in compressed NPZ format."""
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        probabilities=probabilities.astype(np.float32, copy=False),
        **coordinates,
    )
    temporary.replace(path)
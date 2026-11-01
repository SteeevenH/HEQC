"""
Orchestrate WSI inference, heatmap generation, and output management.

Validates configuration, records reproducibility metadata, and coordinates
per-slide processing, result reuse, and failure reporting.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import config as cfg
from constants import (
    LABEL_FILE_SLUGS,
    LABELS,
    MODEL_INPUT_PATCH_SIZE,
    PROJECT_VERSION,
    STATISTICS_SCHEMA_VERSION,
    SUPPORTED_MODEL_NAMES,
    THRESHOLD_PROFILES,
)
from geometry import GridGeometry
from heatmaps import SpatialHeatmaps, build_spatial_heatmaps
from inference import run_patch_inference
from models import DeviceInfo, file_sha256, load_model, resolve_device
from result_statistics import (
    LABEL_STATISTICS_FIELDS,
    build_label_statistics,
    environment_versions,
    file_identity,
    rebuild_run_summary,
    relative_project_path,
    save_csv_atomic,
    save_json_atomic,
    save_patch_predictions_npz,
    stable_signature,
)
from thresholds import load_threshold_profiles, select_threshold_profile
from visualization import (
    format_magnification,
    save_heatmap_overlay,
    save_thumbnail,
)
from wsi import WSIInfo, find_wsi_files, inspect_wsi, read_image_at_magnification


class ExistingOutputError(RuntimeError):
    """Raised when existing slide results cannot be reused or replaced."""
    pass


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_config() -> None:
    """Validate pipeline settings before processing any slides."""
    if cfg.MODEL_NAME not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            f"MODEL_NAME must be one of {SUPPORTED_MODEL_NAMES}, "
            f"received {cfg.MODEL_NAME!r}"
        )
    if cfg.THRESHOLD_PROFILE not in THRESHOLD_PROFILES:
        raise ValueError(f"THRESHOLD_PROFILE must be one of {THRESHOLD_PROFILES}")
    positive_values = {
        "TARGET_MAGNIFICATION": cfg.TARGET_MAGNIFICATION,
        "STRIDE": cfg.STRIDE,
        "BATCH_SIZE": cfg.BATCH_SIZE,
        "PREFETCH_FACTOR": cfg.PREFETCH_FACTOR,
        "HEATMAP_MAGNIFICATION": cfg.HEATMAP_MAGNIFICATION,
        "THUMBNAIL_MAGNIFICATION": cfg.THUMBNAIL_MAGNIFICATION,
        "READ_TILE_SIZE": cfg.READ_TILE_SIZE,
        "HEATMAP_ALPHA_POWER": cfg.HEATMAP_ALPHA_POWER,
    }
    invalid = {name: value for name, value in positive_values.items() if value <= 0}
    if invalid:
        raise ValueError(f"These configuration values must be positive: {invalid}")
    if cfg.STRIDE > MODEL_INPUT_PATCH_SIZE:
        raise ValueError(
            f"STRIDE cannot exceed model patch size {MODEL_INPUT_PATCH_SIZE}"
        )
    if cfg.NUM_WORKERS < 0:
        raise ValueError("NUM_WORKERS cannot be negative")
    if cfg.CUDA_DEVICE_INDEX < 0:
        raise ValueError("CUDA_DEVICE_INDEX cannot be negative")
    if not 0.0 < cfg.RECOMMENDED_MPP_MIN <= cfg.RECOMMENDED_MPP_MAX:
        raise ValueError("The recommended MPP range is invalid")
    if not 0.0 <= cfg.HEATMAP_MAX_ALPHA <= 1.0:
        raise ValueError("HEATMAP_MAX_ALPHA must be within [0, 1]")
    if cfg.HEATMAP_BASE_SATURATION <= 0 or cfg.HEATMAP_BASE_BRIGHTNESS <= 0:
        raise ValueError("Heatmap base saturation and brightness must be positive")
    if cfg.SMOOTHING_RADIUS_OVERRIDE is not None and (
        cfg.SMOOTHING_RADIUS_OVERRIDE < 0
    ):
        raise ValueError("SMOOTHING_RADIUS_OVERRIDE cannot be negative")
    if not 0 <= cfg.PNG_COMPRESSION <= 9:
        raise ValueError("PNG_COMPRESSION must be between 0 and 9")
    if not isinstance(cfg.OVERWRITE, bool):
        raise TypeError("OVERWRITE must be True or False")


def configuration_snapshot(device_info: DeviceInfo) -> dict[str, Any]:
    """Record the effective configuration and selected compute device."""
    return {
        "model_name": cfg.MODEL_NAME,
        "threshold_profile": cfg.THRESHOLD_PROFILE,
        "paths": {
            "wsi_directory": "wsi",
            "model_directory": "model",
            "output_directory": "outputs",
            "model_file": relative_project_path(cfg.model_path(), cfg.PROJECT_ROOT),
            "threshold_file": relative_project_path(
                cfg.threshold_path(), cfg.PROJECT_ROOT
            ),
        },
        "prediction": {
            "target_magnification": float(cfg.TARGET_MAGNIFICATION),
            "patch_size": MODEL_INPUT_PATCH_SIZE,
            "stride": int(cfg.STRIDE),
            "recommended_mpp_min": float(cfg.RECOMMENDED_MPP_MIN),
            "recommended_mpp_max": float(cfg.RECOMMENDED_MPP_MAX),
            "manual_objective_power": cfg.MANUAL_OBJECTIVE_POWER,
            "manual_base_mpp": cfg.MANUAL_BASE_MPP,
        },
        "hardware": {
            **device_info.to_dict(),
            "configured_device": cfg.DEVICE,
            "batch_size": int(cfg.BATCH_SIZE),
            "num_workers": int(cfg.NUM_WORKERS),
            "prefetch_factor": int(cfg.PREFETCH_FACTOR),
            "pin_memory": bool(cfg.PIN_MEMORY),
            "persistent_workers": bool(cfg.PERSISTENT_WORKERS),
            "use_amp": bool(cfg.USE_AMP),
        },
        "images": {
            "heatmap_magnification": float(cfg.HEATMAP_MAGNIFICATION),
            "thumbnail_magnification": float(cfg.THUMBNAIL_MAGNIFICATION),
            "read_tile_size": int(cfg.READ_TILE_SIZE),
            "heatmap_max_alpha": float(cfg.HEATMAP_MAX_ALPHA),
            "heatmap_alpha_power": float(cfg.HEATMAP_ALPHA_POWER),
            "heatmap_base_saturation": float(cfg.HEATMAP_BASE_SATURATION),
            "heatmap_base_brightness": float(cfg.HEATMAP_BASE_BRIGHTNESS),
            "smoothing_radius_override": cfg.SMOOTHING_RADIUS_OVERRIDE,
            "png_compression": int(cfg.PNG_COMPRESSION),
        },
        "runtime": {
            "overwrite": bool(cfg.OVERWRITE),
            "continue_on_error": bool(cfg.CONTINUE_ON_ERROR),
            "save_patch_predictions": bool(cfg.SAVE_PATCH_PREDICTIONS),
            "random_seed": int(cfg.RANDOM_SEED),
        },
    }


def processing_signature_payload(
    config_snapshot: dict[str, Any],
    model_sha256: str,
    threshold_sha256: str,
    selected_thresholds: dict[str, float],
    wsi_path: Path,
) -> dict[str, Any]:
    """Build the metadata used to identify reusable slide results."""
    # Exclude output-management flags from the processing signature.
    signature_config = json.loads(json.dumps(config_snapshot))
    signature_config["runtime"] = {
        "save_patch_predictions": bool(cfg.SAVE_PATCH_PREDICTIONS),
        "random_seed": int(cfg.RANDOM_SEED),
    }
    return {
        "project_version": PROJECT_VERSION,
        "configuration": signature_config,
        "model_sha256": model_sha256,
        "threshold_sha256": threshold_sha256,
        "selected_thresholds": selected_thresholds,
        "wsi_file_identity": file_identity(wsi_path),
    }


def _existing_statistics(output_dir: Path) -> dict[str, Any] | None:
    """Read an existing slide statistics record, if valid."""
    path = output_dir / "statistics.json"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _output_files_exist(output_dir: Path, payload: dict[str, Any]) -> bool:
    """Check that all files declared by a slide statistics record exist."""
    output_files = payload.get("output_files")
    if not isinstance(output_files, list) or not output_files:
        return False
    return all(
        (output_dir / str(relative_path)).is_file() for relative_path in output_files
    )


def _safe_remove_generated_output(output_dir: Path, model_output_dir: Path) -> None:
    """Delete only a direct child of the model output directory."""
    resolved_output = output_dir.resolve()
    resolved_parent = model_output_dir.resolve()
    if resolved_output == resolved_parent or resolved_output.parent != resolved_parent:
        raise RuntimeError(f"Refusing to remove unsafe output path: {resolved_output}")
    shutil.rmtree(resolved_output)


def prepare_output_directory(
    model_output_dir: Path,
    slide_id: str,
    signature: str,
) -> tuple[Path, bool]:
    """
    Create a slide output directory or reuse complete matching results.

    Returns:
        The output directory and whether processing should be skipped.
    """
    model_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = model_output_dir / slide_id
    if output_dir.exists():
        existing = _existing_statistics(output_dir)
        complete = bool(
            existing
            and existing.get("status") == "complete"
            and _output_files_exist(output_dir, existing)
        )
        same_signature = bool(
            existing and existing.get("configuration_signature") == signature
        )
        if complete and same_signature and not cfg.OVERWRITE:
            return output_dir, True
        if not cfg.OVERWRITE:
            if complete and not same_signature:
                reason = "existing results were created with a different configuration"
            else:
                reason = "existing results are incomplete or failed"
            raise ExistingOutputError(
                f"Cannot write {output_dir}: {reason}. Set OVERWRITE = True in "
                "src/config.py to replace this generated WSI output."
            )
        _safe_remove_generated_output(output_dir, model_output_dir)

    output_dir.mkdir(parents=False, exist_ok=False)
    (output_dir / "images").mkdir(parents=False, exist_ok=False)
    return output_dir, False


def _threshold_vector(thresholds: dict[str, float]) -> np.ndarray:
    """Arrange class thresholds in the predefined label order."""
    return np.asarray([thresholds[label] for label in LABELS], dtype=np.float32)


def _write_failure_statistics(
    output_dir: Path,
    slide_id: str,
    signature: str,
    config_snapshot: dict[str, Any],
    error: Exception,
    elapsed_seconds: float,
) -> None:
    """Write a slide statistics record for a failed WSI processing attempt."""
    payload = {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "project_version": PROJECT_VERSION,
        "created_at_utc": _now_utc(),
        "status": "failed",
        "model_name": cfg.MODEL_NAME,
        "slide_id": slide_id,
        "threshold_profile": cfg.THRESHOLD_PROFILE,
        "configuration_signature": signature,
        "run_config": config_snapshot,
        "warnings": [],
        "error": f"{type(error).__name__}: {error}",
        "timing": {
            "model_prediction_seconds": None,
            "other_processing_seconds": float(elapsed_seconds),
            "total_seconds": float(elapsed_seconds),
            "model_only_patches_per_second": None,
            "end_to_end_patches_per_second": None,
        },
        "output_files": ["statistics.json"],
    }
    save_json_atomic(payload, output_dir / "statistics.json")


def _save_images(
    output_dir: Path,
    wsi_path: Path,
    info: WSIInfo,
    heatmaps: SpatialHeatmaps,
    heatmap_base: Image.Image,
    heatmap_image_info: dict[str, Any],
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    """
    Save thumbnail, class-specific heatmaps, and the overall heatmap.

    Returns:
        Saved relative paths and metadata for both image canvases.
    """
    image_dir = output_dir / "images"
    if heatmap_base.size != (
        heatmaps.coverage_mask.shape[1],
        heatmaps.coverage_mask.shape[0],
    ):
        heatmap_base.close()
        raise RuntimeError("Heatmap base dimensions changed during processing")

    if math.isclose(
        cfg.THUMBNAIL_MAGNIFICATION,
        cfg.HEATMAP_MAGNIFICATION,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        thumbnail_base = heatmap_base.copy()
        thumbnail_image_info = dict(heatmap_image_info)
        thumbnail_image_info["reused_heatmap_base"] = True
    else:
        thumbnail_base, thumbnail_image_info = read_image_at_magnification(
            wsi_path,
            info,
            cfg.THUMBNAIL_MAGNIFICATION,
            cfg.READ_TILE_SIZE,
            f"Reading thumbnail for {info.stem}",
        )
        thumbnail_image_info["reused_heatmap_base"] = False

    output_files: list[str] = []
    try:
        thumbnail_name = (
            f"thumbnail_{format_magnification(cfg.THUMBNAIL_MAGNIFICATION)}.png"
        )
        save_thumbnail(
            thumbnail_base,
            image_dir / thumbnail_name,
            info.stem,
            cfg.THUMBNAIL_MAGNIFICATION,
            cfg.PNG_COMPRESSION,
        )
        output_files.append(f"images/{thumbnail_name}")

        for index, label in enumerate(LABELS, start=1):
            filename = f"heatmap_{index:02d}_{LABEL_FILE_SLUGS[label]}.png"
            save_heatmap_overlay(
                heatmap_base,
                heatmaps.relative_maps[label],
                heatmaps.coverage_mask,
                image_dir / filename,
                f"Artifact heatmap {index:02d} — {label}",
                cfg.MODEL_NAME,
                cfg.THRESHOLD_PROFILE,
                cfg.HEATMAP_MAGNIFICATION,
                cfg.HEATMAP_MAX_ALPHA,
                cfg.HEATMAP_ALPHA_POWER,
                cfg.HEATMAP_BASE_SATURATION,
                cfg.HEATMAP_BASE_BRIGHTNESS,
                cfg.PNG_COMPRESSION,
            )
            output_files.append(f"images/{filename}")

        overall_name = "heatmap_overall.png"
        save_heatmap_overlay(
            heatmap_base,
            heatmaps.overall_relative_map,
            heatmaps.coverage_mask,
            image_dir / overall_name,
            "Overall artifact heatmap",
            cfg.MODEL_NAME,
            cfg.THRESHOLD_PROFILE,
            cfg.HEATMAP_MAGNIFICATION,
            cfg.HEATMAP_MAX_ALPHA,
            cfg.HEATMAP_ALPHA_POWER,
            cfg.HEATMAP_BASE_SATURATION,
            cfg.HEATMAP_BASE_BRIGHTNESS,
            cfg.PNG_COMPRESSION,
        )
        output_files.append(f"images/{overall_name}")
    finally:
        thumbnail_base.close()
    return output_files, thumbnail_image_info, heatmap_image_info


def process_one_wsi(
    wsi_path: Path,
    output_dir: Path,
    signature: str,
    config_snapshot: dict[str, Any],
    model: torch.nn.Module,
    device_info: DeviceInfo,
    selected_thresholds: dict[str, float],
    model_sha256: str,
    threshold_sha256: str,
    model_parameter_count: int,
) -> dict[str, Any]:
    """
    Process one WSI from slide inspection through persisted results.

    Runs patch inference, generates spatial heatmaps, and saves images,
    prediction statistics, and reproducibility metadata.
    """
    started = time.perf_counter()
    info = inspect_wsi(
        wsi_path,
        cfg.TARGET_MAGNIFICATION,
        cfg.RECOMMENDED_MPP_MIN,
        cfg.RECOMMENDED_MPP_MAX,
        cfg.MANUAL_OBJECTIVE_POWER,
        cfg.MANUAL_BASE_MPP,
    )
    for warning in info.warnings:
        print(f"Warning [{info.filename}]: {warning}")

    geometry = GridGeometry.create(
        info.prediction_level_width,
        info.prediction_level_height,
        MODEL_INPUT_PATCH_SIZE,
        cfg.STRIDE,
    )
    print(
        f"{info.filename}: prediction level {info.prediction_level}, "
        f"{info.actual_magnification:.3f}x, {len(geometry):,} patches"
    )

    probabilities, model_prediction_seconds = run_patch_inference(
        wsi_path=wsi_path,
        slide_id=info.stem,
        model=model,
        device_info=device_info,
        prediction_level=info.prediction_level,
        prediction_downsample=info.prediction_level_downsample,
        geometry=geometry,
        num_labels=len(LABELS),
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        prefetch_factor=cfg.PREFETCH_FACTOR,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=cfg.PERSISTENT_WORKERS,
        use_amp=cfg.USE_AMP,
    )

    # Keep heatmap geometry independent of thumbnail resolution.
    heatmap_base, heatmap_image_info = read_image_at_magnification(
        wsi_path,
        info,
        cfg.HEATMAP_MAGNIFICATION,
        cfg.READ_TILE_SIZE,
        f"Preparing heatmap geometry for {info.stem}",
    )
    try:
        heatmaps = build_spatial_heatmaps(
            probabilities,
            geometry,
            heatmap_base.size,
            selected_thresholds,
            cfg.SMOOTHING_RADIUS_OVERRIDE,
        )
        image_files, thumbnail_image_info, heatmap_image_info = _save_images(
            output_dir,
            wsi_path,
            info,
            heatmaps,
            heatmap_base,
            heatmap_image_info,
        )
    finally:
        heatmap_base.close()

    label_rows, label_statistics = build_label_statistics(
        probabilities,
        selected_thresholds,
        heatmaps.label_positive_pixel_fraction,
        cfg.MODEL_NAME,
        info.stem,
        cfg.THRESHOLD_PROFILE,
    )
    save_csv_atomic(
        label_rows,
        output_dir / "label_statistics.csv",
        LABEL_STATISTICS_FIELDS,
    )

    output_files = [*image_files, "label_statistics.csv"]
    if cfg.SAVE_PATCH_PREDICTIONS:
        prediction_name = "patch_predictions.npz"
        save_patch_predictions_npz(
            output_dir / prediction_name,
            geometry.arrays(),
            probabilities,
        )
        output_files.append(prediction_name)

    threshold_vector = _threshold_vector(selected_thresholds)
    patch_decisions = probabilities >= threshold_vector.reshape(1, -1)
    abnormal_patch_mask = patch_decisions.any(axis=1)
    abnormal_patch_count = int(abnormal_patch_mask.sum())
    total_patches = len(geometry)
    elapsed = time.perf_counter() - started
    other_processing_seconds = max(0.0, elapsed - model_prediction_seconds)
    model_rate = (
        total_patches / model_prediction_seconds
        if model_prediction_seconds > 0
        else None
    )
    end_to_end_rate = total_patches / elapsed if elapsed > 0 else None

    statistics_payload = {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "project_version": PROJECT_VERSION,
        "created_at_utc": _now_utc(),
        "status": "complete",
        "model_name": cfg.MODEL_NAME,
        "slide_id": info.stem,
        "threshold_profile": cfg.THRESHOLD_PROFILE,
        "configuration_signature": signature,
        "run_config": config_snapshot,
        "wsi": info.to_dict(),
        "model": {
            "filename": cfg.model_path().name,
            "sha256": model_sha256,
            "architecture": cfg.MODEL_NAME,
            "output_count": len(LABELS),
            "parameter_count": int(model_parameter_count),
            "input_patch_size": MODEL_INPUT_PATCH_SIZE,
            "normalization": "ImageNet mean and standard deviation",
        },
        "thresholds": {
            "filename": cfg.threshold_path().name,
            "sha256": threshold_sha256,
            "selected_profile": cfg.THRESHOLD_PROFILE,
            "values": selected_thresholds,
        },
        "device": device_info.to_dict(),
        "software": environment_versions(),
        "grid": geometry.to_dict(),
        "patch_statistics": {
            "total_patches": total_patches,
            "padded_patch_count": int(geometry.arrays()["was_padded"].sum()),
            "abnormal_rule": "one_or_more_labels_at_or_above_threshold",
            "abnormal_patch_count": abnormal_patch_count,
            "abnormal_patch_fraction": float(abnormal_patch_count / total_patches),
            "normal_patch_count": int(total_patches - abnormal_patch_count),
            "normal_patch_fraction": float(
                (total_patches - abnormal_patch_count) / total_patches
            ),
        },
        "label_statistics": label_statistics,
        "heatmap_statistics": {
            "overall_positive_pixel_fraction": (
                heatmaps.overall_positive_pixel_fraction
            ),
            "positive_pixel_fraction_by_label": (
                heatmaps.label_positive_pixel_fraction
            ),
            "spatial_aggregation": heatmaps.metadata,
        },
        "images": {
            "thumbnail": thumbnail_image_info,
            "heatmap": heatmap_image_info,
            "title_and_scale_location": "header_outside_wsi_canvas",
        },
        "timing": {
            "definition": {
                "model_prediction_seconds": (
                    "Pure synchronized model forward time after the input batch "
                    "has been queued on the selected device."
                ),
                "other_processing_seconds": (
                    "WSI inspection, patch reading, preprocessing, transfers, "
                    "probability handling, spatial aggregation, image generation, "
                    "and tabular output, excluding model forward time."
                ),
            },
            "model_prediction_seconds": float(model_prediction_seconds),
            "other_processing_seconds": float(other_processing_seconds),
            "total_seconds": float(elapsed),
            "model_only_patches_per_second": (
                float(model_rate) if model_rate is not None else None
            ),
            "end_to_end_patches_per_second": (
                float(end_to_end_rate) if end_to_end_rate is not None else None
            ),
        },
        "warnings": list(info.warnings),
        "error": None,
        "output_files": [*output_files, "statistics.json"],
    }
    save_json_atomic(statistics_payload, output_dir / "statistics.json")
    return statistics_payload


def run_pipeline() -> int:
    """
    Run inference for all discovered slides using the configured model.

    Returns:
        Exit code 0 if no slides fail, otherwise 1.
    """
    validate_config()
    cfg.WSI_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)

    model_path = cfg.model_path()
    threshold_path = cfg.threshold_path()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    profiles = load_threshold_profiles(threshold_path)
    selected_thresholds = select_threshold_profile(
        profiles,
        cfg.THRESHOLD_PROFILE,
    )
    device_info = resolve_device(cfg.DEVICE, cfg.CUDA_DEVICE_INDEX)
    model_sha256 = file_sha256(model_path)
    threshold_sha256 = file_sha256(threshold_path)
    config_snapshot = configuration_snapshot(device_info)

    print(f"Project: {cfg.PROJECT_ROOT}")
    print(f"Model: {cfg.MODEL_NAME} ({model_path.name})")
    print(f"Threshold profile: {cfg.THRESHOLD_PROFILE}")
    print(f"Device: {device_info.device} — {device_info.name}")
    print(f"Patch size / stride: {MODEL_INPUT_PATCH_SIZE} / {cfg.STRIDE}")
    print(f"Thumbnail: {cfg.THUMBNAIL_MAGNIFICATION:g}x")
    print(f"Heatmaps: {cfg.HEATMAP_MAGNIFICATION:g}x")

    model = load_model(model_path, cfg.MODEL_NAME, len(LABELS), device_info)
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    wsi_files = find_wsi_files(cfg.WSI_DIR)
    model_output_dir = cfg.OUTPUT_DIR / cfg.MODEL_NAME
    model_output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    failures: list[tuple[str, str]] = []
    for wsi_path in wsi_files:
        signature_payload = processing_signature_payload(
            config_snapshot,
            model_sha256,
            threshold_sha256,
            selected_thresholds,
            wsi_path,
        )
        signature = stable_signature(signature_payload)
        output_dir: Path | None = None
        slide_started = time.perf_counter()
        try:
            output_dir, should_skip = prepare_output_directory(
                model_output_dir,
                wsi_path.stem,
                signature,
            )
            if should_skip:
                print(
                    f"Skipping {cfg.MODEL_NAME}/{wsi_path.stem}: complete results "
                    "match the current configuration."
                )
                skipped += 1
                continue
            payload = process_one_wsi(
                wsi_path,
                output_dir,
                signature,
                config_snapshot,
                model,
                device_info,
                selected_thresholds,
                model_sha256,
                threshold_sha256,
                model_parameter_count,
            )
            processed += 1
            print(
                f"Finished {cfg.MODEL_NAME}/{wsi_path.stem}: "
                f"{payload['grid']['total_patches']:,} patches, "
                f"{payload['timing']['total_seconds']:.2f} seconds"
            )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            failures.append((wsi_path.name, message))
            print(f"Failed {wsi_path.name}: {message}")
            if output_dir is not None and output_dir.is_dir():
                try:
                    _write_failure_statistics(
                        output_dir,
                        wsi_path.stem,
                        signature,
                        config_snapshot,
                        error,
                        time.perf_counter() - slide_started,
                    )
                except (OSError, TypeError, ValueError) as write_error:
                    print(
                        f"Could not write failure statistics for {wsi_path.name}: "
                        f"{type(write_error).__name__}: {write_error}"
                    )
            if not cfg.CONTINUE_ON_ERROR:
                rebuild_run_summary(model_output_dir)
                raise

    rebuild_run_summary(model_output_dir)
    print(
        f"Run complete: processed={processed}, skipped={skipped}, "
        f"failed={len(failures)}"
    )
    if failures:
        print("Failures:")
        for filename, message in failures:
            print(f"  {filename}: {message}")
        return 1
    return 0


def main() -> None:
    raise SystemExit(run_pipeline())

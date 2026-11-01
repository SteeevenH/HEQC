"""
Spatial aggregation and visualization of WSI patch predictions.

Generates class-specific and overall heatmaps using center-weighted
overlap averaging, Gaussian smoothing, and threshold normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from constants import HEATMAP_BLEND_BANDS, LABELS
from geometry import GridGeometry


def threshold_relative_scores(
    scores: np.ndarray,
    thresholds: np.ndarray | float,
) -> np.ndarray:
    """Normalize scores relative to thresholds, mapping each threshold to 0.5."""
    score_array = np.asarray(scores, dtype=np.float32)
    threshold_array = np.asarray(thresholds, dtype=np.float32)
    lower = 0.5 * score_array / np.maximum(threshold_array, 1e-8)
    upper = 0.5 + 0.5 * (score_array - threshold_array) / np.maximum(
        1.0 - threshold_array,
        1e-8,
    )
    return np.clip(
        np.where(score_array < threshold_array, lower, upper),
        0.0,
        1.0,
    ).astype(np.float32)


def _add_rectangles(
    difference: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    values: np.ndarray,
) -> None:
    """Accumulate weighted rectangles in a 2D difference array."""
    valid = (x2 > x1) & (y2 > y1)
    if not bool(np.any(valid)):
        return
    x1v = x1[valid]
    y1v = y1[valid]
    x2v = x2[valid]
    y2v = y2[valid]
    vv = values[valid].astype(np.float32, copy=False)
    np.add.at(difference, (y1v, x1v), vv)
    np.add.at(difference, (y1v, x2v), -vv)
    np.add.at(difference, (y2v, x1v), -vv)
    np.add.at(difference, (y2v, x2v), vv)


def _resolve_difference(difference: np.ndarray) -> np.ndarray:
    """Reconstruct pixel values using a 2D cumulative sum."""
    return difference.cumsum(axis=0, dtype=np.float32).cumsum(axis=1, dtype=np.float32)[
        :-1, :-1
    ]


def _smooth_score_map(score_map: np.ndarray, radius: float) -> np.ndarray:
    """Apply Gaussian smoothing to a normalized score map."""
    if radius < 0.5:
        return score_map

    # Convert to 8-bit grayscale for PIL-based Gaussian filtering.
    source = Image.fromarray(
        np.round(np.clip(score_map, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="L",
    )
    blurred = source.filter(ImageFilter.GaussianBlur(radius=radius))
    try:
        return np.asarray(blurred, dtype=np.float32) / 255.0
    finally:
        source.close()
        blurred.close()


@dataclass
class SpatialHeatmaps:
    """Container for spatial heatmaps, coverage statistics, and metadata."""
    relative_maps: dict[str, np.ndarray]
    overall_relative_map: np.ndarray
    coverage_mask: np.ndarray
    label_positive_pixel_fraction: dict[str, float]
    overall_positive_pixel_fraction: float
    metadata: dict[str, Any]


def build_spatial_heatmaps(
    patch_probabilities: np.ndarray,
    geometry: GridGeometry,
    output_size: tuple[int, int],
    thresholds: dict[str, float],
    smoothing_radius_override: float | None,
) -> SpatialHeatmaps:
    """
    Build class-specific and overall heatmaps from patch probabilities.

    Projects patch coordinates onto the output image, aggregates overlapping
    predictions using center-weighted averaging, and applies Gaussian
    smoothing followed by class-specific threshold normalization.
    """
    if patch_probabilities.shape != (len(geometry), len(LABELS)):
        raise ValueError(
            f"patch_probabilities shape is {patch_probabilities.shape}; expected "
            f"{(len(geometry), len(LABELS))}"
        )
    width, height = map(int, output_size)
    if width <= 0 or height <= 0:
        raise ValueError("Heatmap output dimensions must be positive")

    coordinates = geometry.arrays()
    scale_x = width / geometry.level_width
    scale_y = height / geometry.level_height
    raw_x = coordinates["x_level"].astype(np.float64)
    raw_y = coordinates["y_level"].astype(np.float64)
    raw_width = coordinates["valid_width"].astype(np.float64)
    raw_height = coordinates["valid_height"].astype(np.float64)

    x = np.floor(raw_x * scale_x).astype(np.int64)
    y = np.floor(raw_y * scale_y).astype(np.int64)
    patch_widths = np.maximum(
        1,
        np.ceil((raw_x + raw_width) * scale_x).astype(np.int64) - x,
    )
    patch_heights = np.maximum(
        1,
        np.ceil((raw_y + raw_height) * scale_y).astype(np.int64) - y,
    )

    def add_patch_bands(difference: np.ndarray, values: np.ndarray) -> None:
        for inset_fraction, incremental_weight in HEATMAP_BLEND_BANDS:
            inset_x = np.floor(patch_widths * inset_fraction).astype(np.int64)
            inset_y = np.floor(patch_heights * inset_fraction).astype(np.int64)
            x1 = np.clip(x + inset_x, 0, width)
            y1 = np.clip(y + inset_y, 0, height)
            x2 = np.clip(x + patch_widths - inset_x, 0, width)
            y2 = np.clip(y + patch_heights - inset_y, 0, height)
            _add_rectangles(
                difference,
                x1,
                y1,
                x2,
                y2,
                values * float(incremental_weight),
            )

    weight_difference = np.zeros((height + 1, width + 1), dtype=np.float32)
    add_patch_bands(
        weight_difference,
        np.ones(len(geometry), dtype=np.float32),
    )
    weight_map = _resolve_difference(weight_difference)
    coverage_mask = weight_map > 1e-8
    covered_pixels = int(coverage_mask.sum())
    if covered_pixels == 0:
        raise RuntimeError("Heatmap aggregation produced no covered pixels")

    projected_overlap_x = max(
        0.0,
        (geometry.patch_size - geometry.stride) * scale_x,
    )
    projected_overlap_y = max(
        0.0,
        (geometry.patch_size - geometry.stride) * scale_y,
    )
    if smoothing_radius_override is None:
        smoothing_radius = (projected_overlap_x + projected_overlap_y) / 8.0
    else:
        smoothing_radius = float(smoothing_radius_override)
        if smoothing_radius < 0:
            raise ValueError("SMOOTHING_RADIUS_OVERRIDE cannot be negative")

    relative_maps: dict[str, np.ndarray] = {}
    positive_fractions: dict[str, float] = {}
    overall_relative = np.zeros((height, width), dtype=np.float32)
    overall_positive = np.zeros((height, width), dtype=bool)

    for label_index, label in enumerate(LABELS):
        score_difference = np.zeros((height + 1, width + 1), dtype=np.float32)
        add_patch_bands(
            score_difference,
            patch_probabilities[:, label_index].astype(np.float32, copy=False),
        )
        weighted_sum = _resolve_difference(score_difference)
        class_map = np.zeros((height, width), dtype=np.float32)
        np.divide(weighted_sum, weight_map, out=class_map, where=coverage_mask)
        class_map = _smooth_score_map(class_map, smoothing_radius)
        class_map[~coverage_mask] = 0.0

        threshold = float(thresholds[label])
        positive_mask = (class_map >= threshold) & coverage_mask
        relative_map = threshold_relative_scores(class_map, threshold)
        relative_map[~coverage_mask] = 0.0
        relative_maps[label] = relative_map
        positive_fractions[label] = float(positive_mask.sum() / covered_pixels)
        overall_positive |= positive_mask
        np.maximum(overall_relative, relative_map, out=overall_relative)

    overall_relative[~coverage_mask] = 0.0
    metadata = {
        "method": "center_weighted_overlap_average",
        "blend_bands": [
            {
                "inset_fraction": float(inset),
                "incremental_weight": float(weight),
            }
            for inset, weight in HEATMAP_BLEND_BANDS
        ],
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "projected_patch_width": float(geometry.patch_size * scale_x),
        "projected_patch_height": float(geometry.patch_size * scale_y),
        "projected_stride_x": float(geometry.stride * scale_x),
        "projected_stride_y": float(geometry.stride * scale_y),
        "projected_overlap_x": float(projected_overlap_x),
        "projected_overlap_y": float(projected_overlap_y),
        "smoothing_radius_pixels": float(smoothing_radius),
        "covered_pixels": covered_pixels,
        "covered_pixel_fraction": float(coverage_mask.mean()),
    }
    return SpatialHeatmaps(
        relative_maps=relative_maps,
        overall_relative_map=overall_relative,
        coverage_mask=coverage_mask,
        label_positive_pixel_fraction=positive_fractions,
        overall_positive_pixel_fraction=float(overall_positive.sum() / covered_pixels),
        metadata=metadata,
    )

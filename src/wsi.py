"""
WSI inspection, pyramid-level selection, and image extraction.

Handles slide metadata, prediction-level resolution, patch reading,
and tiled image reconstruction at specified magnifications.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openslide
from PIL import Image
from tqdm.auto import tqdm

from constants import SUPPORTED_WSI_EXTENSIONS

Image.MAX_IMAGE_PIXELS = None


def safe_positive_float(value: Any) -> float | None:
    """Convert a value to a finite positive float, or return None."""
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result <= 0:
        return None
    return result


def find_wsi_files(wsi_dir: Path) -> list[Path]:
    """Discover supported WSI files and validate unique slide identifiers."""
    wsi_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        (path.resolve() for path in wsi_dir.iterdir() if path.is_file()),
        key=lambda path: path.name.lower(),
    )
    files = [path for path in files if path.suffix.lower() in SUPPORTED_WSI_EXTENSIONS]
    if not files:
        supported = ", ".join(SUPPORTED_WSI_EXTENSIONS)
        raise FileNotFoundError(
            f"No supported WSI files were found directly inside {wsi_dir}. "
            f"Supported extensions: {supported}"
        )

    by_stem: dict[str, list[str]] = {}
    for path in files:
        by_stem.setdefault(path.stem.casefold(), []).append(path.name)
    duplicates = [names for names in by_stem.values() if len(names) > 1]
    if duplicates:
        raise ValueError(
            "WSI filenames must have unique stems because output directories use "
            f"the stem. Conflicts: {duplicates}"
        )
    return files


def detect_wsi_format(path: Path) -> str:
    """Identify and validate the WSI format using OpenSlide."""
    try:
        detected = openslide.OpenSlide.detect_format(str(path))
    except Exception as error:
        raise ValueError(f"Could not inspect WSI format: {path.name}") from error
    if not detected:
        raise ValueError(
            f"OpenSlide does not recognize {path.name} as a supported WSI. "
            "For .tif/.tiff input, a pyramidal WSI is required."
        )
    return str(detected)


def _nearest_level(level_downsamples: tuple[float, ...], target: float) -> int:
    """Select the pyramid level closest to the target downsample."""
    if target <= 0:
        raise ValueError("Target downsample must be positive")
    return min(
        range(len(level_downsamples)),
        key=lambda index: abs(math.log(float(level_downsamples[index]) / target)),
    )


@dataclass(frozen=True)
class WSIInfo:
    """Store WSI metadata, selected pyramid geometry, and validation warnings."""
    filename: str
    stem: str
    detected_format: str
    vendor: str | None
    level0_width: int
    level0_height: int
    level_dimensions: tuple[tuple[int, int], ...]
    level_downsamples: tuple[float, ...]
    objective_power: float | None
    objective_source: str | None
    base_mpp_x: float | None
    base_mpp_y: float | None
    mpp_source: str | None
    prediction_level: int
    prediction_level_width: int
    prediction_level_height: int
    prediction_level_downsample: float
    actual_magnification: float
    selected_mpp_x: float | None
    selected_mpp_y: float | None
    selected_mpp: float | None
    selection_method: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize WSI metadata and pyramid-level information."""
        return {
            "filename": self.filename,
            "slide_id": self.stem,
            "detected_format": self.detected_format,
            "vendor": self.vendor,
            "level0_width": self.level0_width,
            "level0_height": self.level0_height,
            "level_count": len(self.level_dimensions),
            "levels": [
                {
                    "level": index,
                    "width": width,
                    "height": height,
                    "downsample": self.level_downsamples[index],
                }
                for index, (width, height) in enumerate(self.level_dimensions)
            ],
            "objective_power": self.objective_power,
            "objective_source": self.objective_source,
            "base_mpp_x": self.base_mpp_x,
            "base_mpp_y": self.base_mpp_y,
            "mpp_source": self.mpp_source,
            "prediction_level": self.prediction_level,
            "prediction_level_width": self.prediction_level_width,
            "prediction_level_height": self.prediction_level_height,
            "prediction_level_downsample": self.prediction_level_downsample,
            "actual_magnification": self.actual_magnification,
            "selected_mpp_x": self.selected_mpp_x,
            "selected_mpp_y": self.selected_mpp_y,
            "selected_mpp": self.selected_mpp,
            "selection_method": self.selection_method,
            "warnings": list(self.warnings),
        }


def inspect_wsi(
    path: Path,
    target_magnification: float,
    recommended_mpp_min: float,
    recommended_mpp_max: float,
    manual_objective_power: float | None,
    manual_base_mpp: float | None,
) -> WSIInfo:
    """Inspect slide metadata and select the prediction pyramid level."""
    detected_format = detect_wsi_format(path)
    warnings: list[str] = []

    with openslide.OpenSlide(str(path)) as slide:
        properties = slide.properties
        objective_metadata = safe_positive_float(
            properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
            or properties.get("aperio.AppMag")
            or properties.get("hamamatsu.SourceLens")
        )
        mpp_x_metadata = safe_positive_float(
            properties.get(openslide.PROPERTY_NAME_MPP_X)
            or properties.get("aperio.MPP")
        )
        mpp_y_metadata = safe_positive_float(
            properties.get(openslide.PROPERTY_NAME_MPP_Y)
            or properties.get("aperio.MPP")
        )

        objective_power = objective_metadata
        objective_source: str | None = None
        if objective_power is not None:
            objective_source = "wsi_metadata"
        elif manual_objective_power is not None:
            objective_power = safe_positive_float(manual_objective_power)
            if objective_power is None:
                raise ValueError("MANUAL_OBJECTIVE_POWER must be positive")
            objective_source = "manual_fallback"
            warnings.append(
                "Objective-power metadata was unavailable; the configured manual "
                "fallback was used."
            )

        base_mpp_x = mpp_x_metadata
        base_mpp_y = mpp_y_metadata
        mpp_source: str | None = None
        if base_mpp_x is not None or base_mpp_y is not None:
            base_mpp_x = base_mpp_x or base_mpp_y
            base_mpp_y = base_mpp_y or base_mpp_x
            mpp_source = "wsi_metadata"
        elif manual_base_mpp is not None:
            manual_mpp = safe_positive_float(manual_base_mpp)
            if manual_mpp is None:
                raise ValueError("MANUAL_BASE_MPP must be positive")
            base_mpp_x = manual_mpp
            base_mpp_y = manual_mpp
            mpp_source = "manual_fallback"
            warnings.append(
                "MPP metadata was unavailable; the configured manual fallback was used."
            )

        level_dimensions = tuple(
            (int(width), int(height)) for width, height in slide.level_dimensions
        )
        level_downsamples = tuple(float(value) for value in slide.level_downsamples)
        if objective_power is not None:
            target_downsample = objective_power / target_magnification
            selection_method = "nearest_effective_objective_power"
        elif base_mpp_x is not None and base_mpp_y is not None:
            base_mpp = (base_mpp_x + base_mpp_y) / 2.0
            target_mpp = 10.0 / target_magnification
            target_downsample = target_mpp / base_mpp
            selection_method = "nearest_mpp_fallback"
            warnings.append(
                "Objective-power metadata was unavailable; the prediction level "
                "was selected from MPP instead."
            )
        else:
            raise ValueError(
                "No usable objective-power or MPP metadata was found. Set "
                "MANUAL_OBJECTIVE_POWER or MANUAL_BASE_MPP in src/config.py."
            )

        prediction_level = _nearest_level(level_downsamples, target_downsample)
        prediction_downsample = level_downsamples[prediction_level]
        level_width, level_height = level_dimensions[prediction_level]
        if objective_power is not None:
            actual_magnification = objective_power / prediction_downsample
        else:
            selected_mpp_for_magnification = (
                ((base_mpp_x or 0.0) + (base_mpp_y or 0.0))
                / 2.0
                * prediction_downsample
            )
            actual_magnification = 10.0 / selected_mpp_for_magnification

        selected_mpp_x = (
            base_mpp_x * prediction_downsample if base_mpp_x is not None else None
        )
        selected_mpp_y = (
            base_mpp_y * prediction_downsample if base_mpp_y is not None else None
        )
        selected_values = [
            value for value in (selected_mpp_x, selected_mpp_y) if value is not None
        ]
        selected_mpp = (
            sum(selected_values) / len(selected_values) if selected_values else None
        )
        if selected_mpp is None:
            warnings.append(
                "Selected-level MPP could not be verified because MPP metadata "
                "was unavailable. Inference will continue."
            )
        elif not recommended_mpp_min <= selected_mpp <= recommended_mpp_max:
            warnings.append(
                f"Selected-level MPP is {selected_mpp:.4f} um/px, outside the "
                f"recommended range [{recommended_mpp_min:.2f}, "
                f"{recommended_mpp_max:.2f}]. Inference will continue, but model "
                "performance may be affected."
            )

        vendor = properties.get(openslide.PROPERTY_NAME_VENDOR)
        level0_width, level0_height = level_dimensions[0]

    return WSIInfo(
        filename=path.name,
        stem=path.stem,
        detected_format=detected_format,
        vendor=str(vendor) if vendor else None,
        level0_width=level0_width,
        level0_height=level0_height,
        level_dimensions=level_dimensions,
        level_downsamples=level_downsamples,
        objective_power=objective_power,
        objective_source=objective_source,
        base_mpp_x=base_mpp_x,
        base_mpp_y=base_mpp_y,
        mpp_source=mpp_source,
        prediction_level=prediction_level,
        prediction_level_width=level_width,
        prediction_level_height=level_height,
        prediction_level_downsample=prediction_downsample,
        actual_magnification=float(actual_magnification),
        selected_mpp_x=selected_mpp_x,
        selected_mpp_y=selected_mpp_y,
        selected_mpp=selected_mpp,
        selection_method=selection_method,
        warnings=tuple(warnings),
    )


def read_rgb_patch(
    slide: openslide.OpenSlide,
    level: int,
    downsample: float,
    x_level: int,
    y_level: int,
    patch_size: int,
) -> Image.Image:
    """Read a patch and composite transparent regions onto white."""
    rgba = slide.read_region(
        (
            round(x_level * downsample),
            round(y_level * downsample),
        ),
        level,
        (patch_size, patch_size),
    ).convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    try:
        return Image.alpha_composite(white, rgba).convert("RGB")
    finally:
        rgba.close()
        white.close()


def _target_downsample(info: WSIInfo, target_magnification: float) -> float:
    """Calculate the downsample for targeted magnification."""
    if target_magnification <= 0:
        raise ValueError("Output magnification must be positive")
    if info.objective_power is not None:
        return info.objective_power / target_magnification
    if info.base_mpp_x is not None and info.base_mpp_y is not None:
        base_mpp = (info.base_mpp_x + info.base_mpp_y) / 2.0
        target_mpp = 10.0 / target_magnification
        return target_mpp / base_mpp
    raise ValueError("Cannot resolve output magnification without objective or MPP")


def _read_level_rgb(
    slide: openslide.OpenSlide,
    level: int,
    tile_size: int,
    description: str,
) -> Image.Image:
    """Reconstruct a pyramid-level image through tiled RGB reading."""
    width, height = slide.level_dimensions[level]
    downsample = float(slide.level_downsamples[level])
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    total = math.ceil(width / tile_size) * math.ceil(height / tile_size)
    progress = tqdm(total=total, desc=description, unit="tile", leave=False)
    try:
        for y_level in range(0, height, tile_size):
            for x_level in range(0, width, tile_size):
                tile_width = min(tile_size, width - x_level)
                tile_height = min(tile_size, height - y_level)
                rgba = slide.read_region(
                    (
                        round(x_level * downsample),
                        round(y_level * downsample),
                    ),
                    level,
                    (tile_width, tile_height),
                ).convert("RGBA")
                white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                try:
                    tile = Image.alpha_composite(white, rgba).convert("RGB")
                    canvas.paste(tile, (x_level, y_level))
                    tile.close()
                finally:
                    rgba.close()
                    white.close()
                progress.update(1)
    except Exception:
        canvas.close()
        raise
    finally:
        progress.close()
    return canvas


def read_image_at_magnification(
    path: Path,
    info: WSIInfo,
    target_magnification: float,
    tile_size: int,
    description: str,
) -> tuple[Image.Image, dict[str, Any]]:
    """Read and resize a WSI image to the requested magnification."""
    target_downsample = _target_downsample(info, target_magnification)
    source_level = _nearest_level(info.level_downsamples, target_downsample)
    source_downsample = info.level_downsamples[source_level]
    target_width = max(1, round(info.level0_width / target_downsample))
    target_height = max(1, round(info.level0_height / target_downsample))

    with openslide.OpenSlide(str(path)) as slide:
        image = _read_level_rgb(slide, source_level, tile_size, description)
    if image.size != (target_width, target_height):
        resized = image.resize(
            (target_width, target_height),
            Image.Resampling.LANCZOS,
        )
        image.close()
        image = resized

    metadata = {
        "requested_magnification": float(target_magnification),
        "target_downsample": float(target_downsample),
        "source_level": int(source_level),
        "source_level_downsample": float(source_downsample),
        "source_level_width": int(info.level_dimensions[source_level][0]),
        "source_level_height": int(info.level_dimensions[source_level][1]),
        "output_width": int(image.width),
        "output_height": int(image.height),
        "resize_factor_x": float(image.width / info.level_dimensions[source_level][0]),
        "resize_factor_y": float(image.height / info.level_dimensions[source_level][1]),
    }
    return image, metadata

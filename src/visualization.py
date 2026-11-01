
"""WSI thumbnail generation and spatial heatmap visualization."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

matplotlib.use("Agg")


def format_magnification(value: float) -> str:
    """Format magnification for use in filenames."""
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{text.replace('.', 'p')}x"


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a platform-compatible font with fallback support."""
    candidates = (
        [
            "C:/Windows/Fonts/arialbd.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    """Measure text height using the specified font."""
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    return max(1, bbox[3] - bbox[1])


def _fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    maximum_width: int,
    starting_size: int,
    minimum_size: int,
    bold: bool,
) -> ImageFont.ImageFont:
    """Select the largest font size that fits the available width."""
    for size in range(starting_size, minimum_size - 1, -1):
        font = _load_font(size, bold=bold)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= maximum_width:
            return font
    return _load_font(minimum_size, bold=bold)


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    maximum_width: int,
) -> list[str]:
    """Wrap text into lines within the specified width."""
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= maximum_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _add_header(
    image: Image.Image,
    title: str,
    subtitle: str,
    include_scale: bool,
) -> Image.Image:
    """Add a title, subtitle, and optional heatmap scale above an image."""
    width = image.width
    margin = max(6, min(30, round(width * 0.012)))
    maximum_width = max(1, width - 2 * margin)
    probe = Image.new("RGB", (max(1, width), 1), "white")
    probe_draw = ImageDraw.Draw(probe)
    title_font = _fit_font(
        probe_draw,
        title,
        maximum_width,
        max(14, min(48, round(width / 72))),
        10,
        True,
    )
    subtitle_font = _fit_font(
        probe_draw,
        subtitle,
        maximum_width,
        max(10, min(30, round(width / 112))),
        8,
        False,
    )
    title_lines = _wrap_text(probe_draw, title, title_font, maximum_width)
    subtitle_lines = _wrap_text(
        probe_draw,
        subtitle,
        subtitle_font,
        maximum_width,
    )
    title_height = _text_height(probe_draw, title_font)
    subtitle_height = _text_height(probe_draw, subtitle_font)
    line_gap = max(2, margin // 4)
    block_gap = max(5, margin // 2)
    scale_height = max(14, subtitle_height) if include_scale else 0
    scale_label_height = max(8, int(subtitle_height * 0.85)) if include_scale else 0
    header_height = (
        margin
        + len(title_lines) * title_height
        + max(0, len(title_lines) - 1) * line_gap
        + block_gap
        + len(subtitle_lines) * subtitle_height
        + max(0, len(subtitle_lines) - 1) * line_gap
        + (
            block_gap + scale_height + line_gap + scale_label_height
            if include_scale
            else 0
        )
        + margin
    )
    probe.close()

    canvas = Image.new("RGB", (width, image.height + header_height), "white")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    y_position = margin
    for line in title_lines:
        draw.text((margin, y_position), line, fill=(20, 20, 20), font=title_font)
        y_position += title_height + line_gap
    y_position += block_gap - line_gap
    for line in subtitle_lines:
        draw.text(
            (margin, y_position),
            line,
            fill=(65, 65, 65),
            font=subtitle_font,
        )
        y_position += subtitle_height + line_gap

    if include_scale:
        bar_y = y_position + block_gap - line_gap
        bar_x1 = margin
        bar_x2 = width - margin
        color_map = matplotlib.colormaps["YlOrRd"]
        for x_position in range(bar_x1, bar_x2):
            normalized = (x_position - bar_x1) / max(bar_x2 - bar_x1 - 1, 1)
            red, green, blue, _ = color_map(normalized)
            draw.line(
                [(x_position, bar_y), (x_position, bar_y + scale_height)],
                fill=(int(red * 255), int(green * 255), int(blue * 255)),
            )
        midpoint = round((bar_x1 + bar_x2) / 2)
        draw.line(
            [(midpoint, bar_y - 4), (midpoint, bar_y + scale_height + 4)],
            fill=(15, 15, 15),
            width=max(2, width // 1600),
        )
        small_font = _load_font(max(8, int(subtitle_height * 0.85)))
        label_y = bar_y + scale_height + line_gap
        draw.text(
            (margin, label_y),
            "lower",
            fill=(55, 55, 55),
            font=small_font,
        )
        higher_text = "higher"
        bbox = draw.textbbox((0, 0), higher_text, font=small_font)
        higher_width = bbox[2] - bbox[0]
        draw.text(
            (
                width - margin - higher_width,
                label_y,
            ),
            higher_text,
            fill=(55, 55, 55),
            font=small_font,
        )
        threshold_text = "threshold" if width < 400 else "decision threshold"
        threshold_bbox = draw.textbbox((0, 0), threshold_text, font=small_font)
        threshold_width = threshold_bbox[2] - threshold_bbox[0]
        draw.text(
            (midpoint - threshold_width // 2, label_y),
            threshold_text,
            fill=(30, 30, 30),
            font=small_font,
        )

    draw.line(
        [(0, header_height - 1), (width, header_height - 1)],
        fill=(205, 205, 205),
        width=1,
    )
    return canvas


def _save_png_atomic(image: Image.Image, output_path: Path, compression: int) -> None:
    """Save a PNG image using temporary-file replacement."""
    temporary = output_path.with_name(output_path.name + ".tmp")
    image.save(
        temporary,
        format="PNG",
        optimize=True,
        compress_level=compression,
    )
    temporary.replace(output_path)


def save_thumbnail(
    source_image: Image.Image,
    output_path: Path,
    slide_id: str,
    magnification: float,
    compression: int,
) -> None:
    """Generate and save a standalone WSI thumbnail with a header."""
    subtitle = (
        f"Slide: {slide_id} | Standalone thumbnail | "
        f"Target magnification: {magnification:g}x"
    )
    titled = _add_header(
        source_image,
        "WSI thumbnail",
        subtitle,
        include_scale=False,
    )
    try:
        _save_png_atomic(titled, output_path, compression)
    finally:
        titled.close()


def save_heatmap_overlay(
    base_image: Image.Image,
    heatmap: np.ndarray,
    coverage_mask: np.ndarray,
    output_path: Path,
    title: str,
    model_name: str,
    profile_name: str,
    heatmap_magnification: float,
    max_alpha: float,
    alpha_power: float,
    base_saturation: float,
    base_brightness: float,
    compression: int,
) -> None:
    """Blend a spatial heatmap onto a WSI image and save with a color scale."""
    if heatmap.shape != (base_image.height, base_image.width):
        raise ValueError(
            f"Heatmap shape {heatmap.shape} does not match image size {base_image.size}"
        )
    if coverage_mask.shape != heatmap.shape:
        raise ValueError("Coverage mask does not match heatmap shape")

    prepared = base_image
    if not math.isclose(base_saturation, 1.0, rel_tol=1e-8, abs_tol=1e-8):
        prepared = ImageEnhance.Color(prepared).enhance(base_saturation)
    if not math.isclose(base_brightness, 1.0, rel_tol=1e-8, abs_tol=1e-8):
        brighter = ImageEnhance.Brightness(prepared).enhance(base_brightness)
        if prepared is not base_image:
            prepared.close()
        prepared = brighter

    score = np.clip(heatmap, 0.0, 1.0)
    color_map = matplotlib.colormaps["YlOrRd"]
    heatmap_rgb = color_map(score, bytes=True)[..., :3].astype(np.float32)
    alpha = (float(max_alpha) * score ** float(alpha_power)).astype(np.float32)
    alpha[~coverage_mask] = 0.0
    base_array = np.asarray(prepared, dtype=np.float32)
    blended = Image.fromarray(
        np.clip(
            base_array * (1.0 - alpha[..., None]) + heatmap_rgb * alpha[..., None],
            0,
            255,
        ).astype(np.uint8),
        mode="RGB",
    )
    if prepared is not base_image:
        prepared.close()

    subtitle = (
        f"Model: {model_name} | Profile: {profile_name} | "
        f"Heatmap magnification: {heatmap_magnification:g}x"
    )
    titled = _add_header(blended, title, subtitle, include_scale=True)
    blended.close()
    try:
        _save_png_atomic(titled, output_path, compression)
    finally:
        titled.close()

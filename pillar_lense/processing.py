"""OpenCV implementation of the original FIJI/ImageJ macro pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from .models import DetectionResult, HSBThreshold, ProcessingSettings

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(slots=True)
class Particle:
    area: float
    perimeter: float
    circularity: float
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    contour: np.ndarray
    mask: np.ndarray


@dataclass(slots=True)
class SquareDetection:
    square_index: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    particle: Particle


@dataclass(slots=True)
class BatchOutput:
    results: list[DetectionResult]
    warnings: list[str]
    csv_path: Path


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image as RGB, preserving the filename-driven workflow from FIJI."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: str | Path, image: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def reduce_moire_aliasing(rgb: np.ndarray, strength: int = 0) -> np.ndarray:
    """Suppress screen-photo moiré before thresholding while preserving large region edges.

    Moiré from photographing a display usually appears both as fine pixel-grid
    aliases and as colored ripple bands.  The preprocessing is opt-in and
    deliberately runs on the color image before any HSB/gray thresholding:

    * an area downsample/upsample pass removes display/camera pixel-grid detail,
    * a bilateral luminance pass keeps strong object boundaries usable, and
    * stronger chroma smoothing in LAB space removes colored ripple bands that
      otherwise fragment Hue/Saturation/Value masks.
    """
    strength = max(0, min(100, int(strength)))
    if strength == 0:
        return rgb

    height, width = rgb.shape[:2]
    scale = 1.0 + strength / 35.0
    reduced_width = max(1, int(round(width / scale)))
    reduced_height = max(1, int(round(height / scale)))
    low_pass = cv2.resize(rgb, (reduced_width, reduced_height), interpolation=cv2.INTER_AREA)
    low_pass = cv2.resize(low_pass, (width, height), interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(low_pass, cv2.COLOR_RGB2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)

    bilateral_diameter = 5 + 2 * (strength // 20)
    luminance = cv2.bilateralFilter(
        luminance,
        bilateral_diameter,
        20 + strength,
        8 + strength // 2,
    )

    chroma_kernel = 3 + 2 * max(1, strength // 8)
    channel_a = cv2.GaussianBlur(channel_a, (chroma_kernel, chroma_kernel), 0)
    channel_b = cv2.GaussianBlur(channel_b, (chroma_kernel, chroma_kernel), 0)

    return cv2.cvtColor(cv2.merge((luminance, channel_a, channel_b)), cv2.COLOR_LAB2RGB)


def image_difference_stats(original_rgb: np.ndarray, processed_rgb: np.ndarray) -> tuple[float, int]:
    """Return mean and max absolute RGB-channel delta between two images."""
    delta = cv2.absdiff(original_rgb, processed_rgb)
    return float(delta.mean()), int(delta.max())


def hsb_channels(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ImageJ-style HSB channels on a 0-255 scale from a threshold-input image."""
    denoised = cv2.medianBlur(rgb, 3)
    hsv = cv2.cvtColor(denoised, cv2.COLOR_RGB2HSV)
    hue_ij = np.rint(hsv[:, :, 0].astype(np.float32) * 255.0 / 179.0).astype(np.uint8)
    return hue_ij, hsv[:, :, 1], hsv[:, :, 2]


def threshold_channel(channel: np.ndarray, threshold: HSBThreshold) -> np.ndarray:
    mask = cv2.inRange(channel, threshold.minimum, threshold.maximum)
    return cv2.bitwise_not(mask) if threshold.invert else mask


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill black holes inside white foreground objects."""
    h, w = mask.shape[:2]
    flood = mask.copy()
    floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, floodfill_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def threshold_input_rgb(rgb: np.ndarray, settings: ProcessingSettings) -> np.ndarray:
    """Return the exact RGB image submitted to HSB/gray thresholding."""
    return reduce_moire_aliasing(rgb, settings.moire_reduction_strength)


def hsb_masks(
    rgb: np.ndarray,
    settings: ProcessingSettings,
    threshold_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Hue, Saturation, Brightness, and final AND mask on an ImageJ-like 0-255 scale."""
    threshold_source = threshold_input_rgb(rgb, settings) if threshold_rgb is None else threshold_rgb
    hue_ij, saturation, brightness = hsb_channels(threshold_source)

    hue_mask = threshold_channel(hue_ij, settings.hue)
    saturation_mask = threshold_channel(saturation, settings.saturation)
    brightness_mask = threshold_channel(brightness, settings.brightness)
    final_mask = cv2.bitwise_and(cv2.bitwise_and(hue_mask, saturation_mask), brightness_mask)
    return hue_mask, saturation_mask, brightness_mask, final_mask


def _circular_hue_threshold(values: np.ndarray) -> HSBThreshold:
    """Return the smallest 0-255 circular hue threshold covering values."""
    unique = np.unique(values.astype(np.uint8)).astype(int)
    if unique.size == 0:
        return HSBThreshold(0, 255, False)
    if unique.size == 1:
        value = int(unique[0])
        return HSBThreshold(value, value, False)

    ordered = np.sort(unique)
    gaps = np.diff(np.concatenate([ordered, [ordered[0] + 256]]))
    largest_gap_index = int(np.argmax(gaps))
    largest_gap = int(gaps[largest_gap_index])
    # If the best interval does not wrap around 0, store it directly.
    start = int((ordered[(largest_gap_index + 1) % ordered.size]) % 256)
    end = int(ordered[largest_gap_index])
    if start <= end:
        return HSBThreshold(start, end, False)

    # A wrapping hue interval is represented by inverting the excluded gap.
    excluded_start = (end + 1) % 256
    excluded_end = (start - 1) % 256
    if largest_gap <= 1:
        return HSBThreshold(0, 255, False)
    return HSBThreshold(excluded_start, excluded_end, True)


def hsb_thresholds_from_region(
    rgb: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    moire_reduction_strength: int = 0,
) -> tuple[HSBThreshold, HSBThreshold, HSBThreshold]:
    """Create HSB thresholds from all pixels inside an image rectangle."""
    image_height, image_width = rgb.shape[:2]
    left = max(0, min(image_width, x))
    top = max(0, min(image_height, y))
    right = max(left, min(image_width, x + width))
    bottom = max(top, min(image_height, y + height))
    if right <= left or bottom <= top:
        raise ValueError("Pipette rectangle must cover at least one image pixel")

    threshold_source = reduce_moire_aliasing(rgb[top:bottom, left:right], moire_reduction_strength)
    hue, saturation, brightness = hsb_channels(threshold_source)
    return (
        _circular_hue_threshold(hue.reshape(-1)),
        HSBThreshold(int(saturation.min()), int(saturation.max()), False),
        HSBThreshold(int(brightness.min()), int(brightness.max()), False),
    )


def clean_square_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.dilate(mask, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = fill_holes(cleaned)
    cleaned = cv2.erode(cleaned, kernel, iterations=1)
    return cleaned


def analyze_particles(
    mask: np.ndarray,
    area_min: float,
    area_max: float,
    circularity_min: float,
    circularity_max: float,
) -> list[Particle]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    particles: list[Particle] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < area_min or area > area_max:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0 if perimeter == 0 else float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < circularity_min or circularity > circularity_max:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        x, y, w, h = cv2.boundingRect(contour)
        particle_mask = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(particle_mask, [contour], -1, 255, thickness=cv2.FILLED)
        particles.append(Particle(area, perimeter, circularity, (cx, cy), (x, y, w, h), contour, particle_mask))
    return sorted(particles, key=lambda item: (item.bbox[1], item.bbox[0]))


def square_area_limits_px(settings: ProcessingSettings, scale_mm_per_px: float | None) -> tuple[float, float]:
    """Convert pink-square area limits from mm² to px².

    The FIJI macro used `size=6-7` for pink-square particles. Those values
    are intended to be real-world square areas, not caterpillar-sized pixel
    areas. When no scale is available (for example, a quick threshold preview
    before drawing the scale line), the preview should still show the cleaned
    mask, so particle area filtering is disabled.
    """
    if scale_mm_per_px is None or scale_mm_per_px <= 0:
        return 0.0, float("inf")
    px_per_mm2 = 1.0 / (scale_mm_per_px**2)
    return settings.square_area_min_mm2 * px_per_mm2, settings.square_area_max_mm2 * px_per_mm2


def detect_squares(
    rgb: np.ndarray,
    settings: ProcessingSettings,
    scale_mm_per_px: float | None = None,
    threshold_rgb: np.ndarray | None = None,
) -> tuple[list[Particle], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    masks = hsb_masks(rgb, settings, threshold_rgb)
    cleaned = clean_square_mask(masks[-1])
    square_area_min_px, square_area_max_px = square_area_limits_px(settings, scale_mm_per_px)
    particles = analyze_particles(
        cleaned,
        square_area_min_px,
        square_area_max_px,
        settings.square_circularity_min,
        settings.square_circularity_max,
    )
    return particles, (*masks[:-1], cleaned)


def match_squares_to_layout(squares: Iterable[Particle], layout_points: list[tuple[float, float]]) -> list[SquareDetection]:
    if not layout_points:
        raise ValueError("At least one layout point is required.")
    matched: list[SquareDetection] = []
    used: dict[int, Particle] = {}
    for particle in squares:
        cx, cy = particle.centroid
        distances = [((cx - px) ** 2 + (cy - py) ** 2, idx) for idx, (px, py) in enumerate(layout_points)]
        _, best_idx = min(distances, key=lambda item: item[0])
        if best_idx in used:
            raise ValueError(f"Duplicate square match for layout point {best_idx + 1}")
        used[best_idx] = particle
        matched.append(SquareDetection(best_idx + 1, particle.bbox, particle.centroid, particle))
    return sorted(matched, key=lambda item: item.square_index)


def _threshold_gray_range(gray: np.ndarray, low: int, high: int) -> np.ndarray:
    return cv2.inRange(gray, low, high)


def detect_caterpillars(crop_rgb: np.ndarray, settings: ProcessingSettings) -> tuple[list[Particle], np.ndarray, str]:
    gray = cv2.cvtColor(threshold_input_rgb(crop_rgb, settings), cv2.COLOR_RGB2GRAY)
    attempts = [
        ("threshold_44_143_fill_holes", settings.caterpillar_threshold_high, 0, 0),
        ("threshold_44_143_erode2_dilate2", settings.caterpillar_threshold_high, 2, 2),
        ("threshold_44_123_erode1_dilate1", settings.caterpillar_retry_threshold_high, 1, 1),
    ]
    kernel = np.ones((3, 3), np.uint8)
    last_mask = np.zeros(gray.shape, np.uint8)
    for name, high, erodes, dilates in attempts:
        mask = _threshold_gray_range(gray, settings.caterpillar_threshold_low, high)
        if erodes:
            mask = cv2.erode(mask, kernel, iterations=erodes)
        if dilates:
            mask = cv2.dilate(mask, kernel, iterations=dilates)
        mask = fill_holes(mask)
        particles = analyze_particles(
            mask,
            settings.caterpillar_area_min_px,
            settings.caterpillar_area_max_px,
            settings.caterpillar_circularity_min,
            settings.caterpillar_circularity_max,
        )
        last_mask = mask
        if particles:
            return particles, mask, name
    return [], last_mask, "not_detected"


def annotate_crop(crop_rgb: np.ndarray, mask: np.ndarray, area_mm2: float | None) -> np.ndarray:
    overlay = crop_rgb.copy()
    red = np.zeros_like(overlay)
    red[:, :] = (255, 0, 0)
    alpha = 0.35
    foreground = mask > 0
    overlay[foreground] = (overlay[foreground] * (1 - alpha) + red[foreground] * alpha).astype(np.uint8)
    if area_mm2 is not None:
        cv2.putText(
            overlay,
            f"Area: {area_mm2:.2f} mm^2",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
    return overlay


def particle_mask(particles: Iterable[Particle], shape: tuple[int, int]) -> np.ndarray:
    """Return one binary mask containing the particles that passed square filtering."""
    combined = np.zeros(shape, np.uint8)
    for particle in particles:
        combined = cv2.bitwise_or(combined, particle.mask)
    return combined


def overlay_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a binary mask over an RGB image in a high-contrast color."""
    overlay = rgb.copy()
    color_image = np.zeros_like(overlay)
    color_image[:, :] = color
    foreground = mask > 0
    overlay[foreground] = (overlay[foreground] * (1 - alpha) + color_image[foreground] * alpha).astype(np.uint8)
    return overlay


def make_mask_panel(
    masks: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    rgb: np.ndarray | None = None,
    square_particles: Iterable[Particle] | None = None,
    original_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Build a 2×2 threshold panel with lower-right square/preprocessing context.

    When ``original_rgb`` is supplied, the lower-right panel is split into the
    original image, the exact de-moiré image used for thresholding, and an
    amplified difference view. This makes the preprocessing effect visible in
    the preview while the first three panels still show the actual threshold
    masks.
    """
    hue_mask, saturation_mask, brightness_mask, cleaned_mask = masks
    panels = [
        cv2.cvtColor(hue_mask, cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(saturation_mask, cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(brightness_mask, cv2.COLOR_GRAY2RGB),
    ]
    labels = ["Hue mask", "Saturation mask", "Brightness mask"]
    overlay_source = cleaned_mask if square_particles is None else particle_mask(square_particles, cleaned_mask.shape)
    if rgb is None:
        panels.append(cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2RGB))
        labels.append("Cleaned final mask")
    elif original_rgb is None:
        panels.append(overlay_mask(rgb, overlay_source))
        labels.append("Detected square overlay")
    else:
        original_overlay = overlay_mask(original_rgb, overlay_source)
        processed_overlay = overlay_mask(rgb, overlay_source)
        difference = cv2.convertScaleAbs(cv2.absdiff(original_rgb, rgb), alpha=6.0)
        split_one = original_overlay.shape[1] // 3
        split_two = (original_overlay.shape[1] * 2) // 3
        comparison = original_overlay.copy()
        comparison[:, split_one:split_two] = processed_overlay[:, split_one:split_two]
        comparison[:, split_two:] = difference[:, split_two:]
        comparison[:, max(0, split_one - 1) : split_one + 1] = (255, 255, 255)
        comparison[:, max(0, split_two - 1) : split_two + 1] = (255, 255, 255)
        cv2.putText(comparison, "Original", (10, comparison.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(
            comparison,
            "De-moire input",
            (split_one + 10, comparison.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            comparison,
            "Diff x6",
            (split_two + 10, comparison.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        panels.append(comparison)
        labels.append("Detected overlay: original | de-moire | diff")

    h, w = masks[0].shape[:2]
    canvas = np.zeros((h * 2, w * 2, 3), np.uint8)
    positions = [(0, 0), (w, 0), (0, h), (w, h)]
    for panel, label, (x, y) in zip(panels, labels, positions):
        canvas[y : y + h, x : x + w] = panel
        cv2.putText(canvas, label, (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return canvas


def process_image(
    image_path: str | Path,
    output_dir: str | Path,
    settings: ProcessingSettings,
    scale_mm_per_px: float,
    layout_points: list[tuple[float, float]],
) -> tuple[list[DetectionResult], list[str]]:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    rgb = read_rgb(image_path)
    squares, masks = detect_squares(rgb, settings, scale_mm_per_px)
    warnings: list[str] = []

    if settings.save_debug_masks:
        panel_rgb = threshold_input_rgb(rgb, settings)
        write_rgb(
            output_dir / "debug" / f"{image_path.stem}_threshold_panel.png",
            make_mask_panel(
                masks,
                panel_rgb,
                squares,
                original_rgb=rgb if settings.moire_reduction_strength else None,
            ),
        )

    if not squares:
        warnings.append(f"No pink squares found in {image_path.name}")
        return [], warnings
    try:
        matched = match_squares_to_layout(squares, layout_points)
    except ValueError as exc:
        warnings.append(f"{image_path.name}: {exc}")
        return [], warnings

    results: list[DetectionResult] = []
    for square in matched:
        x, y, w, h = square.bbox
        crop = rgb[y : y + h, x : x + w]
        particles, mask, attempt = detect_caterpillars(crop, settings)
        if not particles:
            warnings.append(f"No caterpillar detected in {image_path.name} square {square.square_index}")
            if settings.save_debug_masks:
                write_rgb(output_dir / f"{image_path.stem}_sq{square.square_index}_mask.png", annotate_crop(crop, mask, None))
            continue
        for obj_idx, particle in enumerate(particles, start=1):
            area_mm2 = particle.area * (scale_mm_per_px**2)
            weight = None
            if settings.regression_slope != 0 or settings.regression_intercept != 0:
                weight = settings.regression_intercept + settings.regression_slope * area_mm2
            results.append(
                DetectionResult(
                    image=image_path.name,
                    square_index=square.square_index,
                    object_in_square=obj_idx,
                    area_px=particle.area,
                    area_mm2=area_mm2,
                    scale_mm_per_px=scale_mm_per_px,
                    x=particle.centroid[0] + x,
                    y=particle.centroid[1] + y,
                    weight_estimate=weight,
                    status=attempt,
                )
            )
        combined_mask = np.zeros(mask.shape, np.uint8)
        for particle in particles:
            combined_mask = cv2.bitwise_or(combined_mask, particle.mask)
        first_area = particles[0].area * (scale_mm_per_px**2)
        write_rgb(output_dir / f"{image_path.stem}_sq{square.square_index}_mask.png", annotate_crop(crop, combined_mask, first_area))
    return results, warnings


def image_files(input_dir: str | Path) -> list[Path]:
    return sorted(path for path in Path(input_dir).iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file())


def process_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    settings: ProcessingSettings,
    scale_mm_per_px: float,
    layout_points: list[tuple[float, float]],
) -> BatchOutput:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[DetectionResult] = []
    warnings: list[str] = []
    for path in image_files(input_dir):
        results, image_warnings = process_image(path, output_dir, settings, scale_mm_per_px, layout_points)
        all_results.extend(results)
        warnings.extend(image_warnings)
    csv_path = output_dir / "AreaMeasurements.csv"
    pd.DataFrame([result.as_csv_row() for result in all_results]).to_csv(csv_path, index=False)
    return BatchOutput(all_results, warnings, csv_path)

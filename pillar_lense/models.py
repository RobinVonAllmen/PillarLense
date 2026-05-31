"""Configuration and result models for PillarLense."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass(slots=True)
class HSBThreshold:
    """ImageJ-style HSB threshold values on a 0-255 channel scale."""

    minimum: int
    maximum: int
    invert: bool = False


@dataclass(slots=True)
class ProcessingSettings:
    """All tunable parameters for reproducing and debugging the FIJI macro pipeline."""

    hue: HSBThreshold = field(default_factory=lambda: HSBThreshold(12, 200, True))
    saturation: HSBThreshold = field(default_factory=lambda: HSBThreshold(40, 80, False))
    brightness: HSBThreshold = field(default_factory=lambda: HSBThreshold(0, 255, False))
    square_area_min_px: float = 6.0
    square_area_max_px: float = 7.0
    square_circularity_min: float = 0.0
    square_circularity_max: float = 1.0
    caterpillar_area_min_px: float = 300.0
    caterpillar_area_max_px: float = 5000.0
    caterpillar_circularity_min: float = 0.25
    caterpillar_circularity_max: float = 0.95
    caterpillar_threshold_low: int = 44
    caterpillar_threshold_high: int = 143
    caterpillar_retry_threshold_high: int = 123
    save_debug_masks: bool = True
    regression_intercept: float = 0.0
    regression_slope: float = 0.0

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "ProcessingSettings":
        raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("hue", "saturation", "brightness"):
            if isinstance(raw.get(key), dict):
                raw[key] = HSBThreshold(**raw[key])
        return cls(**raw)


@dataclass(slots=True)
class DetectionResult:
    image: str
    square_index: int
    object_in_square: int
    area_px: float
    area_mm2: float
    scale_mm_per_px: float
    x: float
    y: float
    weight_estimate: float | None = None
    status: str = "detected"

    def as_csv_row(self) -> dict[str, float | int | str | None]:
        return {
            "Image": self.image,
            "SquareIndex": self.square_index,
            "ObjectInSquare": self.object_in_square,
            "Area_px": self.area_px,
            "Area_mm2": self.area_mm2,
            "Scale_mm_per_px": self.scale_mm_per_px,
            "X": self.x,
            "Y": self.y,
            "Weight_estimate": self.weight_estimate,
            "Status": self.status,
        }

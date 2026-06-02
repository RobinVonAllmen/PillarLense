import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pillar_lense.models import ProcessingSettings


def test_processing_settings_migrates_legacy_square_area_names(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"square_area_min_px": 6.0, "square_area_max_px": 7.0}),
        encoding="utf-8",
    )

    settings = ProcessingSettings.from_json(settings_path)

    assert settings.square_area_min_mm2 == 6.0
    assert settings.square_area_max_mm2 == 7.0

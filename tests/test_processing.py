import importlib.util

import pytest


def needs_imaging_stack():
    if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("cv2") is None:
        pytest.skip("OpenCV and NumPy are not installed in this environment")


def test_detect_squares_and_match_layout_with_imagej_style_hsb_thresholds():
    needs_imaging_stack()
    import cv2  # noqa: F401
    import numpy as np

    from pillar_lense.models import HSBThreshold, ProcessingSettings
    from pillar_lense.processing import detect_squares, match_squares_to_layout

    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[25:75, 30:80] = [255, 0, 255]
    settings = ProcessingSettings(
        hue=HSBThreshold(12, 200, True),
        saturation=HSBThreshold(40, 255, False),
        brightness=HSBThreshold(1, 255, False),
        square_area_min_px=1_000,
        square_area_max_px=4_000,
    )

    squares, _ = detect_squares(image, settings)
    matched = match_squares_to_layout(squares, [(55, 50)])

    assert len(matched) == 1
    assert matched[0].square_index == 1
    assert matched[0].bbox[2] >= 45
    assert matched[0].bbox[3] >= 45


def test_detect_caterpillar_uses_retry_morphology_for_dark_blob():
    needs_imaging_stack()
    import cv2
    import numpy as np

    from pillar_lense.models import ProcessingSettings
    from pillar_lense.processing import detect_caterpillars

    crop = np.full((100, 100, 3), 220, dtype=np.uint8)
    cv2.ellipse(crop, (50, 50), (28, 10), 20, 55, 55, [70, 70, 70], -1)
    settings = ProcessingSettings(caterpillar_area_min_px=300, caterpillar_area_max_px=2_500)

    particles, mask, attempt = detect_caterpillars(crop, settings)

    assert particles
    assert mask.shape == crop.shape[:2]
    assert attempt in {
        "threshold_44_143_fill_holes",
        "threshold_44_143_erode2_dilate2",
        "threshold_44_123_erode1_dilate1",
    }

import importlib.util

import pytest


def needs_imaging_stack():
    if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("cv2") is None:
        pytest.skip("OpenCV and NumPy are not installed in this environment")
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"OpenCV and NumPy are not usable in this environment: {exc}")


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
        square_area_min_mm2=5.0,
        square_area_max_mm2=8.0,
    )

    # The pink-square limits are physical areas. With 0.05 mm/px,
    # a 5-8 mm² square corresponds to roughly 2,000-3,200 px².
    squares, _ = detect_squares(image, settings, scale_mm_per_px=0.05)
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


def test_hsb_thresholds_from_region_uses_selected_rectangle_values():
    needs_imaging_stack()
    import numpy as np

    from pillar_lense.processing import hsb_thresholds_from_region

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :] = [0, 0, 0]
    image[5:15, 4:12] = [255, 0, 255]

    hue, saturation, brightness = hsb_thresholds_from_region(image, 4, 5, 8, 10)

    assert hue.minimum == hue.maximum
    assert not hue.invert
    assert saturation.minimum == 255
    assert saturation.maximum == 255
    assert brightness.minimum == 255
    assert brightness.maximum == 255


def test_hsb_thresholds_from_region_represents_wrapping_hues_with_inversion():
    needs_imaging_stack()
    import numpy as np

    from pillar_lense.processing import hsb_thresholds_from_region

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :4] = [255, 0, 0]
    image[:, 4:] = [255, 0, 16]

    hue, _, _ = hsb_thresholds_from_region(image, 0, 0, 8, 8)

    assert hue.invert
    assert hue.minimum > 0
    assert hue.maximum < 255


def test_make_mask_panel_overlays_detected_square_particles_on_lower_right_quadrant():
    needs_imaging_stack()
    import numpy as np

    from pillar_lense.models import HSBThreshold, ProcessingSettings
    from pillar_lense.processing import detect_squares, make_mask_panel

    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[25:75, 30:80] = [255, 0, 255]
    image[90:95, 90:95] = [255, 0, 255]
    settings = ProcessingSettings(
        hue=HSBThreshold(12, 200, True),
        saturation=HSBThreshold(40, 255, False),
        brightness=HSBThreshold(1, 255, False),
        square_area_min_mm2=5.0,
        square_area_max_mm2=8.0,
    )

    squares, masks = detect_squares(image, settings, scale_mm_per_px=0.05)
    panel = make_mask_panel(masks, image, squares)

    h, w = image.shape[:2]
    highlighted_square_pixel = panel[h + 50, w + 55]
    rejected_speck_pixel = panel[h + 92, w + 92]

    assert len(squares) == 1
    assert highlighted_square_pixel[1] > highlighted_square_pixel[0]
    assert highlighted_square_pixel[2] > highlighted_square_pixel[0]
    assert rejected_speck_pixel.tolist() == image[92, 92].tolist()

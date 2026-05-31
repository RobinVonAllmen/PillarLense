# PillarLense

PillarLense is a PyQt6 desktop application that replaces the attached FIJI/ImageJ macro with an interactive, repeatable, and debuggable Python workflow for measuring caterpillar projected area inside hot-pink square regions.

## What the app does

The application implements the same major stages as the macro:

1. **Calibrate image scale** by drawing a known-length line on a reference image.
2. **Define the plate layout** by clicking expected pink-square centers in the order you want reported as square 1, square 2, etc.
3. **Tune hot-pink square segmentation** with ImageJ-style HSB thresholds, including per-channel inversion and a 2×2 mask preview panel.
4. **Batch process image folders** by detecting pink squares, matching them to the user-defined layout, cropping each square, detecting caterpillars, measuring area in pixels and mm², and saving annotated mask PNG files.
5. **Export `AreaMeasurements.csv`** with image name, square index, object index, area, scale, centroid coordinates, optional weight estimate, and the successful detection attempt.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run the desktop app

```bash
pillar-lense
```

Or:

```bash
python -m pillar_lense.app
```

## Practical workflow

1. Click **Open scale/layout reference image** and choose a representative image.
2. Click **Draw scale line** and click twice on the image to mark a straight line of known length.
3. Enter the known line length in millimetres. The app displays the computed `mm/px` scale.
4. Click **Add square centers** and click each expected square center in the desired order. Press **Backspace** or **Delete** to remove the most recent point.
5. Adjust HSB and particle settings in the **Thresholds** tab. The defaults mirror the macro, but the pink-square area limits are often the first values to adapt to your camera resolution.
6. Use **Preview pink-square mask** to inspect Hue, Saturation, Brightness, and final HSB masks.
7. Choose input and output folders, then click **Run batch analysis**.

## Outputs

The output folder contains:

- `AreaMeasurements.csv` with one row per detected caterpillar object.
- `<image>_sq<index>_mask.png` annotated crop overlays for each processed square.
- `debug/<image>_threshold_panel.png` HSB threshold panels when debug-mask saving is enabled.

## Optional weight estimation

If you already have an R linear model such as:

```text
weight = intercept + slope * area_mm2
```

enter the intercept and slope in the **Thresholds** tab. The app writes the calculated value to the `Weight_estimate` CSV column. Leave both fields at zero to skip weight estimation.

## Development checks

```bash
python -m compileall pillar_lense tests
pytest
```

The tests skip image-processing assertions automatically when optional runtime dependencies such as OpenCV and NumPy are not installed.

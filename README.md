# PillarLense

PillarLense is a PyQt6 desktop application that replaces the attached FIJI/ImageJ macro with an interactive, repeatable, and debuggable Python workflow for measuring caterpillar projected area inside hot-pink square regions.

## What the app does

The application implements the same major stages as the macro:

1. **Calibrate image scale** by drawing a known-length line on a reference image.
2. **Define the plate layout** by clicking expected pink-square centers in the order you want reported as square 1, square 2, etc.
3. **Tune hot-pink square segmentation** with optional pre-threshold de-moiré smoothing, ImageJ-style HSB thresholds, per-channel inversion, and a 2×2 mask preview panel.
4. **Batch process image folders** by detecting pink squares, matching them to the user-defined layout, cropping each square, detecting caterpillars, measuring area in pixels and mm², and saving annotated mask PNG files.
5. **Export `AreaMeasurements.csv`** with image name, square index, object index, area, scale, centroid coordinates, optional weight estimate, and the successful detection attempt.

## Installation / virtual environment

A Python virtual environment should be created locally as `.venv`. Virtual environments contain many platform-specific binaries, so the repository includes a repeatable setup script instead of committing the generated `.venv` folder.

```bash
./scripts/setup_venv.sh
```

On Windows, run this from Command Prompt or PowerShell in the repository root:

```bat
scripts\setup_venv.bat
```

The script creates `.venv`, upgrades `pip/setuptools/wheel`, and installs PillarLense plus the required packages from `pyproject.toml`/`requirements.txt`:

- `PyQt6`
- `opencv-python`
- `numpy`
- `pandas`

If your system uses a specific Python executable, pass it with `PYTHON`, for example:

```bash
PYTHON=python3.11 ./scripts/setup_venv.sh
```

On Windows Command Prompt, the equivalent is:

```bat
set PYTHON=C:\Path\To\python.exe
scripts\setup_venv.bat
```

## How to open the desktop app

After setup, the simplest macOS/Linux command is:

```bash
./scripts/run_app.sh
```

On Windows, run this from Command Prompt or PowerShell in the repository root:

```bat
scripts\run_app.bat
```

Equivalent macOS/Linux commands are:

```bash
.venv/bin/pillar-lense
```

or:

```bash
source .venv/bin/activate
pillar-lense
```

Equivalent Windows commands are:

```bat
.venv\Scripts\pillar-lense.exe
```

or:

```bat
.venv\Scripts\activate
pillar-lense
```

You can also run the module directly from an activated environment:

```bash
python -m pillar_lense.app
```

If you run the file directly, this is now supported too:

```bash
python pillar_lense/app.py
```

On Windows, the same direct-file launch is:

```bat
python pillar_lense\app.py
```

## Practical workflow

1. Click **Open scale/layout reference image** and choose a representative image.
2. Click **Draw scale line** and click twice on the image to mark a straight line of known length.
3. Enter the known line length in millimetres. The app displays the computed `mm/px` scale.
4. Click **Add square centers** and click each expected square center in the desired order. Press **Backspace** or **Delete** to remove the most recent point.
5. Adjust HSB and particle settings in the **Thresholds** tab. If your images were photographed from a screen, raise **Pre-threshold de-moiré strength** from `0` to about `40-80` before changing thresholds; the app low-passes the camera/display pixel-grid pattern, smooths colored ripple bands, and then generates Hue, Saturation, Brightness, and caterpillar gray-threshold masks from that processed image. The pink-square area defaults (`6-7 mm²`) are physical square areas from the macro and are converted to pixel area using your drawn scale; they are not caterpillar pixel-area limits.
6. Use **Preview pink-square mask** to inspect Hue, Saturation, Brightness, and the cleaned final mask after dilate/close/fill-holes/erode. The preview is scaled to fit your screen, with the original panel dimensions shown below it. Draw the scale line before previewing if you want the `6-7 mm²` square-area filter applied; without a scale, the preview shows the cleaned mask without square-area filtering.
7. Choose input and output folders, then click **Run batch analysis**.

## Outputs

The output folder contains:

- `AreaMeasurements.csv` with one row per detected caterpillar object.
- `<image>_sq<index>_mask.png` annotated crop overlays for each processed square.
- `debug/<image>_threshold_panel.png` HSB threshold panels when debug-mask saving is enabled. The HSB masks are generated after any configured pre-threshold de-moiré smoothing. When de-moiré is enabled, the fourth panel shows a split original/de-moiré-input overlay so the preprocessing effect is visible; otherwise it shows the detected square overlay.

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

"""PyQt6 desktop interface for the PillarLense segmentation workflow."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if __package__ in {None, ""}:
    # Support launching this file directly, e.g. `python pillar_lense/app.py`.
    # In that mode Python does not know the package parent, so add the
    # repository root before importing package modules.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pillar_lense.models import HSBThreshold, ProcessingSettings
    from pillar_lense.processing import (
        BatchOutput,
        detect_squares,
        hsb_thresholds_from_region,
        make_mask_panel,
        process_batch,
        read_rgb,
        reduce_moire_aliasing,
    )
else:
    from .models import HSBThreshold, ProcessingSettings
    from .processing import (
        BatchOutput,
        detect_squares,
        hsb_thresholds_from_region,
        make_mask_panel,
        process_batch,
        read_rgb,
        reduce_moire_aliasing,
    )


def rgb_to_qpixmap(image) -> QPixmap:
    h, w, channels = image.shape
    bytes_per_line = channels * w
    qimage = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


class ZoomablePixmapView(QGraphicsView):
    """Graphics view with mouse-wheel/touchpad zoom and drag panning."""

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pixmap_item: QGraphicsPixmapItem | None = None
        self._zoom_steps = 0
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self.scene().clear()
        self.pixmap_item = self.scene().addPixmap(pixmap)
        self.setSceneRect(self.pixmap_item.boundingRect())
        self._zoom_steps = 0
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        if self.pixmap_item is None:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        factor = 1.25 if delta > 0 else 0.8
        self._zoom_steps += 1 if delta > 0 else -1
        if self._zoom_steps < -10:
            self._zoom_steps = -10
            return
        self.scale(factor, factor)
        event.accept()


class ImageCanvas(ZoomablePixmapView):
    """Interactive image widget supporting annotation and pipette rectangle sampling."""

    changed = pyqtSignal()
    pipette_selected = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.mode = "view"
        self.scale_line: tuple[QPointF, QPointF] | None = None
        self.layout_points: list[QPointF] = []
        self._pending_line_start: QPointF | None = None
        self._pipette_start: QPointF | None = None
        self._pipette_current: QPointF | None = None

    def set_pixmap(self, pixmap: QPixmap) -> None:
        super().set_pixmap(pixmap)
        self.scale_line = None
        self.layout_points = []
        self._pending_line_start = None
        self._pipette_start = None
        self._pipette_current = None
        self.changed.emit()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._pipette_start = None
        self._pipette_current = None
        if mode in {"scale", "layout", "pipette"}:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.viewport().update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.pixmap_item is None or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        point = self.mapToScene(event.position().toPoint())
        if not self.sceneRect().contains(point):
            return
        if self.mode == "scale":
            if self._pending_line_start is None:
                self._pending_line_start = point
            else:
                self.scale_line = (self._pending_line_start, point)
                self._pending_line_start = None
                self.changed.emit()
                self.viewport().update()
            return
        if self.mode == "layout":
            self.layout_points.append(point)
            self.changed.emit()
            self.viewport().update()
            return
        if self.mode == "pipette":
            self._pipette_start = point
            self._pipette_current = point
            self.viewport().update()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.mode == "pipette" and self._pipette_start is not None:
            self._pipette_current = self.mapToScene(event.position().toPoint())
            self.viewport().update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (
            self.mode == "pipette"
            and event.button() == Qt.MouseButton.LeftButton
            and self._pipette_start is not None
        ):
            end = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._pipette_start, end).normalized().intersected(self.sceneRect())
            self._pipette_start = None
            self._pipette_current = None
            self.viewport().update()
            if rect.width() >= 1 and rect.height() >= 1:
                self.pipette_selected.emit(rect)
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete) and self.mode == "layout" and self.layout_points:
            self.layout_points.pop()
            self.changed.emit()
            self.viewport().update()
            return
        super().keyPressEvent(event)

    def drawForeground(self, painter: QPainter, rect) -> None:
        super().drawForeground(painter, rect)
        painter.setPen(QPen(Qt.GlobalColor.cyan, 3))
        if self.scale_line:
            painter.drawLine(*self.scale_line)
        painter.setPen(QPen(Qt.GlobalColor.yellow, 3))
        for idx, point in enumerate(self.layout_points, start=1):
            painter.drawEllipse(point, 7, 7)
            painter.drawText(point + QPointF(9, -9), str(idx))
        if self._pipette_start is not None and self._pipette_current is not None:
            pen = QPen(Qt.GlobalColor.green, 2)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRectF(self._pipette_start, self._pipette_current).normalized())

    def scale_length_px(self) -> float | None:
        if not self.scale_line:
            return None
        p1, p2 = self.scale_line
        return math.hypot(p1.x() - p2.x(), p1.y() - p2.y())

    def layout_as_tuples(self) -> list[tuple[float, float]]:
        return [(point.x(), point.y()) for point in self.layout_points]


class BatchWorker(QThread):
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, input_dir: Path, output_dir: Path, settings: ProcessingSettings, scale: float, layout: list[tuple[float, float]]):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.settings = settings
        self.scale = scale
        self.layout = layout

    def run(self) -> None:
        try:
            self.finished_ok.emit(process_batch(self.input_dir, self.output_dir, self.settings, self.scale, self.layout))
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PillarLense - Caterpillar area segmentation")
        self.settings = ProcessingSettings()
        self.current_image_path: Path | None = None
        self.current_rgb = None
        self.worker: BatchWorker | None = None
        self._build_ui()
        self._build_menu()

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("Settings")
        load_action = QAction("Load settings JSON", self)
        load_action.triggered.connect(self.load_settings)
        save_action = QAction("Save settings JSON", self)
        save_action.triggered.connect(self.save_settings)
        menu.addAction(load_action)
        menu.addAction(save_action)

    def _build_ui(self) -> None:
        splitter = QSplitter()
        self.canvas = ImageCanvas()
        self.canvas.pipette_selected.connect(self.apply_pipette_thresholds)
        splitter.addWidget(self.canvas)

        right = QTabWidget()
        self.tabs = right
        right.addTab(self._workflow_tab(), "Workflow")
        right.addTab(self._threshold_tab(), "Thresholds")
        right.addTab(self._results_tab(), "Results")
        splitter.addWidget(right)
        splitter.setSizes([900, 430])
        self.setCentralWidget(splitter)

    def _workflow_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        load_image = QPushButton("Open scale/layout reference image")
        load_image.clicked.connect(self.open_reference_image)
        layout.addWidget(load_image)

        modes = QHBoxLayout()
        scale_mode = QPushButton("Draw scale line")
        scale_mode.clicked.connect(lambda: self.canvas.set_mode("scale"))
        layout_mode = QPushButton("Add square centers")
        layout_mode.clicked.connect(lambda: self.canvas.set_mode("layout"))
        pipette_mode = QPushButton("Pipette")
        pipette_mode.clicked.connect(lambda: self.canvas.set_mode("pipette"))
        view_mode = QPushButton("Pan/zoom")
        view_mode.clicked.connect(lambda: self.canvas.set_mode("view"))
        modes.addWidget(scale_mode)
        modes.addWidget(layout_mode)
        modes.addWidget(pipette_mode)
        modes.addWidget(view_mode)
        layout.addLayout(modes)

        scale_box = QGroupBox("Scale")
        scale_form = QFormLayout(scale_box)
        self.real_length = QDoubleSpinBox()
        self.real_length.setRange(0.0001, 1_000_000)
        self.real_length.setValue(10.0)
        self.real_length.setSuffix(" mm")
        self.scale_label = QLabel("Draw a line with two clicks")
        scale_form.addRow("Known line length", self.real_length)
        scale_form.addRow("Computed scale", self.scale_label)
        self.canvas.changed.connect(self.update_scale_label)
        self.real_length.valueChanged.connect(self.update_scale_label)
        layout.addWidget(scale_box)

        dirs = QGroupBox("Batch folders")
        dirs_form = QFormLayout(dirs)
        self.input_dir = QLineEdit()
        self.output_dir = QLineEdit()
        input_button = QPushButton("Browse")
        output_button = QPushButton("Browse")
        input_button.clicked.connect(lambda: self.choose_dir(self.input_dir))
        output_button.clicked.connect(lambda: self.choose_dir(self.output_dir))
        dirs_form.addRow("Input image folder", self._line_with_button(self.input_dir, input_button))
        dirs_form.addRow("Output folder", self._line_with_button(self.output_dir, output_button))
        layout.addWidget(dirs)

        preview = QPushButton("Preview pink-square mask")
        preview.clicked.connect(self.preview_square_mask)
        run = QPushButton("Run batch analysis")
        run.clicked.connect(self.run_batch)
        layout.addWidget(preview)
        layout.addWidget(run)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, stretch=1)
        return panel

    def _threshold_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.h_min, self.h_max, self.h_inv = self._threshold_group(layout, "Hue", self.settings.hue)
        self.s_min, self.s_max, self.s_inv = self._threshold_group(layout, "Saturation", self.settings.saturation)
        self.b_min, self.b_max, self.b_inv = self._threshold_group(layout, "Brightness", self.settings.brightness)

        particle_box = QGroupBox("Particle filters")
        form = QFormLayout(particle_box)
        self.square_min = self._double_spin(self.settings.square_area_min_mm2, 0, 1_000_000)
        self.square_max = self._double_spin(self.settings.square_area_max_mm2, 0, 1_000_000)
        self.cat_min = self._double_spin(self.settings.caterpillar_area_min_px, 0, 1_000_000)
        self.cat_max = self._double_spin(self.settings.caterpillar_area_max_px, 0, 1_000_000)
        self.cat_low = self._spin(self.settings.caterpillar_threshold_low)
        self.cat_high = self._spin(self.settings.caterpillar_threshold_high)
        self.cat_retry_high = self._spin(self.settings.caterpillar_retry_threshold_high)
        self.moire_strength = self._spin(self.settings.moire_reduction_strength)
        self.moire_strength.setToolTip("0 disables preprocessing; try 40-80 for screen-photo ripple/moire artifacts, then re-preview and adjust thresholds.")
        form.addRow("Pink square area min mm²", self.square_min)
        form.addRow("Pink square area max mm²", self.square_max)
        form.addRow("Caterpillar area min px²", self.cat_min)
        form.addRow("Caterpillar area max px²", self.cat_max)
        form.addRow("Caterpillar gray low", self.cat_low)
        form.addRow("Caterpillar gray high", self.cat_high)
        form.addRow("Retry gray high", self.cat_retry_high)
        form.addRow("Pre-threshold de-moiré strength", self.moire_strength)
        layout.addWidget(particle_box)

        regression_box = QGroupBox("Optional area-to-weight regression")
        regression_form = QFormLayout(regression_box)
        self.reg_intercept = self._double_spin(0, -1_000_000, 1_000_000, decimals=6)
        self.reg_slope = self._double_spin(0, -1_000_000, 1_000_000, decimals=6)
        regression_form.addRow("Intercept", self.reg_intercept)
        regression_form.addRow("Slope × area_mm²", self.reg_slope)
        layout.addWidget(regression_box)
        layout.addStretch(1)
        return panel

    def _results_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(["Image", "Square", "Object", "Area px", "Area mm²", "Scale", "X", "Y", "Weight", "Status"])
        layout.addWidget(self.table)
        return panel

    def _threshold_group(self, parent: QVBoxLayout, title: str, threshold: HSBThreshold):
        box = QGroupBox(title)
        form = QFormLayout(box)
        minimum = self._spin(threshold.minimum)
        maximum = self._spin(threshold.maximum)
        invert = QCheckBox("Invert mask")
        invert.setChecked(threshold.invert)
        form.addRow("Minimum", minimum)
        form.addRow("Maximum", maximum)
        form.addRow(invert)
        parent.addWidget(box)
        return minimum, maximum, invert

    def _line_with_button(self, line: QLineEdit, button: QPushButton) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line)
        layout.addWidget(button)
        return widget

    def _spin(self, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, 255)
        spin.setValue(value)
        return spin

    def _double_spin(self, value: float, minimum: float, maximum: float, decimals: int = 2) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    def collect_settings(self) -> ProcessingSettings:
        return ProcessingSettings(
            hue=HSBThreshold(self.h_min.value(), self.h_max.value(), self.h_inv.isChecked()),
            saturation=HSBThreshold(self.s_min.value(), self.s_max.value(), self.s_inv.isChecked()),
            brightness=HSBThreshold(self.b_min.value(), self.b_max.value(), self.b_inv.isChecked()),
            square_area_min_mm2=self.square_min.value(),
            square_area_max_mm2=self.square_max.value(),
            caterpillar_area_min_px=self.cat_min.value(),
            caterpillar_area_max_px=self.cat_max.value(),
            caterpillar_threshold_low=self.cat_low.value(),
            caterpillar_threshold_high=self.cat_high.value(),
            caterpillar_retry_threshold_high=self.cat_retry_high.value(),
            moire_reduction_strength=self.moire_strength.value(),
            regression_intercept=self.reg_intercept.value(),
            regression_slope=self.reg_slope.value(),
        )

    def apply_settings(self, settings: ProcessingSettings) -> None:
        self.h_min.setValue(settings.hue.minimum)
        self.h_max.setValue(settings.hue.maximum)
        self.h_inv.setChecked(settings.hue.invert)
        self.s_min.setValue(settings.saturation.minimum)
        self.s_max.setValue(settings.saturation.maximum)
        self.s_inv.setChecked(settings.saturation.invert)
        self.b_min.setValue(settings.brightness.minimum)
        self.b_max.setValue(settings.brightness.maximum)
        self.b_inv.setChecked(settings.brightness.invert)
        self.square_min.setValue(settings.square_area_min_mm2)
        self.square_max.setValue(settings.square_area_max_mm2)
        self.cat_min.setValue(settings.caterpillar_area_min_px)
        self.cat_max.setValue(settings.caterpillar_area_max_px)
        self.cat_low.setValue(settings.caterpillar_threshold_low)
        self.cat_high.setValue(settings.caterpillar_threshold_high)
        self.cat_retry_high.setValue(settings.caterpillar_retry_threshold_high)
        self.moire_strength.setValue(settings.moire_reduction_strength)
        self.reg_intercept.setValue(settings.regression_intercept)
        self.reg_slope.setValue(settings.regression_slope)

    def choose_dir(self, line: QLineEdit) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choose folder")
        if directory:
            line.setText(directory)

    def open_reference_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open reference image", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self.current_image_path = Path(path)
        self.current_rgb = read_rgb(path)
        self.canvas.set_pixmap(rgb_to_qpixmap(self.current_rgb))
        self.log.append(f"Loaded reference image: {path}")

    def update_scale_label(self) -> None:
        length = self.canvas.scale_length_px()
        if not length:
            self.scale_label.setText("Draw a line with two clicks")
            return
        self.scale_label.setText(f"{self.real_length.value() / length:.6f} mm/px ({length:.2f} px)")

    def computed_scale(self) -> float | None:
        length = self.canvas.scale_length_px()
        if not length:
            return None
        return self.real_length.value() / length

    def preview_square_mask(self) -> None:
        if self.current_rgb is None:
            QMessageBox.warning(self, "No image", "Open a reference image first.")
            return
        settings = self.collect_settings()
        scale = self.computed_scale()
        squares, masks = detect_squares(self.current_rgb, settings, scale)
        preview_rgb = reduce_moire_aliasing(self.current_rgb, settings.moire_reduction_strength)
        panel = make_mask_panel(
            masks,
            preview_rgb,
            squares,
            original_rgb=self.current_rgb if settings.moire_reduction_strength else None,
        )
        panel_pixmap = rgb_to_qpixmap(panel)
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else self.geometry()
        max_width = max(400, int(available.width() * 0.85))
        max_height = max(300, int(available.height() * 0.85))
        dialog = QDialog(self)
        dialog.setWindowTitle("Pink-square threshold preview")
        dialog_layout = QVBoxLayout(dialog)
        preview_view = ZoomablePixmapView()
        preview_view.set_pixmap(panel_pixmap)
        dialog_layout.addWidget(preview_view)
        dialog_layout.addWidget(
            QLabel(
                "Use the mouse wheel or two-finger touchpad gesture to zoom; "
                "drag to pan "
                f"(panel: {panel_pixmap.width()}×{panel_pixmap.height()} px)."
            )
        )
        dialog.resize(
            min(panel_pixmap.width() + 60, max_width),
            min(panel_pixmap.height() + 110, max_height),
        )
        dialog.exec()
        if scale is None:
            area_note = "no scale drawn yet; square area filtering disabled for this preview"
        else:
            min_px = settings.square_area_min_mm2 / (scale**2)
            max_px = settings.square_area_max_mm2 / (scale**2)
            area_note = f"square area filter {min_px:.0f}-{max_px:.0f} px² from {settings.square_area_min_mm2:g}-{settings.square_area_max_mm2:g} mm²"
        self.log.append(
            f"Preview found {len(squares)} pink-square candidate(s) after de-moiré strength "
            f"{settings.moire_reduction_strength} and dilate/close/fill holes/erode ({area_note})."
        )

    def apply_pipette_thresholds(self, rect: QRectF) -> None:
        if self.current_rgb is None:
            QMessageBox.warning(self, "No image", "Open a reference image before using the pipette.")
            return
        x = math.floor(rect.left())
        y = math.floor(rect.top())
        width = max(1, math.ceil(rect.right()) - x)
        height = max(1, math.ceil(rect.bottom()) - y)
        try:
            hue, saturation, brightness = hsb_thresholds_from_region(
                self.current_rgb, x, y, width, height, self.collect_settings().moire_reduction_strength
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Empty pipette selection", str(exc))
            return

        self.h_min.setValue(hue.minimum)
        self.h_max.setValue(hue.maximum)
        self.h_inv.setChecked(hue.invert)
        self.s_min.setValue(saturation.minimum)
        self.s_max.setValue(saturation.maximum)
        self.s_inv.setChecked(saturation.invert)
        self.b_min.setValue(brightness.minimum)
        self.b_max.setValue(brightness.maximum)
        self.b_inv.setChecked(brightness.invert)
        self.tabs.setCurrentIndex(1)
        self.log.append(
            "Pipette set HSB thresholds from "
            f"{width}×{height} px rectangle at ({x}, {y}): "
            f"H {hue.minimum}-{hue.maximum}{' inverted' if hue.invert else ''}, "
            f"S {saturation.minimum}-{saturation.maximum}, "
            f"V {brightness.minimum}-{brightness.maximum} "
            f"(de-moiré strength {self.moire_strength.value()})."
        )

    def run_batch(self) -> None:
        scale = self.computed_scale()
        layout = self.canvas.layout_as_tuples()
        if scale is None:
            QMessageBox.warning(self, "Missing scale", "Draw the scale line and enter its real length first.")
            return
        if not layout:
            QMessageBox.warning(self, "Missing layout", "Click the expected pink-square centers in order first.")
            return
        if not self.input_dir.text() or not self.output_dir.text():
            QMessageBox.warning(self, "Missing folders", "Choose both input and output folders.")
            return
        self.worker = BatchWorker(Path(self.input_dir.text()), Path(self.output_dir.text()), self.collect_settings(), scale, layout)
        self.worker.finished_ok.connect(self.batch_finished)
        self.worker.failed.connect(self.batch_failed)
        self.worker.start()
        self.log.append("Batch analysis started...")

    def batch_finished(self, output: BatchOutput) -> None:
        self.log.append(f"Finished. Results saved to {output.csv_path}")
        for warning in output.warnings:
            self.log.append(f"WARNING: {warning}")
        frame = pd.DataFrame([result.as_csv_row() for result in output.results])
        self.populate_table(frame)

    def batch_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Batch failed", message)
        self.log.append(f"ERROR: {message}")

    def populate_table(self, frame: pd.DataFrame) -> None:
        self.table.setRowCount(len(frame))
        for row_idx, (_, row) in enumerate(frame.iterrows()):
            values = [row.get(column, "") for column in frame.columns]
            for col_idx, value in enumerate(values):
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()

    def load_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", "", "JSON (*.json)")
        if path:
            self.apply_settings(ProcessingSettings.from_json(path))

    def save_settings(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", "pillar_lense_settings.json", "JSON (*.json)")
        if path:
            self.collect_settings().to_json(path)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1350, 850)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

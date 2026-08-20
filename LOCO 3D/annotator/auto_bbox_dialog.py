"""
auto_bbox_dialog.py

3-step validation dialog for the Autonomous 3D Bounding Box Generation feature.
"""

import os
import warnings
import numpy as np

# Suppress noisy deprecation warnings from PyTorch internals / upstream libraries
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*", category=FutureWarning)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QPushButton, QStackedWidget, QWidget, QSizePolicy, QComboBox, QCheckBox,
    QDoubleSpinBox, QRadioButton, QGroupBox, QFileDialog, QMessageBox,
    QLineEdit, QFrame, QApplication,
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, QEvent
from PyQt5.QtGui import QPixmap, QImage

from pose_estimation_pipeline import (
    apply_hist_depth_filter,
    estimate_3d_pose,
    make_label_object,
)

# Known class dimensions (L, W, H) in metres — must match DROPDOWN_OPTIONS in label_editor_gui.py
_CLASS_DIMS: dict[str, tuple[float, float, float]] = {
    "pallet":        (1.200, 0.800, 0.144),
    "KLT small":     (0.400, 0.300, 0.147),
    "KLT large":     (0.600, 0.400, 0.147),
    "stillage":      (1.200, 0.800, 0.970),
    "forklift":      (2.800, 1.300, 2.150),
}

# Classes whose footprint is bounded but not fixed.
# Format: ((L_min, L_max), (W_min, W_max), H)
# H may be a float (upper cap, measured value kept when shorter)
# or a (H_min, H_max) tuple (measured height clamped to the range).
_CLASS_DIMS_RANGE: dict[str, tuple] = {
    "small_load_carrier": ((0.400, 0.600), (0.300, 0.400), 0.147),
    "pallet truck":       ((1.375, 2.200), (0.550, 1.000), (1.000, 1.200)),
    "pallet_truck":       ((1.375, 2.200), (0.550, 1.000), (1.000, 1.200)),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ndarray_to_qpixmap(rgb_array: np.ndarray) -> QPixmap:
    """Convert an H×W×3 uint8 numpy array to QPixmap."""
    arr = np.ascontiguousarray(rgb_array, dtype=np.uint8)
    h, w, ch = arr.shape
    if h == 0 or w == 0:
        return QPixmap()
    img = QImage(arr.tobytes(), w, h, w * ch, QImage.Format_RGB888)
    return QPixmap.fromImage(img)


_SEG_PALETTE = [
    (0, 200, 255), (255, 100, 0), (0, 255, 100),
    (200, 0, 255), (255, 200, 0), (0, 100, 255),
]


def _annotate_detections(rgb_img: np.ndarray, detections, selected_idx: int = -1) -> np.ndarray:
    """Blend detection masks into rgb_img using the same style as the main window.

    All masks use a 50/50 colour blend. The selected mask uses a brighter cyan
    and gets a thin border; all others use the shared palette.
    """
    import cv2
    out = rgb_img.copy()
    if detections is None or len(detections) == 0:
        return out

    masks = detections.mask if detections.mask is not None else [None] * len(detections)

    for i, mask in enumerate(masks):
        if mask is None:
            continue
        is_sel = (i == selected_idx)
        c = np.array((0, 200, 255) if is_sel else _SEG_PALETTE[i % len(_SEG_PALETTE)], dtype=np.int32)
        where = mask > 0
        out[where] = np.clip(out[where].astype(np.int32) // 2 + c // 2, 0, 255).astype(np.uint8)

    # Draw a border around the selected detection bbox so it's easy to spot
    if 0 <= selected_idx < len(detections.xyxy):
        x1, y1, x2, y2 = detections.xyxy[selected_idx].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 3)

    return out


# ---------------------------------------------------------------------------
# Background worker — keeps the Qt event loop alive during slow inference
# ---------------------------------------------------------------------------

class _SegWorker(QThread):
    done  = pyqtSignal(object)  # sv.Detections
    error = pyqtSignal(str)

    def __init__(self, func):
        super().__init__()
        self._func = func

    def run(self):
        try:
            self.done.emit(self._func())
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# AutoBBoxValidationDialog
# ---------------------------------------------------------------------------

class AutoBBoxValidationDialog(QDialog):
    """
    3-step modal dialog for validating autonomous 3D bounding box generation.

    Pages:
      0 — Method selection + segmentation run  (with prev/next image browsing)
      1 — Depth histogram with HDF bounds
      2 — Top-down X-Z projection + aligned bounding box

    Parameters
    ----------
    load_fn : callable
        ``load_fn(idx) -> (rgb_img, depth_img, fx, fy, cx, cy)``
        Called whenever the user navigates to a different preview image.
    n_frames : int
        Total number of frames available for browsing.
    """

    def __init__(self, load_fn, n_frames: int,
                 default_yolo_path: str = "", precomputed=None, parent=None):
        """
        Parameters
        ----------
        precomputed : dict or list[dict] or None
            Single dict (backward-compatible merge mode) or list of dicts
            (multi-mask mode from the mask editor). Each dict has keys:
              ``bbox``       – [x1, y1, x2, y2] in RGB image coordinates
              ``mask``       – H×W uint8/bool array (full RGB resolution)
              ``class_name`` – string label
            When a list is passed the dialog lets the user process each mask
            one by one; Accept saves that mask's result and returns to the
            list; Finish collects all saved results and closes.
        """
        super().__init__(parent)
        self.setWindowTitle("Autonomous 3D BB — Validation")
        self.setModal(True)
        self.setMinimumSize(700, 600)

        self._load_fn   = load_fn
        self._n_frames  = n_frames
        self._preview_idx = 0

        self._default_yolo_path = default_yolo_path
        self._class_name = "object"
        self._precomputed_mode = precomputed is not None

        self._selected_idx   = 0
        self._filtered_depth = None
        self._mask_crop      = None
        self._x1 = self._y1 = 0
        self._hdf_params     = {"resolution": 1, "max_height_percent": 5, "ignore_background": False}
        self._depth_crop_cache   = None
        self._depth_masked_cache = None
        self._masks_pix_cache    = None

        # Segmentation state
        self._detections        = None
        self._segmentation_run  = False
        self._yolo_model        = None
        self._seg_worker        = None

        self.conf_threshold: float = 0.5
        self._active_detections = None

        self.validated_frame_idx: int = 0
        self.result: dict | list | None = None

        # Multi-mask mode state
        self._multi_mode = isinstance(precomputed, list)
        self._multi_results: list[dict] = []
        self._done_detection_indices: set[int] = set()

        # Load the first (and only) frame
        self._load_preview_frame(0)

        # Pre-populate detections from precomputed mask(s) before building UI
        if isinstance(precomputed, list):
            import supervision as sv
            xyxy_list, masks_list, classes_list = [], [], []
            for item in precomputed:
                x1, y1, x2, y2 = item["bbox"]
                xyxy_list.append([x1, y1, x2, y2])
                masks_list.append(np.array(item["mask"] > 0, dtype=bool))
                classes_list.append(item.get("class_name", "object"))
            self._detections = sv.Detections(
                xyxy=np.array(xyxy_list, dtype=np.float32),
                mask=np.array(masks_list),
                confidence=np.ones(len(xyxy_list), dtype=np.float32),
                data={"class_name": np.array(classes_list)},
            )
            self._active_detections = self._detections
            self._segmentation_run  = True
            if classes_list:
                self._class_name = classes_list[0]
        elif precomputed is not None:
            import supervision as sv
            x1, y1, x2, y2 = precomputed["bbox"]
            pmask   = np.array(precomputed["mask"] > 0, dtype=bool)
            pclass  = precomputed.get("class_name", "merged")
            self._class_name = pclass
            self._detections = sv.Detections(
                xyxy=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                mask=np.array([pmask]),
                confidence=np.array([1.0]),
                data={"class_name": np.array([pclass])},
            )
            self._active_detections = self._detections
            self._segmentation_run  = True

        self._build_ui()

        # In precomputed mode: jump straight to mask selection page
        if self._precomputed_mode:
            self._stack.setCurrentIndex(1)
            self._back_btn.setVisible(False)
            if self._multi_mode:
                self._masks_header_lbl.setText(
                    "Step 2 / 4 — Click a mask in the image or list to select it, then click Next →:"
                )
            else:
                self._masks_header_lbl.setText(
                    "Merged mask preview — click Next → to run HDF and orientation:"
                )

        self._init_page0()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self._stack = QStackedWidget()
        root.addWidget(self._stack, stretch=1)

        # Page 0: segmentation method selection + run
        self._page0 = QWidget()
        self._stack.addWidget(self._page0)
        self._build_page0()

        # Page 1: mask/detection selection with clickable image
        self._page_masks = QWidget()
        self._stack.addWidget(self._page_masks)
        self._build_page_masks()

        # Page 2: depth histogram filter
        self._page1 = QWidget()
        self._stack.addWidget(self._page1)
        self._build_page1()

        # Page 3: top-down X-Z pose view
        self._page2 = QWidget()
        self._stack.addWidget(self._page2)
        self._build_page2()

        # Navigation buttons (shared)
        nav = QHBoxLayout()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        self._back_btn   = QPushButton("← Back")
        self._back_btn.clicked.connect(self._on_back)
        self._back_btn.setVisible(False)
        self._next_btn   = QPushButton("Next →")
        self._next_btn.clicked.connect(self._on_next)
        self._accept_btn = QPushButton("Accept")
        self._accept_btn.clicked.connect(self._on_accept)
        self._accept_btn.setVisible(False)

        self._finish_btn = QPushButton("✓ Finish")
        self._finish_btn.clicked.connect(self._on_finish_multi)
        self._finish_btn.setVisible(False)

        nav.addWidget(self._cancel_btn)
        nav.addStretch()
        nav.addWidget(self._finish_btn)
        nav.addWidget(self._back_btn)
        nav.addWidget(self._next_btn)
        nav.addWidget(self._accept_btn)
        root.addLayout(nav)

    # ---- page 0: method selection + segmentation run ----
    def _build_page0(self):
        layout = QVBoxLayout(self._page0)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Step 1 / 4 — Choose segmentation method and run:"))

        # ── Image preview with prev/next navigation ──────────────────
        nav_row = QHBoxLayout()
        self._prev_frame_btn = QPushButton("◀")
        self._prev_frame_btn.setFixedWidth(32)
        self._prev_frame_btn.setToolTip("Previous image")
        self._prev_frame_btn.clicked.connect(self._on_prev_frame)
        self._frame_counter_lbl = QLabel("1 / 1")
        self._frame_counter_lbl.setAlignment(Qt.AlignCenter)
        self._next_frame_btn = QPushButton("▶")
        self._next_frame_btn.setFixedWidth(32)
        self._next_frame_btn.setToolTip("Next image")
        self._next_frame_btn.clicked.connect(self._on_next_frame)
        nav_row.addWidget(self._prev_frame_btn)
        nav_row.addStretch()
        nav_row.addWidget(QLabel("Browse to a frame with visible objects, then run segmentation:"))
        nav_row.addStretch()
        nav_row.addWidget(self._frame_counter_lbl)
        nav_row.addWidget(self._next_frame_btn)
        layout.addLayout(nav_row)

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        layout.addWidget(self._img_label, stretch=1)

        # ── Method selector ──────────────────────────────────────────
        self._method_box = QGroupBox("Segmentation method")
        method_layout = QVBoxLayout(self._method_box)
        method_layout.setSpacing(4)

        self._rb_yolo_default = QRadioButton("YOLO — default model  (YOLO11-seg trained on the LOCO dataset)")
        self._rb_yolo_custom  = QRadioButton("YOLO — custom weights")
        self._rb_yolo_default.setChecked(True)

        self._rb_yolo_default.toggled.connect(self._on_method_changed)
        self._rb_yolo_custom.toggled.connect(self._on_method_changed)

        method_layout.addWidget(self._rb_yolo_default)
        method_layout.addWidget(self._rb_yolo_custom)

        # Custom YOLO path row (hidden by default)
        self._custom_path_row = QWidget()
        path_row_layout = QHBoxLayout(self._custom_path_row)
        path_row_layout.setContentsMargins(16, 0, 0, 0)
        path_row_layout.setSpacing(4)
        self._custom_path_edit = QLineEdit()
        self._custom_path_edit.setPlaceholderText("Path to custom .pt weights file…")
        self._custom_path_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._on_browse_weights)
        path_row_layout.addWidget(self._custom_path_edit)
        path_row_layout.addWidget(browse_btn)
        self._custom_path_row.setVisible(False)
        method_layout.addWidget(self._custom_path_row)

        layout.addWidget(self._method_box)

        # ── Options ──────────────────────────────────────────────────
        self._blur_faces_cb = QCheckBox("Blur faces in processed images  (GDPR / data protection)")
        self._blur_faces_cb.setToolTip(
            "After 3D BB generation, automatically detect and blur faces\n"
            "in every RGB image that was processed. Overwrites image files on disk."
        )
        layout.addWidget(self._blur_faces_cb)

        # ── Run Segmentation button ──────────────────────────────────
        self._run_btn = QPushButton("Run Segmentation")
        self._run_btn.setMinimumHeight(36)
        self._run_btn.clicked.connect(self._on_run_segmentation)
        layout.addWidget(self._run_btn)

    # ---- page 1 (NEW): mask selection with clickable image ----
    def _build_page_masks(self):
        layout = QVBoxLayout(self._page_masks)
        layout.setSpacing(6)

        self._masks_header_lbl = QLabel(
            "Step 2 / 4 — Click a mask in the image or the list to select it, then click Next →:"
        )
        layout.addWidget(self._masks_header_lbl)

        # Confidence threshold row
        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Confidence threshold:"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.0, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setValue(0.5)
        self._conf_spin.setFixedWidth(80)
        self._conf_spin.valueChanged.connect(self._apply_conf_filter)
        thr_row.addWidget(self._conf_spin)
        thr_row.addStretch()
        layout.addLayout(thr_row)

        # Main split: image left, list right
        split = QHBoxLayout()
        split.setSpacing(8)

        self._masks_img_label = QLabel()
        self._masks_img_label.setAlignment(Qt.AlignCenter)
        self._masks_img_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self._masks_img_label.installEventFilter(self)
        split.addWidget(self._masks_img_label, stretch=3)

        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("Detected objects:"))
        self._detection_list = QListWidget()
        self._detection_list.currentRowChanged.connect(self._on_detection_selected)
        right_col.addWidget(self._detection_list, stretch=1)

        split.addLayout(right_col, stretch=1)
        layout.addLayout(split, stretch=1)

    def _init_page0(self):
        self._refresh_page0_image()
        self._update_frame_counter()
        if self._precomputed_mode:
            self._apply_conf_filter()

    # ------------------------------------------------------------------
    # Frame navigation (page 0)
    # ------------------------------------------------------------------
    def _load_preview_frame(self, idx: int):
        """Load frame idx via load_fn and update internal rgb/depth/intrinsics."""
        rgb, depth, fx, fy, cx, cy = self._load_fn(idx)
        self._rgb   = rgb
        self._depth = depth
        self._fx, self._fy, self._cx, self._cy = fx, fy, cx, cy
        self._preview_idx = idx

    def _reset_segmentation(self):
        """Clear all segmentation state so the user must re-run on the new frame."""
        self._detections        = None
        self._active_detections = None
        self._segmentation_run  = False
        self._detection_list.clear()
        # Return to seg setup page
        self._stack.setCurrentIndex(0)
        self._back_btn.setVisible(False)
        self._next_btn.setEnabled(True)

    def _on_prev_frame(self):
        if self._preview_idx <= 0:
            return
        self._load_preview_frame(self._preview_idx - 1)
        self._reset_segmentation()
        self._refresh_page0_image()
        self._update_frame_counter()

    def _on_next_frame(self):
        if self._preview_idx >= self._n_frames - 1:
            return
        self._load_preview_frame(self._preview_idx + 1)
        self._reset_segmentation()
        self._refresh_page0_image()
        self._update_frame_counter()

    def _update_frame_counter(self):
        self._frame_counter_lbl.setText(f"{self._preview_idx + 1} / {self._n_frames}")
        self._prev_frame_btn.setEnabled(self._preview_idx > 0)
        self._next_frame_btn.setEnabled(self._preview_idx < self._n_frames - 1)

    def _on_method_changed(self):
        self._custom_path_row.setVisible(self._rb_yolo_custom.isChecked())

    def _on_browse_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO weights file", "", "PyTorch weights (*.pt);;All files (*)"
        )
        if path:
            self._custom_path_edit.setText(path)

    def _get_selected_method(self) -> str:
        if self._rb_yolo_custom.isChecked():
            return "yolo_custom"
        return "yolo_default"

    def _on_run_segmentation(self):
        method = self._get_selected_method()

        if method == "yolo_custom":
            path = self._custom_path_edit.text().strip()
            if not path or not os.path.exists(path):
                QMessageBox.warning(self, "Custom weights missing",
                                    "Please browse to a valid .pt weights file.")
                return

        # Disable buttons while running
        self._next_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        self._run_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)

        rgb_copy   = self._rgb.copy()
        method_cap = method

        def _do_inference():
            import supervision as sv
            from PIL import Image as _PIL

            model_path = (self._default_yolo_path
                          if method_cap == "yolo_default"
                          else self._custom_path_edit.text().strip())
            if self._yolo_model is None or getattr(self, "_yolo_model_path", None) != model_path:
                from ultralytics import YOLO
                self._yolo_model = YOLO(model_path)
                self._yolo_model_path = model_path
            result = self._yolo_model.predict(_PIL.fromarray(rgb_copy), conf=0.25)[0]
            return sv.Detections.from_ultralytics(result)

        self._seg_worker = _SegWorker(_do_inference)
        self._seg_worker.done.connect(self._on_segmentation_done)
        self._seg_worker.error.connect(self._on_segmentation_error)
        self._seg_worker.start()

    def _on_segmentation_done(self, detections):
        QApplication.restoreOverrideCursor()
        self._next_btn.setEnabled(True)
        self._cancel_btn.setEnabled(True)
        self._run_btn.setEnabled(True)
        self._detections = detections
        self._segmentation_run = True
        # Auto-advance to mask selection page
        self._apply_conf_filter()
        self._stack.setCurrentIndex(1)
        self._back_btn.setVisible(True)

    def _on_segmentation_error(self, msg: str):
        QApplication.restoreOverrideCursor()
        self._next_btn.setEnabled(True)
        self._cancel_btn.setEnabled(True)
        self._run_btn.setEnabled(True)
        if "import" in msg.lower() or "no module" in msg.lower():
            QMessageBox.critical(self, "Import error", f"Required package not available:\n{msg}")
        else:
            QMessageBox.critical(self, "Segmentation error", msg)

    def _apply_conf_filter(self):
        """Filter detections by current threshold and refresh page 0."""
        thr = self._conf_spin.value()
        self.conf_threshold = thr

        det = self._detections
        if det is None or len(det) == 0:
            self._active_detections = det
            self._detection_list.clear()
            self._refresh_page0_image()
            self._refresh_masks_image()   # keep page 1 image populated if we advance there
            return

        confs = det.confidence if det.confidence is not None else [1.0] * len(det)
        mask = np.array([float(c) >= thr for c in confs])
        self._active_detections = det[mask]

        names = det.data.get("class_name", [f"det_{i}" for i in range(len(det))])
        self._detection_list.blockSignals(True)
        self._detection_list.clear()
        for i, (nm, cf, keep) in enumerate(zip(names, confs, mask)):
            if keep:
                done = "✓ " if i in self._done_detection_indices else ""
                self._detection_list.addItem(f"{done}[{i}] {nm}  {cf*100:.1f}%")
        self._detection_list.blockSignals(False)
        self._selected_idx = 0
        if self._detection_list.count() > 0:
            self._detection_list.setCurrentRow(0)
        self._refresh_masks_image()

    def _refresh_page0_image(self):
        pix = _ndarray_to_qpixmap(self._rgb)
        if pix.isNull():
            return
        w = self._img_label.width()
        h = self._img_label.height()
        if w <= 0 or h <= 0:
            self._img_label.setPixmap(pix)
            return
        self._img_label.setPixmap(pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _refresh_masks_image(self):
        if self._active_detections is not None:
            annotated = _annotate_detections(self._rgb, self._active_detections, self._selected_idx)
        else:
            annotated = self._rgb
        pix = _ndarray_to_qpixmap(annotated)
        if pix.isNull():
            return
        self._masks_pix_cache = pix          # always keep the latest full-res pixmap
        w = self._masks_img_label.width()
        h = self._masks_img_label.height()
        if w <= 0 or h <= 0:
            return                           # defer until showEvent / resizeEvent sizes us
        self._masks_img_label.setPixmap(
            pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def eventFilter(self, obj, event):
        if obj is self._masks_img_label and event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                self._on_masks_image_clicked(event.pos())
        return super().eventFilter(obj, event)

    def _on_masks_image_clicked(self, pos):
        det = self._active_detections
        if det is None or len(det) == 0:
            return
        img_h, img_w = self._rgb.shape[:2]
        lbl_w = self._masks_img_label.width()
        lbl_h = self._masks_img_label.height()
        scale = min(lbl_w / img_w, lbl_h / img_h)
        off_x = (lbl_w - img_w * scale) / 2
        off_y = (lbl_h - img_h * scale) / 2
        ix = (pos.x() - off_x) / scale
        iy = (pos.y() - off_y) / scale
        if ix < 0 or iy < 0 or ix >= img_w or iy >= img_h:
            return
        # Find the detection whose mask/bbox contains the click
        best = -1
        for i in range(len(det)):
            x1, y1, x2, y2 = det.xyxy[i].astype(int)
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                if det.mask is not None and i < len(det.mask):
                    if det.mask[i][int(iy), int(ix)]:
                        best = i
                        break
                else:
                    best = i
                    break
        if best < 0:
            # Fall back to whichever bbox contains the click
            for i in range(len(det)):
                x1, y1, x2, y2 = det.xyxy[i].astype(int)
                if x1 <= ix <= x2 and y1 <= iy <= y2:
                    best = i
                    break
        if best >= 0:
            self._detection_list.setCurrentRow(best)

    def _on_detection_selected(self, row: int):
        if row < 0:
            return
        self._selected_idx = row
        det = self._active_detections
        if det is not None and len(det) > row:
            names = det.data.get("class_name", [])
            if row < len(names):
                self._class_name = str(names[row])
        self._refresh_masks_image()

    @property
    def blur_faces(self) -> bool:
        return self._blur_faces_cb.isChecked()

    def run_on_image(self, rgb_img: np.ndarray):
        """Run the same YOLO model on a new image. For batch use."""
        import supervision as sv
        from PIL import Image as _PIL
        result = self._yolo_model.predict(_PIL.fromarray(rgb_img), conf=0.25)[0]
        return sv.Detections.from_ultralytics(result)

    # ---- page 1: depth histogram ----
    def _build_page1(self):
        layout = QVBoxLayout(self._page1)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Step 3 / 4 — Histogram depth filter:"))

        h_split = QHBoxLayout()
        h_split.setSpacing(10)

        self._hist_fig    = Figure(tight_layout=True)
        self._hist_canvas = FigureCanvasQTAgg(self._hist_fig)
        h_split.addWidget(self._hist_canvas, stretch=1)

        overlay_col = QVBoxLayout()
        overlay_col.setSpacing(4)
        overlay_col.addWidget(QLabel("Depth mask on image:"))
        self._overlay_label = QLabel()
        self._overlay_label.setAlignment(Qt.AlignCenter)
        self._overlay_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self._overlay_label.setStyleSheet("background-color: #1a1a2e; border: 1px solid #3c3c3c;")
        overlay_col.addWidget(self._overlay_label, stretch=1)
        h_split.addLayout(overlay_col, stretch=1)

        param_panel = QVBoxLayout()
        param_panel.setSpacing(8)

        param_panel.addWidget(QLabel("Resolution (bins/m):"))
        self._res_combo = QComboBox()
        self._res_combo.addItems(["1", "10", "100"])
        self._res_combo.setCurrentIndex(0)
        param_panel.addWidget(self._res_combo)

        param_panel.addWidget(QLabel("Min height %:"))
        self._mhp_combo = QComboBox()
        self._mhp_combo.addItems(["5", "10", "20"])
        self._mhp_combo.setCurrentIndex(0)
        param_panel.addWidget(self._mhp_combo)

        self._ignore_bg_cb = QCheckBox("Ignore background")
        self._ignore_bg_cb.setChecked(False)
        param_panel.addWidget(self._ignore_bg_cb)

        rerun_btn = QPushButton("Re-run")
        rerun_btn.clicked.connect(self._apply_hdf_and_draw)
        param_panel.addWidget(rerun_btn)

        param_panel.addStretch()

        param_widget = QWidget()
        param_widget.setFixedWidth(160)
        param_widget.setLayout(param_panel)
        h_split.addWidget(param_widget)

        layout.addLayout(h_split, stretch=1)

    def _run_hdf_and_show(self):
        import cv2 as _cv2
        det = self._active_detections
        idx = self._selected_idx

        # Bounding box is in RGB-image space (YOLO ran on rgb)
        x1, y1, x2, y2 = det.xyxy[idx].astype(int)

        # Scale box to depth-image space (resolutions may differ, e.g. 1920×1080 → 1280×720)
        rgb_h, rgb_w = self._rgb.shape[:2]
        dep_h, dep_w = self._depth.shape[:2]
        sx = dep_w / rgb_w
        sy = dep_h / rgb_h
        dx1 = int(x1 * sx);  dy1 = int(y1 * sy)
        dx2 = int(x2 * sx);  dy2 = int(y2 * sy)

        # Store depth-space origin — used by estimate_3d_pose
        self._x1, self._y1 = dx1, dy1

        depth_crop = self._depth[dy1:dy2, dx1:dx2].astype(float)
        valid_px = depth_crop[depth_crop > 0]
        if valid_px.size > 0 and valid_px.max() <= 100:
            depth_crop = depth_crop * 1000.0

        if det.mask is not None and idx < len(det.mask):
            full_mask = det.mask[idx].astype(np.uint8)
            # Resize mask from RGB resolution to depth resolution if needed
            if rgb_h != dep_h or rgb_w != dep_w:
                full_mask = _cv2.resize(full_mask, (dep_w, dep_h),
                                        interpolation=_cv2.INTER_NEAREST)
            raw_mask = full_mask[dy1:dy2, dx1:dx2]
        else:
            raw_mask = np.ones_like(depth_crop, dtype=np.uint8)

        self._depth_crop_cache   = depth_crop
        self._depth_masked_cache = np.where(raw_mask, depth_crop, 0)

        self._apply_hdf_and_draw()

    def _apply_hdf_and_draw(self):
        res_bins_per_m = int(self._res_combo.currentText())
        mhp            = int(self._mhp_combo.currentText())
        ignore_bg      = self._ignore_bg_cb.isChecked()

        effective_resolution = res_bins_per_m / 1000.0

        self._hdf_params = {"resolution": effective_resolution,
                            "max_height_percent": mhp,
                            "ignore_background": ignore_bg}

        filtered, hist, bin_edges, lb, ub, threshold, min_h = apply_hist_depth_filter(
            self._depth_masked_cache,
            ignore_background=ignore_bg,
            resolution=effective_resolution,
            max_height_percent=mhp,
        )

        self._filtered_depth = filtered
        self._mask_crop = (filtered > 0).astype(np.uint8)

        self._update_overlay_image()

        ax = (self._hist_fig.clf(), self._hist_fig.add_subplot(1, 1, 1))[1]
        ax.plot(bin_edges[:-1], hist, color="#4fc3f7", label="Histogram")
        ax.axvline(lb,        color="red",    linestyle="--", linewidth=1.5, label=f"LB {lb:.0f}")
        ax.axvline(ub,        color="green",  linestyle="--", linewidth=1.5, label=f"UB {ub:.0f}")
        ax.axhline(threshold, color="orange", linestyle="--", linewidth=1,   label="Avg")
        ax.axhline(min_h,     color="blue",   linestyle="--", linewidth=1,   label="Min-h")
        ax.set_xlabel("Depth (mm)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.set_facecolor("#1a1a2e")
        self._hist_fig.set_facecolor("#1e1e1e")
        ax.tick_params(colors="#d4d4d4")
        ax.xaxis.label.set_color("#d4d4d4")
        ax.yaxis.label.set_color("#d4d4d4")
        self._hist_canvas.draw()

    def _update_overlay_image(self):
        """Build the colour crop + yellow HDF mask overlay."""
        det = self._active_detections
        idx = self._selected_idx
        x1, y1, x2, y2 = det.xyxy[idx].astype(int)

        rgb_crop = self._rgb[y1:y2, x1:x2].copy()

        mask = self._mask_crop
        if mask.shape == rgb_crop.shape[:2]:
            overlay = rgb_crop.copy()
            where = mask > 0
            overlay[where] = np.clip(
                rgb_crop[where].astype(np.int32) // 2 + np.array([255, 255, 0]) // 2,
                0, 255,
            ).astype(np.uint8)
            rgb_crop = overlay

        self._overlay_pixmap = _ndarray_to_qpixmap(rgb_crop)
        self._refresh_overlay_pixmap()

    def _refresh_overlay_pixmap(self):
        if not hasattr(self, "_overlay_pixmap") or self._overlay_pixmap is None:
            return
        if self._overlay_pixmap.isNull():
            return
        w = self._overlay_label.width()
        h = self._overlay_label.height()
        if w > 0 and h > 0:
            self._overlay_label.setPixmap(
                self._overlay_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    # Fallback HDF params used when aggressive settings leave too few points.
    # resolution=1 bin/m, max_height_percent=5 → most permissive histogram filter.
    _FALLBACK_HDF = {"resolution": 1 / 1000.0, "max_height_percent": 5,
                     "ignore_background": False}

    def _prepare_mask_depth(self, i):
        """Return (dep_masked, x1, y1, x2, y2, dx1, dy1) for detection index i, or None on error."""
        import cv2 as _cv2
        det = self._detections
        x1, y1, x2, y2 = det.xyxy[i].astype(int)
        rgb_h, rgb_w = self._rgb.shape[:2]
        dep_h, dep_w = self._depth.shape[:2]
        sx, sy = dep_w / rgb_w, dep_h / rgb_h
        dx1, dy1 = int(x1 * sx), int(y1 * sy)
        dx2, dy2 = int(x2 * sx), int(y2 * sy)

        depth_crop = self._depth[dy1:dy2, dx1:dx2].astype(float)
        if depth_crop.size == 0:
            return None
        valid_px = depth_crop[depth_crop > 0]
        if valid_px.size > 0 and valid_px.max() <= 100:
            depth_crop = depth_crop * 1000.0

        if det.mask is not None and i < len(det.mask):
            full_mask = det.mask[i].astype(np.uint8)
            if rgb_h != dep_h or rgb_w != dep_w:
                full_mask = _cv2.resize(full_mask, (dep_w, dep_h),
                                        interpolation=_cv2.INTER_NEAREST)
            raw_mask = full_mask[dy1:dy2, dx1:dx2]
        else:
            raw_mask = np.ones_like(depth_crop, dtype=np.uint8)

        return np.where(raw_mask, depth_crop, 0), x1, y1, x2, y2, dx1, dy1

    def _count_mask_points(self, i, hdf_params) -> int:
        """Count valid filtered pixels for detection i with given HDF params."""
        prep = self._prepare_mask_depth(i)
        if prep is None:
            return 0
        dep_masked = prep[0]
        filtered, *_ = apply_hist_depth_filter(
            dep_masked,
            ignore_background=hdf_params.get("ignore_background", False),
            resolution=hdf_params["resolution"],
            max_height_percent=hdf_params["max_height_percent"],
        )
        return int((filtered > 0).sum())

    def _process_masks_with_params(self, hdf_params, skip_indices=None):
        """Run HDF + pose for all masks.
        If HDF leaves no points, falls back to all raw depth pixels.
        If pose still fails, saves the annotation with a rough or zero 3D BB
        so the segmentation mask is never discarded.
        Returns (results, errors, []) — retryable list is always empty."""
        det = self._detections
        if det is None or len(det) == 0:
            return [], [], []

        skip = set(skip_indices or [])
        results, errors = [], []
        names = det.data.get("class_name", [])

        for i in range(len(det)):
            if i in skip:
                continue

            class_name = str(names[i]) if i < len(names) else "object"

            prep = self._prepare_mask_depth(i)
            if prep is None:
                errors.append(f"Mask {i + 1} ({class_name}): empty depth crop — skipped.")
                continue

            dep_masked, x1, y1, x2, y2, dx1, dy1 = prep

            filtered, *_ = apply_hist_depth_filter(
                dep_masked,
                ignore_background=hdf_params.get("ignore_background", False),
                resolution=hdf_params["resolution"],
                max_height_percent=hdf_params["max_height_percent"],
            )

            no_3d_bb = False
            center = dims = yaw_deg = None
            if (filtered > 0).sum() == 0:
                no_3d_bb = True
                errors.append(f"Mask {i + 1} ({class_name}): no depth points after filtering — mask saved, add 3D BB manually.")
            else:
                mask_crop = (filtered > 0).astype(np.uint8)
                try:
                    _, center, dims, yaw_deg, _ = estimate_3d_pose(
                        filtered, mask_crop, dx1, dy1,
                        self._fx, self._fy, self._cx, self._cy,
                        class_dims=_CLASS_DIMS.get(class_name),
                        class_dims_range=_CLASS_DIMS_RANGE.get(class_name),
                    )
                except Exception as exc:
                    no_3d_bb = True
                    errors.append(f"Mask {i + 1} ({class_name}): {exc} — mask saved, add 3D BB manually.")

            if no_3d_bb:
                obj = {"name": class_name, "no_3d_bb": True,
                       "bbox_2d": [int(x1), int(y1), int(x2), int(y2)],
                       "centroid": {"x": 0.0, "y": 0.0, "z": 0.0},
                       "dimensions": {"length": 0.0, "width": 0.0, "height": 0.0},
                       "rotations": {"x": 0.0, "y": 0.0, "z": 0.0}}
            else:
                obj = make_label_object(class_name, center, dims, yaw_deg)
                obj["bbox_2d"] = [int(x1), int(y1), int(x2), int(y2)]
            results.append(obj)

        return results, errors, []   # retryable always empty — handled inline

    def _finish_with_results(self, results, errors, n_total):
        if not results:
            detail = "\n".join(errors) if errors else "No valid depth data under any mask."
            QMessageBox.warning(self, "No results",
                f"Could not process any masks.\n\n{detail}")
            return
        if errors:
            QMessageBox.information(self, "Some masks saved with incomplete 3D BB",
                f"Saved {len(results)} mask(s). Notes:\n\n"
                + "\n".join(errors)
                + "\n\nAnnotations with zero or rough 3D BBs can be adjusted manually in the editor.")
        self.validated_frame_idx = self._preview_idx
        self.result = results
        self.accept()

    def _finish_or_retry(self, results, perm_errors, retryable, n_total):
        """Retryable list is now always empty (handled inline). Delegate directly."""
        self._finish_with_results(results, perm_errors, n_total)

    def _finish_or_retry_UNUSED(self, results, perm_errors, retryable, n_total):
        """Kept for reference — retry dialog removed; fallback is now built into
        _process_masks_with_params."""
        if not retryable:
            self._finish_with_results(results, perm_errors, n_total)
            return

        fb = self._FALLBACK_HDF
        lines = []
        for idx, cls in retryable:
            n_fb = self._count_mask_points(idx, fb)
            lines.append(
                f"  • Mask {idx + 1} ({cls}): {n_fb} points with fallback params"
            )

        msg = (
            f"{len(retryable)} mask(s) failed because the current HDF settings "
            f"left too few points:\n\n"
            + "\n".join(lines)
            + "\n\nWould you like to retry these masks with more permissive "
              "parameters?\n  Resolution: 1 bin/m   Max height: 5 %"
        )
        reply = QMessageBox.question(
            self, "Retry with fallback parameters?", msg,
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            retry_skip = (set(range(len(self._detections)))
                          - {idx for idx, _ in retryable})
            retry_results, retry_perm_errors, still_failed = \
                self._process_masks_with_params(fb, skip_indices=retry_skip)
            retry_errors = retry_perm_errors + [
                f"Mask {idx + 1} ({cls}): still too few points even with fallback params"
                for idx, cls in still_failed
            ]
            self._finish_with_results(
                results + retry_results,
                perm_errors + retry_errors,
                n_total,
            )
        else:
            retryable_errors = [
                f"Mask {idx + 1} ({cls}): too few points (skipped retry)"
                for idx, cls in retryable
            ]
            self._finish_with_results(results, perm_errors + retryable_errors, n_total)

    # ---- page 2: top-down X-Z view ----
    def _build_page2(self):
        layout = QVBoxLayout(self._page2)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Step 4 / 4 — Top-down X-Z view. Verify alignment:"))

        self._topdown_fig    = Figure(figsize=(5, 5), tight_layout=True)
        self._topdown_canvas = FigureCanvasQTAgg(self._topdown_fig)
        layout.addWidget(self._topdown_canvas, stretch=1)

        self._pose_label = QLabel("")
        self._pose_label.setStyleSheet("color: #9cdcfe; font-size: 11px;")
        layout.addWidget(self._pose_label)

    def _run_pose_and_show(self):
        try:
            _, center, dimensions, yaw_deg, bbox_result = estimate_3d_pose(
                self._filtered_depth, self._mask_crop,
                self._x1, self._y1,
                self._fx, self._fy, self._cx, self._cy,
                class_dims=_CLASS_DIMS.get(self._class_name),
                class_dims_range=_CLASS_DIMS_RANGE.get(self._class_name),
            )
        except Exception as e:
            # Try to get a rough centroid from whatever sparse depth pixels exist
            rough_center = None
            try:
                ys, xs = np.where(self._mask_crop > 0)
                if len(xs) > 0:
                    zs = self._filtered_depth[ys, xs].astype(np.float32) / 1000.0
                    valid = zs > 0
                    if valid.any():
                        gxs = (xs + self._x1)[valid]
                        gys = (ys + self._y1)[valid]
                        zvs = zs[valid]
                        Xs = (gxs - self._cx) * zvs / self._fx
                        Ys = (gys - self._cy) * zvs / self._fy
                        rough_center = np.array([float(np.median(Xs)),
                                                 float(np.median(Ys)),
                                                 float(np.median(zvs))])
            except Exception:
                pass

            # No 3D BB — mask is kept, user adds BB manually
            self._estimated_center     = None
            self._estimated_dimensions = None
            self._estimated_yaw        = None
            note = "Mask saved without 3D BB — add it manually in the editor."
            self._pose_label.setText(f"Pose failed: {e}  |  {note}")
            ax = (self._topdown_fig.clf(), self._topdown_fig.add_subplot(1, 1, 1))[1]
            ax.text(0.5, 0.5, f"Pose failed:\n{e}\n\n{note}",
                    ha="center", va="center", color="#e0a060",
                    transform=ax.transAxes, fontsize=9)
            ax.set_facecolor("#1a1a2e")
            self._topdown_fig.set_facecolor("#1e1e1e")
            ax.set_xticks([])
            ax.set_yticks([])
            self._topdown_canvas.draw()
            self._accept_btn.setEnabled(True)
            return

        self._estimated_center     = center
        self._estimated_dimensions = dimensions
        self._estimated_yaw        = yaw_deg

        self._pose_label.setText(
            f"Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m   "
            f"Dims (L×W×H): {dimensions[0]:.3f}×{dimensions[2]:.3f}×{dimensions[1]:.3f} m   "
            f"Yaw: {yaw_deg:.1f}°"
        )

        ax = (self._topdown_fig.clf(), self._topdown_fig.add_subplot(1, 1, 1))[1]

        if bbox_result is not None:
            pts   = bbox_result["points_xz"]
            hull  = bbox_result["hull_pts"]
            p1, p2 = bbox_result["edge"]
            bbox  = bbox_result["bbox"]
            u     = bbox_result["direction"]
            mean_xz = pts.mean(axis=0)

            ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.25, color="#4fc3f7")
            ax.scatter(hull[:, 0], hull[:, 1], c="white", s=8)

            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "r", linewidth=3, label="Ref edge")

            bb = np.vstack([bbox, bbox[0]])
            ax.plot(bb[:, 0], bb[:, 1], "g--", linewidth=1.5, label="Aligned BB")

            scale = 0.5 * np.linalg.norm(pts.max(0) - pts.min(0))
            ax.plot([mean_xz[0] - scale * u[0], mean_xz[0] + scale * u[0]],
                    [mean_xz[1] - scale * u[1], mean_xz[1] + scale * u[1]],
                    "r--", linewidth=1, label=f"Yaw {yaw_deg:.1f}°")

        ax.set_xlabel("X (m)", color="#d4d4d4")
        ax.set_ylabel("Z (m)", color="#d4d4d4")
        ax.set_title(f"Top-down view — yaw = {yaw_deg:.1f}°", color="#d4d4d4")
        ax.axis("equal")
        ax.grid(True, color="#333")
        ax.legend(fontsize=8)
        ax.set_facecolor("#1a1a2e")
        self._topdown_fig.set_facecolor("#1e1e1e")
        ax.tick_params(colors="#d4d4d4")
        self._topdown_canvas.draw()

        self._accept_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _on_next(self):
        page = self._stack.currentIndex()
        if page == 0:
            # Seg setup → mask selection (only reachable if user navigated back)
            if not self._segmentation_run:
                QMessageBox.warning(self, "Run segmentation first",
                    "Please select a method and click 'Run Segmentation'.")
                return
            self._stack.setCurrentIndex(1)
            self._back_btn.setVisible(True)
        elif page == 1:
            # Mask selection → HDF
            if self._active_detections is None or len(self._active_detections) == 0:
                QMessageBox.warning(self, "No detections",
                    "No detections above the confidence threshold.\n"
                    "Adjust the threshold or re-run segmentation.")
                return
            self._run_hdf_and_show()
            self._stack.setCurrentIndex(2)
            # Re-render the overlay after the layout pass has distributed space
            # within page 2 (the split between histogram and overlay label).
            # Without this, _refresh_overlay_pixmap uses a pre-layout width that
            # is too large, making the overlay appear zoomed-in on first show.
            QTimer.singleShot(0, self._refresh_overlay_pixmap)
        elif page == 2:
            # HDF → pose
            self._run_pose_and_show()
            self._stack.setCurrentIndex(3)
            self._next_btn.setVisible(False)
            self._accept_btn.setVisible(True)

    def _on_back(self):
        page = self._stack.currentIndex()
        if page == 1:
            self._stack.setCurrentIndex(0)
            self._back_btn.setVisible(False)
        elif page == 2:
            self._stack.setCurrentIndex(1)
        elif page == 3:
            self._stack.setCurrentIndex(2)
            self._next_btn.setVisible(True)
            self._accept_btn.setVisible(False)

    def _on_accept(self):
        det = self._active_detections
        idx = self._selected_idx
        if self._estimated_center is None:
            # Pose failed — save mask only, no 3D BB
            result = {"name": self._class_name, "no_3d_bb": True,
                      "centroid": {"x": 0.0, "y": 0.0, "z": 0.0},
                      "dimensions": {"length": 0.0, "width": 0.0, "height": 0.0},
                      "rotations": {"x": 0.0, "y": 0.0, "z": 0.0}}
        else:
            result = make_label_object(
                self._class_name,
                self._estimated_center,
                self._estimated_dimensions,
                self._estimated_yaw,
            )
        if det is not None and idx < len(det.xyxy):
            x1, y1, x2, y2 = det.xyxy[idx].astype(int).tolist()
            result["bbox_2d"] = [x1, y1, x2, y2]

        if self._multi_mode:
            # Save the reviewed mask's result, then auto-run all remaining masks
            # with the same HDF params and close.
            self._multi_results.append(result)
            self._done_detection_indices.add(idx)
            n_total = len(self._detections) if self._detections is not None else 0
            remaining, perm_errors, retryable = self._process_masks_with_params(
                self._hdf_params, skip_indices=self._done_detection_indices
            )
            all_results = self._multi_results + remaining
            self._finish_or_retry(all_results, perm_errors, retryable, n_total)
        else:
            self.validated_frame_idx = self._preview_idx
            self.result = result
            self.accept()

    def _on_finish_multi(self):
        if not self._multi_results:
            QMessageBox.warning(self, "Nothing processed",
                                "Process at least one mask before finishing.")
            return
        self.validated_frame_idx = self._preview_idx
        self.result = list(self._multi_results)
        self.accept()

    def reject(self):
        self.result = None
        super().reject()

    def showEvent(self, event):
        super().showEvent(event)
        # Now that the dialog is laid out, render images at their correct sizes.
        self._refresh_page0_image()
        if hasattr(self, "_masks_pix_cache") and self._masks_pix_cache is not None:
            self._repaint_masks_label()

    def _repaint_masks_label(self):
        """Scale and display the cached masks pixmap to the current label size."""
        pix = getattr(self, "_masks_pix_cache", None)
        if pix is None or pix.isNull():
            return
        w = self._masks_img_label.width()
        h = self._masks_img_label.height()
        if w <= 0 or h <= 0:
            return
        self._masks_img_label.setPixmap(
            pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        idx = self._stack.currentIndex()
        if idx == 0:
            self._refresh_page0_image()
        elif idx == 1:
            self._repaint_masks_label()
        elif idx == 2:
            self._refresh_overlay_pixmap()

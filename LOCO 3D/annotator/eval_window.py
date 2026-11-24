"""
Evaluation window — run registered 3D OD methods against ground-truth annotations
and display per-frame and aggregate metrics.
"""

import os, sys, json, glob, math, time, subprocess, tempfile, copy
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QLabel, QPushButton,
    QDoubleSpinBox, QTableWidget, QTableWidgetItem, QMenu,
    QSizePolicy, QGroupBox, QScrollArea, QMessageBox, QComboBox,
    QProgressBar, QAbstractItemView, QFrame,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QColor, QFont

try:
    import pyvista as pv
    import open3d as o3d
    from pyvistaqt import QtInteractor
    _HAS_PYVISTA = True
except ImportError:
    _HAS_PYVISTA = False

# ---------------------------------------------------------------------------
# 3D geometry helpers
# ---------------------------------------------------------------------------

def _box_bev_corners(cx: float, cz: float, length: float, width: float,
                     yaw_rad: float) -> np.ndarray:
    """Return the 4 corners of a 2D oriented rectangle in the x-z plane."""
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[ hl,  hw],
                      [-hl,  hw],
                      [-hl, -hw],
                      [ hl, -hw]], dtype=float)
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([cx, cz])


def _polygon_area(poly: np.ndarray) -> float:
    """Shoelace formula."""
    n = len(poly)
    if n < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip_polygon_by_half_plane(poly: np.ndarray, p1: np.ndarray,
                                 p2: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of poly by the left half-plane of edge p1→p2."""
    if len(poly) == 0:
        return poly
    result = []
    n = len(poly)
    for i in range(n):
        cur, nxt = poly[i], poly[(i + 1) % n]
        d_cur = (p2[0] - p1[0]) * (cur[1] - p1[1]) - (p2[1] - p1[1]) * (cur[0] - p1[0])
        d_nxt = (p2[0] - p1[0]) * (nxt[1] - p1[1]) - (p2[1] - p1[1]) * (nxt[0] - p1[0])
        if d_cur >= 0:
            result.append(cur)
        if (d_cur >= 0) != (d_nxt >= 0):
            denom = (p2[0]-p1[0])*(nxt[1]-cur[1]) - (p2[1]-p1[1])*(nxt[0]-cur[0])
            if abs(denom) > 1e-12:
                t = ((p1[0]-cur[0])*(nxt[1]-cur[1])-(p1[1]-cur[1])*(nxt[0]-cur[0])) / denom
                result.append(cur + t * (nxt - cur))
    return np.array(result) if result else np.empty((0, 2))


def _bev_intersection_area(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """Area of intersection of two convex polygons via Sutherland-Hodgman."""
    clip = corners_b.copy()
    n = len(corners_a)
    for i in range(n):
        if len(clip) == 0:
            return 0.0
        clip = _clip_polygon_by_half_plane(clip, corners_a[i], corners_a[(i+1) % n])
    return _polygon_area(clip)


def iou_3d(box_a: dict, box_b: dict) -> float:
    """
    3D IoU = BEV IoU × height overlap ratio.
    Both boxes are dicts with keys 'center', 'dimensions', 'yaw'.
    """
    ca, cb = box_a.get("center", {}), box_b.get("center", {})
    da, db = box_a.get("dimensions", {}), box_b.get("dimensions", {})
    yaw_a  = float(box_a.get("yaw", 0.0))
    yaw_b  = float(box_b.get("yaw", 0.0))

    la, wa, ha = da.get("length", 0), da.get("width", 0), da.get("height", 0)
    lb, wb, hb = db.get("length", 0), db.get("width", 0), db.get("height", 0)
    if la <= 0 or wa <= 0 or ha <= 0 or lb <= 0 or wb <= 0 or hb <= 0:
        return 0.0

    # BEV intersection
    corners_a = _box_bev_corners(ca.get("x", 0), ca.get("z", 0), la, wa, yaw_a)
    corners_b = _box_bev_corners(cb.get("x", 0), cb.get("z", 0), lb, wb, yaw_b)
    inter_bev = _bev_intersection_area(corners_a, corners_b)
    area_a    = la * wa
    area_b    = lb * wb
    union_bev = area_a + area_b - inter_bev
    if union_bev <= 0:
        return 0.0
    bev_iou = inter_bev / union_bev

    # Height overlap (Y axis in camera frame = down)
    y_a = ca.get("y", 0)
    y_b = cb.get("y", 0)
    top_a, bot_a = y_a - ha / 2, y_a + ha / 2
    top_b, bot_b = y_b - hb / 2, y_b + hb / 2
    h_inter = max(0.0, min(bot_a, bot_b) - max(top_a, top_b))
    h_union = max(bot_a, bot_b) - min(top_a, top_b)
    h_iou   = h_inter / h_union if h_union > 0 else 0.0

    return bev_iou * h_iou


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_ap(recalls: list[float], precisions: list[float]) -> float:
    """11-point interpolated average precision (PASCAL VOC)."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p_at_r = [p for r, p in zip(recalls, precisions) if r >= t]
        ap += max(p_at_r) if p_at_r else 0.0
    return ap / 11.0


def compute_metrics(gt_by_frame: dict, pred_by_frame: dict,
                    cat_ids: list[int], iou_thresh: float) -> dict:
    """
    Returns {
        "ap":        {cat_id: float},
        "ap_25":     {cat_id: float},   # always at 0.25
        "ate":       float,
        "aoe":       float,
        "n_matched": int,
    }
    """
    results = {}
    for thresh, key in [(iou_thresh, "ap"), (0.25, "ap_25")]:
        ap_per_class = {}
        for cat in cat_ids:
            # collect all predictions for this class sorted by score descending
            all_preds = []  # (score, frame_id, pred_idx)
            n_gt = 0
            gt_matched = {}  # frame_id → set of matched gt indices

            for fid, gts in gt_by_frame.items():
                gt_this = [g for g in gts if g.get("category_id") == cat]
                n_gt += len(gt_this)
                gt_matched[fid] = set()
                preds = pred_by_frame.get(fid, [])
                for pi, p in enumerate(preds):
                    if p.get("category_id") == cat:
                        all_preds.append((p.get("score", 0.0), fid, pi))

            if n_gt == 0:
                ap_per_class[cat] = float("nan")
                continue

            all_preds.sort(key=lambda x: -x[0])
            tp_list, fp_list = [], []

            for _score, fid, pi in all_preds:
                pred_box = pred_by_frame[fid][pi]["bbox_3d"]
                gts_this = [g for g in gt_by_frame[fid]
                             if g.get("category_id") == cat]
                best_iou, best_gi = 0.0, -1
                for gi, gt in enumerate(gts_this):
                    if gi in gt_matched[fid]:
                        continue
                    iou = iou_3d(pred_box, gt["bbox_3d"])
                    if iou > best_iou:
                        best_iou, best_gi = iou, gi
                if best_iou >= thresh and best_gi >= 0:
                    tp_list.append(1)
                    fp_list.append(0)
                    gt_matched[fid].add(best_gi)
                else:
                    tp_list.append(0)
                    fp_list.append(1)

            cum_tp = np.cumsum(tp_list)
            cum_fp = np.cumsum(fp_list)
            recalls    = (cum_tp / n_gt).tolist()
            precisions = (cum_tp / (cum_tp + cum_fp + 1e-9)).tolist()
            ap_per_class[cat] = compute_ap(recalls, precisions)

        results[key] = ap_per_class

    # ATE and AOE on matched pairs at iou_thresh
    ate_vals, aoe_vals = [], []
    for fid, gts in gt_by_frame.items():
        preds = pred_by_frame.get(fid, [])
        matched_gt = set()
        for p in sorted(preds, key=lambda x: -x.get("score", 0)):
            best_iou, best_gi = 0.0, -1
            for gi, gt in enumerate(gts):
                if gi in matched_gt:
                    continue
                if gt.get("category_id") != p.get("category_id"):
                    continue
                iou = iou_3d(p["bbox_3d"], gt["bbox_3d"])
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_thresh and best_gi >= 0:
                matched_gt.add(best_gi)
                pc = p["bbox_3d"]["center"]
                gc = gts[best_gi]["bbox_3d"]["center"]
                ate_vals.append(math.sqrt((pc["x"]-gc["x"])**2 +
                                          (pc["y"]-gc["y"])**2 +
                                          (pc["z"]-gc["z"])**2))
                py = p["bbox_3d"].get("yaw", 0.0)
                gy = gts[best_gi]["bbox_3d"].get("yaw", 0.0)
                diff = abs(py - gy) % math.pi
                aoe_vals.append(min(diff, math.pi - diff))

    results["ate"]       = float(np.mean(ate_vals)) if ate_vals else float("nan")
    results["aoe"]       = float(np.mean(aoe_vals)) if aoe_vals else float("nan")
    results["n_matched"] = len(ate_vals)
    return results


# ---------------------------------------------------------------------------
# Method discovery
# ---------------------------------------------------------------------------

def discover_methods(annotator_dir: str) -> list[dict]:
    """Return list of method dicts from annotator/methods/*/method.json."""
    methods_dir = os.path.join(annotator_dir, "methods")
    if not os.path.isdir(methods_dir):
        return []
    result = []
    for json_path in sorted(glob.glob(os.path.join(methods_dir, "*", "method.json"))):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception:
            data = {"name": os.path.basename(os.path.dirname(json_path))}
        data["_dir"]     = os.path.dirname(json_path)
        data["_predict"] = os.path.join(data["_dir"], "predict.py")
        result.append(data)
    return result


# ---------------------------------------------------------------------------
# Worker thread — runs a method subprocess
# ---------------------------------------------------------------------------

class MethodRunWorker(QThread):
    done  = pyqtSignal(str, dict, float)   # method_name, {frame_id: [preds]}, elapsed_s
    error = pyqtSignal(str, str)           # method_name, error_message

    def __init__(self, method: dict, frames: list, parent=None):
        super().__init__(parent)
        self._method = method
        self._frames = frames

    def run(self):
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            inp = os.path.join(tmp, "input.json")
            out = os.path.join(tmp, "output.json")
            with open(inp, "w") as f:
                json.dump({"frames": self._frames}, f)
            try:
                proc = subprocess.run(
                    [sys.executable, self._method["_predict"],
                     "--input", inp, "--output", out],
                    capture_output=True, text=True, timeout=600,
                )
                if proc.returncode != 0:
                    self.error.emit(self._method["name"],
                                    proc.stderr[-2000:] or "Unknown error")
                    return
                with open(out) as f:
                    predictions = json.load(f)
            except subprocess.TimeoutExpired:
                self.error.emit(self._method["name"], "Timed out after 10 minutes")
                return
            except Exception as e:
                self.error.emit(self._method["name"], str(e))
                return
        elapsed = time.time() - t0
        self.done.emit(self._method["name"], predictions, elapsed)


# ---------------------------------------------------------------------------
# Eval window
# ---------------------------------------------------------------------------

_TREE_SS = (
    "QTreeWidget::item:selected { background:#094771; color:#fff; }"
    "QTreeWidget::item:hover { background:#2a2d2e; }"
)
_HDR_SS = (
    "QHeaderView::section { background:#2a2a2a; color:#888; "
    "border:none; border-bottom:1px solid #3c3c3c; padding:2px 4px; }"
)


class EvalWindow(QMainWindow):

    def __init__(self, dataset_root: str, pcd_dir: str, img_dir: str,
                 ann_dir: str, all_frame_files: list, current_user: dict | None,
                 coco_cat_id_to_name: dict, dataset_axes: dict,
                 style_sheet: str, annotator_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LOCO3D — Evaluation")
        self.resize(1600, 900)
        self.setStyleSheet(style_sheet)

        self._dataset_root      = dataset_root
        self._pcd_dir           = pcd_dir
        self._img_dir           = img_dir
        self._ann_dir           = ann_dir
        self._all_frame_files   = all_frame_files  # list of "XXXXXX.json" basenames
        self._current_user      = current_user
        self._cat_id_to_name    = coco_cat_id_to_name
        self._cat_name_to_id    = {v: k for k, v in coco_cat_id_to_name.items()}
        self._dataset_axes      = dataset_axes
        self._annotator_dir     = annotator_dir

        self._methods           = discover_methods(annotator_dir)
        self._current_frame_id: str | None = None
        self._gt_objects: list  = []
        self._method_results: dict[str, dict] = {}  # name → {frame_id: [preds]}
        self._method_metrics: dict[str, dict] = {}  # name → metrics dict
        self._method_elapsed: dict[str, float] = {} # name → elapsed seconds
        self._run_worker: MethodRunWorker | None = None
        self._camera_needs_reset = True

        self._build_ui()
        self._populate_frame_list()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([220, 820, 380])
        self.setCentralWidget(splitter)

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Filter bar
        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.addWidget(QLabel("Frames:"))
        filter_row.addStretch()
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All frames", "With annotations"])
        if self._current_user and self._current_user.get("role") != "admin":
            self._filter_combo.addItem("My frames")
        self._filter_combo.currentIndexChanged.connect(self._populate_frame_list)
        filter_row.addWidget(self._filter_combo)
        lay.addLayout(filter_row)

        # Frame list
        self._frame_list = QTreeWidget()
        self._frame_list.setHeaderLabels(["Frame", "Annotator"])
        self._frame_list.setRootIsDecorated(False)
        self._frame_list.setSelectionBehavior(QTreeWidget.SelectRows)
        hdr = self._frame_list.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        self._frame_list.setStyleSheet(_TREE_SS)
        self._frame_list.header().setStyleSheet(_HDR_SS)
        self._frame_list.currentItemChanged.connect(self._on_frame_selected)
        lay.addWidget(self._frame_list)

        self._frame_count_lbl = QLabel("")
        self._frame_count_lbl.setStyleSheet("color:#888; font-size:11px;")
        lay.addWidget(self._frame_count_lbl)

        # Navigation
        nav = QHBoxLayout()
        nav.setContentsMargins(0, 0, 0, 0)
        prev_btn = QPushButton("←")
        prev_btn.setFixedWidth(36)
        prev_btn.clicked.connect(self._on_prev_frame)
        rnd_btn = QPushButton("⟳ Random")
        rnd_btn.clicked.connect(self._on_random_frame)
        next_btn = QPushButton("→")
        next_btn.setFixedWidth(36)
        next_btn.clicked.connect(self._on_next_frame)
        nav.addWidget(prev_btn)
        nav.addWidget(rnd_btn, stretch=1)
        nav.addWidget(next_btn)
        lay.addLayout(nav)
        return w

    def _build_center_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        vsplit = QSplitter(Qt.Vertical)

        # 3D viewer
        if _HAS_PYVISTA:
            self.plotter = QtInteractor(parent=w)
            self.plotter.set_background("#1a1a2e", top="#0d0d1a")
            vsplit.addWidget(self.plotter)
        else:
            lbl = QLabel("PyVista not available")
            lbl.setAlignment(Qt.AlignCenter)
            vsplit.addWidget(lbl)

        # Bottom: RGB image + metrics panel
        bottom = QSplitter(Qt.Horizontal)

        img_w = QWidget()
        img_lay = QVBoxLayout(img_w)
        img_lay.setContentsMargins(4, 4, 4, 4)
        self._img_lbl = QLabel()
        self._img_lbl.setAlignment(Qt.AlignCenter)
        self._img_lbl.setStyleSheet("background:#111; border:1px solid #3c3c3c;")
        self._img_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img_lay.addWidget(self._img_lbl)
        bottom.addWidget(img_w)

        bottom.addWidget(self._build_metrics_panel())
        bottom.setSizes([420, 400])
        vsplit.addWidget(bottom)
        vsplit.setSizes([450, 280])
        lay.addWidget(vsplit)
        return w

    def _build_metrics_panel(self) -> QWidget:
        w = QGroupBox("Method metrics")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 8, 6, 8)
        lay.setSpacing(6)

        self._metrics_table = QTableWidget(0, 7)
        self._metrics_table.setHorizontalHeaderLabels(
            ["Method", "AP@0.5", "AP@0.25", "mAP/class", "ATE (m)", "AOE (rad)", "FPS"]
        )
        self._metrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 7):
            self._metrics_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self._metrics_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._metrics_table.setAlternatingRowColors(True)
        self._metrics_table.verticalHeader().setVisible(False)
        lay.addWidget(self._metrics_table)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # indeterminate
        self._progress_bar.setVisible(False)
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setStyleSheet(
            "QProgressBar { border:none; background:#2a2a2a; }"
            "QProgressBar::chunk { background:#007acc; }"
        )
        lay.addWidget(self._progress_bar)

        run_btn = QPushButton("▶  Run Method…")
        run_btn.setFixedHeight(38)
        run_btn.setStyleSheet(
            "QPushButton { background:#1a7a30; color:#fff; font-weight:bold; "
            "border:1px solid #2a9a40; border-radius:4px; font-size:13px; }"
            "QPushButton:hover { background:#22963a; }"
            "QPushButton:pressed { background:#155f24; }"
        )
        run_btn.clicked.connect(self._on_run_method_clicked)
        lay.addWidget(run_btn)

        self._run_status_lbl = QLabel("")
        self._run_status_lbl.setStyleSheet("color:#888; font-size:11px;")
        self._run_status_lbl.setWordWrap(True)
        lay.addWidget(self._run_status_lbl)
        return w

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        # IoU threshold
        iou_row = QHBoxLayout()
        iou_row.setContentsMargins(0, 0, 0, 0)
        iou_row.addWidget(QLabel("IoU threshold:"))
        self._iou_spin = QDoubleSpinBox()
        self._iou_spin.setRange(0.05, 0.95)
        self._iou_spin.setSingleStep(0.05)
        self._iou_spin.setDecimals(2)
        self._iou_spin.setValue(0.50)
        self._iou_spin.setFixedWidth(70)
        self._iou_spin.valueChanged.connect(self._refresh_comparison_table)
        iou_row.addWidget(self._iou_spin)
        iou_row.addStretch()
        lay.addLayout(iou_row)

        # Objects list (ground truth)
        obj_hdr = QLabel("Ground-truth objects")
        obj_hdr.setStyleSheet("color:#9cdcfe; font-weight:bold;")
        lay.addWidget(obj_hdr)

        self._obj_list = QTreeWidget()
        self._obj_list.setHeaderLabels(["#", "Class", "Score"])
        self._obj_list.setRootIsDecorated(False)
        self._obj_list.setMaximumHeight(160)
        hdr = self._obj_list.header()
        hdr.setSectionResizeMode(0, QHeaderView.Fixed)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._obj_list.setColumnWidth(0, 28)
        self._obj_list.setStyleSheet(_TREE_SS)
        self._obj_list.header().setStyleSheet(_HDR_SS)
        self._obj_list.currentItemChanged.connect(self._refresh_comparison_table)
        lay.addWidget(self._obj_list)

        # Comparison table (GT vs predictions)
        cmp_hdr = QLabel("Dimension / pose comparison")
        cmp_hdr.setStyleSheet("color:#9cdcfe; font-weight:bold;")
        lay.addWidget(cmp_hdr)

        self._cmp_table = QTableWidget(0, 1)
        self._cmp_table.setHorizontalHeaderLabels(["GT"])
        self._cmp_table.verticalHeader().setDefaultSectionSize(22)
        self._cmp_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._cmp_table.setAlternatingRowColors(True)
        self._cmp_table.horizontalHeader().setStyleSheet(_HDR_SS)
        self._cmp_table.verticalHeader().setStyleSheet(_HDR_SS)
        lay.addWidget(self._cmp_table, stretch=1)
        return w

    # ------------------------------------------------------------------
    # Frame list
    # ------------------------------------------------------------------

    def _visible_frames(self) -> list[str]:
        mode = self._filter_combo.currentText()
        result = []
        username = (self._current_user or {}).get("username", "")
        for fname in self._all_frame_files:
            fid = os.path.splitext(fname)[0]
            ann_path = os.path.join(self._ann_dir, fid + ".json")
            has_ann = os.path.exists(ann_path)
            if mode == "With annotations" and not has_ann:
                continue
            if mode == "My frames":
                if not has_ann:
                    continue
                try:
                    with open(ann_path) as f:
                        doc = json.load(f)
                    if doc.get("annotated_by") != username:
                        continue
                except Exception:
                    continue
            result.append(fid)
        return result

    def _populate_frame_list(self):
        self._frame_list.blockSignals(True)
        self._frame_list.clear()
        self._visible = self._visible_frames()
        for fid in self._visible:
            ann_path = os.path.join(self._ann_dir, fid + ".json")
            annotator = ""
            if os.path.exists(ann_path):
                try:
                    with open(ann_path) as f:
                        doc = json.load(f)
                    annotator = doc.get("annotated_by", "")
                except Exception:
                    pass
            it = QTreeWidgetItem([fid, annotator])
            self._frame_list.addTopLevelItem(it)
        self._frame_count_lbl.setText(f"{len(self._visible)} frames")
        self._frame_list.blockSignals(False)

    # ------------------------------------------------------------------
    # Frame navigation
    # ------------------------------------------------------------------

    def _on_frame_selected(self, item, _prev):
        if item is None:
            return
        self._load_frame(item.text(0))

    def _on_prev_frame(self):
        idx = self._current_list_index()
        if idx > 0:
            self._frame_list.setCurrentItem(self._frame_list.topLevelItem(idx - 1))

    def _on_next_frame(self):
        idx = self._current_list_index()
        n = self._frame_list.topLevelItemCount()
        if idx < n - 1:
            self._frame_list.setCurrentItem(self._frame_list.topLevelItem(idx + 1))

    def _on_random_frame(self):
        import random
        n = self._frame_list.topLevelItemCount()
        if n:
            self._frame_list.setCurrentItem(
                self._frame_list.topLevelItem(random.randrange(n)))

    def _current_list_index(self) -> int:
        item = self._frame_list.currentItem()
        if item is None:
            return -1
        return self._frame_list.indexOfTopLevelItem(item)

    # ------------------------------------------------------------------
    # Frame loading
    # ------------------------------------------------------------------

    def _load_frame(self, frame_id: str):
        self._current_frame_id = frame_id
        self._gt_objects = []
        ann_path = os.path.join(self._ann_dir, frame_id + ".json")
        if os.path.exists(ann_path):
            try:
                with open(ann_path) as f:
                    doc = json.load(f)
                for ann in doc.get("annotations", []):
                    bbox3d = ann.get("bbox_3d", {})
                    cen  = bbox3d.get("center", {})
                    dims = bbox3d.get("dimensions", {})
                    self._gt_objects.append({
                        "category_id": ann.get("category_id", 0),
                        "bbox_3d": {
                            "center":     {"x": cen.get("x",0), "y": cen.get("y",0), "z": cen.get("z",0)},
                            "dimensions": {"length": dims.get("length",0), "width": dims.get("width",0), "height": dims.get("height",0)},
                            "yaw":        bbox3d.get("yaw", 0.0),
                        }
                    })
            except Exception:
                pass

        self._populate_obj_list()
        self._refresh_rgb_image()
        self._render_scene(reset_camera=True)

    def _populate_obj_list(self):
        self._obj_list.blockSignals(True)
        self._obj_list.clear()
        for i, obj in enumerate(self._gt_objects):
            cat_name = self._cat_id_to_name.get(obj["category_id"], f"cat{obj['category_id']}")
            it = QTreeWidgetItem([str(i), cat_name, "GT"])
            it.setForeground(1, QColor("#9cdcfe"))
            self._obj_list.addTopLevelItem(it)
            # Also add prediction rows for each loaded method
            for mname, preds_by_frame in self._method_results.items():
                if self._current_frame_id not in preds_by_frame:
                    continue
                matched = self._best_pred_for_gt(i, preds_by_frame[self._current_frame_id])
                score_str = f"{matched['score']:.2f}" if matched else "—"
                child = QTreeWidgetItem(["", f"  ↳ {mname[:20]}", score_str])
                child.setForeground(1, QColor("#888"))
                it.addChild(child)
        self._obj_list.expandAll()
        self._obj_list.blockSignals(False)
        self._refresh_comparison_table()

    def _best_pred_for_gt(self, gt_idx: int, preds: list) -> dict | None:
        """Return the highest-scoring prediction that best matches this GT object."""
        if gt_idx >= len(self._gt_objects):
            return None
        gt = self._gt_objects[gt_idx]
        thresh = self._iou_spin.value()
        best, best_iou = None, 0.0
        for p in preds:
            if p.get("category_id") != gt["category_id"]:
                continue
            iou = iou_3d(p["bbox_3d"], gt["bbox_3d"])
            if iou > best_iou:
                best_iou, best = iou, p
        return best if best_iou >= thresh else None

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def _refresh_comparison_table(self):
        item = self._obj_list.currentItem()
        if item is None or item.parent() is not None:
            # deselect child rows
            self._cmp_table.setRowCount(0)
            return

        gt_idx = int(item.text(0)) if item.text(0).isdigit() else -1
        if gt_idx < 0 or gt_idx >= len(self._gt_objects):
            self._cmp_table.setRowCount(0)
            return

        gt = self._gt_objects[gt_idx]
        methods_with_data = [
            m for m in self._method_results
            if self._current_frame_id in self._method_results[m]
        ]

        # Columns: GT + one per method
        col_labels = ["GT"] + methods_with_data
        self._cmp_table.setColumnCount(len(col_labels))
        self._cmp_table.setHorizontalHeaderLabels(col_labels)
        self._cmp_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        rows = ["Length (m)", "Width (m)", "Height (m)",
                "X (m)", "Y (m)", "Z (m)", "Yaw (°)", "IoU"]
        self._cmp_table.setRowCount(len(rows))
        self._cmp_table.setVerticalHeaderLabels(rows)

        gc = gt["bbox_3d"]["center"]
        gd = gt["bbox_3d"]["dimensions"]
        gy = gt["bbox_3d"].get("yaw", 0.0)
        gt_vals = [gd.get("length",0), gd.get("width",0), gd.get("height",0),
                   gc.get("x",0), gc.get("y",0), gc.get("z",0),
                   math.degrees(gy), "—"]

        def _cell(val):
            txt = f"{val:.3f}" if isinstance(val, float) else str(val)
            it = QTableWidgetItem(txt)
            it.setTextAlignment(Qt.AlignCenter)
            return it

        for ri, v in enumerate(gt_vals):
            self._cmp_table.setItem(ri, 0, _cell(v))

        for ci, mname in enumerate(methods_with_data, start=1):
            pred = self._best_pred_for_gt(gt_idx, self._method_results[mname].get(self._current_frame_id, []))
            if pred is None:
                for ri in range(len(rows)):
                    self._cmp_table.setItem(ri, ci, _cell("—"))
            else:
                pc = pred["bbox_3d"]["center"]
                pd = pred["bbox_3d"]["dimensions"]
                py = pred["bbox_3d"].get("yaw", 0.0)
                iou_val = iou_3d(pred["bbox_3d"], gt["bbox_3d"])
                vals = [pd.get("length",0), pd.get("width",0), pd.get("height",0),
                        pc.get("x",0), pc.get("y",0), pc.get("z",0),
                        math.degrees(py), iou_val]
                for ri, v in enumerate(vals):
                    cell = _cell(v)
                    if ri == 7:  # IoU
                        iou_color = "#4ec94e" if iou_val >= self._iou_spin.value() else "#e05252"
                        cell.setForeground(QColor(iou_color))
                    self._cmp_table.setItem(ri, ci, cell)

    # ------------------------------------------------------------------
    # RGB image
    # ------------------------------------------------------------------

    def _refresh_rgb_image(self):
        if not self._current_frame_id or not self._img_dir:
            self._img_lbl.clear()
            return
        for ext in (".png", ".jpg", ".jpeg"):
            path = os.path.join(self._img_dir, self._current_frame_id + ext)
            if os.path.exists(path):
                pix = QPixmap(path)
                self._img_lbl.setPixmap(
                    pix.scaled(self._img_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                return
        self._img_lbl.clear()

    # ------------------------------------------------------------------
    # 3D rendering
    # ------------------------------------------------------------------

    def _render_scene(self, reset_camera: bool = False):
        if not _HAS_PYVISTA:
            return
        do_reset = reset_camera or self._camera_needs_reset
        cam_pos = None if do_reset else self.plotter.camera_position
        self.plotter.clear()
        if self._current_frame_id is None:
            return

        # Load point cloud
        for ext in (".pcd", ".ply"):
            pcd_path = os.path.join(self._pcd_dir, self._current_frame_id + ext)
            if os.path.exists(pcd_path):
                try:
                    pcd_o3d = o3d.io.read_point_cloud(pcd_path)
                    pts = np.asarray(pcd_o3d.points)
                    if pts.shape[0] > 0:
                        cloud = pv.PolyData(pts)
                        cols = np.asarray(pcd_o3d.colors)
                        if cols.shape[0] == pts.shape[0]:
                            cloud["RGB"] = (cols * 255).astype(np.uint8)
                            self.plotter.add_mesh(cloud, scalars="RGB", rgb=True,
                                                  point_size=2, style="points",
                                                  render_points_as_spheres=False)
                        else:
                            self.plotter.add_mesh(cloud, color="white", point_size=2, style="points")
                except Exception:
                    pass
                break

        # Draw GT boxes (solid)
        from label_editor_gui import build_pv_bbox
        for obj_data in self._gt_objects:
            cat_name = self._cat_id_to_name.get(obj_data["category_id"], "object")
            bbox3d = obj_data["bbox_3d"]
            cen = bbox3d["center"]
            dims = bbox3d["dimensions"]
            # Build obj dict in the format build_pv_bbox expects
            obj = {
                "name": cat_name,
                "centroid":   {"x": cen["x"], "y": cen["y"], "z": cen["z"]},
                "dimensions": {"length": dims["length"], "width": dims["width"], "height": dims["height"]},
                "rotations":  {"x": 0.0, "y": math.degrees(bbox3d.get("yaw", 0.0)), "z": 0.0},
            }
            mesh, color = build_pv_bbox(obj)
            self.plotter.add_mesh(mesh, color=color, line_width=3, opacity=1.0)

        # Draw prediction boxes (dashed / transparent) for each loaded method
        pred_colors = ["#ff6666", "#66ccff", "#ffcc66", "#cc66ff", "#66ff99"]
        for mi, (mname, preds_by_frame) in enumerate(self._method_results.items()):
            preds = preds_by_frame.get(self._current_frame_id, [])
            pcol = pred_colors[mi % len(pred_colors)]
            for p in preds:
                bbox3d = p["bbox_3d"]
                cat_name = self._cat_id_to_name.get(p.get("category_id", 0), "object")
                cen  = bbox3d["center"]
                dims = bbox3d["dimensions"]
                obj = {
                    "name": cat_name,
                    "centroid":   {"x": cen["x"], "y": cen["y"], "z": cen["z"]},
                    "dimensions": {"length": dims["length"], "width": dims["width"], "height": dims["height"]},
                    "rotations":  {"x": 0.0, "y": math.degrees(bbox3d.get("yaw", 0.0)), "z": 0.0},
                }
                mesh, _ = build_pv_bbox(obj)
                self.plotter.add_mesh(mesh, color=pcol, line_width=2, opacity=0.7)

        self.plotter.add_axes()
        if do_reset:
            self.plotter.reset_camera()
            self._camera_needs_reset = False
        else:
            self.plotter.camera_position = cam_pos

    # ------------------------------------------------------------------
    # Run method
    # ------------------------------------------------------------------

    def _on_run_method_clicked(self):
        if self._run_worker and self._run_worker.isRunning():
            QMessageBox.information(self, "Busy", "A method is already running.")
            return
        if not self._methods:
            QMessageBox.warning(self, "No methods",
                                "No method.json files found in annotator/methods/.")
            return

        menu = QMenu(self)
        # Scope options
        scope_all    = menu.addAction("▶ Run on ALL visible frames")
        scope_single = menu.addAction(f"▶ Run on current frame only ({self._current_frame_id or '—'})")
        menu.addSeparator()
        for m in self._methods:
            menu.addSeparator()
            menu.addSection(m.get("name", "?"))

        # Actually build proper sub-menu: pick method then scope
        menu2 = QMenu(self)
        method_actions = {}
        for m in self._methods:
            sub = menu2.addMenu(m.get("name", m["_dir"]))
            a_all    = sub.addAction("Run on all visible frames")
            a_single = sub.addAction(f"Run on current frame ({self._current_frame_id or '—'})")
            method_actions[id(a_all)]    = (m, "all")
            method_actions[id(a_single)] = (m, "single")

        btn = self.sender()
        gpos = btn.mapToGlobal(btn.rect().bottomLeft())
        chosen = menu2.exec_(gpos)
        if chosen is None or id(chosen) not in method_actions:
            return

        method, scope = method_actions[id(chosen)]
        if scope == "single":
            if not self._current_frame_id:
                return
            frame_ids = [self._current_frame_id]
        else:
            frame_ids = self._visible

        frames_payload = []
        for fid in frame_ids:
            pcd_path = None
            for ext in (".pcd", ".ply"):
                p = os.path.join(self._pcd_dir, fid + ext)
                if os.path.exists(p):
                    pcd_path = p
                    break
            img_path = None
            for ext in (".png", ".jpg", ".jpeg"):
                p = os.path.join(self._img_dir, fid + ext)
                if os.path.exists(p):
                    img_path = p
                    break
            frames_payload.append({
                "frame_id":    fid,
                "pcd_path":    pcd_path or "",
                "image_path":  img_path  or "",
                "dataset_root": self._dataset_root,
            })

        self._progress_bar.setVisible(True)
        self._run_status_lbl.setText(f"Running {method['name']} on {len(frames_payload)} frames…")
        self._run_worker = MethodRunWorker(method, frames_payload, parent=self)
        self._run_worker.done.connect(self._on_method_done)
        self._run_worker.error.connect(self._on_method_error)
        self._run_worker.start()

    def _on_method_done(self, method_name: str, predictions: dict, elapsed: float):
        self._progress_bar.setVisible(False)
        fps = len(predictions) / elapsed if elapsed > 0 else 0.0
        self._method_results[method_name] = predictions
        self._method_elapsed[method_name] = elapsed

        # Compute metrics
        gt_all = {}
        for fid, preds in predictions.items():
            ann_path = os.path.join(self._ann_dir, fid + ".json")
            if not os.path.exists(ann_path):
                continue
            try:
                with open(ann_path) as f:
                    doc = json.load(f)
                gt_all[fid] = []
                for ann in doc.get("annotations", []):
                    bbox3d = ann.get("bbox_3d", {})
                    cen  = bbox3d.get("center", {})
                    dims = bbox3d.get("dimensions", {})
                    gt_all[fid].append({
                        "category_id": ann.get("category_id", 0),
                        "bbox_3d": {
                            "center":     cen,
                            "dimensions": dims,
                            "yaw":        bbox3d.get("yaw", 0.0),
                        }
                    })
            except Exception:
                pass

        all_cats = list(set(
            obj["category_id"]
            for gts in gt_all.values() for obj in gts
        ))
        metrics = compute_metrics(gt_all, predictions, all_cats, self._iou_spin.value())
        metrics["fps"] = fps
        self._method_metrics[method_name] = metrics
        self._refresh_metrics_table()

        self._run_status_lbl.setText(
            f"{method_name}: done in {elapsed:.1f}s ({fps:.1f} FPS)"
        )
        self._populate_obj_list()
        self._render_scene()

    def _on_method_error(self, method_name: str, error: str):
        self._progress_bar.setVisible(False)
        self._run_status_lbl.setText(f"Error in {method_name}")
        QMessageBox.critical(self, f"Method error — {method_name}", error)

    # ------------------------------------------------------------------
    # Metrics table
    # ------------------------------------------------------------------

    def _refresh_metrics_table(self):
        self._metrics_table.setRowCount(len(self._method_metrics))
        for row, (mname, m) in enumerate(self._method_metrics.items()):
            ap_vals = [v for v in m.get("ap", {}).values()
                       if not math.isnan(v)]
            ap25_vals = [v for v in m.get("ap_25", {}).values()
                         if not math.isnan(v)]
            ap50   = f"{np.mean(ap_vals):.3f}"   if ap_vals   else "—"
            ap25   = f"{np.mean(ap25_vals):.3f}" if ap25_vals else "—"
            # mAP per class: list each class
            cat_aps = {self._cat_id_to_name.get(k, str(k)): v
                       for k, v in m.get("ap", {}).items()
                       if not math.isnan(v)}
            map_str = "  ".join(f"{n}:{v:.2f}" for n, v in cat_aps.items()) or "—"
            ate    = f"{m['ate']:.3f}" if not math.isnan(m.get("ate", float("nan"))) else "—"
            aoe    = f"{m['aoe']:.3f}" if not math.isnan(m.get("aoe", float("nan"))) else "—"
            fps    = f"{m.get('fps', 0):.1f}"

            vals = [mname, ap50, ap25, map_str, ate, aoe, fps]
            for col, v in enumerate(vals):
                cell = QTableWidgetItem(v)
                cell.setTextAlignment(Qt.AlignCenter if col > 0 else Qt.AlignLeft | Qt.AlignVCenter)
                self._metrics_table.setItem(row, col, cell)

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if _HAS_PYVISTA:
            try:
                self.plotter.close()
            except Exception:
                pass
        super().closeEvent(event)

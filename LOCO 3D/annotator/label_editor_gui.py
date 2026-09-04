import sys
import os
import re
import json
import copy
import random
import hashlib
import datetime
import subprocess
import cv2

# Allow sibling-module imports (pose_estimation_pipeline, auto_bbox_dialog)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import open3d as o3d
import pyvista as pv
from pyvistaqt import QtInteractor
from PIL import Image as _PIL_Image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QLineEdit, QLabel,
    QFileDialog, QMessageBox, QSplitter, QGroupBox, QGridLayout,
    QDialog, QComboBox, QDialogButtonBox, QFormLayout, QDoubleSpinBox,
    QStyle, QCheckBox, QSizePolicy, QScrollArea, QFrame, QProgressBar,
    QInputDialog, QTabWidget, QTreeWidget, QTreeWidgetItem, QHeaderView,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect, QEvent
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QIcon

pv.set_plot_theme("dark")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_DIR         = os.path.expanduser("~/.3d_label_editor")
PROJECTS_DIR    = os.path.join(APP_DIR, "projects")
CONFIG_PATH     = os.path.join(APP_DIR, "config.json")
_OLD_CONFIG_PATH = os.path.expanduser("~/.3d_label_editor.json")
MAX_RECENT      = 5
YOLO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best.pt")

PALLET_CLASSES = {"pallet", "pallet truck", "pallet_truck"}

BBOX_LINES = [
    [0,1], [1,2], [2,3], [3,0],
    [4,5], [5,6], [6,7], [7,4],
    [0,4], [1,5], [2,6], [3,7],
]

CLASS_COLORS = {
    "forklift":           (0.0, 1.0, 0.0),
    "pallet_truck":       (1.0, 0.5, 0.0),
    "pallet":             (0.0, 1.0, 1.0),
    "small_load_carrier": (1.0, 0.0, 1.0),
    "stillage":           (1.0, 1.0, 0.0),
    "person":             (1.0, 1.0, 0.0),
}
DEFAULT_COLOR = (0.0, 0.5, 1.0)

FIELD_KEYS = [
    ("centroid",   "x"), ("centroid",   "y"), ("centroid",   "z"),
    ("dimensions", "length"), ("dimensions", "width"), ("dimensions", "height"),
    ("rotations",  "x"), ("rotations",  "y"), ("rotations",  "z"),
]
FIELD_LABELS = [
    "centroid x", "centroid y", "centroid z",
    "dim length",  "dim width",  "dim height",
    "rot x",       "rot y",      "rot z",
]

DROPDOWN_OPTIONS = ["pallet", "small load carrier", "stillage", "forklift", "pallet truck", "Custom..."]

DROPDOWN_NAME_MAP = {
    "pallet":             "pallet",
    "small load carrier": "small_load_carrier",
    "stillage":           "stillage",
    "forklift":           "forklift",
    "pallet truck":       "pallet truck",
}

DEFAULT_DIMENSIONS = {
    "pallet":             {"length": 1.200, "width": 0.800, "height": 0.144},
    "small load carrier": {"length": 0.400, "width": 0.300, "height": 0.147},
    "stillage":           {"length": 1.200, "width": 0.800, "height": 0.970},
    "forklift":           {"length": 2.800, "width": 1.300, "height": 2.150},
    "pallet truck":       {"length": 1.800, "width": 0.550, "height": 1.200},
}


_YOLO_TO_COCO_NAME: dict[str, str] = {}

def _canonical_class_name(name: str) -> str:
    return _YOLO_TO_COCO_NAME.get(name, name)


def _class_dims_for(cls_name: str):
    """Return (L, W, H) tuple in metres for a known fixed-size class, or None."""
    if cls_name in ("pallet truck", "pallet_truck"):
        return None
    if _class_dims_range_for(cls_name) is not None:
        return None   # variable-size class — use range instead
    d = DEFAULT_DIMENSIONS.get(cls_name)
    if d is None:
        d = DEFAULT_DIMENSIONS.get(cls_name.replace("_", " "))
    if d is None:
        return None
    return (d["length"], d["width"], d["height"])

# Classes whose size is not fixed but bounded — ((L_min,L_max),(W_min,W_max), H)
_DIMS_RANGE: dict[str, tuple] = {
    "small_load_carrier": ((0.400, 0.600), (0.300, 0.400), (0.147, 0.280)),
}

def _class_dims_range_for(cls_name: str):
    """Return ((L_min,L_max),(W_min,W_max),H) for variable-size classes, else None."""
    return _DIMS_RANGE.get(cls_name)


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-size: 12px;
}
QPushButton {
    background-color: #3c3c3c;
    color: #d4d4d4;
    border: 1px solid #555555;
    padding: 5px 10px;
    border-radius: 4px;
    min-height: 22px;
}
QPushButton:hover  { background-color: #4c4c4c; border-color: #888; }
QPushButton:pressed { background-color: #2a2a2a; }
QListWidget {
    background-color: #252526;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    outline: none;
}
QListWidget::item:selected { background-color: #094771; color: #ffffff; }
QListWidget::item:hover    { background-color: #2a2d2e; }
QLineEdit {
    background-color: #3c3c3c;
    color: #d4d4d4;
    border: 1px solid #555555;
    border-radius: 3px;
    padding: 2px 4px;
}
QLineEdit:focus { border-color: #007acc; }
QGroupBox {
    color: #9cdcfe;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 4px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
}
QLabel { color: #d4d4d4; }
QScrollArea { background-color: #1e1e1e; border: none; }
QScrollBar:vertical {
    background: #252526; width: 10px; border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #555; border-radius: 5px; min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #888; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QStatusBar { background-color: #007acc; color: #ffffff; font-weight: bold; }
QSplitter::handle { background-color: #3c3c3c; width: 2px; }
"""

# ---------------------------------------------------------------------------
# Icon helper
# ---------------------------------------------------------------------------
_ICON_MAP: dict[str, tuple[str, int]] = {
    "add":      ("list-add",                  QStyle.SP_FileDialogNewFolder),
    "remove":   ("list-remove",               QStyle.SP_TrashIcon),
    "save":     ("document-save",             QStyle.SP_DialogSaveButton),
    "cancel":   ("window-close",              QStyle.SP_DialogCancelButton),
    "edit":     ("document-edit",             QStyle.SP_FileDialogDetailedView),
    "users":    ("system-users",              QStyle.SP_ComputerIcon),
    "verify":   ("emblem-default",            QStyle.SP_DialogApplyButton),
    "prev":     ("go-previous",               QStyle.SP_ArrowLeft),
    "next":     ("go-next",                   QStyle.SP_ArrowRight),
    "random":   ("media-playlist-shuffle",    QStyle.SP_MediaSkipForward),
    "merge":    ("edit-copy",                 QStyle.SP_CommandLink),
    "open":     ("document-open",             QStyle.SP_DialogOpenButton),
    "generate": ("system-run",                QStyle.SP_MediaPlay),
    "masks":    ("image-edit",                QStyle.SP_FileDialogContentsView),
    "browse":   ("folder-open",               QStyle.SP_DirOpenIcon),
    "delete":   ("edit-delete",               QStyle.SP_TrashIcon),
    "new":      ("document-new",              QStyle.SP_FileIcon),
    "load":     ("document-open",             QStyle.SP_DialogOpenButton),
}

def _icon(name: str) -> QIcon:
    theme_name, sp = _ICON_MAP.get(name, ("", QStyle.SP_CustomBase))
    if theme_name:
        ico = QIcon.fromTheme(theme_name)
        if not ico.isNull():
            return ico
    from PyQt5.QtWidgets import QApplication
    return QApplication.style().standardIcon(sp)


# ---------------------------------------------------------------------------
# Config persistence  (~/.3d_label_editor/config.json)
# ---------------------------------------------------------------------------
def _ensure_app_dirs():
    os.makedirs(APP_DIR, exist_ok=True)
    os.makedirs(PROJECTS_DIR, exist_ok=True)


def load_config() -> dict:
    _ensure_app_dirs()
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    # One-time migration from the old single-file config
    if not cfg and os.path.isfile(_OLD_CONFIG_PATH):
        try:
            with open(_OLD_CONFIG_PATH) as f:
                old = json.load(f)
            # Copy only global keys; project_users are migrated on first project load
            cfg = {k: v for k, v in old.items() if k in ("recent_projects", "last_users")}
            save_config(cfg)
        except Exception:
            pass
    cfg.setdefault("recent_projects", [])
    return cfg


def save_config(cfg: dict):
    """Merge cfg into the on-disk config so unrelated keys are never wiped."""
    _ensure_app_dirs()
    try:
        with open(CONFIG_PATH) as f:
            on_disk = json.load(f)
    except Exception:
        on_disk = {}
    on_disk.update(cfg)
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(on_disk, f, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Per-project file helpers  (~/.3d_label_editor/projects/<id>.json)
# ---------------------------------------------------------------------------
def _project_id(root: str) -> str:
    """Stable filesystem-safe ID derived from the dataset root path."""
    import hashlib as _hl
    h = _hl.md5(root.encode()).hexdigest()[:10]
    safe = re.sub(r"[^\w]", "_", os.path.basename(root.rstrip("/").rstrip("\\")))[:40]
    return f"{h}_{safe}"


def _project_path(root: str) -> str:
    _ensure_app_dirs()
    return os.path.join(PROJECTS_DIR, _project_id(root) + ".json")


def _load_project_file(root: str) -> dict | None:
    p = _project_path(root)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _save_project_file(root: str, data: dict):
    p = _project_path(root)
    _ensure_app_dirs()
    try:
        # Merge with existing file to avoid clobbering unrelated keys
        existing = {}
        if os.path.isfile(p):
            with open(p) as f:
                existing = json.load(f)
        existing.update(data)
        with open(p, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception:
        pass


def discover_dataset_paths(root: str) -> dict:
    """Auto-discover all dataset folder paths from a root directory.

    Expected layout (produced by migrate_dataset.py):
      root/
      ├── dataset_info.json
      ├── calib/                 → camera_params_dir
      ├── annotations/           → annotations_dir  (must contain categories.json)
      ├── images/                → rgb_dir          (falls back to color/)
      ├── depth_files/           → depth_dir
      └── point_clouds/          → pcd_dir          (falls back to point_cloud/)
    """
    result: dict = {"dataset_root": root, "name": os.path.basename(root),
                    "pcd_dir": "", "rgb_dir": "", "depth_dir": "",
                    "camera_params_dir": "", "annotations_dir": "", "labels_dir": ""}

    for key, candidates in [
        ("pcd_dir",           ["point_clouds", "point_cloud"]),
        ("rgb_dir",           ["images", "color"]),
        ("depth_dir",         ["depth_files", "depth"]),
        ("camera_params_dir", ["calib"]),
        ("annotations_dir",   ["annotations"]),
    ]:
        for name in candidates:
            d = os.path.join(root, name)
            if os.path.isdir(d):
                # For annotations_dir require categories.json
                if key == "annotations_dir" and not os.path.isfile(
                        os.path.join(d, "categories.json")):
                    continue
                result[key] = d
                break
    return result


def push_recent_project(cfg: dict, project: dict):
    """Insert project dict at the front of recent_projects (dedup by root or pcd_dir)."""
    def _key(p):
        return p.get("dataset_root") or p.get("pcd_dir", "")
    k = _key(project)
    recent = [p for p in cfg.get("recent_projects", []) if _key(p) != k]
    recent.insert(0, project)
    cfg["recent_projects"] = recent[:MAX_RECENT]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Extrinsic Rx @ Ry @ Rz rotation matrix (degrees). Y is up/yaw axis."""
    rx, ry, rz = np.deg2rad(rx_deg), np.deg2rad(ry_deg), np.deg2rad(rz_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,  0,   0 ], [0,  cx, -sx], [0,  sx,  cx]])
    Ry = np.array([[cy, 0, -sy ], [0,  1,   0 ], [ sy, 0,  cy]])
    Rz = np.array([[cz, -sz, 0 ], [sz, cz,  0 ], [0,   0,   1]])
    return Rx @ Ry @ Rz


def build_pv_bbox(obj) -> tuple[pv.PolyData, tuple]:
    cx, cy, cz = obj["centroid"]["x"], obj["centroid"]["y"], obj["centroid"]["z"]
    center = np.array([cx, cy, cz])
    L = obj["dimensions"]["length"]
    W = obj["dimensions"]["width"]
    H = obj["dimensions"]["height"]

    # Convention: length (L) = long side → local X; width (W) = short side → local Z.
    # yaw aligns local X with the object's long axis.
    local_corners = np.array([
        [-L/2, -H/2, -W/2], [ L/2, -H/2, -W/2],
        [ L/2,  H/2, -W/2], [-L/2,  H/2, -W/2],
        [-L/2, -H/2,  W/2], [ L/2, -H/2,  W/2],
        [ L/2,  H/2,  W/2], [-L/2,  H/2,  W/2],
    ])

    rots = obj.get("rotations", {})
    R = _rotation_matrix(rots.get("x", 0.0), rots.get("y", 0.0), rots.get("z", 0.0))
    bbox_points = (R @ local_corners.T).T + center

    lines_conn = []
    for a, b in BBOX_LINES:
        lines_conn.extend([2, a, b])

    mesh = pv.PolyData()
    mesh.points = bbox_points
    mesh.lines = np.array(lines_conn)

    color = CLASS_COLORS.get(obj["name"].replace(" ", "_"), DEFAULT_COLOR)
    return mesh, color



_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def _find_rgb_image(rgb_dir: str, frame_id: str) -> str | None:
    # Fast path: exact name match with common suffixes
    for suffix in ("", "_color"):
        for ext in _IMAGE_EXTS:
            path = os.path.join(rgb_dir, f"{frame_id}{suffix}{ext}")
            if os.path.exists(path):
                return path

    # Fallback: match by numeric ID only (handles different naming conventions,
    # e.g. pointcloud_000 <-> rgb_000, frame_0019 <-> image_19)
    numbers = re.findall(r"\d+", frame_id)
    if not numbers:
        return None
    target = int(numbers[-1])  # use last number, compare as int (ignores zero-padding)

    for fname in sorted(os.listdir(rgb_dir)):
        if os.path.splitext(fname)[1].lower() not in _IMAGE_EXTS:
            continue
        file_nums = re.findall(r"\d+", fname)
        if file_nums and int(file_nums[-1]) == target:
            return os.path.join(rgb_dir, fname)

    return None


def _detect_and_blur_faces(img_bgr: np.ndarray) -> tuple:
    """Detect faces with Haar cascades and apply Gaussian blur. Returns (result_bgr, n_faces)."""
    gray = cv2.equalizeHist(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    cascade_names = [
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
    ]
    raw_boxes = []
    for name in cascade_names:
        path = os.path.join(cv2.data.haarcascades, name)
        cc = cv2.CascadeClassifier(path)
        dets = cc.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4,
                                   minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(dets):
            raw_boxes.extend(dets.tolist())

    if not raw_boxes:
        return img_bgr, 0

    # NMS to suppress heavily overlapping boxes from the two cascades
    rects = np.array(raw_boxes, dtype=np.float32)
    x1 = rects[:, 0];  y1 = rects[:, 1]
    x2 = x1 + rects[:, 2];  y2 = y1 + rects[:, 3]
    scores = np.ones(len(rects), dtype=np.float32)
    keep_indices = cv2.dnn.NMSBoxes(
        raw_boxes, scores.tolist(), score_threshold=0.0, nms_threshold=0.4
    )
    if len(keep_indices):
        keep_indices = keep_indices.flatten()
        kept = [raw_boxes[i] for i in keep_indices]
    else:
        kept = raw_boxes

    out = img_bgr.copy()
    img_h, img_w = out.shape[:2]
    for (x, y, bw, bh) in kept:
        pad = int(0.15 * max(bw, bh))
        x1c = max(0, x - pad);  y1c = max(0, y - pad)
        x2c = min(img_w, x + bw + pad);  y2c = min(img_h, y + bh + pad)
        k = max(51, (bh // 5) | 1)  # odd kernel, proportional to face height
        out[y1c:y2c, x1c:x2c] = cv2.GaussianBlur(out[y1c:y2c, x1c:x2c], (k, k), 0)

    return out, len(kept)


def _seg_class_id(name: str, seg_labels_dir: str) -> int:
    """Return (and persist) the integer class-ID for *name* in seg_labels/classes.txt."""
    classes_path = os.path.join(seg_labels_dir, "classes.txt")
    classes: list[str] = []
    if os.path.exists(classes_path):
        with open(classes_path) as _f:
            classes = [l.strip() for l in _f if l.strip()]
    if name not in classes:
        classes.append(name)
        with open(classes_path, "w") as _f:
            _f.write("\n".join(classes) + "\n")
    return classes.index(name)


_SEG_PALETTE = [
    (0, 200, 255), (255, 100, 0), (0, 255, 100),
    (200, 0, 255), (255, 200, 0), (0, 100, 255),
]


def _build_seg_overlay(rgb_arr: np.ndarray, txt_path: str) -> np.ndarray:
    """Return a copy of rgb_arr with YOLO-polygon masks blended in from txt_path."""
    import cv2 as _cv2_ov
    h, w = rgb_arr.shape[:2]
    overlay = rgb_arr.copy()
    try:
        with open(txt_path) as _f:
            lines = [l.strip() for l in _f if l.strip()]
        for mi, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 7:
                continue
            pts = np.array([float(v) for v in parts[1:]], dtype=np.float64).reshape(-1, 2)
            pts_px = (pts * np.array([w, h])).astype(np.int32)
            m = np.zeros((h, w), dtype=np.uint8)
            _cv2_ov.fillPoly(m, [pts_px], 255)
            mask_bool = m > 0
            c = _SEG_PALETTE[mi % len(_SEG_PALETTE)]
            overlay[mask_bool] = np.clip(
                overlay[mask_bool].astype(np.int32) // 2 +
                np.array([c[0], c[1], c[2]]) // 2,
                0, 255,
            ).astype(np.uint8)
    except Exception:
        pass
    return overlay


# Spinbox config: (min, max, step, decimals) per field section
_SPIN_CFG = {
    "centroid":   (-50.0, 50.0,  0.01, 3),
    "dimensions": ( 0.001, 10.0, 0.01, 3),
    "rotations":  (-360.0, 360.0, 1.0, 1),
}

# ---------------------------------------------------------------------------
# User-management helpers
# ---------------------------------------------------------------------------
def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def _migrate_users_format(stored: dict) -> list:
    """Normalise any legacy users structure to a plain list of user dicts."""
    if isinstance(stored, list):
        return stored
    if "users" in stored:
        return stored["users"]
    users_list = []
    admin = stored.get("admin") or {}
    if admin.get("username"):
        users_list.append({"username": admin["username"], "role": "admin",
                           "password_hash": admin.get("password_hash", "")})
    for ann in stored.get("annotators", []):
        if ann.get("username"):
            users_list.append({"username": ann["username"], "role": "annotator",
                               "password_hash": ann.get("password_hash", "")})
    return users_list


def _load_project_users(root: str) -> list | None:
    """Return list of user dicts for this project, or None if none configured."""
    data = _load_project_file(root)
    if data is not None and "users" in data:
        return _migrate_users_format(data["users"])

    # Migration 1: old ~/.3d_label_editor.json project_users key
    try:
        with open(_OLD_CONFIG_PATH) as f:
            old_cfg = json.load(f)
        stored = old_cfg.get("project_users", {}).get(root)
        if stored is not None:
            users = _migrate_users_format(stored)
            _save_project_users(root, users)
            return users
    except Exception:
        pass

    # Migration 2: legacy project_users.json in dataset folder
    legacy = os.path.join(root, "project_users.json")
    if os.path.isfile(legacy):
        try:
            with open(legacy) as f:
                raw = json.load(f)
            users = _migrate_users_format(raw)
            _save_project_users(root, users)
            os.remove(legacy)
            return users
        except Exception:
            pass

    return None


def _save_project_users(root: str, users: list | dict):
    """Persist users for a project into its project file."""
    if isinstance(users, dict):
        users = _migrate_users_format(users)
    _save_project_file(root, {"dataset_root": root, "users": users})


class LoginDialog(QDialog):
    """Sign-in dialog: select user from dropdown, enter password."""

    def __init__(self, users: list, dataset_root: str = "", default_user: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sign in")
        self.setModal(True)
        self.setMinimumWidth(340)
        self.logged_in_user: dict | None = None
        self._users = users          # plain list of user dicts
        self._dataset_root = dataset_root

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)

        title = QLabel("Sign in to project")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #9cdcfe;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(8)

        self._user_combo = QComboBox()
        self._all_users = [u["username"] for u in users if u.get("username")]
        self._user_combo.addItems(self._all_users)
        if default_user and default_user in self._all_users:
            self._user_combo.setCurrentIndex(self._all_users.index(default_user))
        self._user_combo.currentIndexChanged.connect(self._on_user_changed)
        form.addRow("User:", self._user_combo)

        self._pw_edit = QLineEdit()
        self._pw_edit.setEchoMode(QLineEdit.Password)
        self._pw_edit.returnPressed.connect(self._on_accept)
        form.addRow("Password:", self._pw_edit)
        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Sign in")
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._on_user_changed()

    def _user_record(self, username: str) -> dict | None:
        return next((u for u in self._users if u.get("username") == username), None)

    def _on_user_changed(self):
        username = self._user_combo.currentText()
        record = self._user_record(username)
        has_pw = bool(record and record.get("password_hash"))
        self._pw_edit.setPlaceholderText("Password" if has_pw else "Set a new password")
        self._pw_edit.clear()

    def _on_accept(self):
        username = self._user_combo.currentText()
        entered_pw = self._pw_edit.text()
        record = self._user_record(username)
        if record is None:
            QMessageBox.warning(self, "Login failed", "User not found.")
            return
        role = record.get("role", "annotator")

        if not record.get("password_hash"):
            if not entered_pw:
                QMessageBox.information(self, "Set password",
                                        "No password set yet. Enter a new password to continue.")
                self._pw_edit.setFocus()
                return
            from PyQt5.QtWidgets import QInputDialog, QLineEdit as _LE
            confirm, ok = QInputDialog.getText(self, "Confirm password",
                                               "Confirm your new password:", _LE.Password)
            if not ok:
                return
            if entered_pw != confirm:
                QMessageBox.warning(self, "Mismatch", "Passwords do not match.")
                self._pw_edit.clear()
                self._pw_edit.setFocus()
                return
            record["password_hash"] = _hash_pw(entered_pw)
            if self._dataset_root:
                _save_project_users(self._dataset_root, self._users)
        else:
            if _hash_pw(entered_pw) != record["password_hash"]:
                QMessageBox.warning(self, "Login failed", "Incorrect password.")
                self._pw_edit.clear()
                self._pw_edit.setFocus()
                return

        self.logged_in_user = {"username": username, "role": role}
        self.accept()


class _AddUserDialog(QDialog):
    """Small dialog to add or edit a user entry."""

    def __init__(self, username: str = "", role: str = "annotator", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add user" if not username else "Edit user")
        self.setModal(True)
        self.setMinimumWidth(320)

        form = QFormLayout(self)
        form.setContentsMargins(14, 14, 14, 14)
        form.setSpacing(8)

        self._name_edit = QLineEdit(username)
        self._name_edit.setPlaceholderText("Username")
        self._name_edit.setReadOnly(bool(username))   # can't rename existing users here
        form.addRow("Username:", self._name_edit)

        self._role_combo = QComboBox()
        self._role_combo.addItems(["annotator", "admin"])
        self._role_combo.setCurrentText(role)
        form.addRow("Role:", self._role_combo)

        self._pw_edit = QLineEdit()
        self._pw_edit.setEchoMode(QLineEdit.Password)
        self._pw_edit.setPlaceholderText("Leave blank — user sets on first login")
        form.addRow("Password:", self._pw_edit)

        self._pw_confirm = QLineEdit()
        self._pw_confirm.setEchoMode(QLineEdit.Password)
        self._pw_confirm.setPlaceholderText("Confirm password")
        form.addRow("Confirm:", self._pw_confirm)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def _on_accept(self):
        if not self._name_edit.text().strip():
            QMessageBox.warning(self, "Error", "Username cannot be empty.")
            return
        pw = self._pw_edit.text()
        if pw and pw != self._pw_confirm.text():
            QMessageBox.warning(self, "Mismatch", "Passwords do not match.")
            self._pw_confirm.clear()
            return
        self.accept()

    def get_values(self) -> tuple[str, str, str]:
        """Returns (username, role, password_or_empty)."""
        return (self._name_edit.text().strip(),
                self._role_combo.currentText(),
                self._pw_edit.text())


class ManageUsersDialog(QDialog):
    """Admin-only dialog to manage project users and their roles."""

    def __init__(self, dataset_root: str, users_doc: list | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Users")
        self.setModal(True)
        self.setMinimumWidth(420)
        self._root = dataset_root
        self._users = list(users_doc or [])   # working copy — plain list of user dicts

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(14, 14, 14, 14)

        self._list = QListWidget()
        self._list.setMinimumHeight(150)
        self._refresh_list()
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        add_btn = QPushButton(_icon("add"), "Add user")
        add_btn.clicked.connect(self._on_add)
        rem_btn = QPushButton(_icon("remove"), "Remove selected")
        rem_btn.clicked.connect(self._on_remove)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(rem_btn)
        btn_row.addStretch()
        save_btn = QPushButton(_icon("save"), "Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._on_save)
        cancel_btn = QPushButton(_icon("cancel"), "Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _refresh_list(self):
        self._list.clear()
        for u in self._users:
            name = u.get("username", "")
            role = u.get("role", "annotator")
            pw_status = "password set" if u.get("password_hash") else "no password yet"
            self._list.addItem(f"{name}  [{role}]  ({pw_status})")

    def _on_add(self):
        dlg = _AddUserDialog(parent=self)
        dlg.setStyleSheet(self.styleSheet())
        if dlg.exec_() != QDialog.Accepted:
            return
        username, role, pw = dlg.get_values()
        if any(u["username"] == username for u in self._users):
            QMessageBox.warning(self, "Duplicate", f"User '{username}' already exists.")
            return
        self._users.append({"username": username, "role": role,
                            "password_hash": _hash_pw(pw) if pw else ""})
        self._refresh_list()

    def _on_remove(self):
        row = self._list.currentRow()
        if row < 0:
            return
        name = self._users[row]["username"]
        reply = QMessageBox.question(self, "Remove user",
                                     f"Remove user '{name}'?",
                                     QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        if reply == QMessageBox.Yes:
            self._users.pop(row)
            self._refresh_list()

    def _on_save(self):
        if not any(u.get("role") == "admin" for u in self._users):
            QMessageBox.warning(self, "Error", "At least one admin user is required.")
            return
        _save_project_users(self._root, self._users)
        self.accept()


class _FlagCommentDialog(QDialog):
    """Shows the flag comment for a frame; lets admin/annotator resolve it."""

    def __init__(self, meta: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🚩 Flagged frame")
        self.setModal(True)
        self.setMinimumWidth(420)
        self.resolved = False

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        who = meta.get("flagged_by", "unknown")
        when = meta.get("flagged_at", "")
        info = QLabel(f"<b>Flagged by:</b> {who}   <b>at:</b> {when}")
        info.setWordWrap(True)
        layout.addWidget(info)

        comment_box = QLabel(meta.get("flag_comment", "(no comment)"))
        comment_box.setWordWrap(True)
        comment_box.setStyleSheet(
            "background:#2d2d2d; color:#e0e0e0; padding:10px;"
            "border-radius:4px; font-size:13px;"
        )
        layout.addWidget(comment_box)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        accept_btn = QPushButton("Accept")
        accept_btn.setToolTip("Acknowledge and close")
        accept_btn.clicked.connect(self.accept)
        resolve_btn = QPushButton("✓ Resolve")
        resolve_btn.setToolTip("Mark issue as corrected and remove the flag")
        resolve_btn.clicked.connect(self._on_resolve)
        btn_row.addWidget(accept_btn)
        btn_row.addWidget(resolve_btn)
        layout.addLayout(btn_row)

    def _on_resolve(self):
        self.resolved = True
        self.accept()


class _BootstrapAdminDialog(QDialog):
    """Shown when a project has no admin yet.
    Lets the user pick a previously used admin or create a new one."""

    def __init__(self, known_admins: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set up admin account")
        self.setModal(True)
        self.setMinimumWidth(380)
        self._known = known_admins
        self.chosen_admin: dict | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        # ── Known admins list ──────────────────────────────────────────
        self._list = None
        if known_admins:
            layout.addWidget(QLabel("Select a previous admin account:"))
            self._list = QListWidget()
            self._list.setMaximumHeight(120)
            for a in known_admins:
                self._list.addItem(a["username"])
            self._list.setCurrentRow(0)
            self._list.itemClicked.connect(self._on_list_clicked)
            layout.addWidget(self._list)

            sep = QLabel("— or create a new admin below —")
            sep.setAlignment(Qt.AlignCenter)
            sep.setStyleSheet("color: #666; font-size: 11px;")
            layout.addWidget(sep)

        # ── New admin fields ───────────────────────────────────────────
        new_box = QGroupBox("New admin")
        new_form = QFormLayout(new_box)
        self._new_name = QLineEdit()
        self._new_name.setPlaceholderText("Username")
        self._new_name.textChanged.connect(self._on_new_name_changed)
        self._new_pw = QLineEdit()
        self._new_pw.setEchoMode(QLineEdit.Password)
        self._new_pw.setPlaceholderText("Password")
        new_form.addRow("Username:", self._new_name)
        new_form.addRow("Password:", self._new_pw)
        layout.addWidget(new_box)

        # ── Buttons ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._ok_btn = QPushButton(_icon("save"), "OK")
        self._ok_btn.setDefault(True)
        self._ok_btn.clicked.connect(self._on_accept)
        skip_btn = QPushButton(_icon("cancel"), "Skip")
        skip_btn.setToolTip("Load project without user management")
        skip_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._ok_btn)
        btn_row.addWidget(skip_btn)
        layout.addLayout(btn_row)

    def _on_list_clicked(self, _item):
        self._new_name.clear()
        self._new_pw.clear()

    def _on_new_name_changed(self, text):
        if text.strip() and self._list is not None:
            self._list.clearSelection()

    def _on_accept(self):
        new_name = self._new_name.text().strip()
        if new_name:
            pw = self._new_pw.text()
            if not pw:
                QMessageBox.warning(self, "Error", "Enter a password for the new admin.")
                return
            self.chosen_admin = {"username": new_name, "password_hash": _hash_pw(pw)}
        elif self._list is not None and self._list.currentItem() and self._list.currentRow() >= 0:
            row = self._list.currentRow()
            if self._list.selectedItems():
                self.chosen_admin = self._known[row]
            else:
                QMessageBox.warning(self, "Nothing selected",
                                    "Select an existing admin or fill in the new admin fields.")
                return
        else:
            QMessageBox.warning(self, "Nothing selected",
                                "Select an existing admin or fill in the new admin fields.")
            return
        self.accept()


# ---------------------------------------------------------------------------
# Project editor dialog — pick a dataset root folder; paths are auto-discovered
# ---------------------------------------------------------------------------
class ProjectEditorDialog(QDialog):

    def __init__(self, dataset_root: str = "", name: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Dataset")
        self.setModal(True)
        self.setMinimumWidth(560)
        self._dataset_root = dataset_root
        self._name         = name
        self._discovered: dict = {}
        self._build_ui()
        if dataset_root:
            self._refresh_discovery(dataset_root)

    def _build_ui(self):
        ROW_H = 30
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setSpacing(6)
        self._name_edit = QLineEdit(self._name)
        self._name_edit.setPlaceholderText("e.g. Realsense D435 — warehouse run 1")
        self._name_edit.setFixedHeight(ROW_H)
        form.addRow("Project name:", self._name_edit)
        layout.addLayout(form)

        row = QHBoxLayout()
        self._root_edit = QLineEdit(self._dataset_root)
        self._root_edit.setPlaceholderText("Dataset root folder…")
        self._root_edit.setReadOnly(True)
        self._root_edit.setFixedHeight(ROW_H)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedHeight(ROW_H)
        browse_btn.clicked.connect(self._browse_root)
        row.addWidget(self._root_edit)
        row.addWidget(browse_btn)
        layout.addLayout(row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #888; font-size: 11px; font-family: monospace;")
        layout.addWidget(self._status_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn  = QPushButton("Cancel")
        self._ok_btn = QPushButton("Load")
        cancel_btn.setFixedHeight(ROW_H)
        self._ok_btn.setFixedHeight(ROW_H)
        cancel_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogCancelButton))
        self._ok_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogOkButton))
        self._ok_btn.setEnabled(bool(self._dataset_root))
        cancel_btn.clicked.connect(self.reject)
        self._ok_btn.clicked.connect(self._on_confirm)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self._ok_btn)
        layout.addLayout(btn_row)

    def _browse_root(self):
        start = self._root_edit.text() or os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(self, "Select dataset root folder", start)
        if path:
            self._root_edit.setText(path)
            self._refresh_discovery(path)

    def _refresh_discovery(self, root: str):
        self._discovered = discover_dataset_paths(root)
        lines = []
        checks = [
            ("point_clouds",  "pcd_dir",           "Point clouds"),
            ("images",        "rgb_dir",            "RGB images"),
            ("depth_files",   "depth_dir",          "Depth files"),
            ("calib",         "camera_params_dir",  "Calibration"),
            ("annotations",   "annotations_dir",    "Annotations"),
        ]
        for _, key, label in checks:
            val = self._discovered.get(key, "")
            icon = "✓" if val else "✗"
            lines.append(f"  {icon}  {label}: {os.path.basename(val) if val else 'not found'}")
        self._status_label.setText("\n".join(lines))
        self._ok_btn.setEnabled(bool(self._discovered.get("pcd_dir")))
        # Auto-fill name from folder name if blank
        if not self._name_edit.text().strip():
            self._name_edit.setText(os.path.basename(root))

    def _on_confirm(self):
        root = self._root_edit.text().strip()
        if not root or not self._discovered.get("pcd_dir"):
            QMessageBox.warning(self, "No point clouds found",
                                "The selected folder has no point_clouds/ subdirectory.")
            return
        self.accept()

    def get_project(self) -> dict:
        name = self._name_edit.text().strip()
        proj = dict(self._discovered)
        proj["name"] = name or os.path.basename(self._root_edit.text().strip())
        return proj


# ---------------------------------------------------------------------------
# Load Project dialog
# ---------------------------------------------------------------------------
class LoadProjectDialog(QDialog):

    def __init__(self, recent_projects: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Project")
        self.setModal(True)
        self.setMinimumWidth(540)
        self._recent = list(recent_projects)
        self._selected: dict | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Recent projects:"))
        self._list = QListWidget()
        self._list.setMinimumHeight(150)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.itemDoubleClicked.connect(lambda _: self._on_accept())
        layout.addWidget(self._list)

        # Action buttons row — equal fixed width so they all match
        action_row = QHBoxLayout()
        new_btn = QPushButton("+ New")
        new_btn.clicked.connect(self._on_new)
        self._edit_btn   = QPushButton("Edit")
        self._edit_btn.clicked.connect(self._on_edit)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._on_delete)
        for b in (new_btn, self._edit_btn, self._delete_btn):
            b.setFixedWidth(80)
            action_row.addWidget(b)
        action_row.addStretch()
        layout.addLayout(action_row)

        self._detail = QLabel("")
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._detail)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Load")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Populate list only after _detail exists
        self._refresh_list()
        self._update_button_state()

    def _refresh_list(self, keep_row: int = 0):
        self._list.blockSignals(True)
        self._list.clear()
        for p in self._recent:
            name = p.get("name", "").strip()
            fallback = p.get("dataset_root") or p.get("pcd_dir", "")
            label = name if name else os.path.basename(fallback)
            self._list.addItem(f"  {label}")
        if not self._recent:
            self._list.addItem("No recent projects — click '+ New' to add one")
        self._list.blockSignals(False)
        row = min(keep_row, max(0, len(self._recent) - 1))
        self._list.setCurrentRow(row)
        self._on_row_changed(row)
        self._update_button_state()

    def _update_button_state(self):
        has_selection = 0 <= self._list.currentRow() < len(self._recent)
        self._edit_btn.setEnabled(has_selection)
        self._delete_btn.setEnabled(has_selection)

    def _on_row_changed(self, row: int):
        if 0 <= row < len(self._recent):
            p = self._recent[row]
            root = p.get("dataset_root") or p.get("pcd_dir", "")
            ann  = p.get("annotations_dir", "")
            ann_tag = f"\nAnnotations: {os.path.basename(ann)}" if ann else ""
            self._detail.setText(f"Root: {root}{ann_tag}")
        else:
            self._detail.setText("")
        self._update_button_state()

    def _unique_name(self, name: str, skip_index: int = -1) -> str:
        """Return name, appending v2/v3/… if it conflicts with an existing entry."""
        existing = {p.get("name", "").strip()
                    for i, p in enumerate(self._recent) if i != skip_index}
        if name not in existing:
            return name
        import re as _re
        base = _re.sub(r'\s+v\d+$', '', name)
        i = 2
        while True:
            candidate = f"{base} v{i}"
            if candidate not in existing:
                return candidate
            i += 1

    def _on_new(self):
        dlg = ProjectEditorDialog(parent=self)
        dlg.setStyleSheet(self.styleSheet())
        if dlg.exec_() != QDialog.Accepted:
            return
        entry = dlg.get_project()
        # Remove any existing entry with same root path, then ensure unique name
        self._recent = [p for p in self._recent
                        if (p.get("dataset_root") or p.get("pcd_dir", "")) !=
                           (entry.get("dataset_root") or entry.get("pcd_dir", ""))]
        entry["name"] = self._unique_name(entry.get("name", "").strip())
        self._recent.insert(0, entry)
        self._recent = self._recent[:MAX_RECENT]
        self._refresh_list(keep_row=0)

    def _on_edit(self):
        row = self._list.currentRow()
        if row < 0 or row >= len(self._recent):
            return
        p = self._recent[row]
        root = p.get("dataset_root") or os.path.dirname(p.get("pcd_dir", ""))
        dlg = ProjectEditorDialog(dataset_root=root, name=p.get("name", ""), parent=self)
        dlg.setStyleSheet(self.styleSheet())
        if dlg.exec_() != QDialog.Accepted:
            return
        entry = dlg.get_project()
        entry["name"] = self._unique_name(entry.get("name", "").strip(), skip_index=row)
        self._recent[row] = entry
        self._refresh_list(keep_row=row)

    def _on_delete(self):
        row = self._list.currentRow()
        if row < 0 or row >= len(self._recent):
            return
        p = self._recent[row]
        name = p.get("name") or os.path.basename(p.get("dataset_root") or p.get("pcd_dir", ""))
        reply = QMessageBox.question(self, "Delete project",
                                     f"Remove '{name}' from recent projects?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self._recent.pop(row)
        self._refresh_list(keep_row=max(0, row - 1))

    def _on_accept(self):
        row = self._list.currentRow()
        if row < 0 or row >= len(self._recent):
            return
        self._selected = self._recent[row]
        self.accept()

    def get_project(self) -> dict | None:
        return self._selected

    def get_updated_recent(self) -> list:
        return self._recent


# ---------------------------------------------------------------------------
# Per-object editor widget
# ---------------------------------------------------------------------------
class ObjectFieldWidget(QGroupBox):

    def __init__(self, index: int, obj: dict, on_change=None, parent=None):
        super().__init__(f"[{index}]  {obj['name']}", parent)
        self._obj = obj
        self.fields: dict[tuple, QDoubleSpinBox] = {}

        CELL_H = 24

        grid = QGridLayout()
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(3)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 1)

        def _spin(section, key):
            lo, hi, step, dec = _SPIN_CFG[section]
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(dec)
            s.setFixedHeight(CELL_H)
            s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            s.setValue(float(obj[section][key]))
            if on_change:
                s.valueChanged.connect(lambda _: on_change())
            return s

        def _btn(text, slot):
            b = QPushButton(text)
            b.setFixedHeight(CELL_H)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setStyleSheet("min-height: 0; padding: 0;")
            b.clicked.connect(slot)
            if on_change:
                b.clicked.connect(on_change)
            return b

        for row_idx, ((section, key), label_text) in enumerate(zip(FIELD_KEYS, FIELD_LABELS)):
            spin = _spin(section, key)
            self.fields[(section, key)] = spin

            grid.addWidget(QLabel(label_text), row_idx, 0)
            grid.addWidget(spin,               row_idx, 1)

            if (section, key) == ("dimensions", "length"):
                grid.addWidget(_btn("L↔W", self.swap_lw), row_idx, 2)
                grid.addWidget(_btn("L↔H", self.swap_lh), row_idx, 3)
            elif (section, key) == ("dimensions", "width"):
                grid.addWidget(_btn("W↔H", self.swap_wh), row_idx, 2)
            elif (section, key) == ("rotations", "y"):
                grid.addWidget(_btn("+90°", self.rotate_90), row_idx, 2)
                grid.addWidget(_btn("-90°", self.rotate_neg90), row_idx, 3)

        self.setLayout(grid)

    def get_values(self) -> dict | None:
        result = copy.deepcopy(self._obj)
        for (section, key), spin in self.fields.items():
            result[section][key] = spin.value()
        return result

    def clear_highlights(self):
        pass

    def swap_wh(self):
        tmp = self.fields[("dimensions", "width")].value()
        self.fields[("dimensions", "width")].setValue(self.fields[("dimensions", "height")].value())
        self.fields[("dimensions", "height")].setValue(tmp)

    def swap_lw(self):
        tmp = self.fields[("dimensions", "length")].value()
        self.fields[("dimensions", "length")].setValue(self.fields[("dimensions", "width")].value())
        self.fields[("dimensions", "width")].setValue(tmp)

    def swap_lh(self):
        tmp = self.fields[("dimensions", "length")].value()
        self.fields[("dimensions", "length")].setValue(self.fields[("dimensions", "height")].value())
        self.fields[("dimensions", "height")].setValue(tmp)

    def rotate_90(self):
        spin = self.fields[("rotations", "y")]
        new_val = (spin.value() + 90.0) % 360.0
        if new_val > 180.0:
            new_val -= 360.0
        spin.setValue(new_val)

    def rotate_neg90(self):
        spin = self.fields[("rotations", "y")]
        new_val = (spin.value() - 90.0) % 360.0
        if new_val > 180.0:
            new_val -= 360.0
        spin.setValue(new_val)


# ---------------------------------------------------------------------------
# Add-object dialog
# ---------------------------------------------------------------------------
class AddObjectDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Object")
        self.setModal(True)
        self.setMinimumWidth(320)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        form = QFormLayout()
        form.setSpacing(6)

        self._combo = QComboBox()
        self._combo.addItems(DROPDOWN_OPTIONS)
        self._combo.currentTextChanged.connect(self._on_type_changed)
        form.addRow("Object type:", self._combo)

        self._custom_name = QLineEdit()
        self._custom_name.setPlaceholderText("Enter custom name…")
        self._custom_name.hide()
        form.addRow("Custom name:", self._custom_name)

        self._f_length = QLineEdit("0.000")
        self._f_width  = QLineEdit("0.000")
        self._f_height = QLineEdit("0.000")
        form.addRow("Length (m):", self._f_length)
        form.addRow("Width  (m):", self._f_width)
        form.addRow("Height (m):", self._f_height)

        self._f_cx = QLineEdit("0.000")
        self._f_cy = QLineEdit("0.000")
        self._f_cz = QLineEdit("0.000")
        form.addRow("Centroid X:", self._f_cx)
        form.addRow("Centroid Y:", self._f_cy)
        form.addRow("Centroid Z:", self._f_cz)

        self._f_rx = QLineEdit("0.000")
        self._f_ry = QLineEdit("0.000")
        self._f_rz = QLineEdit("0.000")
        form.addRow("Rotation X:", self._f_rx)
        form.addRow("Rotation Y:", self._f_ry)
        form.addRow("Rotation Z:", self._f_rz)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Add")
        buttons.accepted.connect(self._on_add)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._on_type_changed(DROPDOWN_OPTIONS[0])

    def _on_type_changed(self, text: str):
        is_custom = (text == "Custom...")
        self._custom_name.setVisible(is_custom)
        if is_custom:
            self._f_length.clear()
            self._f_width.clear()
            self._f_height.clear()
        else:
            dims = DEFAULT_DIMENSIONS.get(text, {})
            self._f_length.setText(f"{dims.get('length', 0.0):.3f}")
            self._f_width.setText(f"{dims.get('width',  0.0):.3f}")
            self._f_height.setText(f"{dims.get('height', 0.0):.3f}")

    def _on_add(self):
        fields = [
            self._f_length, self._f_width, self._f_height,
            self._f_cx, self._f_cy, self._f_cz,
            self._f_rx, self._f_ry, self._f_rz,
        ]
        invalid = False
        for edit in fields:
            try:
                float(edit.text().strip())
                edit.setStyleSheet("")
            except ValueError:
                edit.setStyleSheet("background-color: #7a2020; color: #ffcccc;")
                invalid = True

        if self._combo.currentText() == "Custom..." and not self._custom_name.text().strip():
            self._custom_name.setStyleSheet("background-color: #7a2020; color: #ffcccc;")
            invalid = True

        if invalid:
            QMessageBox.warning(self, "Invalid values", "Please fix the highlighted fields.")
            return
        self.accept()

    def get_object(self) -> dict:
        label = self._combo.currentText()
        if label == "Custom...":
            name = self._custom_name.text().strip()
        else:
            name = DROPDOWN_NAME_MAP.get(label, label)

        return {
            "name": name,
            "centroid": {
                "x": float(self._f_cx.text()),
                "y": float(self._f_cy.text()),
                "z": float(self._f_cz.text()),
            },
            "dimensions": {
                "length": float(self._f_length.text()),
                "width":  float(self._f_width.text()),
                "height": float(self._f_height.text()),
            },
            "rotations": {
                "x": float(self._f_rx.text()),
                "y": float(self._f_ry.text()),
                "z": float(self._f_rz.text()),
            },
        }


# ---------------------------------------------------------------------------
# Background worker for the batch processing loop in _on_auto_bbox
# ---------------------------------------------------------------------------
class _BatchWorker(QThread):
    frame_started   = pyqtSignal(int, str)   # (frame_idx, frame_id)
    object_progress = pyqtSignal(int, int)   # (det_idx, total)
    batch_done      = pyqtSignal(list, int)  # (current_frame_objects, total_saved)
    batch_cancelled = pyqtSignal(list)       # (written_fids)

    def __init__(self, run_fn):
        super().__init__()
        self._run_fn = run_fn
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        self._run_fn(self)


# ---------------------------------------------------------------------------
# Dual progress dialog used by _on_auto_bbox
# ---------------------------------------------------------------------------
class _AutoBBoxProgressDialog(QDialog):
    def __init__(self, n_frames: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Autonomous 3D BB Generation")
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        self.setFixedWidth(440)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._frame_label = QLabel("Starting…")
        layout.addWidget(self._frame_label)
        self._frame_bar = QProgressBar()
        self._frame_bar.setRange(0, n_frames)
        self._frame_bar.setValue(0)
        layout.addWidget(self._frame_bar)

        self._obj_label = QLabel("Objects: —")
        layout.addWidget(self._obj_label)
        self._obj_bar = QProgressBar()
        self._obj_bar.setRange(0, 1)
        self._obj_bar.setValue(0)
        layout.addWidget(self._obj_bar)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedHeight(28)
        layout.addWidget(self._cancel_btn)

    def set_frame(self, idx: int, frame_id: str):
        total = self._frame_bar.maximum()
        self._frame_label.setText(f"Frame {idx + 1} / {total}:  {frame_id}")
        self._frame_bar.setValue(idx + 1)
        self._obj_bar.setValue(0)
        self._obj_bar.setRange(0, 1)
        self._obj_label.setText("Objects: detecting…")
        QApplication.processEvents()

    def set_object(self, idx: int, total: int):
        self._obj_bar.setRange(0, total)
        self._obj_bar.setValue(idx + 1)
        self._obj_label.setText(f"Object {idx + 1} / {total}")
        QApplication.processEvents()


# ---------------------------------------------------------------------------
# Git sync helpers
# ---------------------------------------------------------------------------

def _git_root() -> str | None:
    """Return the repo root that contains this file, or None if not in a git repo."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


class _GitRun:
    """Helper: run git commands rooted at a given directory."""
    def __init__(self, root: str):
        self.root = root

    def __call__(self, *args, timeout=30) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", self.root] + list(args),
            capture_output=True, text=True, timeout=timeout,
        )


class GitPullWorker(QThread):
    """Fetch and fast-forward pull on startup. Runs in a background thread."""
    done = pyqtSignal(str, bool)   # (message, is_error)

    def __init__(self, root: str, parent=None):
        super().__init__(parent)
        self._git = _GitRun(root)

    def run(self):
        try:
            if not self._git("remote").stdout.strip():
                return   # no remote → nothing to do, stay silent

            r = self._git("fetch", "--quiet", timeout=30)
            if r.returncode != 0:
                self.done.emit(f"Git fetch failed: {r.stderr.strip()}", True)
                return

            # Commits remote is ahead of us
            r = self._git("rev-list", "HEAD..@{u}", "--count")
            ahead = int(r.stdout.strip() or "0") if r.returncode == 0 else 0
            if ahead == 0:
                self.done.emit("Git: already up to date.", False)
                return

            # Check for local uncommitted changes that would block a pull
            local_dirty = bool(self._git("status", "--porcelain").stdout.strip())
            if local_dirty:
                self.done.emit(
                    f"Git: remote has {ahead} new commit(s) but you have uncommitted local "
                    "changes.\nCommit and push your work first, then pull manually.",
                    True,
                )
                return

            r = self._git("pull", "--ff-only", timeout=60)
            if r.returncode == 0:
                self.done.emit(f"Git: pulled {ahead} new commit(s).", False)
            elif "CONFLICT" in r.stdout + r.stderr:
                self.done.emit(
                    "Git pull produced conflicts.\n"
                    "Please resolve them manually in a terminal, then reopen the app.",
                    True,
                )
            else:
                self.done.emit(f"Git pull failed:\n{r.stderr.strip()}", True)

        except subprocess.TimeoutExpired:
            self.done.emit("Git sync timed out — check your network.", True)
        except Exception as e:
            self.done.emit(f"Git sync error: {e}", True)


class GitCommitPushWorker(QThread):
    """Stage all annotation changes, commit with a message, and push."""
    progress = pyqtSignal(str)
    done = pyqtSignal(bool, str)   # (success, message)

    def __init__(self, root: str, message: str, parent=None):
        super().__init__(parent)
        self._git = _GitRun(root)
        self._message = message

    def run(self):
        try:
            self.progress.emit("Staging changes…")
            # Stage everything under LOCO 3D/LOCO_3D (annotations + any other tracked files)
            r = self._git("add", "--", "LOCO 3D/LOCO_3D/")
            if r.returncode != 0:
                self.done.emit(False, f"git add failed:\n{r.stderr.strip()}")
                return

            self.progress.emit("Committing…")
            r = self._git("commit", "-m", self._message)
            if r.returncode != 0:
                if "nothing to commit" in r.stdout + r.stderr:
                    self.done.emit(True, "Nothing to commit — already up to date.")
                    return
                self.done.emit(False, f"git commit failed:\n{r.stderr.strip()}")
                return

            self.progress.emit("Pushing to remote…")
            r = self._git("push", timeout=90)
            if r.returncode == 0:
                self.done.emit(True, "Annotations committed and pushed successfully.")
            else:
                self.done.emit(False, f"git push failed:\n{r.stderr.strip()}")

        except subprocess.TimeoutExpired:
            self.done.emit(False, "Git operation timed out.")
        except Exception as e:
            self.done.emit(False, f"Git error: {e}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class LabelEditorWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.pcd_dir:            str | None = None
        self.labels_dir:         str | None = None
        self.rgb_dir:            str | None = None
        self.depth_dir:          str | None = None
        self.camera_params_dir:  str | None = None
        self.current_frame_id:   str | None = None
        self.current_label_data: dict | None = None
        self.current_objects:    list = []
        self._selected_obj_idx:  int = -1
        self._active_widget:     ObjectFieldWidget | None = None
        self._dirty = False
        self._cfg = load_config()
        self._orig_pixmap = None
        self._cur_fx = self._cur_fy = self._cur_cx = self._cur_cy = None
        self._dataset_axes: dict | None = None   # bbox_3d_axes from dataset_info.json
        self._camera_needs_reset = True          # set True on project load → orient on first render
        self._dataset_root: str = ""
        self._current_user: dict | None = None   # {"username": str, "role": "admin"|"annotator"}
        self._3d_bb_manually_touched = False     # True when user edits spinboxes manually
        self._block_manual_tracking = False      # True during programmatic field updates
        self._copied_pose: dict | None = None    # {"x": float, "z": float, "yaw": float}
        # COCO per-frame format support
        self._coco_db:            dict | None = None   # {"categories": [...]} when COCO mode active
        self._annotations_dir:    str  | None = None   # path to annotations/ directory
        self._coco_categories:    list = []             # [{id, name, ...}]
        self._coco_cat_id_to_name: dict = {}
        self._coco_name_to_cat_id: dict = {}
        self._coco_frame_cache:   dict = {}            # frame_id → loaded frame doc
        self._all_frame_files:    list = []            # full sorted list of pcd/ply filenames

        self._git_root: str | None = _git_root()
        self._git_pull_worker: GitPullWorker | None = None

        self.setWindowTitle("3D Label Editor")
        self.resize(1500, 900)
        self._build_ui()
        self.statusBar().showMessage("Ready")
        # Defer until after the main window is shown so modal dialogs have a visible parent.
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, self._auto_load_last_project)
        QTimer.singleShot(200, self._start_git_pull)

    def _auto_load_last_project(self):
        recent = self._cfg.get("recent_projects", [])
        if recent:
            self._apply_project(recent[0])

    @property
    def _seg_labels_dir(self) -> str | None:
        """Working directory for segmentation TXT files (mask editor scratchpad)."""
        if self._annotations_dir:
            return os.path.join(self._annotations_dir, "seg_labels")
        if self.labels_dir:
            return os.path.join(os.path.dirname(self.labels_dir), "seg_labels")
        return None

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([220, 800, 360])
        self.setCentralWidget(splitter)

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        load_btn = QPushButton("Load Project…")
        load_btn.clicked.connect(self._on_load_project)
        top_row.addWidget(load_btn)
        self._manage_users_btn = QPushButton("Manage Users")
        self._manage_users_btn.setToolTip("Manage project users (admin only)")
        self._manage_users_btn.clicked.connect(self._on_manage_users)
        self._manage_users_btn.setEnabled(False)
        top_row.addWidget(self._manage_users_btn)
        layout.addLayout(top_row)

        self._project_label = QLabel("No project loaded")
        self._project_label.setWordWrap(True)
        self._project_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._project_label)

        self._user_label = QLabel("")
        self._user_label.setStyleSheet("color: #6a9955; font-size: 11px; font-style: italic;")
        layout.addWidget(self._user_label)

        files_header = QHBoxLayout()
        files_header.setContentsMargins(0, 0, 0, 0)
        files_header.addWidget(QLabel("PCD / PLY files:"))
        files_header.addStretch()
        self._filter_combo = QComboBox()
        self._filter_combo.addItems([
            "All frames",
            "Without annotations",
            "With annotations",
            "Manually annotated",
            "Flagged frames",
            "Verified",
        ])
        self._filter_combo.setToolTip("Filter the frame list by annotation status")
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        files_header.addWidget(self._filter_combo)
        layout.addLayout(files_header)

        self._select_all_cb = QCheckBox("Select all")
        self._select_all_cb.setTristate(True)
        self._select_all_cb.clicked.connect(self._on_select_all_clicked)
        layout.addWidget(self._select_all_cb)

        self.file_list = QTreeWidget()
        self.file_list.setHeaderLabels(["Frame", "Annotator", ""])
        self.file_list.setRootIsDecorated(False)
        self.file_list.setAllColumnsShowFocus(True)
        self.file_list.setSelectionBehavior(QTreeWidget.SelectRows)
        hdr = self.file_list.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.Fixed)
        self.file_list.setColumnWidth(2, 40)
        hdr.setMinimumSectionSize(80)
        self.file_list.setStyleSheet(
            "QTreeWidget::item:selected { background:#094771; color:#fff; }"
            "QTreeWidget::item:hover { background:#2a2d2e; }"
        )
        self.file_list.header().setStyleSheet(
            "QHeaderView::section { background:#2a2a2a; color:#888; "
            "border:none; border-bottom:1px solid #3c3c3c; padding:2px 4px; }"
        )
        self.file_list.currentItemChanged.connect(self._on_file_selected)
        self.file_list.itemChanged.connect(self._on_item_check_changed)
        self.file_list.itemDoubleClicked.connect(self._on_file_list_double_clicked)
        layout.addWidget(self.file_list)

        self._file_count_label = QLabel("")
        self._file_count_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._file_count_label)

        bottom_btn_row = QHBoxLayout()
        self._verify_btn = QPushButton("✓ Verify frame")
        self._verify_btn.setToolTip("Mark checked frames as verified (admin only); falls back to current frame if none checked)")
        self._verify_btn.setEnabled(False)
        self._verify_btn.clicked.connect(self._on_verify_frame)
        bottom_btn_row.addWidget(self._verify_btn)

        self._flag_btn = QPushButton("🚩 Flag")
        self._flag_btn.setToolTip("Flag this frame with a comment describing the issue")
        self._flag_btn.setEnabled(False)
        self._flag_btn.clicked.connect(self._on_flag_frame)
        bottom_btn_row.addWidget(self._flag_btn)
        layout.addLayout(bottom_btn_row)

        self._remove_frame_btn = QPushButton("🗑 Remove frame")
        self._remove_frame_btn.setToolTip("Permanently delete this frame and all its data (admin only)")
        self._remove_frame_btn.setEnabled(False)
        self._remove_frame_btn.setStyleSheet("color: #e05252;")
        self._remove_frame_btn.clicked.connect(self._on_remove_frame)
        layout.addWidget(self._remove_frame_btn)

        return w

    def _build_center_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.plotter = QtInteractor(parent=w)
        self.plotter.set_background("#1a1a2e", top="#0d0d1a")
        layout.addWidget(self.plotter, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(4, 4, 4, 4)

        prev_btn = QPushButton("←")
        prev_btn.clicked.connect(self._on_prev_frame)
        btn_row.addWidget(prev_btn)

        rand_btn = QPushButton("Random Frame")
        rand_btn.clicked.connect(self._on_random_frame)
        btn_row.addWidget(rand_btn)

        next_btn = QPushButton("→")
        next_btn.clicked.connect(self._on_next_frame)
        btn_row.addWidget(next_btn)

        layout.addLayout(btn_row)
        return w

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        w_layout = QVBoxLayout(w)
        w_layout.setContentsMargins(8, 8, 8, 8)
        w_layout.setSpacing(0)

        self._right_splitter = QSplitter(Qt.Vertical)
        self._right_splitter.setChildrenCollapsible(False)
        self._right_splitter.setHandleWidth(2)

        # ── Panel 1: Objects ───────────────────────────────────────────────
        self._obj_panel = QWidget()
        obj_layout = QVBoxLayout(self._obj_panel)
        obj_layout.setContentsMargins(0, 0, 0, 4)
        obj_layout.setSpacing(4)

        obj_layout.addWidget(QLabel("Objects:"))
        self.object_list = QListWidget()
        self.object_list.currentRowChanged.connect(self._on_object_selected)
        self.object_list.itemChanged.connect(self._on_obj_check_changed)
        self.object_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.object_list.customContextMenuRequested.connect(self._on_obj_list_context_menu)
        obj_layout.addWidget(self.object_list)

        obj_btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add Object")
        add_btn.clicked.connect(self._on_add_object)
        obj_btn_row.addWidget(add_btn)
        remove_btn = QPushButton("− Remove Object")
        remove_btn.clicked.connect(self._on_remove_object)
        obj_btn_row.addWidget(remove_btn)
        obj_layout.addLayout(obj_btn_row)

        self._merge_btn = QPushButton("Merge Objects")
        self._merge_btn.setEnabled(False)
        self._merge_btn.setToolTip("Check 2 or more objects to merge their masks")
        self._merge_btn.clicked.connect(self._on_merge_objects)
        obj_layout.addWidget(self._merge_btn)

        self._right_splitter.addWidget(self._obj_panel)

        # ── Panel 2: BB dimensions ─────────────────────────────────────────
        self._fields_panel = QWidget()
        fields_outer = QVBoxLayout(self._fields_panel)
        fields_outer.setContentsMargins(0, 0, 0, 0)
        fields_outer.setSpacing(0)

        self._fields_container = QWidget()
        self._fields_layout = QVBoxLayout(self._fields_container)
        self._fields_layout.setAlignment(Qt.AlignTop)
        self._fields_layout.setSpacing(8)

        fields_scroll = QScrollArea()
        fields_scroll.setWidget(self._fields_container)
        fields_scroll.setWidgetResizable(True)
        fields_scroll.setFrameShape(QFrame.NoFrame)
        fields_outer.addWidget(fields_scroll)

        self._right_splitter.addWidget(self._fields_panel)

        # ── Panel 3: RGB image + action buttons ────────────────────────────
        self._bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(self._bottom_panel)
        bottom_layout.setContentsMargins(0, 4, 0, 0)
        bottom_layout.setSpacing(4)

        bottom_layout.addWidget(QLabel("RGB image:"))
        self._image_label = QLabel("No image loaded")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._image_label.setStyleSheet(
            "background-color: #252526; border: 1px solid #3c3c3c; color: #555;"
        )
        self._image_label.installEventFilter(self)
        self.object_list.viewport().installEventFilter(self)
        bottom_layout.addWidget(self._image_label, stretch=1)

        # Row 1: [Generate 3D BB] | [Edit Masks]  — equal width
        top_btn_row = QHBoxLayout()
        top_btn_row.setSpacing(6)
        self._auto_btn = QPushButton("Generate 3D BB")
        self._auto_btn.clicked.connect(self._on_auto_bbox)
        self._auto_btn.setEnabled(False)
        self._auto_btn.setToolTip("Runs on the current frame; check multiple files to batch-process them")
        top_btn_row.addWidget(self._auto_btn, 1)
        edit_masks_btn = QPushButton("Edit Masks")
        edit_masks_btn.clicked.connect(self._on_edit_masks)
        top_btn_row.addWidget(edit_masks_btn, 1)
        bottom_layout.addLayout(top_btn_row)

        # Row 2: [Save results] spanning full width
        save_btn = QPushButton("Save results")
        save_btn.clicked.connect(self._on_save)
        bottom_layout.addWidget(save_btn)

        self._right_splitter.addWidget(self._bottom_panel)

        w_layout.addWidget(self._right_splitter)
        return w

    # ------------------------------------------------------------------
    # Project loading
    # ------------------------------------------------------------------
    def _on_load_project(self):
        recent = self._cfg.get("recent_projects", [])
        dlg = LoadProjectDialog(recent, parent=self)
        dlg.setStyleSheet(self.styleSheet())
        result = dlg.exec_()
        # Always persist edits/deletions, even if the user cancels
        self._cfg["recent_projects"] = dlg.get_updated_recent()
        save_config(self._cfg)
        if result != QDialog.Accepted:
            return
        project = dlg.get_project()
        if not project:
            return
        self._apply_project(project)

    def _apply_project(self, project: dict):
        # If the project has a dataset_root, re-discover paths (handles moved dirs)
        root = project.get("dataset_root", "")
        if root and os.path.isdir(root):
            discovered = discover_dataset_paths(root)
            # Explicit paths in the stored project override auto-discovery
            for key in ("pcd_dir", "rgb_dir", "depth_dir", "camera_params_dir",
                        "annotations_dir", "labels_dir"):
                if project.get(key) and os.path.isdir(project[key]):
                    discovered[key] = project[key]
            project = {**discovered, **{k: v for k, v in project.items()
                                        if k in ("name", "dataset_root")
                                        or (k not in discovered and v)}}

        pcd_dir    = project.get("pcd_dir", "")
        labels_dir = project.get("labels_dir", "")
        if not os.path.isdir(pcd_dir):
            QMessageBox.warning(self, "Folder not found", f"Point clouds folder not found:\n{pcd_dir}")
            return
        self.pcd_dir    = pcd_dir
        self.labels_dir = labels_dir if os.path.isdir(labels_dir) else None
        rgb_dir         = project.get("rgb_dir", "")
        self.rgb_dir    = rgb_dir if os.path.isdir(rgb_dir) else None
        depth_dir       = project.get("depth_dir", "")
        self.depth_dir  = depth_dir if os.path.isdir(depth_dir) else None
        cam_dir         = project.get("camera_params_dir", "")
        self.camera_params_dir = cam_dir if (os.path.isdir(cam_dir) or os.path.isfile(cam_dir)) else None

        # Store dataset root; save discovered paths into the project file
        root = project.get("dataset_root", "") or os.path.dirname(pcd_dir)
        self._dataset_root = root
        if root:
            _save_project_file(root, {
                "dataset_root": root,
                "name": project.get("name", os.path.basename(root)),
                "pcd_dir": pcd_dir,
                "rgb_dir": rgb_dir,
                "depth_dir": depth_dir,
                "camera_params_dir": cam_dir,
            })

        # Authentication
        self._current_user = None
        self._user_label.setText("Not signed in")
        self._verify_btn.setEnabled(False)
        self._remove_frame_btn.setEnabled(False)
        self._manage_users_btn.setEnabled(True)   # always available so users can be set up
        if root:
            users = _load_project_users(root)
            if users:
                last_user = self._cfg.get("last_users", {}).get(root, "")
                dlg = LoginDialog(users, dataset_root=root, default_user=last_user, parent=self)
                dlg.setStyleSheet(self.styleSheet())
                if dlg.exec_() != QDialog.Accepted or dlg.logged_in_user is None:
                    return   # cancelled — do not load the project
                self._current_user = dlg.logged_in_user
                role = self._current_user["role"]
                name = self._current_user["username"]
                self._user_label.setText(f"Signed in: {name} ({role})")
                self._verify_btn.setEnabled(role == "admin")
                self._remove_frame_btn.setEnabled(role == "admin")
                self._cfg.setdefault("last_users", {})[root] = name
                save_config(self._cfg)

        # Load dataset_info.json to get coordinate axis convention
        self._dataset_axes = None
        self._camera_needs_reset = True
        info_path = os.path.join(root, "dataset_info.json") if root else ""
        if os.path.isfile(info_path):
            try:
                with open(info_path) as _f:
                    _info = json.load(_f)
                self._dataset_axes = (
                    _info.get("annotation_format", {}).get("bbox_3d_axes")
                    or _info.get("sensor", {}).get("axes")
                    or _info.get("info", {}).get("bbox_3d_axes")
                    or _info.get("axes")
                )
            except Exception:
                pass

        # Load COCO annotations dir — fall back to <root>/annotations if not discovered
        ann_dir = (project.get("annotations_dir", "")
                   or os.path.dirname(project.get("annotations_file", ""))).strip()
        if not ann_dir:
            root = project.get("dataset_root", "") or os.path.dirname(pcd_dir)
            if root:
                ann_dir = os.path.join(root, "annotations")
        self._load_coco_db(ann_dir)

        project_name = project.get("name", "").strip()
        root_display = project.get("dataset_root", "") or pcd_dir
        coco_tag = " [COCO]" if self._coco_db else ""
        header = f"{project_name}\n" if project_name else ""
        self._project_label.setText(
            f"{header}{os.path.basename(root_display)}{coco_tag}\n"
            f"PCD: {os.path.basename(pcd_dir)}\n"
            f"RGB: {os.path.basename(rgb_dir) if self.rgb_dir else '—'}\n"
            f"Depth: {os.path.basename(depth_dir) if self.depth_dir else '—'}\n"
            f"Calib: {os.path.basename(cam_dir) if self.camera_params_dir else '—'}"
        )

        push_recent_project(self._cfg, {**project,
                                        "pcd_dir": pcd_dir, "rgb_dir": rgb_dir,
                                        "depth_dir": depth_dir,
                                        "camera_params_dir": cam_dir,
                                        "annotations_dir": ann_dir})
        save_config(self._cfg)

        self._populate_file_list()
        if self.file_list.topLevelItemCount() > 0:
            self.file_list.setCurrentItem(self.file_list.topLevelItem(0))

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # COCO per-frame helpers
    # ------------------------------------------------------------------
    def _load_coco_db(self, ann_dir: str):
        """Load categories.json from the annotations directory and enter COCO mode."""
        self._coco_db = None
        self._annotations_dir = None
        self._coco_categories = []
        self._coco_cat_id_to_name = {}
        self._coco_name_to_cat_id = {}
        self._coco_frame_cache = {}
        _DEFAULT_CATEGORIES = [
            {"id": 1, "name": "stillage",           "supercategory": "load_carrier"},
            {"id": 2, "name": "pallet_truck",        "supercategory": "vehicle"},
            {"id": 3, "name": "pallet",              "supercategory": "load_carrier"},
            {"id": 4, "name": "forklift",            "supercategory": "vehicle"},
            {"id": 5, "name": "small_load_carrier",  "supercategory": "container"},
        ]
        _DEFAULT_CLASSES = "stillage\npallet_truck\npallet\nforklift\nsmall_load_carrier\n"

        if not ann_dir:
            return
        if os.path.isfile(ann_dir):
            ann_dir = os.path.dirname(ann_dir)

        # Auto-create the annotations directory and required files if missing
        try:
            os.makedirs(ann_dir, exist_ok=True)
            cat_file = os.path.join(ann_dir, "categories.json")
            if not os.path.isfile(cat_file):
                with open(cat_file, "w") as _f:
                    json.dump(_DEFAULT_CATEGORIES, _f, indent=2)
            seg_dir = os.path.join(ann_dir, "seg_labels")
            os.makedirs(seg_dir, exist_ok=True)
            classes_file = os.path.join(seg_dir, "classes.txt")
            if not os.path.isfile(classes_file):
                with open(classes_file, "w") as _f:
                    _f.write(_DEFAULT_CLASSES)
        except Exception as e:
            QMessageBox.warning(self, "Annotations setup error",
                                f"Could not initialise annotations folder:\n{ann_dir}\n{e}")
            return

        cat_file = os.path.join(ann_dir, "categories.json")
        try:
            with open(cat_file) as f:
                cats = json.load(f)
            self._coco_categories = cats
            self._coco_cat_id_to_name = {c["id"]: c["name"] for c in cats}
            self._coco_name_to_cat_id = {c["name"]: c["id"] for c in cats}
            self._coco_db = {"categories": cats}
            self._annotations_dir = ann_dir
        except Exception as e:
            QMessageBox.warning(self, "Annotations load error",
                                f"Failed to load {cat_file}:\n{e}")

    def _load_frame_doc(self, frame_id: str) -> dict | None:
        """Load (and cache) the per-frame annotation JSON. Returns None if not found."""
        if self._annotations_dir is None:
            return None
        if frame_id in self._coco_frame_cache:
            return self._coco_frame_cache[frame_id]
        path = os.path.join(self._annotations_dir, f"{frame_id}.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path) as f:
                doc = json.load(f)
            self._coco_frame_cache[frame_id] = doc
            return doc
        except Exception:
            return None

    def _coco_obj_to_internal(self, ann: dict) -> dict:
        """Convert a per-frame COCO annotation to the internal object dict."""
        import math as _math
        b3d = ann.get("bbox_3d") or {}
        cen  = b3d.get("center", {})
        dims = b3d.get("dimensions", {})
        yaw_rad = b3d.get("yaw", 0.0)
        rx_deg  = b3d.get("rx", 0.0)
        rz_deg  = b3d.get("rz", 0.0)
        bbox = ann.get("bbox", [0, 0, 0, 0])  # [x, y, w, h]
        name = self._coco_cat_id_to_name.get(ann.get("category_id", 0), "object")
        return {
            "name": name,
            "bbox_2d": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
            "centroid":   {"x": cen.get("x", 0.0), "y": cen.get("y", 0.0), "z": cen.get("z", 0.0)},
            "dimensions": {"height": dims.get("height", 0.0),
                           "width":  dims.get("width",  0.0),
                           "length": dims.get("length", 0.0)},
            "rotations":  {"x": rx_deg, "y": _math.degrees(yaw_rad), "z": rz_deg},
        }

    def _internal_obj_to_coco(self, obj: dict, ann_id: int,
                               segmentation: list | None = None) -> dict:
        """Convert an internal object dict to a per-frame COCO annotation (no image_id)."""
        import math as _math
        b2d = obj.get("bbox_2d", [0, 0, 0, 0])
        x1, y1, x2, y2 = int(b2d[0]), int(b2d[1]), int(b2d[2]), int(b2d[3])
        cen  = obj.get("centroid", {})
        dims = obj.get("dimensions", {})
        rots    = obj.get("rotations", {})
        yaw_rad = _math.radians(rots.get("y", 0.0))
        rx_deg  = rots.get("x", 0.0)
        rz_deg  = rots.get("z", 0.0)
        name = _canonical_class_name(obj.get("name", "object"))
        cat_id = self._coco_name_to_cat_id.get(name, 0)
        area = (x2 - x1) * (y2 - y1)
        if segmentation:
            try:
                pts = [(segmentation[0][i], segmentation[0][i+1])
                       for i in range(0, len(segmentation[0]) - 1, 2)]
                area = int(abs(sum(pts[i][0]*pts[(i+1)%len(pts)][1]
                                   - pts[(i+1)%len(pts)][0]*pts[i][1]
                                   for i in range(len(pts)))) / 2)
            except Exception:
                pass
        ann = {
            "id":           ann_id,
            "category_id":  cat_id,
            "bbox":         [x1, y1, x2 - x1, y2 - y1],
            "segmentation": segmentation or [],
            "area":         area,
            "iscrowd":      0,
        }
        if not obj.get("no_3d_bb"):
            ann["bbox_3d"] = {
                "center":     {"x": cen.get("x", 0.0), "y": cen.get("y", 0.0), "z": cen.get("z", 0.0)},
                "dimensions": {"height": dims.get("height", 0.0),
                               "width":  dims.get("width",  0.0),
                               "length": dims.get("length", 0.0)},
                "yaw": yaw_rad,
                "rx":  rx_deg,
                "rz":  rz_deg,
            }
        return ann

    def _coco_ann_for_frame(self, frame_id: str) -> list[dict]:
        """Return all annotations for the given frame_id from its per-frame JSON."""
        doc = self._load_frame_doc(frame_id)
        return doc.get("annotations", []) if doc else []

    def _ensure_seg_txt_from_coco(self, frame_id: str):
        """Write the seg TXT working file from per-frame annotations (for mask editor)."""
        seg_dir = self._seg_labels_dir
        if not seg_dir or self._coco_db is None:
            return
        os.makedirs(seg_dir, exist_ok=True)
        doc = self._load_frame_doc(frame_id)
        if doc is None:
            return
        img_w = doc.get("width", 1920)
        img_h = doc.get("height", 1080)
        cat_names = [c["name"] for c in self._coco_categories]
        lines = []
        for ann in doc.get("annotations", []):
            seg = ann.get("segmentation", [])
            if not seg or not seg[0]:
                continue
            name = self._coco_cat_id_to_name.get(ann.get("category_id", 0), "object")
            cid = cat_names.index(name) if name in cat_names else 0
            flat = seg[0]
            norm = []
            for i in range(0, len(flat) - 1, 2):
                norm.extend([flat[i] / img_w, flat[i+1] / img_h])
            lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in norm))
        txt_path = os.path.join(seg_dir, f"{frame_id}.txt")
        if lines:
            with open(txt_path, "w") as f:
                f.write("\n".join(lines) + "\n")
        elif os.path.exists(txt_path):
            os.remove(txt_path)

    def _save_to_coco(self, frame_id: str, objects: list, seg_txt_path: str | None = None):
        """Write/update the per-frame annotation JSON for frame_id."""
        if self._coco_db is None or not self._annotations_dir:
            return
        # Image dimensions from cached doc or RGB
        doc = self._load_frame_doc(frame_id)
        img_w = doc.get("width",  1920) if doc else 1920
        img_h = doc.get("height", 1080) if doc else 1080
        if doc is None and self.rgb_dir:
            rgb_path = _find_rgb_image(self.rgb_dir, frame_id)
            if rgb_path:
                try:
                    img_w, img_h = _PIL_Image.open(rgb_path).size
                except Exception:
                    pass

        # Read seg TXT → pixel-space segmentation polygons
        txt_segs: dict[int, list] = {}
        if seg_txt_path and os.path.isfile(seg_txt_path):
            with open(seg_txt_path) as f:
                txt_lines = [l.strip() for l in f if l.strip()]
            for li, line in enumerate(txt_lines):
                parts = line.split()
                if len(parts) < 5:
                    continue
                vals = [float(v) for v in parts[1:]]
                flat_px = []
                for i in range(0, len(vals) - 1, 2):
                    flat_px.extend([vals[i] * img_w, vals[i+1] * img_h])
                txt_segs[li] = [flat_px]

        def _iou2(b1, b2):
            ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            a1 = max(0.0, (b1[2]-b1[0]) * (b1[3]-b1[1]))
            a2 = max(0.0, (b2[2]-b2[0]) * (b2[3]-b2[1]))
            return inter / (a1 + a2 - inter + 1e-9)

        txt_used = [False] * len(txt_segs)
        new_anns = []
        for ann_id, obj in enumerate(objects, 1):
            b2d = obj.get("bbox_2d", [0, 0, 0, 0])
            best_iou, best_li = 0.0, -1
            for li, seg_flat in txt_segs.items():
                if txt_used[li]:
                    continue
                flat = seg_flat[0]
                xs, ys = flat[0::2], flat[1::2]
                iou = _iou2(b2d, [min(xs), min(ys), max(xs), max(ys)])
                if iou > best_iou:
                    best_iou, best_li = iou, li
            seg = None
            if best_li >= 0 and best_iou > 0.2:
                txt_used[best_li] = True
                seg = txt_segs[best_li]
            new_anns.append(self._internal_obj_to_coco(obj, ann_id, seg))

        frame_doc = {
            "file_name":   f"{frame_id}.png",
            "width":       img_w,
            "height":      img_h,
            "annotations": new_anns,
        }
        out_path = os.path.join(self._annotations_dir, f"{frame_id}.json")
        with open(out_path, "w") as f:
            json.dump(frame_doc, f, indent=2)
        self._coco_frame_cache[frame_id] = frame_doc

    def _update_frame_meta(self, frame_id: str, updates: dict):
        """Merge updates into the per-frame status stored in the project file."""
        if not self._dataset_root:
            return
        data = _load_project_file(self._dataset_root) or {}
        status = data.setdefault("frame_status", {})
        status.setdefault(frame_id, {}).update(updates)
        _save_project_file(self._dataset_root, data)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _populate_file_list(self):
        if not self.pcd_dir:
            return
        self._all_frame_files = sorted(
            f for f in os.listdir(self.pcd_dir)
            if f.endswith(".pcd") or f.endswith(".ply")
        )
        self._apply_file_filter()
        self._update_status(f"Found {len(self._all_frame_files)} point cloud files")

    def _frame_has_annotations(self, frame_id: str) -> bool:
        if self._coco_db is not None:
            return bool(self._coco_ann_for_frame(frame_id))
        if self.labels_dir:
            path = os.path.join(self.labels_dir, f"{frame_id}.json")
            if not os.path.isfile(path):
                return False
            try:
                with open(path) as f:
                    doc = json.load(f)
                return bool(doc.get("objects"))
            except Exception:
                return False
        return False

    def _frame_meta(self, frame_id: str) -> dict:
        """Return the status dict for a frame from the project file, or {} if absent."""
        if not self._dataset_root:
            return {}
        data = _load_project_file(self._dataset_root)
        return (data or {}).get("frame_status", {}).get(frame_id, {})

    def _apply_file_filter(self):
        mode = self._filter_combo.currentText()
        if mode == "All frames":
            visible = self._all_frame_files
        elif mode in ("With annotations", "Without annotations"):
            want = (mode == "With annotations")
            visible = [
                f for f in self._all_frame_files
                if self._frame_has_annotations(os.path.splitext(f)[0]) == want
            ]
        elif mode == "Manually annotated":
            visible = [
                f for f in self._all_frame_files
                if self._frame_meta(os.path.splitext(f)[0]).get("manually_modified")
                and not self._frame_meta(os.path.splitext(f)[0]).get("verified_by")
            ]
        elif mode == "Flagged frames":
            visible = [
                f for f in self._all_frame_files
                if self._frame_meta(os.path.splitext(f)[0]).get("flagged")
            ]
        elif mode == "Verified":
            visible = [
                f for f in self._all_frame_files
                if self._frame_meta(os.path.splitext(f)[0]).get("verified_by")
            ]
        else:
            visible = self._all_frame_files

        cur_item = self.file_list.currentItem()
        cur_fname = self._item_fname(cur_item) if cur_item else None

        self._select_all_cb.blockSignals(True)
        self._select_all_cb.setCheckState(Qt.Unchecked)
        self._select_all_cb.blockSignals(False)
        self.file_list.blockSignals(True)
        self.file_list.clear()
        restore_item = None
        for f in visible:
            frame_id = os.path.splitext(f)[0]
            meta              = self._frame_meta(frame_id)
            flagged           = meta.get("flagged", False)
            verified_by       = meta.get("verified_by", "")
            manually_modified = meta.get("manually_modified", False)
            annotated_by      = meta.get("annotated_by", "") if manually_modified else ""
            has_ann           = self._frame_has_annotations(frame_id)

            # Annotator column
            if not has_ann:
                annotator_col = "no annotation"
            elif manually_modified and annotated_by:
                annotator_col = annotated_by
            else:
                annotator_col = "auto"

            # Icon column
            icon_col = ""
            if verified_by:
                icon_col += "✓"
            if flagged:
                icon_col += "🚩"

            item = QTreeWidgetItem([f, annotator_col, icon_col])
            item.setData(0, Qt.UserRole, f)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Unchecked)

            if flagged:
                color = QColor("#e05252")   # red   – flagged
            elif verified_by:
                color = QColor("#4ec94e")   # green – verified
            elif manually_modified:
                color = QColor("#d4d4d4")   # white – manually annotated
            elif has_ann:
                color = QColor("#9cdcfe")   # blue  – auto/pseudo
            else:
                color = QColor("#666666")   # gray  – unannotated
            for col in range(3):
                item.setForeground(col, color)

            self.file_list.addTopLevelItem(item)
            if f == cur_fname:
                restore_item = item
        self.file_list.blockSignals(False)

        if restore_item is not None:
            self.file_list.setCurrentItem(restore_item)

        total = len(self._all_frame_files)
        shown = len(visible)
        if shown == total:
            self._file_count_label.setText(f"{total} frames")
        else:
            self._file_count_label.setText(f"{shown} of {total} frames")

    def _on_filter_changed(self):
        self._apply_file_filter()

    @staticmethod
    def _item_fname(item) -> str:
        """Return the clean filename stored in column-0 UserRole of a file_list item."""
        return item.data(0, Qt.UserRole) or item.text(0)

    def _on_file_selected(self, item: QTreeWidgetItem | None):
        if item is None:
            return
        self._load_frame(os.path.splitext(self._item_fname(item))[0])
        self._auto_btn.setEnabled(True)
        self._flag_btn.setEnabled(True)

    def _load_frame(self, frame_id: str):
        self._3d_bb_manually_touched = False
        if self._dirty and not self._check_unsaved():
            return

        if not self.pcd_dir:
            self._update_status("Please load a project first")
            return

        pcd_path = next(
            (os.path.join(self.pcd_dir, f"{frame_id}{ext}") for ext in (".pcd", ".ply")
             if os.path.exists(os.path.join(self.pcd_dir, f"{frame_id}{ext}"))),
            None,
        )
        if pcd_path is None:
            QMessageBox.warning(self, "Missing file", f"No .pcd or .ply found for:\n{frame_id}")
            return

        if self._coco_db is not None:
            # COCO format: load from per-frame JSON
            self.current_label_data = None
            anns = self._coco_ann_for_frame(frame_id)
            self.current_objects = [self._coco_obj_to_internal(a) for a in anns
                                    if a.get("bbox_3d") is not None]
            # Write seg TXT working file so mask editor and overlay work correctly
            self._ensure_seg_txt_from_coco(frame_id)
            if not anns:
                self._update_status(f"{frame_id} — no COCO annotations yet")
        elif self.labels_dir:
            label_path = os.path.join(self.labels_dir, f"{frame_id}.json")
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    self.current_label_data = json.load(f)
                self.current_objects = copy.deepcopy(self.current_label_data.get("objects", []))
            else:
                self._update_status(f"No JSON for {frame_id} — showing point cloud only")
                self.current_label_data = None
                self.current_objects = []
        else:
            self.current_label_data = None
            self.current_objects = []

        self.current_frame_id = frame_id
        self._dirty = False
        self._rebuild_object_panel()
        self.setWindowTitle(f"3D Label Editor — {frame_id}")
        self._sync_file_list_selection(frame_id)
        self._render_scene()
        self._load_current_intrinsics(frame_id)
        self._load_rgb_image(frame_id)
        self._update_status(f"{frame_id} — {len(self.current_objects)} object(s)")

    def _rebuild_object_panel(self):
        self._block_manual_tracking = True
        self.object_list.blockSignals(True)
        self.object_list.clear()
        for i, obj in enumerate(self.current_objects):
            item = QListWidgetItem(f"[{i}]  {obj['name']}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.object_list.addItem(item)
        self.object_list.blockSignals(False)

        self._merge_btn.setEnabled(False)
        self._clear_active_widget()
        self._selected_obj_idx = -1

        if self.current_objects:
            self.object_list.setCurrentRow(0)
        self._block_manual_tracking = False

    def _clear_active_widget(self):
        if self._active_widget is not None:
            self._active_widget.hide()
            self._fields_layout.removeWidget(self._active_widget)
            self._active_widget.deleteLater()
            self._active_widget = None

    def _sync_active(self) -> str | None:
        if self._active_widget is None or self._selected_obj_idx < 0:
            return None
        result = self._active_widget.get_values()
        if result is None:
            return f"Object [{self._selected_obj_idx}]: invalid numeric value in one or more fields"
        self.current_objects[self._selected_obj_idx] = result
        return None

    def _on_object_selected(self, row: int):
        if row < 0 or row >= len(self.current_objects):
            return
        self._sync_active()
        self._uncheck_all_objects()
        self._selected_obj_idx = row
        self._clear_active_widget()
        widget = ObjectFieldWidget(row, self.current_objects[row], on_change=self._on_regenerate)
        self._fields_layout.addWidget(widget)
        self._active_widget = widget
        self._render_scene()
        self._refresh_image_pixmap()

    def _on_obj_list_context_menu(self, pos):
        """Right-click menu on the object list: Copy / Paste pose (X, Z, Yaw)."""
        from PyQt5.QtWidgets import QMenu
        item = self.object_list.itemAt(pos)
        if item is None:
            return
        row = self.object_list.row(item)
        if row < 0 or row >= len(self.current_objects):
            return

        # Make sure the active widget values are synced before reading
        self._sync_active()
        obj = self.current_objects[row]
        cen  = obj.get("centroid",  {})
        rots = obj.get("rotations", {})

        menu = QMenu(self)
        copy_act  = menu.addAction("Copy pose (X, Z, Yaw)")
        paste_act = menu.addAction("Paste pose (X, Z, Yaw)")
        paste_act.setEnabled(self._copied_pose is not None)

        action = menu.exec_(self.object_list.viewport().mapToGlobal(pos))
        if action is None:
            return

        if action is copy_act:
            self._copied_pose = {
                "x":   cen.get("x",  0.0),
                "z":   cen.get("z",  0.0),
                "yaw": rots.get("y", 0.0),
            }
            self._update_status(
                f"Copied pose from [{row}] {obj.get('name','')}: "
                f"X={self._copied_pose['x']:.3f}, Z={self._copied_pose['z']:.3f}, "
                f"Yaw={self._copied_pose['yaw']:.1f}°"
            )

        elif action is paste_act and self._copied_pose is not None:
            obj["centroid"]["x"]  = self._copied_pose["x"]
            obj["centroid"]["z"]  = self._copied_pose["z"]
            obj["rotations"]["y"] = self._copied_pose["yaw"]
            # Select that object so the field widget refreshes
            self.object_list.setCurrentRow(row)
            self._clear_active_widget()
            widget = ObjectFieldWidget(row, obj, on_change=self._on_regenerate)
            self._fields_layout.addWidget(widget)
            self._active_widget = widget
            self._selected_obj_idx = row
            self._3d_bb_manually_touched = True
            self._dirty = True
            self._render_scene()
            self._update_status(
                f"Pasted pose to [{row}] {obj.get('name','')}: "
                f"X={self._copied_pose['x']:.3f}, Z={self._copied_pose['z']:.3f}, "
                f"Yaw={self._copied_pose['yaw']:.1f}°"
            )

    def _on_regenerate(self):
        if self.current_frame_id is None:
            return
        err = self._sync_active()
        if err:
            QMessageBox.warning(self, "Invalid field values", err)
            return
        if not self._block_manual_tracking:
            self._3d_bb_manually_touched = True
        self._dirty = True
        if self._active_widget:
            self._active_widget.clear_highlights()
        self._render_scene()
        self._refresh_image_pixmap()

    def _render_scene(self, reset_camera: bool = False):
        do_reset = reset_camera or self._camera_needs_reset
        cam_pos = None if do_reset else self.plotter.camera_position

        self.plotter.clear()
        if self.current_frame_id is None or not self.pcd_dir:
            return

        pcd_path = next(
            (os.path.join(self.pcd_dir, f"{self.current_frame_id}{ext}") for ext in (".pcd", ".ply")
             if os.path.exists(os.path.join(self.pcd_dir, f"{self.current_frame_id}{ext}"))),
            None,
        )
        if pcd_path is None:
            return
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
        except Exception as e:
            self._update_status(f"Error loading point cloud: {e}")
            return

        checked_set = {
            i for i in range(self.object_list.count())
            if self.object_list.item(i) and
               self.object_list.item(i).checkState() == Qt.Checked
        }
        for i, obj in enumerate(self.current_objects):
            mesh, color = build_pv_bbox(obj)
            active   = (i == self._selected_obj_idx)
            checked  = (i in checked_set)
            if active:
                lw, op, draw_color = 4, 1.0, color
            elif checked:
                lw, op, draw_color = 3, 0.85, "yellow"
            else:
                lw, op, draw_color = 2, 0.3, color
            self.plotter.add_mesh(mesh, color=draw_color, line_width=lw, opacity=op)

        self.plotter.add_axes()
        if do_reset:
            self.plotter.reset_camera()
            self._set_initial_camera()
            self._camera_needs_reset = False
        else:
            self.plotter.camera_position = cam_pos

    def _set_initial_camera(self):
        """Orient the camera according to the dataset's coordinate axes (from dataset_info.json).
        Falls back to PyVista's default reset if no axis info is available.
        """
        axes = self._dataset_axes  # e.g. {"x": "right", "y": "down", "z": "forward"}
        if not axes:
            return

        forward = next((k for k, v in axes.items() if v == "forward"), None)
        up_axis = next((k for k, v in axes.items() if v == "up"), None)
        down_axis = next((k for k, v in axes.items() if v == "down"), None)

        if forward is None:
            return

        # Build the "up" vector: if Y is down, screen-up is -Y; if Z is up, screen-up is +Z, etc.
        _axis_vec = {"x": np.array([1,0,0]), "y": np.array([0,1,0]), "z": np.array([0,0,1])}
        if up_axis:
            view_up = _axis_vec[up_axis]
        elif down_axis:
            view_up = -_axis_vec[down_axis]
        else:
            view_up = np.array([0, 1, 0])

        # Forward direction vector
        fwd = _axis_vec[forward]

        # Place the camera behind the scene (at negative-forward side) looking toward scene center
        bounds = self.plotter.bounds   # (xmin, xmax, ymin, ymax, zmin, zmax)
        cx = (bounds[0] + bounds[1]) / 2
        cy = (bounds[2] + bounds[3]) / 2
        cz = (bounds[4] + bounds[5]) / 2
        focal = np.array([cx, cy, cz])

        extent = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4], 0.1)
        cam_pos = focal - fwd * extent * 1.5

        self.plotter.camera_position = [
            tuple(cam_pos),
            tuple(focal),
            tuple(view_up),
        ]

    def _on_add_object(self):
        self._on_edit_masks()

    def _remove_seg_masks_for_objs(self, objs: list):
        """Remove the TXT polygon line(s) that best match each object's bbox_2d."""
        if not self.current_frame_id or not self.rgb_dir:
            return
        seg_labels_dir = self._seg_labels_dir
        if not seg_labels_dir:
            return
        txt_path = os.path.join(seg_labels_dir, f"{self.current_frame_id}.txt")
        if not os.path.exists(txt_path):
            return
        rgb_path = _find_rgb_image(self.rgb_dir, self.current_frame_id)
        if rgb_path is None:
            return
        with _PIL_Image.open(rgb_path) as _img:
            rgb_w, rgb_h = _img.size
        with open(txt_path) as _f:
            lines = [l.strip() for l in _f if l.strip()]
        to_remove: set[int] = set()
        for obj in objs:
            bb2d = obj.get("bbox_2d")
            if bb2d is None:
                continue
            x1, y1, x2, y2 = float(bb2d[0]), float(bb2d[1]), float(bb2d[2]), float(bb2d[3])
            best_iou, best_li = 0.0, None
            for li, line in enumerate(lines):
                if li in to_remove:
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                pts = np.array([float(v) for v in parts[1:]], dtype=np.float64).reshape(-1, 2)
                px, py = pts[:, 0] * rgb_w, pts[:, 1] * rgb_h
                bx1, bx2 = px.min(), px.max()
                by1, by2 = py.min(), py.max()
                ix1, iy1 = max(x1, bx1), max(y1, by1)
                ix2, iy2 = min(x2, bx2), min(y2, by2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = (x2-x1)*(y2-y1) + (bx2-bx1)*(by2-by1) - inter
                iou = inter / max(union, 1.0)
                if iou > best_iou:
                    best_iou, best_li = iou, li
            if best_li is not None and best_iou > 0.2:
                to_remove.add(best_li)
        if not to_remove:
            return
        new_lines = [l for i, l in enumerate(lines) if i not in to_remove]
        if new_lines:
            with open(txt_path, "w") as _f:
                _f.write("\n".join(new_lines) + "\n")
        else:
            os.remove(txt_path)

    def _on_remove_object(self):
        checked_indices = [
            i for i in range(self.object_list.count())
            if self.object_list.item(i).checkState() == Qt.Checked
        ]
        if checked_indices:
            self._sync_active()
            objs_removed = [self.current_objects[i] for i in checked_indices]
            self._active_widget = None
            self._selected_obj_idx = -1
            self.object_list.blockSignals(True)
            for idx in sorted(checked_indices, reverse=True):
                del self.current_objects[idx]
                self.object_list.takeItem(idx)
            for i in range(self.object_list.count()):
                self.object_list.item(i).setText(f"[{i}]  {self.current_objects[i]['name']}")
            self.object_list.blockSignals(False)
            self._dirty = True
            new_row = min(checked_indices[0], self.object_list.count() - 1)
            if new_row >= 0:
                self.object_list.setCurrentRow(new_row)
            self._remove_seg_masks_for_objs(objs_removed)
            self._render_scene()
            if self.current_frame_id:
                self._load_rgb_image(self.current_frame_id)
            self._update_status(
                f"Removed {len(checked_indices)} object(s) — {len(self.current_objects)} remaining")
        else:
            row = self.object_list.currentRow()
            if row < 0:
                return
            self._sync_active()
            obj_removed = self.current_objects[row]
            removed_name = obj_removed["name"]
            del self.current_objects[row]
            self.object_list.blockSignals(True)
            self.object_list.takeItem(row)
            for i in range(self.object_list.count()):
                self.object_list.item(i).setText(f"[{i}]  {self.current_objects[i]['name']}")
            self.object_list.blockSignals(False)
            # Reset stale widget/index BEFORE setCurrentRow so _sync_active is a no-op
            self._active_widget = None
            self._selected_obj_idx = -1
            self._dirty = True
            new_row = min(row, self.object_list.count() - 1)
            self.object_list.setCurrentRow(new_row)
            self._remove_seg_masks_for_objs([obj_removed])
            self._render_scene()
            if self.current_frame_id:
                self._load_rgb_image(self.current_frame_id)
            self._update_status(f"Removed '{removed_name}' — {len(self.current_objects)} object(s)")

    def _ensure_active_checked(self):
        """On first Ctrl+click, also check the currently active object so it joins the selection."""
        if 0 <= self._selected_obj_idx < self.object_list.count():
            it = self.object_list.item(self._selected_obj_idx)
            if it and it.checkState() != Qt.Checked:
                it.setCheckState(Qt.Checked)

    def _uncheck_all_objects(self):
        self.object_list.blockSignals(True)
        for i in range(self.object_list.count()):
            it = self.object_list.item(i)
            if it:
                it.setCheckState(Qt.Unchecked)
                it.setBackground(QColor(0, 0, 0, 0))
        self.object_list.blockSignals(False)
        self._merge_btn.setEnabled(False)

    def _on_obj_check_changed(self, item):
        n_checked = sum(
            1 for i in range(self.object_list.count())
            if self.object_list.item(i).checkState() == Qt.Checked
        )
        self._merge_btn.setEnabled(n_checked >= 2)
        # Highlight checked items in the list with a distinct background
        for i in range(self.object_list.count()):
            it = self.object_list.item(i)
            if it:
                if it.checkState() == Qt.Checked:
                    it.setBackground(QColor(180, 140, 0, 120))
                else:
                    it.setBackground(QColor(0, 0, 0, 0))
        self._render_scene()
        self._refresh_image_pixmap()


    def _on_merge_objects(self):
        """OR checked objects' masks, re-run HDF+orientation, replace with merged BB."""
        if not self.current_frame_id or (not self.labels_dir and self._coco_db is None):
            return

        checked_indices = [
            i for i in range(self.object_list.count())
            if self.object_list.item(i) and
               self.object_list.item(i).checkState() == Qt.Checked
        ]
        if len(checked_indices) < 2:
            QMessageBox.warning(self, "Need 2+ objects",
                                "Check at least 2 objects to merge.")
            return

        objs_to_merge = [self.current_objects[i] for i in checked_indices]

        if not self.rgb_dir:
            QMessageBox.warning(self, "No RGB folder",
                                "RGB images folder is not configured.")
            return
        rgb_path = _find_rgb_image(self.rgb_dir, self.current_frame_id)
        if rgb_path is None:
            QMessageBox.warning(self, "No RGB image",
                                f"Cannot find RGB image for {self.current_frame_id}.")
            return
        rgb_arr = np.array(_PIL_Image.open(rgb_path).convert("RGB"))
        rgb_h, rgb_w = rgb_arr.shape[:2]

        # Build merged mask (OR) from YOLO TXT polygons; fall back to bbox fill
        merged_mask = np.zeros((rgb_h, rgb_w), dtype=bool)
        all_bboxes = []
        import cv2 as _cv2_mg
        seg_labels_dir = self._seg_labels_dir or ""
        _txt_path = os.path.join(seg_labels_dir, f"{self.current_frame_id}.txt") if seg_labels_dir else ""
        _all_txt_lines: list[str] = []
        if os.path.exists(_txt_path):
            with open(_txt_path) as _f:
                _all_txt_lines = [l.strip() for l in _f if l.strip()]
        _line_indices_to_remove: set = set()
        for obj in objs_to_merge:
            bb2d = obj.get("bbox_2d")
            if bb2d is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in bb2d]
            all_bboxes.append((x1, y1, x2, y2))
            # Match TXT line by polygon-bbox IoU
            _best_iou, _best_li = 0.0, None
            for _li, _line in enumerate(_all_txt_lines):
                _parts = _line.split()
                if len(_parts) < 7:
                    continue
                _pts = np.array([float(v) for v in _parts[1:]], dtype=np.float64).reshape(-1, 2)
                _px, _py = _pts[:, 0] * rgb_w, _pts[:, 1] * rgb_h
                _bx1, _bx2 = float(_px.min()), float(_px.max())
                _by1, _by2 = float(_py.min()), float(_py.max())
                _ix1, _iy1 = max(x1, _bx1), max(y1, _by1)
                _ix2, _iy2 = min(x2, _bx2), min(y2, _by2)
                if _ix2 <= _ix1 or _iy2 <= _iy1:
                    continue
                _inter = (_ix2 - _ix1) * (_iy2 - _iy1)
                _union = (x2-x1)*(y2-y1) + (_bx2-_bx1)*(_by2-_by1) - _inter
                _iou = _inter / max(_union, 1.0)
                if _iou > _best_iou:
                    _best_iou, _best_li = _iou, _li
            if _best_li is not None and _best_iou > 0.2:
                _line_indices_to_remove.add(_best_li)
                _parts = _all_txt_lines[_best_li].split()
                _pts = np.array([float(v) for v in _parts[1:]], dtype=np.float64).reshape(-1, 2)
                _pts_px = (_pts * np.array([rgb_w, rgb_h])).astype(np.int32)
                _pm = np.zeros((rgb_h, rgb_w), dtype=np.uint8)
                _cv2_mg.fillPoly(_pm, [_pts_px], 255)
                merged_mask |= (_pm > 0)
            else:
                merged_mask[max(0, y1):min(rgb_h, y2), max(0, x1):min(rgb_w, x2)] = True

        if not all_bboxes:
            QMessageBox.warning(self, "No 2D bbox",
                                "Selected objects have no 2D bounding box.\n"
                                "Run autonomous annotation first.")
            return

        mx1 = min(b[0] for b in all_bboxes)
        my1 = min(b[1] for b in all_bboxes)
        mx2 = max(b[2] for b in all_bboxes)
        my2 = max(b[3] for b in all_bboxes)
        cls_names = list(dict.fromkeys(o["name"] for o in objs_to_merge))
        merged_cls = cls_names[0]

        # Load depth + intrinsics
        from pose_estimation_pipeline import (
            find_depth_file, load_depth_file, find_camera_params_file,
            load_intrinsics, align_depth_to_color,
            apply_hist_depth_filter, estimate_3d_pose, make_label_object,
        )
        dep_arr = None
        fx_v = fy_v = cx_v = cy_v = None
        if self.depth_dir and self.camera_params_dir:
            dep_path = find_depth_file(self.depth_dir, self.current_frame_id)
            p_path   = find_camera_params_file(self.camera_params_dir, self.current_frame_id)
            if dep_path:
                dep_raw = load_depth_file(dep_path)
                if p_path and p_path.lower().endswith(".json"):
                    dep_arr, fx_v, fy_v, cx_v, cy_v = align_depth_to_color(dep_raw, p_path)
                else:
                    dep_arr = dep_raw
                    if p_path:
                        fx_v, fy_v, cx_v, cy_v = load_intrinsics(p_path)
        if dep_arr is None:
            dep_arr = np.zeros((rgb_h, rgb_w), dtype=np.float32)
        if fx_v is None:
            fx_v = self._cur_fx or 1382.0
            fy_v = self._cur_fy or 1382.0
            cx_v = self._cur_cx or 960.0
            cy_v = self._cur_cy or 540.0

        def _load_fn(_idx):
            return rgb_arr, dep_arr, fx_v, fy_v, cx_v, cy_v

        # Detect Z convention
        z_backward = False
        if self.pcd_dir:
            for _ext in (".pcd", ".ply"):
                _pc_path = os.path.join(self.pcd_dir, f"{self.current_frame_id}{_ext}")
                if os.path.exists(_pc_path):
                    try:
                        _pts = pv.read(_pc_path).points
                        if len(_pts) > 0 and _pts[:, 2].max() < 0:
                            z_backward = True
                    except Exception:
                        pass
                    break

        from auto_bbox_dialog import AutoBBoxValidationDialog
        dlg = AutoBBoxValidationDialog(
            _load_fn, n_frames=1,
            precomputed={
                "bbox":       [mx1, my1, mx2, my2],
                "mask":       merged_mask.astype(np.uint8),
                "class_name": merged_cls,
            },
            parent=self,
        )
        dlg.setStyleSheet(self.styleSheet())
        if dlg.exec_() != QDialog.Accepted or dlg.result is None:
            return

        hdf_params = dlg._hdf_params

        # Re-run 3D estimation with merged mask + dialog's HDF params
        dep_h, dep_w = dep_arr.shape[:2]
        sx, sy = dep_w / rgb_w, dep_h / rgb_h
        dx1, dy1 = int(mx1 * sx), int(my1 * sy)
        dx2, dy2 = int(mx2 * sx), int(my2 * sy)
        dep_crop = dep_arr[dy1:dy2, dx1:dx2].astype(float)
        if dep_crop.size == 0:
            QMessageBox.warning(self, "Empty depth crop",
                                "Depth crop for merged bbox is empty.")
            return

        if rgb_h != dep_h or rgb_w != dep_w:
            full_mask_dep = _cv2_mg.resize(
                merged_mask.astype(np.uint8), (dep_w, dep_h),
                interpolation=_cv2_mg.INTER_NEAREST)
        else:
            full_mask_dep = merged_mask.astype(np.uint8)
        raw_mask = full_mask_dep[dy1:dy2, dx1:dx2]

        valid_px = dep_crop[dep_crop > 0]
        if valid_px.size > 0 and valid_px.max() <= 100:
            dep_crop = dep_crop * 1000.0
        dep_masked = np.where(raw_mask, dep_crop, 0)
        filtered, *_ = apply_hist_depth_filter(
            dep_masked,
            resolution=hdf_params["resolution"],
            max_height_percent=hdf_params["max_height_percent"],
            ignore_background=hdf_params.get("ignore_background", False),
        )
        mask_crop = (filtered > 0).astype(np.uint8)
        try:
            _, center, dims, yaw_deg, _ = estimate_3d_pose(
                filtered, mask_crop, dx1, dy1, fx_v, fy_v, cx_v, cy_v,
                class_dims=_class_dims_for(merged_cls),
                class_dims_range=_class_dims_range_for(merged_cls))
        except Exception as e:
            QMessageBox.warning(self, "3D estimation failed", str(e))
            return
        if z_backward:
            center[1] = -center[1]
            center[2] = -center[2]
            yaw_deg   = -yaw_deg

        merged_obj = make_label_object(merged_cls, center, dims, yaw_deg)
        merged_obj["bbox_2d"] = [mx1, my1, mx2, my2]

        # Extract polygon from merged mask, update YOLO TXT
        _merged_u8 = (merged_mask > 0).astype(np.uint8) * 255
        _conts, _ = _cv2_mg.findContours(_merged_u8, _cv2_mg.RETR_EXTERNAL,
                                           _cv2_mg.CHAIN_APPROX_SIMPLE)
        if _conts:
            _lc = max(_conts, key=_cv2_mg.contourArea)
            _eps = 0.003 * _cv2_mg.arcLength(_lc, True)
            _ap = _cv2_mg.approxPolyDP(_lc, _eps, True)
            _pts = _ap.reshape(-1, 2).astype(np.float64)
            _pts[:, 0] /= rgb_w
            _pts[:, 1] /= rgb_h
            if seg_labels_dir:
                _cid = _seg_class_id(merged_cls, seg_labels_dir)
                _poly_line = f"{_cid} " + " ".join(f"{v:.6f}" for v in _pts.flatten())
                _new_lines = [l for i, l in enumerate(_all_txt_lines)
                              if i not in _line_indices_to_remove]
                _new_lines.append(_poly_line)
                os.makedirs(seg_labels_dir, exist_ok=True)
                with open(_txt_path, "w") as _f:
                    _f.write("\n".join(_new_lines) + "\n")

        # Sync the currently active field widget, then detach it so _on_save()
        # won't re-sync against objects that no longer exist.
        self._sync_active()
        self._clear_active_widget()
        self._selected_obj_idx = -1

        # Remove merged objects, append result
        for idx in sorted(checked_indices, reverse=True):
            del self.current_objects[idx]
        self.current_objects.append(merged_obj)
        self._dirty = True

        self._on_save()
        self._rebuild_object_panel()
        self._load_rgb_image(self.current_frame_id)
        self._render_scene()
        self._update_status(
            f"Merged {len(checked_indices)} objects into '{merged_cls}' — saved."
        )

    def _on_save(self):
        if self.current_frame_id is None:
            self._update_status("No frame loaded")
            return
        if not self.labels_dir and self._coco_db is None:
            self._update_status("No labels folder or annotations file configured")
            return
        err = self._sync_active()
        if err:
            QMessageBox.warning(self, "Invalid field values", err)
            return

        if self._coco_db is not None:
            # If 3D BBs were manually edited, ask user to confirm before saving
            manually_confirmed = False
            if self._3d_bb_manually_touched:
                reply = QMessageBox.question(
                    self, "Manual annotation",
                    "You have manually adjusted 3D bounding boxes in this frame.\n\n"
                    "Confirm that these changes are intentional and should be marked\n"
                    "as manually annotated?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                    QMessageBox.Yes,
                )
                if reply == QMessageBox.Cancel:
                    return
                manually_confirmed = (reply == QMessageBox.Yes)

            # COCO format: write per-frame JSON
            seg_dir = self._seg_labels_dir
            txt_path = os.path.join(seg_dir, f"{self.current_frame_id}.txt") if seg_dir else None
            self._save_to_coco(self.current_frame_id, self.current_objects, txt_path)

            # Update meta for manual annotation flag
            if manually_confirmed:
                meta_updates: dict = {
                    "manually_modified": True,
                    "manually_modified_at": datetime.datetime.now().isoformat(timespec="seconds"),
                }
                if self._current_user:
                    meta_updates["annotated_by"] = self._current_user["username"]
                self._update_frame_meta(self.current_frame_id, meta_updates)

            self._3d_bb_manually_touched = False
            self._dirty = False
            self._update_status(f"Saved {self.current_frame_id}.json")
            self._refresh_file_list_item(self.current_frame_id)
            if self._filter_combo.currentIndex() != 0:
                self._apply_file_filter()
            return

        is_new = self.current_label_data is None
        if is_new:
            pcd_filename = f"{self.current_frame_id}.pcd"
            pcd_path = os.path.join(self.pcd_dir, pcd_filename) if self.pcd_dir else ""
            save_data = {
                "folder":   os.path.basename(self.labels_dir),
                "filename": f"{self.current_frame_id}.ply",
                "path":     pcd_path,
                "objects":  self.current_objects,
            }
        else:
            save_data = copy.deepcopy(self.current_label_data)
            save_data["objects"] = self.current_objects

        out_path = os.path.join(self.labels_dir, f"{self.current_frame_id}.json")
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent="\t")

        self.current_label_data = save_data
        self._dirty = False
        if is_new:
            self._update_status(f"New label file created: {self.current_frame_id}.json")
        else:
            self._update_status(f"Saved {self.current_frame_id}.json successfully")
        self._refresh_file_list_item(self.current_frame_id)
        if self._filter_combo.currentIndex() != 0:
            self._apply_file_filter()

    # ------------------------------------------------------------------
    # Face blurring (GDPR / data protection)
    # ------------------------------------------------------------------
    def _on_blur_faces(self):
        if not self.rgb_dir:
            QMessageBox.warning(self, "No RGB folder",
                                "Load a project with RGB images first.")
            return

        choice = QMessageBox.question(
            self, "Blur Faces — scope",
            "Apply face blurring to:\n\n"
            "  [Yes]  Current frame only\n"
            "  [No]   All frames in this dataset\n\n"
            "Warning: this permanently overwrites the image files on disk.",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if choice == QMessageBox.Cancel:
            return

        if choice == QMessageBox.Yes:
            self._blur_faces_frame(self.current_frame_id)
        else:
            ans = QMessageBox.warning(
                self, "Overwrite all images?",
                "This will permanently blur faces in every RGB image in the dataset.\n"
                "There is no undo. Continue?",
                QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel,
            )
            if ans != QMessageBox.Ok:
                return
            self._blur_faces_all_frames()

    def _blur_faces_frame(self, frame_id: str | None, *, silent: bool = False) -> int:
        """Detect and blur faces in one frame. Returns number of faces blurred."""
        if not frame_id or not self.rgb_dir:
            return 0
        rgb_path = _find_rgb_image(self.rgb_dir, frame_id)
        if rgb_path is None:
            if not silent:
                QMessageBox.information(self, "No image", f"No RGB image found for {frame_id}.")
            return 0

        img_bgr = cv2.imread(rgb_path)
        if img_bgr is None:
            return 0
        blurred, n = _detect_and_blur_faces(img_bgr)

        if n == 0:
            if not silent:
                QMessageBox.information(self, "No faces detected",
                                        f"No faces were found in {frame_id}.")
            return 0

        cv2.imwrite(rgb_path, blurred)

        if frame_id == self.current_frame_id:
            self._load_rgb_image(frame_id)

        if not silent:
            QMessageBox.information(self, "Done",
                                    f"{n} face(s) blurred and saved in {frame_id}.")
        return n

    def _blur_faces_all_frames(self):
        """Blur faces across every frame that has an RGB image."""
        all_ids = [os.path.splitext(f)[0] for f in os.listdir(self.rgb_dir)
                   if os.path.splitext(f)[1].lower() in _IMAGE_EXTS]
        all_ids.sort()

        from PyQt5.QtWidgets import QProgressDialog
        prog = QProgressDialog("Detecting and blurring faces…", "Cancel",
                               0, len(all_ids), self)
        prog.setWindowTitle("Blur Faces")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)

        total_faces = 0
        processed = 0
        for i, fid in enumerate(all_ids):
            prog.setValue(i)
            prog.setLabelText(f"Processing {fid}  ({i + 1}/{len(all_ids)})")
            if prog.wasCanceled():
                break
            total_faces += self._blur_faces_frame(fid, silent=True)
            processed += 1

        prog.setValue(len(all_ids))

        if self.current_frame_id:
            self._load_rgb_image(self.current_frame_id)

        QMessageBox.information(
            self, "Done",
            f"Processed {processed} image(s).\n"
            f"{total_faces} face(s) blurred in total."
        )

    def _on_edit_masks(self):
        if not self.current_frame_id:
            QMessageBox.warning(self, "No frame", "Please load a frame first.")
            return
        if not self.rgb_dir:
            QMessageBox.warning(self, "No RGB folder",
                                "RGB images folder is not configured.")
            return
        rgb_path = _find_rgb_image(self.rgb_dir, self.current_frame_id)
        if rgb_path is None:
            QMessageBox.warning(self, "No RGB image",
                                f"Cannot find RGB image for {self.current_frame_id}.")
            return
        if not self.labels_dir and self._coco_db is None:
            QMessageBox.warning(self, "No labels folder",
                                "Please configure a labels folder or annotations file first.")
            return

        seg_labels_dir = self._seg_labels_dir
        if seg_labels_dir is None:
            QMessageBox.warning(self, "No seg labels dir",
                                "Cannot determine segmentation labels directory.")
            return
        os.makedirs(seg_labels_dir, exist_ok=True)
        txt_path = os.path.join(seg_labels_dir, f"{self.current_frame_id}.txt")

        # Detect z-axis convention from first available point cloud
        z_backward = False
        if self.pcd_dir:
            for _ext in (".pcd", ".ply"):
                _pc = os.path.join(self.pcd_dir, f"{self.current_frame_id}{_ext}")
                if os.path.exists(_pc):
                    try:
                        import pyvista as _pv
                        _pts = _pv.read(_pc).points
                        if len(_pts) > 0 and _pts[:, 2].max() < 0:
                            z_backward = True
                    except Exception:
                        pass
                    break

        from mask_edit_dialog import MaskEditDialog
        dlg = MaskEditDialog(
            frame_id=self.current_frame_id,
            rgb_path=rgb_path,
            txt_path=txt_path,
            seg_labels_dir=seg_labels_dir,
            depth_dir=self.depth_dir,
            camera_params_dir=self.camera_params_dir,
            hdf_params=getattr(self, "_last_hdf_params", None),
            z_backward=z_backward,
            parent=self,
        )
        dlg.setStyleSheet(self.styleSheet())
        dlg.objects_generated.connect(self._on_masks_generated)
        dlg.masks_saved.connect(self._on_masks_class_updated)
        dlg.exec_()
        # Reload overlay in case masks were edited/saved without generating BBs
        self._load_rgb_image(self.current_frame_id)

    def _on_masks_generated(self, new_objs: list):
        """Replace current frame's objects with results from mask-editor pipeline."""
        if not new_objs:
            QMessageBox.warning(
                self, "No objects estimated",
                "All masks were rejected by the depth/size filters.\n"
                "Existing annotations have been kept."
            )
            return

        n_prev = len(self.current_objects)
        # Clear current objects from UI
        self._clear_active_widget()
        self._selected_obj_idx = -1
        self.current_objects.clear()

        # Write new labels
        if self._coco_db is not None:
            seg_dir = self._seg_labels_dir
            txt_path = os.path.join(seg_dir, f"{self.current_frame_id}.txt") if seg_dir else None
            self._save_to_coco(self.current_frame_id, new_objs, txt_path)
        elif self.labels_dir:
            label_path = os.path.join(self.labels_dir, f"{self.current_frame_id}.json")
            import json as _json
            data = {
                "folder":   os.path.basename(self.labels_dir),
                "filename": f"{self.current_frame_id}.pcd",
                "path":     os.path.join(self.pcd_dir or "", f"{self.current_frame_id}.pcd"),
                "objects":  new_objs,
            }
            with open(label_path, "w") as f:
                _json.dump(data, f, indent="\t")

        # Data already saved above — clear dirty flag so _load_frame's unsaved-changes
        # guard doesn't fire and accidentally overwrite the just-written annotations.
        self._dirty = False
        # Reload frame so UI reflects new objects
        self._load_frame(self.current_frame_id)

        # Record that this frame was annotated (masks → 3D BBs pipeline)
        meta_updates: dict = {"manually_modified": True,
                              "manually_modified_at": datetime.datetime.now().isoformat(timespec="seconds")}
        if self._current_user:
            meta_updates["annotated_by"] = self._current_user["username"]
        self._update_frame_meta(self.current_frame_id, meta_updates)

        skipped = n_prev - len(new_objs) if n_prev > len(new_objs) else 0
        msg = f"Generated {len(new_objs)} 3D BB(s) from masks for '{self.current_frame_id}'"
        if skipped > 0:
            msg += f" ({skipped} mask(s) rejected by depth/size filters)"
        self._update_status(msg)

    def _on_masks_class_updated(self, mask_info: list):
        """Update object names when masks are saved with changed class IDs."""
        if not mask_info or not self.current_objects:
            return
        changed = False
        for mi in mask_info:
            new_name = mi["class_name"]
            mx1, my1, mx2, my2 = [float(v) for v in mi["bbox"]]
            best_iou, best_oi = 0.0, -1
            for oi, obj in enumerate(self.current_objects):
                bb = obj.get("bbox_2d")
                if bb is None:
                    continue
                ox1, oy1, ox2, oy2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
                ix1, iy1 = max(mx1, ox1), max(my1, oy1)
                ix2, iy2 = min(mx2, ox2), min(my2, oy2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = (mx2 - mx1) * (my2 - my1) + (ox2 - ox1) * (oy2 - oy1) - inter
                iou = inter / max(union, 1.0)
                if iou > best_iou:
                    best_iou, best_oi = iou, oi
            if best_oi >= 0 and best_iou > 0.2 and self.current_objects[best_oi]["name"] != new_name:
                self.current_objects[best_oi]["name"] = new_name
                changed = True
        if changed:
            self._dirty = True
            self._rebuild_object_panel()
            self._on_save()

    def _on_verify_frame(self):
        """Admin-only: mark checked frames (or current frame) as verified."""
        if not self._current_user or self._current_user.get("role") != "admin":
            QMessageBox.warning(self, "Admin only", "Only admins can verify annotations.")
            return
        if not self._annotations_dir:
            QMessageBox.warning(self, "No project", "No annotation directory found.")
            return

        # Collect target frames — checked frames first, fall back to current frame
        frame_ids = self.get_checked_frame_ids()
        if not frame_ids:
            if self.current_frame_id is None:
                QMessageBox.information(self, "No frame", "Load a frame first.")
                return
            frame_ids = [self.current_frame_id]

        # Re-authenticate once for the whole batch
        from PyQt5.QtWidgets import QInputDialog, QLineEdit
        pw, ok = QInputDialog.getText(self, "Verify — confirm password",
                                      "Enter your password to confirm verification:",
                                      QLineEdit.Password)
        if not ok or not pw:
            return
        users = _load_project_users(self._dataset_root) if self._dataset_root else None
        if not users:
            QMessageBox.warning(self, "No users", "No users configured for this project.")
            return
        current_name = self._current_user["username"]
        admin_rec = next((u for u in users if u.get("username") == current_name), None)
        if admin_rec is None or _hash_pw(pw) != admin_rec.get("password_hash", ""):
            QMessageBox.warning(self, "Wrong password", "Incorrect password.")
            return

        # Warn once if any frames are already verified
        already = [fid for fid in frame_ids if self._frame_meta(fid).get("verified_by")]
        if already:
            _n = len(already)
            reply = QMessageBox.question(
                self, "Already verified",
                f"{_n} frame{'s' if _n > 1 else ''} {'are' if _n > 1 else 'is'} already verified.\n"
                "Re-verify (overwrite) those frames?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                frame_ids = [fid for fid in frame_ids if fid not in already]
            if not frame_ids:
                return

        updates = {
            "verified_by": self._current_user["username"],
            "verified_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        for fid in frame_ids:
            self._update_frame_meta(fid, updates)
            self._refresh_file_list_item(fid)

        n = len(frame_ids)
        self._update_status(f"{n} frame{'s' if n > 1 else ''} marked as verified.")
        if self._filter_combo.currentIndex() != 0:
            self._apply_file_filter()

    def _on_manage_users(self):
        """Open the ManageUsers dialog.
        Anyone can open it when no users are configured yet (first-time setup).
        Once users exist, only an admin can make changes."""
        if not self._dataset_root:
            QMessageBox.warning(self, "No project", "Load a project first.")
            return

        existing = _load_project_users(self._dataset_root)
        if existing and (not self._current_user or self._current_user.get("role") != "admin"):
            QMessageBox.warning(self, "Admin only", "Only admins can manage users.")
            return

        dlg = ManageUsersDialog(self._dataset_root, existing, parent=self)
        dlg.setStyleSheet(self.styleSheet())
        if dlg.exec_() == QDialog.Accepted:
            updated = _load_project_users(self._dataset_root) or []
            if self._current_user:
                current_rec = next((u for u in updated
                                    if u["username"] == self._current_user["username"]), None)
                if current_rec:
                    self._current_user["role"] = current_rec["role"]

    def _on_flag_frame(self):
        if self.current_frame_id is None:
            QMessageBox.information(self, "No frame", "Load a frame first.")
            return
        if not self._annotations_dir:
            QMessageBox.warning(self, "No project", "No annotation directory found.")
            return
        already = self._frame_meta(self.current_frame_id).get("flagged", False)
        prompt = ("Update flag comment:" if already
                  else "Describe the issue with this frame:")
        from PyQt5.QtWidgets import QInputDialog
        comment, ok = QInputDialog.getMultiLineText(self, "🚩 Flag frame", prompt,
                                                    self._frame_meta(self.current_frame_id)
                                                        .get("flag_comment", "") if already else "")
        if not ok:
            return
        updates = {
            "flagged": True,
            "flag_comment": comment.strip(),
            "flagged_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "flagged_by": self._current_user["username"] if self._current_user else "unknown",
        }
        self._update_frame_meta(self.current_frame_id, updates)
        self._update_status(f"Frame {self.current_frame_id} flagged.")
        self._apply_file_filter()

    def _on_remove_frame(self):
        if self.current_frame_id is None:
            QMessageBox.information(self, "No frame", "Load a frame first.")
            return
        if not self._current_user or self._current_user.get("role") != "admin":
            QMessageBox.warning(self, "Admin only", "Only admins can remove frames.")
            return

        frame_id = self.current_frame_id
        reply = QMessageBox.warning(
            self, "Remove frame — irreversible",
            f"This will permanently delete frame  {frame_id}  and ALL its associated data:\n"
            "  • RGB image\n  • Depth file (.npy)\n  • Depth image\n"
            "  • Point cloud (.pcd)\n  • Annotation JSON\n  • Segmentation label\n\n"
            "This action CANNOT be undone.\n\nProceed?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return

        removed = []
        failed = []

        def _try_remove(path):
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                    removed.append(os.path.basename(path))
                except Exception as e:
                    failed.append(f"{os.path.basename(path)}: {e}")

        # Annotation JSON + segmentation label
        if self._annotations_dir:
            _try_remove(os.path.join(self._annotations_dir, f"{frame_id}.json"))
            seg_dir = self._seg_labels_dir
            if seg_dir:
                _try_remove(os.path.join(seg_dir, f"{frame_id}.png"))

        # RGB image
        if self.rgb_dir:
            for ext in (".png", ".jpg", ".jpeg"):
                p = os.path.join(self.rgb_dir, f"{frame_id}{ext}")
                if os.path.isfile(p):
                    _try_remove(p)
                    break

        # Depth .npy
        if self.depth_dir:
            _try_remove(os.path.join(self.depth_dir, f"{frame_id}.npy"))

        # Depth image (PNG, sibling of depth_files dir)
        if self.depth_dir:
            depth_images_dir = os.path.join(os.path.dirname(self.depth_dir), "depth_images")
            _try_remove(os.path.join(depth_images_dir, f"{frame_id}.png"))

        # Point cloud
        if self.pcd_dir:
            for ext in (".pcd", ".ply"):
                p = os.path.join(self.pcd_dir, f"{frame_id}{ext}")
                if os.path.isfile(p):
                    _try_remove(p)
                    break

        # Remove from cache
        self._coco_frame_cache.pop(frame_id, None)

        if failed:
            QMessageBox.warning(self, "Partial deletion",
                                "Some files could not be deleted:\n" + "\n".join(failed))

        # Move to the next item in the list before refreshing
        row = self.file_list.indexOfTopLevelItem(self.file_list.currentItem())
        self.current_frame_id = None
        self._apply_file_filter()
        count = self.file_list.topLevelItemCount()
        if count > 0:
            self.file_list.setCurrentItem(self.file_list.topLevelItem(min(row, count - 1)))
        self._update_status(f"Frame {frame_id} removed ({len(removed)} files deleted).")

    def _on_file_list_double_clicked(self, item):
        frame_id = os.path.splitext(self._item_fname(item))[0]
        meta = self._frame_meta(frame_id)
        if not meta.get("flagged"):
            return
        dlg = _FlagCommentDialog(meta, parent=self)
        dlg.setStyleSheet(self.styleSheet())
        dlg.exec_()
        if dlg.resolved:
            # Remove flag from meta
            updates = {"flagged": False, "flag_comment": "", "flagged_by": "",
                       "flagged_at": "", "resolved_by": self._current_user["username"]
                       if self._current_user else "unknown",
                       "resolved_at": datetime.datetime.now().isoformat(timespec="seconds")}
            self._update_frame_meta(frame_id, updates)
            self._update_status(f"Flag resolved on {frame_id}.")
            self._apply_file_filter()

    def _on_random_frame(self):
        if not self.pcd_dir:
            self._update_status("Please load a project first")
            return
        available = [
            os.path.splitext(self._item_fname(self.file_list.topLevelItem(i)))[0]
            for i in range(self.file_list.topLevelItemCount())
        ]
        if available:
            self._load_frame(random.choice(available))

    def _check_unsaved(self) -> bool:
        if not self._dirty:
            return True
        reply = QMessageBox.question(
            self, "Unsaved changes",
            f"Save changes to '{self.current_frame_id}' before leaving?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        )
        if reply == QMessageBox.Cancel:
            return False
        if reply == QMessageBox.Save:
            self._on_save()
        return True

    def _on_prev_frame(self):
        if not self._check_unsaved():
            return
        row = self.file_list.indexOfTopLevelItem(self.file_list.currentItem())
        if row > 0:
            self.file_list.setCurrentItem(self.file_list.topLevelItem(row - 1))

    def _on_next_frame(self):
        if not self._check_unsaved():
            return
        row = self.file_list.indexOfTopLevelItem(self.file_list.currentItem())
        if row < self.file_list.topLevelItemCount() - 1:
            self.file_list.setCurrentItem(self.file_list.topLevelItem(row + 1))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_current_intrinsics(self, frame_id: str):
        """Cache color-camera intrinsics for the current frame (used for 3D→2D projection)."""
        self._cur_fx = self._cur_fy = self._cur_cx = self._cur_cy = None
        if not self.camera_params_dir:
            return
        try:
            # camera_params_dir may be a directory or a direct file path
            if os.path.isfile(self.camera_params_dir):
                p_path = self.camera_params_dir
            else:
                target = int(re.findall(r"\d+", frame_id)[-1]) if re.findall(r"\d+", frame_id) else None
                all_files = [f for f in os.listdir(self.camera_params_dir)
                             if f.lower().endswith((".json", ".npz"))]
                p_path = None
                if len(all_files) == 1:
                    p_path = os.path.join(self.camera_params_dir, all_files[0])
                elif target is not None:
                    for fname in sorted(all_files):
                        nums = re.findall(r"\d+", fname)
                        if nums and int(nums[-1]) == target:
                            p_path = os.path.join(self.camera_params_dir, fname)
                            break
                if not p_path:
                    return
            if p_path.lower().endswith(".json"):
                with open(p_path) as f:
                    data = json.load(f)
                # Prefer color camera — 3D labels are expressed in color-camera space.
                # Supports old ZED2 keys (right_camera) and new COCO keys (color_camera).
                cam = (data.get("color_camera") or data.get("right_camera")
                       or data.get("left_camera") or data)
                self._cur_fx = float(cam["fx"])
                self._cur_fy = float(cam["fy"])
                self._cur_cx = float(cam["cx"])
                self._cur_cy = float(cam["cy"])
            elif p_path.lower().endswith(".npz"):
                params = np.load(p_path)
                intr = params["rgb_intrinsics"]
                self._cur_fx = float(intr[0, 0])
                self._cur_fy = float(intr[1, 1])
                self._cur_cx = float(intr[0, 2])
                self._cur_cy = float(intr[1, 2])
        except Exception:
            pass

    def _project_3d_bbox_to_2d(self, obj: dict) -> list | None:
        """Project the 8 corners of a 3D bounding box to image coords and return the
        axis-aligned 2D bounding rect [x1, y1, x2, y2].  Returns None if intrinsics are
        not available or the object is behind the camera."""
        if self._cur_fx is None:
            return None
        try:
            L = obj["dimensions"]["length"]
            W = obj["dimensions"]["width"]
            H = obj["dimensions"]["height"]
            cX = obj["centroid"]["x"]
            cY = obj["centroid"]["y"]
            cZ = obj["centroid"]["z"]
            if cZ <= 0:
                return None
            yaw = np.deg2rad(obj["rotations"]["y"])
            corners = np.array([
                [-L/2, -H/2, -W/2], [ L/2, -H/2, -W/2],
                [ L/2,  H/2, -W/2], [-L/2,  H/2, -W/2],
                [-L/2, -H/2,  W/2], [ L/2, -H/2,  W/2],
                [ L/2,  H/2,  W/2], [-L/2,  H/2,  W/2],
            ])
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
            pts = (R @ corners.T).T + np.array([cX, cY, cZ])
            in_front = pts[:, 2] > 0
            if not in_front.any():
                return None
            u = self._cur_fx * pts[in_front, 0] / pts[in_front, 2] + self._cur_cx
            v = self._cur_fy * pts[in_front, 1] / pts[in_front, 2] + self._cur_cy
            return [int(u.min()), int(v.min()), int(u.max()), int(v.max())]
        except Exception:
            return None

    def _load_rgb_image(self, frame_id: str):
        self._orig_pixmap = None
        if not self.rgb_dir:
            self._image_label.setText("No RGB folder set")
            self._image_label.setPixmap(QPixmap())
            return
        path = _find_rgb_image(self.rgb_dir, frame_id)
        if path is None:
            self._image_label.setText(f"Image not found for {frame_id}")
            self._image_label.setPixmap(QPixmap())
            return
        rgb_arr = np.array(_PIL_Image.open(path).convert("RGB"))
        seg_dir = self._seg_labels_dir
        if seg_dir:
            txt_path = os.path.join(seg_dir, f"{frame_id}.txt")
            if os.path.exists(txt_path):
                rgb_arr = _build_seg_overlay(rgb_arr, txt_path)
        h, w = rgb_arr.shape[:2]
        qimg = QImage(rgb_arr.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        self._orig_pixmap = QPixmap.fromImage(qimg)
        self._image_label.setText("")
        self._refresh_image_pixmap()

    def _refresh_image_pixmap(self):
        if self._orig_pixmap is None:
            return
        w = self._image_label.width()
        h = self._image_label.height()
        if w <= 0 or h <= 0:
            return
        scaled = self._orig_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Overlay 2D bboxes: cyan for active selection, yellow for checked objects.
        sw, sh = scaled.width(), scaled.height()
        ow, oh = self._orig_pixmap.width(), self._orig_pixmap.height()

        checked_indices = [
            i for i in range(self.object_list.count())
            if self.object_list.item(i) and
               self.object_list.item(i).checkState() == Qt.Checked
        ]

        painter = QPainter(scaled)
        for ci in checked_indices:
            if ci == self._selected_obj_idx or ci >= len(self.current_objects):
                continue
            obj_c = self.current_objects[ci]
            bbox = obj_c.get("bbox_2d") or self._project_3d_bbox_to_2d(obj_c)
            if bbox:
                x1, y1, x2, y2 = bbox
                sx1 = int(x1 * sw / ow); sy1 = int(y1 * sh / oh)
                sx2 = int(x2 * sw / ow); sy2 = int(y2 * sh / oh)
                painter.setPen(QPen(QColor(255, 200, 0), 3))
                painter.drawRect(QRect(sx1, sy1, sx2 - sx1, sy2 - sy1))

        if 0 <= self._selected_obj_idx < len(self.current_objects):
            obj_sel = self.current_objects[self._selected_obj_idx]
            bbox = obj_sel.get("bbox_2d") or self._project_3d_bbox_to_2d(obj_sel)
            if bbox:
                x1, y1, x2, y2 = bbox
                sx1 = int(x1 * sw / ow); sy1 = int(y1 * sh / oh)
                sx2 = int(x2 * sw / ow); sy2 = int(y2 * sh / oh)
                painter.setPen(QPen(QColor(0, 200, 255), 3))
                painter.drawRect(QRect(sx1, sy1, sx2 - sx1, sy2 - sy1))
        painter.end()

        self._image_label.setPixmap(scaled)

    # ------------------------------------------------------------------
    # Image click → select matching object
    # ------------------------------------------------------------------
    def eventFilter(self, obj, event):
        if obj is self._image_label and event.type() == QEvent.Resize:
            self._refresh_image_pixmap()
        if obj is self._image_label and event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                ctrl = bool(event.modifiers() & Qt.ControlModifier)
                if ctrl:
                    self._on_image_ctrl_clicked(event.pos())
                else:
                    self._on_image_clicked(event.pos())
            elif event.button() == Qt.RightButton:
                self._on_image_right_click(event.pos())
                return True
        if obj is self.object_list.viewport() and event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton and (event.modifiers() & Qt.ControlModifier):
                item = self.object_list.itemAt(event.pos())
                if item is not None:
                    self._ensure_active_checked()
                    new_state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
                    item.setCheckState(new_state)
                    return True
        return super().eventFilter(obj, event)

    def _on_image_clicked(self, pos):
        if (self._orig_pixmap is None or self._orig_pixmap.isNull()
                or not self.current_frame_id
                or (not self.labels_dir and self._coco_db is None)):
            return

        orig_w = self._orig_pixmap.width()
        orig_h = self._orig_pixmap.height()
        label_w = self._image_label.width()
        label_h = self._image_label.height()

        # Compute scale and letterbox offset (Qt.KeepAspectRatio + AlignCenter)
        scale = min(label_w / orig_w, label_h / orig_h)
        disp_w = orig_w * scale
        disp_h = orig_h * scale
        off_x = (label_w - disp_w) / 2
        off_y = (label_h - disp_h) / 2

        img_x = (pos.x() - off_x) / scale
        img_y = (pos.y() - off_y) / scale
        if img_x < 0 or img_y < 0 or img_x >= orig_w or img_y >= orig_h:
            return

        seg_labels_dir = self._seg_labels_dir
        if not seg_labels_dir:
            return
        txt_path = os.path.join(seg_labels_dir, f"{self.current_frame_id}.txt")
        if not os.path.exists(txt_path):
            return

        import cv2 as _cv2_click
        with open(txt_path) as _f:
            lines = [l.strip() for l in _f if l.strip()]

        # Find which polygon the click landed in
        clicked_li = None
        clicked_bbox = None
        for li, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 7:
                continue
            pts = np.array([float(v) for v in parts[1:]], dtype=np.float64).reshape(-1, 2)
            pts_px = (pts * np.array([orig_w, orig_h])).astype(np.int32)
            if _cv2_click.pointPolygonTest(pts_px, (float(img_x), float(img_y)), False) >= 0:
                clicked_li = li
                clicked_bbox = (pts_px[:, 0].min(), pts_px[:, 1].min(),
                                pts_px[:, 0].max(), pts_px[:, 1].max())
                break

        if clicked_li is None or clicked_bbox is None:
            return

        # Match to the best-IoU object in the list
        mx1, my1, mx2, my2 = [float(v) for v in clicked_bbox]
        best_iou, best_idx = 0.0, -1
        for oi, obj in enumerate(self.current_objects):
            bb = obj.get("bbox_2d")
            if bb is None:
                continue
            ox1, oy1, ox2, oy2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
            ix1, iy1 = max(mx1, ox1), max(my1, oy1)
            ix2, iy2 = min(mx2, ox2), min(my2, oy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = (mx2 - mx1) * (my2 - my1) + (ox2 - ox1) * (oy2 - oy1) - inter
            iou = inter / max(union, 1.0)
            if iou > best_iou:
                best_iou, best_idx = iou, oi

        if best_idx >= 0 and best_iou > 0.2:
            self.object_list.setCurrentRow(best_idx)

    def _on_image_ctrl_clicked(self, pos):
        """Ctrl+click on the image: toggle the checkbox of the object under the cursor."""
        if (self._orig_pixmap is None or self._orig_pixmap.isNull()
                or not self.current_frame_id
                or (not self.labels_dir and self._coco_db is None)):
            return

        orig_w = self._orig_pixmap.width()
        orig_h = self._orig_pixmap.height()
        label_w = self._image_label.width()
        label_h = self._image_label.height()

        scale = min(label_w / orig_w, label_h / orig_h)
        disp_w = orig_w * scale
        disp_h = orig_h * scale
        off_x = (label_w - disp_w) / 2
        off_y = (label_h - disp_h) / 2

        img_x = (pos.x() - off_x) / scale
        img_y = (pos.y() - off_y) / scale
        if img_x < 0 or img_y < 0 or img_x >= orig_w or img_y >= orig_h:
            return

        _sld = self._seg_labels_dir
        if not _sld:
            return
        txt_path = os.path.join(_sld, f"{self.current_frame_id}.txt")
        if not os.path.exists(txt_path):
            return

        import cv2 as _cv2_ctrl
        with open(txt_path) as _f:
            lines = [l.strip() for l in _f if l.strip()]

        clicked_bbox = None
        for line in lines:
            parts = line.split()
            if len(parts) < 7:
                continue
            pts = np.array([float(v) for v in parts[1:]], dtype=np.float64).reshape(-1, 2)
            pts_px = (pts * np.array([orig_w, orig_h])).astype(np.int32)
            if _cv2_ctrl.pointPolygonTest(pts_px, (float(img_x), float(img_y)), False) >= 0:
                clicked_bbox = (pts_px[:, 0].min(), pts_px[:, 1].min(),
                                pts_px[:, 0].max(), pts_px[:, 1].max())
                break

        if clicked_bbox is None:
            return

        mx1, my1, mx2, my2 = [float(v) for v in clicked_bbox]
        best_iou, best_idx = 0.0, -1
        for oi, obj in enumerate(self.current_objects):
            bb = obj.get("bbox_2d")
            if bb is None:
                continue
            ox1, oy1, ox2, oy2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
            ix1, iy1 = max(mx1, ox1), max(my1, oy1)
            ix2, iy2 = min(mx2, ox2), min(my2, oy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = (mx2 - mx1) * (my2 - my1) + (ox2 - ox1) * (oy2 - oy1) - inter
            iou = inter / max(union, 1.0)
            if iou > best_iou:
                best_iou, best_idx = iou, oi

        if best_idx >= 0 and best_iou > 0.2:
            self._ensure_active_checked()
            item = self.object_list.item(best_idx)
            if item is not None:
                new_state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
                item.setCheckState(new_state)

    def _on_image_right_click(self, pos):
        """Right-click on the RGB preview: select the mask under the cursor, then show Copy/Paste pose menu."""
        from PyQt5.QtWidgets import QMenu
        if (self._orig_pixmap is None or self._orig_pixmap.isNull()
                or not self.current_frame_id):
            return

        # Select the object under the cursor (reuse left-click logic)
        self._on_image_clicked(pos)

        row = self.object_list.currentRow()
        if row < 0 or row >= len(self.current_objects):
            return

        self._sync_active()
        obj = self.current_objects[row]
        cen  = obj.get("centroid",  {})
        rots = obj.get("rotations", {})

        menu = QMenu(self)
        copy_act  = menu.addAction("Copy pose (X, Z, Yaw)")
        paste_act = menu.addAction("Paste pose (X, Z, Yaw)")
        paste_act.setEnabled(self._copied_pose is not None)

        action = menu.exec_(self._image_label.mapToGlobal(pos))
        if action is None:
            return

        if action is copy_act:
            self._copied_pose = {
                "x":   cen.get("x",  0.0),
                "z":   cen.get("z",  0.0),
                "yaw": rots.get("y", 0.0),
            }
            self._update_status(
                f"Copied pose from [{row}] {obj.get('name','')}: "
                f"X={self._copied_pose['x']:.3f}, Z={self._copied_pose['z']:.3f}, "
                f"Yaw={self._copied_pose['yaw']:.1f}°"
            )

        elif action is paste_act and self._copied_pose is not None:
            obj["centroid"]["x"]  = self._copied_pose["x"]
            obj["centroid"]["z"]  = self._copied_pose["z"]
            obj["rotations"]["y"] = self._copied_pose["yaw"]
            self.object_list.setCurrentRow(row)
            self._clear_active_widget()
            widget = ObjectFieldWidget(row, obj, on_change=self._on_regenerate)
            self._fields_layout.addWidget(widget)
            self._active_widget = widget
            self._selected_obj_idx = row
            self._3d_bb_manually_touched = True
            self._dirty = True
            self._render_scene()
            self._update_status(
                f"Pasted pose to [{row}] {obj.get('name','')}: "
                f"X={self._copied_pose['x']:.3f}, Z={self._copied_pose['z']:.3f}, "
                f"Yaw={self._copied_pose['yaw']:.1f}°"
            )

    # -----------------------------------------------------------------------
    # Git sync
    # -----------------------------------------------------------------------

    def _start_git_pull(self):
        if not self._git_root:
            return
        self.statusBar().showMessage("Git: checking for updates…")
        self._git_pull_worker = GitPullWorker(self._git_root, parent=self)
        self._git_pull_worker.done.connect(self._on_git_pull_done)
        self._git_pull_worker.start()

    def _on_git_pull_done(self, message: str, is_error: bool):
        self.statusBar().showMessage(message, 8000)
        if is_error:
            QMessageBox.warning(self, "Git sync", message)

    def _git_has_annotation_changes(self) -> bool:
        """Return True if there are staged/unstaged changes under LOCO_3D/."""
        if not self._git_root:
            return False
        try:
            git = _GitRun(self._git_root)
            r = git("status", "--porcelain", "--", "LOCO 3D/LOCO_3D/")
            return bool(r.stdout.strip())
        except Exception:
            return False

    def _run_git_commit_push(self):
        """Show a blocking progress dialog while committing and pushing."""
        user = self._current_user or {}
        username = user.get("username", "annotator")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        msg = f"Update annotations [{timestamp}] ({username})"

        dlg = QDialog(self)
        dlg.setWindowTitle("Git — committing…")
        dlg.setFixedWidth(400)
        layout = QVBoxLayout(dlg)
        status_label = QLabel("Starting…")
        layout.addWidget(status_label)
        bar = QProgressBar()
        bar.setRange(0, 0)   # indeterminate
        layout.addWidget(bar)
        dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowCloseButtonHint)

        worker = GitCommitPushWorker(self._git_root, msg, parent=self)
        worker.progress.connect(status_label.setText)
        worker.done.connect(lambda ok, m: (
            dlg.accept() if ok else (
                status_label.setText(m),
                bar.setRange(0, 1),
                bar.setValue(0),
            )
        ))
        worker.done.connect(lambda ok, m: (
            QMessageBox.information(dlg, "Git", m) if ok
            else QMessageBox.warning(dlg, "Git error", m)
        ))
        worker.start()
        dlg.exec_()

    # -----------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        h = self._right_splitter.height()
        # Fixed targets: list ~15%, fields enough for all 10 form rows (~340px),
        # image gets the rest.  Clamp so each panel keeps a sensible minimum.
        list_h   = max(120, h * 22 // 100)
        fields_h = max(300, h * 35 // 100)
        image_h  = max(100, h - list_h - fields_h)
        self._right_splitter.setSizes([list_h, fields_h, image_h])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_image_pixmap()

    def _on_select_all_clicked(self, checked: bool):
        new_state = Qt.Checked if checked else Qt.Unchecked
        self._select_all_cb.blockSignals(True)
        self._select_all_cb.setCheckState(new_state)
        self._select_all_cb.blockSignals(False)
        self.file_list.blockSignals(True)
        for i in range(self.file_list.topLevelItemCount()):
            self.file_list.topLevelItem(i).setCheckState(0, new_state)
        self.file_list.blockSignals(False)
        self._auto_btn.setEnabled(self.file_list.topLevelItemCount() > 0)

    def _on_item_check_changed(self, _item, _col=0):
        count = self.file_list.topLevelItemCount()
        if count == 0:
            return
        n_checked = sum(
            1 for i in range(count)
            if self.file_list.topLevelItem(i).checkState(0) == Qt.Checked
        )
        self._select_all_cb.blockSignals(True)
        if n_checked == 0:
            self._select_all_cb.setCheckState(Qt.Unchecked)
        elif n_checked == count:
            self._select_all_cb.setCheckState(Qt.Checked)
        else:
            self._select_all_cb.setCheckState(Qt.PartiallyChecked)
        self._select_all_cb.blockSignals(False)
        self._auto_btn.setEnabled(n_checked > 0 or self.current_frame_id is not None)

    def get_checked_frame_ids(self) -> list:
        return [
            os.path.splitext(self._item_fname(self.file_list.topLevelItem(i)))[0]
            for i in range(self.file_list.topLevelItemCount())
            if self.file_list.topLevelItem(i).checkState(0) == Qt.Checked
        ]

    def _sync_file_list_selection(self, frame_id: str):
        for ext in (".pcd", ".ply"):
            target = f"{frame_id}{ext}"
            for i in range(self.file_list.topLevelItemCount()):
                if self._item_fname(self.file_list.topLevelItem(i)) == target:
                    self.file_list.blockSignals(True)
                    self.file_list.setCurrentItem(self.file_list.topLevelItem(i))
                    self.file_list.blockSignals(False)
                    return

    def _refresh_file_list_item(self, frame_id: str):
        """Update color and annotator text for a single file-list row in-place."""
        for i in range(self.file_list.topLevelItemCount()):
            item = self.file_list.topLevelItem(i)
            if item and os.path.splitext(self._item_fname(item))[0] == frame_id:
                meta              = self._frame_meta(frame_id)
                flagged           = meta.get("flagged", False)
                verified_by       = meta.get("verified_by", "")
                manually_modified = meta.get("manually_modified", False)
                annotated_by      = meta.get("annotated_by", "") if manually_modified else ""
                has_ann           = self._frame_has_annotations(frame_id)

                if not has_ann:
                    annotator_col = "no annotation"
                elif manually_modified and annotated_by:
                    annotator_col = annotated_by
                else:
                    annotator_col = "auto"

                icon_col = ""
                if verified_by:
                    icon_col += "✓"
                if flagged:
                    icon_col += "🚩"

                if flagged:
                    color = QColor("#e05252")
                elif verified_by:
                    color = QColor("#4ec94e")
                elif manually_modified:
                    color = QColor("#d4d4d4")
                elif has_ann:
                    color = QColor("#9cdcfe")
                else:
                    color = QColor("#666666")

                self.file_list.blockSignals(True)
                item.setText(1, annotator_col)
                item.setText(2, icon_col)
                for col in range(3):
                    item.setForeground(col, color)
                self.file_list.blockSignals(False)
                return

    # ------------------------------------------------------------------
    # Autonomous 3D BB generation
    # ------------------------------------------------------------------
    def _on_auto_bbox(self):
        # 1. Collect target frames (checked, or fall back to current frame)
        frame_ids = self.get_checked_frame_ids()
        if not frame_ids:
            if self.current_frame_id:
                frame_ids = [self.current_frame_id]
            else:
                QMessageBox.warning(self, "No frame", "Please load a frame first.")
                return

        # 2. Validate required folders
        missing = []
        if not self.rgb_dir:
            missing.append("RGB images folder")
        if not self.depth_dir:
            missing.append("Depth maps folder")
        if not self.camera_params_dir:
            missing.append("Camera parameters folder")
        if not self.labels_dir and self._coco_db is None:
            missing.append("Labels folder or COCO annotations file")
        if missing:
            QMessageBox.warning(self, "Missing folders",
                                "Please configure these project folders first:\n• " +
                                "\n• ".join(missing))
            return

        from pose_estimation_pipeline import (
            find_depth_file, load_depth_file, find_camera_params_file,
            load_intrinsics, align_depth_to_color,
            apply_hist_depth_filter, estimate_3d_pose, make_label_object,
        )
        from auto_bbox_dialog import AutoBBoxValidationDialog

        # 3. Build per-frame path list and loader for the validation dialog
        frame_paths = []
        for fid in frame_ids:
            r = _find_rgb_image(self.rgb_dir, fid) if self.rgb_dir else None
            d = find_depth_file(self.depth_dir, fid)
            p = find_camera_params_file(self.camera_params_dir, fid)
            frame_paths.append((fid, r, d, p))

        def _load_frame(idx):
            fid, r_path, d_path, p_path = frame_paths[idx]
            rgb = np.array(_PIL_Image.open(r_path).convert("RGB")) if r_path else np.zeros((480, 640, 3), dtype=np.uint8)
            if d_path:
                depth_raw = load_depth_file(d_path)
                if p_path and p_path.lower().endswith(".json"):
                    # Reproject depth into color-camera frame so bboxes align exactly
                    depth, fx_i, fy_i, cx_i, cy_i = align_depth_to_color(depth_raw, p_path)
                else:
                    depth = depth_raw
                    fx_i, fy_i, cx_i, cy_i = load_intrinsics(p_path) if p_path else (644.145, 644.145, 640., 360.)
            else:
                depth = np.zeros((480, 640), dtype=np.float32)
                fx_i, fy_i, cx_i, cy_i = (644.145, 644.145, 640., 360.)
            return rgb, depth, fx_i, fy_i, cx_i, cy_i

        # Detect Z-axis convention from point cloud of the first frame
        z_backward = False
        first_id = frame_ids[0]
        if self.pcd_dir:
            for _ext in (".pcd", ".ply"):
                _pc_path = os.path.join(self.pcd_dir, f"{first_id}{_ext}")
                if os.path.exists(_pc_path):
                    try:
                        _pts = pv.read(_pc_path).points
                        if len(_pts) > 0 and _pts[:, 2].max() < 0:
                            z_backward = True
                    except Exception:
                        pass
                    break

        # Show validation dialog — user browses to a good frame then runs segmentation
        dlg = AutoBBoxValidationDialog(
            _load_frame, len(frame_ids),
            default_yolo_path=YOLO_MODEL_PATH, parent=self
        )
        dlg.setStyleSheet(self.styleSheet())
        if dlg.exec_() != QDialog.Accepted or dlg.result is None:
            return

        validated_idx = dlg.validated_frame_idx
        # Capture the validated frame's FID *before* frame_ids is filtered below.
        # Index-based comparison breaks when manually-annotated frames are removed.
        validated_fid = frame_ids[validated_idx] if validated_idx < len(frame_ids) else None
        detections    = dlg._detections

        hdf_params     = dlg._hdf_params
        conf_threshold = dlg.conf_threshold
        self._last_hdf_params = hdf_params  # share with mask editor

        # ── helper: run pipeline on one detection, return label dict or None ──
        def _process_detection(box, det_mask, dep_full, rgb_shape, fx_i, fy_i, cx_i, cy_i, cls_name):
            import cv2 as _cv2_pd
            x1, y1, x2, y2 = box.astype(int)
            # Scale bbox from RGB space to depth space (may differ, e.g. 1920×1080 → 1280×720)
            rgb_h, rgb_w = rgb_shape
            dep_h, dep_w = dep_full.shape[:2]
            sx = dep_w / rgb_w
            sy = dep_h / rgb_h
            dx1, dy1 = int(x1 * sx), int(y1 * sy)
            dx2, dy2 = int(x2 * sx), int(y2 * sy)
            dep_crop = dep_full[dy1:dy2, dx1:dx2].astype(float)
            if dep_crop.size == 0:
                return None
            if det_mask is not None:
                if rgb_h != dep_h or rgb_w != dep_w:
                    full_mask_dep = _cv2_pd.resize(det_mask.astype(np.uint8), (dep_w, dep_h),
                                                   interpolation=_cv2_pd.INTER_NEAREST)
                else:
                    full_mask_dep = det_mask.astype(np.uint8)
                raw_mask = full_mask_dep[dy1:dy2, dx1:dx2]
            else:
                raw_mask = np.ones(dep_crop.shape, dtype=np.uint8)
            # Normalize to mm
            valid_px = dep_crop[dep_crop > 0]
            if valid_px.size > 0 and valid_px.max() <= 100:
                dep_crop = dep_crop * 1000.0
            dep_masked = np.where(raw_mask, dep_crop, 0)
            filtered, *_ = apply_hist_depth_filter(
                dep_masked,
                resolution=hdf_params["resolution"],
                max_height_percent=hdf_params["max_height_percent"],
                ignore_background=hdf_params.get("ignore_background", False),
            )
            mask_crop = (filtered > 0).astype(np.uint8)
            try:
                _, center, dims, yaw_deg, _ = estimate_3d_pose(
                    filtered, mask_crop, dx1, dy1, fx_i, fy_i, cx_i, cy_i,
                    class_dims=_class_dims_for(cls_name),
                    class_dims_range=_class_dims_range_for(cls_name))
            except Exception:
                return None
            if z_backward:
                center[1] = -center[1]
                center[2] = -center[2]
                yaw_deg   = -yaw_deg
            obj = make_label_object(cls_name, center, dims, yaw_deg)
            obj["bbox_2d"] = [int(x1), int(y1), int(x2), int(y2)]
            return obj

        def _get_class_names(frame_detections):
            if "class_name" in frame_detections.data:
                return [str(n) for n in frame_detections.data["class_name"]]
            return ["object"] * len(frame_detections)

        def _get_masks(frame_detections):
            if frame_detections.mask is not None:
                return list(frame_detections.mask)
            return [None] * len(frame_detections)

        # ── batch worker runs all frame processing in a background thread ────
        save_fn            = self._save_auto_result_to_file
        current_fid_cap    = self.current_frame_id
        validated_fid_cap  = validated_fid      # FID-based; survives frame_ids filtering
        _frame_orig_idx    = {fp[0]: i for i, fp in enumerate(frame_paths)}
        load_frame_cap     = _load_frame
        coco_db_cap       = self._coco_db  # None if old format
        annotations_dir_cap = self._annotations_dir
        blur_faces_cap    = dlg.blur_faces

        seg_labels_dir_cap = self._seg_labels_dir
        if seg_labels_dir_cap:
            os.makedirs(seg_labels_dir_cap, exist_ok=True)

        labels_dir_cap = self.labels_dir

        # Frames marked manually_modified are protected from accidental batch overwrites.
        # Single-frame case: ask the user explicitly. Batch case: skip and inform.
        _manually_annotated = [
            fid for fid in frame_ids
            if self._frame_meta(fid).get("manually_modified")
        ]
        if _manually_annotated:
            if len(frame_ids) == 1:
                # Single frame targeted — ask the user whether to overwrite.
                _reply = QMessageBox.question(
                    self, "Frame has manual annotations",
                    "This frame was manually annotated.\n\n"
                    "Do you want to overwrite it with auto-annotation?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if _reply != QMessageBox.Yes:
                    return
                # User confirmed — proceed without skipping.
            else:
                frame_ids = [fid for fid in frame_ids if fid not in _manually_annotated]
                _nm = len(_manually_annotated)
                QMessageBox.information(
                    self, "Skipping manually annotated frames",
                    f"{_nm} frame{'s' if _nm > 1 else ''} {'have' if _nm > 1 else 'has'} "
                    "manual annotations and will be skipped.\n\n"
                    "To re-run auto-annotation on one of those frames, "
                    "load it and click Generate 3D BB with only that frame active.",
                )
                if not frame_ids:
                    return

        _existing_frames = [
            fid for fid in frame_ids
            if (annotations_dir_cap and os.path.exists(os.path.join(annotations_dir_cap, f"{fid}.json")))
            or (labels_dir_cap and os.path.exists(os.path.join(labels_dir_cap, f"{fid}.json")))
            or (seg_labels_dir_cap and os.path.exists(os.path.join(seg_labels_dir_cap, f"{fid}.txt")))
        ]
        replace_mode_cap = False
        if _existing_frames:
            _n = len(_existing_frames)
            _reply = QMessageBox.warning(
                self, "Existing predictions",
                f"{_n} frame{'s' if _n > 1 else ''} already "
                f"{'have' if _n > 1 else 'has'} existing predictions.\n"
                "Running detection will replace all results for those frames.\n\nContinue?",
                QMessageBox.Ok | QMessageBox.Cancel,
            )
            if _reply == QMessageBox.Cancel:
                return
            replace_mode_cap = True

        self._batch_replace_mode = replace_mode_cap
        self._batch_frame_ids    = frame_ids
        self._batch_coco_accumulated: dict = {}  # fid → {objs: [], txt_lines: []}

        def _run_batch(worker):
            current_frame_objs: list = []
            total = 0
            written_fids: list = []

            # Replace mode: wipe existing data for all target frames upfront
            if replace_mode_cap and labels_dir_cap:
                for _fid in frame_ids:
                    _jp = os.path.join(labels_dir_cap, f"{_fid}.json")
                    if os.path.exists(_jp):
                        os.remove(_jp)
                    if seg_labels_dir_cap:
                        _tp = os.path.join(seg_labels_dir_cap, f"{_fid}.txt")
                        if os.path.exists(_tp):
                            os.remove(_tp)

            for frame_idx, fid in enumerate(frame_ids):
                if worker._cancel:
                    break
                worker.frame_started.emit(frame_idx, fid)
                # Look up the ORIGINAL position of this fid in frame_paths so
                # _load_frame uses the correct path even after frame_ids was filtered.
                _orig_idx = _frame_orig_idx.get(fid, frame_idx)
                try:
                    if fid == validated_fid_cap:
                        # Re-use the image and detections from the validation step
                        rgb_f, dep_f, fx_f, fy_f, cx_f, cy_f = load_frame_cap(_orig_idx)
                        frame_det = detections
                    else:
                        rgb_f, dep_f, fx_f, fy_f, cx_f, cy_f = load_frame_cap(_orig_idx)
                        frame_det = dlg.run_on_image(rgb_f)

                    # Blur faces in this frame's RGB before continuing
                    if blur_faces_cap:
                        rgb_path_f = frame_paths[_orig_idx][1]  # (fid, rgb, depth, params)
                        if rgb_path_f:
                            img_bgr = cv2.imread(rgb_path_f)
                            if img_bgr is not None:
                                blurred, n_faces = _detect_and_blur_faces(img_bgr)
                                if n_faces:
                                    cv2.imwrite(rgb_path_f, blurred)

                    if frame_det.confidence is not None:
                        frame_det = frame_det[frame_det.confidence >= conf_threshold]
                    if len(frame_det) == 0:
                        continue

                    cls_names  = _get_class_names(frame_det)
                    masks      = _get_masks(frame_det)
                    frame_objs: list = []
                    frame_obj_masks: list = []

                    # For the validated frame in single-object mode, identify the
                    # validated detection by bbox overlap so the stored yaw matches
                    # what the user saw in step 4.  Match by FID (survives frame_ids
                    # filtering) and bbox IoU (survives confidence-filter index shifts).
                    _validated_bbox = None
                    if (fid == validated_fid_cap
                            and not dlg._multi_mode
                            and dlg.result is not None):
                        _vb = dlg.result.get("bbox_2d")
                        if _vb and len(_vb) == 4:
                            _validated_bbox = [int(v) for v in _vb]

                    for det_idx, (box, det_mask, cls_name) in enumerate(
                            zip(frame_det.xyxy, masks, cls_names)):
                        worker.object_progress.emit(det_idx, len(frame_det))
                        cls_name = _canonical_class_name(cls_name)

                        # Check whether this detection is the validated one.
                        _use_validated = False
                        if _validated_bbox is not None:
                            bx1, by1, bx2, by2 = box.astype(int)
                            vx1, vy1, vx2, vy2 = _validated_bbox
                            ix1 = max(bx1, vx1); iy1 = max(by1, vy1)
                            ix2 = min(bx2, vx2); iy2 = min(by2, vy2)
                            if ix2 > ix1 and iy2 > iy1:
                                inter = (ix2 - ix1) * (iy2 - iy1)
                                area_b = max((bx2 - bx1) * (by2 - by1), 1)
                                if inter / area_b > 0.5:
                                    _use_validated = True

                        if _use_validated:
                            obj = dict(dlg.result)
                            obj["rotations"] = dict(obj["rotations"])
                            bx1, by1, bx2, by2 = box.astype(int)
                            obj["bbox_2d"] = [int(bx1), int(by1), int(bx2), int(by2)]
                            # Dialog result has raw yaw (no z_backward); apply it now.
                            if z_backward:
                                cen = dict(obj.get("centroid", {}))
                                cen["y"] = -cen.get("y", 0.0)
                                cen["z"] = -cen.get("z", 0.0)
                                obj["centroid"] = cen
                                obj["rotations"]["y"] = -obj["rotations"]["y"]
                        else:
                            obj = _process_detection(
                                box, det_mask, dep_f, rgb_f.shape[:2], fx_f, fy_f, cx_f, cy_f, cls_name)
                        if obj is not None:
                            frame_objs.append(obj)
                            frame_obj_masks.append(det_mask)

                    # Build segmentation TXT lines (used for both old and COCO format)
                    _txt_lines: list = []
                    if seg_labels_dir_cap and frame_objs:
                        import cv2 as _cv2_sg
                        _rgb_h, _rgb_w = rgb_f.shape[:2]
                        for _obj_m, _m in zip(frame_objs, frame_obj_masks):
                            if _m is None:
                                continue
                            _mask_u8 = (_m > 0).astype(np.uint8) * 255
                            if _mask_u8.shape != (_rgb_h, _rgb_w):
                                _mask_u8 = _cv2_sg.resize(
                                    _mask_u8, (_rgb_w, _rgb_h),
                                    interpolation=_cv2_sg.INTER_NEAREST)
                            _conts, _ = _cv2_sg.findContours(
                                _mask_u8, _cv2_sg.RETR_EXTERNAL,
                                _cv2_sg.CHAIN_APPROX_SIMPLE)
                            if _conts:
                                _lc = max(_conts, key=_cv2_sg.contourArea)
                                _eps = 0.003 * _cv2_sg.arcLength(_lc, True)
                                _ap = _cv2_sg.approxPolyDP(_lc, _eps, True)
                                if len(_ap) >= 3:
                                    _pts = _ap.reshape(-1, 2).astype(np.float64)
                                    _pts[:, 0] /= _rgb_w
                                    _pts[:, 1] /= _rgb_h
                                    _cid = _seg_class_id(
                                        _obj_m.get("name", "object"),
                                        seg_labels_dir_cap)
                                    _txt_lines.append(
                                        f"{_cid} " +
                                        " ".join(f"{v:.6f}" for v in _pts.flatten()))
                        if _txt_lines:
                            with open(os.path.join(seg_labels_dir_cap, f"{fid}.txt"), "w") as _f:
                                _f.write("\n".join(_txt_lines) + "\n")

                    if fid == current_fid_cap:
                        current_frame_objs.extend(frame_objs)
                    elif coco_db_cap is not None:
                        # COCO mode: accumulate; main thread saves after batch completes
                        self._batch_coco_accumulated[fid] = {
                            "objs": list(frame_objs),
                            "txt_lines": list(_txt_lines),
                        }
                    else:
                        for obj in frame_objs:
                            save_fn(fid, obj)

                    total += len(frame_objs)
                    if frame_objs:
                        written_fids.append(fid)
                except Exception:
                    continue

            if worker._cancel:
                worker.batch_cancelled.emit(written_fids)
            else:
                worker.batch_done.emit(current_frame_objs, total)

        self._batch_prog    = _AutoBBoxProgressDialog(len(frame_ids), parent=self)
        self._batch_n_frames = len(frame_ids)
        self._batch_prog.show()

        self._batch_worker = _BatchWorker(_run_batch)
        self._batch_worker.frame_started.connect(self._batch_prog.set_frame)
        self._batch_worker.object_progress.connect(self._batch_prog.set_object)
        self._batch_worker.batch_done.connect(self._finish_batch)
        self._batch_worker.batch_cancelled.connect(self._on_batch_cancelled)
        self._batch_prog._cancel_btn.clicked.connect(self._cancel_batch)
        self._batch_worker.start()

    def _finish_batch(self, current_frame_objs: list, total_saved: int):
        """Slot called on the main thread when _BatchWorker finishes."""
        self._batch_prog.close()
        self._sync_active()

        # COCO format: save all accumulated non-current-frame results now (main thread)
        if self._coco_db is not None:
            accumulated = getattr(self, '_batch_coco_accumulated', {})
            seg_dir = self._seg_labels_dir
            for fid, data in accumulated.items():
                self._coco_frame_cache.pop(fid, None)  # force reload after write
                txt_path = os.path.join(seg_dir, f"{fid}.txt") if seg_dir else None
                self._save_to_coco(fid, data["objs"], txt_path)
            self._batch_coco_accumulated = {}

        # Replace mode: clear existing objects for the current frame before adding new results
        _replace_mode_active = (
            getattr(self, '_batch_replace_mode', False) and
            self.current_frame_id in getattr(self, '_batch_frame_ids', [])
        )
        if _replace_mode_active:
            self.current_objects.clear()
            self.object_list.blockSignals(True)
            self.object_list.clear()
            self.object_list.blockSignals(False)
            self._clear_active_widget()   # properly hide and remove old widget from layout
            self._selected_obj_idx = -1

        self.object_list.blockSignals(True)
        for obj in current_frame_objs:
            self.current_objects.append(obj)
            _fi = QListWidgetItem(f"[{len(self.current_objects) - 1}]  {obj['name']}")
            _fi.setFlags(_fi.flags() | Qt.ItemIsUserCheckable)
            _fi.setCheckState(Qt.Unchecked)
            self.object_list.addItem(_fi)
        self.object_list.blockSignals(False)
        if current_frame_objs:
            self._dirty = True
            if _replace_mode_active and self.object_list.count() > 0:
                # Auto-select first new object so the user can edit it immediately
                self.object_list.setCurrentRow(0)
        self._render_scene()
        if self.current_frame_id:
            self._load_rgb_image(self.current_frame_id)  # pick up freshly saved seg mask
        self._update_status(
            f"Done — {total_saved} object(s) saved across {self._batch_n_frames} frame(s)"
        )

    def _cancel_batch(self):
        if hasattr(self, '_batch_worker'):
            self._batch_worker.cancel()

    def _on_batch_cancelled(self, written_fids: list):
        self._batch_prog.close()
        seg_dir = self._seg_labels_dir
        for fid in written_fids:
            if self._coco_db is not None and self._annotations_dir:
                # COCO: delete the per-frame JSON and evict from cache
                jp = os.path.join(self._annotations_dir, f"{fid}.json")
                if os.path.exists(jp):
                    os.remove(jp)
                self._coco_frame_cache.pop(fid, None)
            elif self.labels_dir:
                jp = os.path.join(self.labels_dir, f"{fid}.json")
                if os.path.exists(jp):
                    os.remove(jp)
            if seg_dir:
                tp = os.path.join(seg_dir, f"{fid}.txt")
                if os.path.exists(tp):
                    os.remove(tp)
        self._batch_coco_accumulated = {}
        self._update_status(
            f"Batch cancelled — removed results for {len(written_fids)} frame(s)")

    def _save_auto_result_to_file(self, frame_id: str, obj: dict):
        """Append obj to the label JSON for frame_id (creates file if needed)."""
        if not self.labels_dir:
            return
        os.makedirs(self.labels_dir, exist_ok=True)
        label_path = os.path.join(self.labels_dir, f"{frame_id}.json")
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                data = json.load(f)
            data.setdefault("objects", []).append(obj)
        else:
            data = {
                "folder":   os.path.basename(self.labels_dir),
                "filename": f"{frame_id}.pcd",
                "path":     os.path.join(self.pcd_dir or "", f"{frame_id}.pcd"),
                "objects":  [obj],
            }
        with open(label_path, "w") as f:
            json.dump(data, f, indent="\t")

    def _update_status(self, msg: str):
        self.statusBar().showMessage(msg)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S and (event.modifiers() & Qt.ControlModifier):
            self._on_save()
            return
        if event.key() == Qt.Key_Delete:
            focused = QApplication.focusWidget()
            # Only fire if focus is not inside a text/spin input
            from PyQt5.QtWidgets import QAbstractSpinBox, QLineEdit, QTextEdit
            if not isinstance(focused, (QAbstractSpinBox, QLineEdit, QTextEdit)):
                self._on_remove_object()
                return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved changes",
                f"Save changes to '{self.current_frame_id}' before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                event.ignore()
                return
            if reply == QMessageBox.Save:
                self._on_save()

        if self._git_root and self._git_has_annotation_changes():
            reply = QMessageBox.question(
                self, "Commit annotations",
                "You have uncommitted annotation changes.\n"
                "Commit and push them to git before closing?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                event.ignore()
                return
            if reply == QMessageBox.Yes:
                self._run_git_commit_push()

        self.plotter.close()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    win = LabelEditorWindow()
    win.show()
    sys.exit(app.exec_())

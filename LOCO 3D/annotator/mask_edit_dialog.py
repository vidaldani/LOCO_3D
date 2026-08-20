"""Interactive segmentation-mask editor with polygon editing and 3D-BB generation."""
from __future__ import annotations
import os
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QDialog, QGraphicsScene, QGraphicsView, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsLineItem,
    QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QComboBox, QSizePolicy, QMessageBox, QApplication,
    QFrame, QMenu,
)
from PyQt5.QtCore import Qt, QPointF, QEvent, pyqtSignal
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence
from PyQt5.QtGui import (
    QPolygonF, QColor, QPen, QBrush, QPainter, QPixmap, QImage, QCursor,
    QTransform,
)
from PIL import Image as _PIL_Image

HANDLE_R = 6
SNAP_R   = 16   # scene-px: distance to first point that triggers loop-close
EDGE_THR = 10   # scene-px: distance to edge line that triggers point insertion

_PALETTE = [
    (0, 200, 255), (255, 100, 0), (0, 255, 100),
    (200, 0, 255), (255, 200, 0), (0, 100, 255),
]
_FILL_A = 70


# ---------------------------------------------------------------------------
# TXT helpers
# ---------------------------------------------------------------------------
def _load_classes(seg_labels_dir: str) -> list[str]:
    p = os.path.join(seg_labels_dir, "classes.txt")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [l.strip() for l in f if l.strip()]


def _load_txt(txt_path: str) -> list[dict]:
    if not os.path.exists(txt_path):
        return []
    result = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cid = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]
            result.append({"class_id": cid, "pts": pts})
    return result


def _save_txt(txt_path: str, masks: list[dict]):
    lines = []
    for m in masks:
        pts_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in m["pts"])
        lines.append(f"{m['class_id']} {pts_str}")
    os.makedirs(os.path.dirname(os.path.abspath(txt_path)), exist_ok=True)
    if lines:
        with open(txt_path, "w") as f:
            f.write("\n".join(lines) + "\n")
    elif os.path.exists(txt_path):
        os.remove(txt_path)


# ---------------------------------------------------------------------------
# PointHandle — draggable vertex
# ---------------------------------------------------------------------------
class PointHandle(QGraphicsEllipseItem):
    def __init__(self, x: float, y: float, mask_item: "MaskItem"):
        super().__init__(-HANDLE_R, -HANDLE_R, 2 * HANDLE_R, 2 * HANDLE_R)
        self.setPos(x, y)
        self.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemSendsGeometryChanges,
        )
        self.setBrush(QBrush(QColor(255, 255, 0)))
        self.setPen(QPen(QColor(0, 0, 0), 1))
        self.setZValue(4)
        self._mask = mask_item

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self._mask.sync_from_handles()
        return super().itemChange(change, value)


# ---------------------------------------------------------------------------
# MaskItem — one polygon and its visual representation
# ---------------------------------------------------------------------------
class MaskItem:
    def __init__(self, scene: QGraphicsScene, pts_px: list[tuple],
                 class_id: int, color: tuple, on_modified=None):
        self.scene = scene
        self.pts_px: list[tuple] = list(pts_px)
        self.class_id = class_id
        self.color = color
        self.on_modified = on_modified  # callable() called on any point edit
        self._poly = None
        self._handles: list[PointHandle] = []
        self._edges: list[QGraphicsLineItem] = []
        self._moving = False             # suppresses per-handle itemChange during move_by
        self._rebuild_poly()

    # ── polygon display ────────────────────────────────────────────────────
    def _rebuild_poly(self):
        pts = [QPointF(x, y) for x, y in self.pts_px]
        poly = QPolygonF(pts)
        c = self.color
        if self._poly is None:
            self._poly = self.scene.addPolygon(
                poly,
                QPen(QColor(c[0], c[1], c[2]), 2),
                QBrush(QColor(c[0], c[1], c[2], _FILL_A)),
            )
            self._poly.setZValue(1)
        else:
            self._poly.setPolygon(poly)

    def update_color(self, color: tuple):
        self.color = color
        c = color
        if self._poly:
            self._poly.setPen(QPen(QColor(c[0], c[1], c[2]), 2))
            self._poly.setBrush(QBrush(QColor(c[0], c[1], c[2], _FILL_A)))

    # ── handles & edges ────────────────────────────────────────────────────
    def set_active(self, active: bool):
        if active:
            self._show_handles()
        else:
            self._hide_handles()

    def _show_handles(self):
        self._hide_handles()
        for x, y in self.pts_px:
            h = PointHandle(x, y, self)
            self.scene.addItem(h)
            self._handles.append(h)
        self._rebuild_edges()

    def _hide_handles(self):
        for h in self._handles:
            self.scene.removeItem(h)
        self._handles.clear()
        self._clear_edges()

    def _clear_edges(self):
        for e in self._edges:
            self.scene.removeItem(e)
        self._edges.clear()

    def _rebuild_edges(self):
        n = len(self.pts_px)
        # Grow / shrink edge list in place to avoid flickering during drag
        while len(self._edges) < n:
            e = self.scene.addLine(0, 0, 0, 0, QPen(QColor(255, 255, 0), 2))
            e.setZValue(2)
            self._edges.append(e)
        while len(self._edges) > n:
            self.scene.removeItem(self._edges.pop())
        for i in range(n):
            x1, y1 = self.pts_px[i]
            x2, y2 = self.pts_px[(i + 1) % n]
            self._edges[i].setLine(x1, y1, x2, y2)

    def sync_from_handles(self):
        if self._moving:
            return                        # bulk move in progress; skip per-handle rebuild
        self.pts_px = [(h.pos().x(), h.pos().y()) for h in self._handles]
        self._rebuild_poly()
        self._rebuild_edges()
        if self.on_modified:
            self.on_modified()

    def move_by(self, dx: float, dy: float):
        """Translate the entire mask by (dx, dy) pixels."""
        self._moving = True
        for h in self._handles:
            h.setPos(h.pos().x() + dx, h.pos().y() + dy)
        self._moving = False
        self.pts_px = [(h.pos().x(), h.pos().y()) for h in self._handles]
        self._rebuild_poly()
        self._rebuild_edges()
        if self.on_modified:
            self.on_modified()

    # ── point manipulation ─────────────────────────────────────────────────
    def remove_handle(self, handle: PointHandle):
        if len(self.pts_px) <= 3:
            return
        idx = self._handles.index(handle)
        self.scene.removeItem(handle)
        self._handles.pop(idx)
        del self.pts_px[idx]
        self._rebuild_poly()
        self._rebuild_edges()
        if self.on_modified:
            self.on_modified()

    def insert_on_edge(self, edge_idx: int, x: float, y: float):
        self.pts_px.insert(edge_idx + 1, (x, y))
        self._show_handles()
        self._rebuild_poly()
        if self.on_modified:
            self.on_modified()

    # ── hit detection ──────────────────────────────────────────────────────
    def edge_hit(self, pos: QPointF) -> tuple | None:
        """Return (edge_idx, ix, iy) if pos is within EDGE_THR of a segment."""
        px, py = pos.x(), pos.y()
        n = len(self.pts_px)
        for i in range(n):
            x1, y1 = self.pts_px[i]
            x2, y2 = self.pts_px[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            dlen2 = dx * dx + dy * dy
            if dlen2 < 1e-6:
                continue
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / dlen2))
            cx, cy = x1 + t * dx, y1 + t * dy
            if ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 <= EDGE_THR:
                return (i, cx, cy)
        return None

    def near_handle(self, pos: QPointF) -> bool:
        px, py = pos.x(), pos.y()
        for h in self._handles:
            if ((px - h.pos().x()) ** 2 + (py - h.pos().y()) ** 2) ** 0.5 <= HANDLE_R * 1.5:
                return True
        return False

    # ── cleanup ────────────────────────────────────────────────────────────
    def remove_from_scene(self):
        self._hide_handles()
        if self._poly:
            self.scene.removeItem(self._poly)
            self._poly = None

    def get_normalized(self, w: int, h: int) -> list[tuple]:
        return [(x / w, y / h) for x, y in self.pts_px]


# ---------------------------------------------------------------------------
# MaskScene — custom scene with draw + edit logic
# ---------------------------------------------------------------------------
class MaskScene(QGraphicsScene):
    polygon_finished = pyqtSignal(list)   # list of (x_px, y_px)
    drawing_cancelled = pyqtSignal()
    mask_selection_requested = pyqtSignal(int)  # index in _all_masks
    delete_active_mask = pyqtSignal()           # Del with no handle selected
    copy_mask_requested = pyqtSignal()
    paste_mask_at_requested = pyqtSignal(float, float)  # scene x, y
    blur_region_requested = pyqtSignal(float, float, float, float)  # x1,y1,x2,y2 in px
    blur_mode_exit_requested = pyqtSignal()

    def __init__(self, img_w: int, img_h: int, parent=None):
        super().__init__(parent)
        self._img_w = img_w
        self._img_h = img_h
        self._drawing = False
        self._draw_pts: list[tuple] = []
        self._draw_items = []        # temp QGraphicsItems
        self._preview_line = None
        self._first_handle = None    # highlighted first point handle
        self._active_mask: MaskItem | None = None
        self._all_masks: list = []   # reference to MaskEditDialog._masks (shared list)
        self._mask_drag_active = False
        self._mask_drag_last: QPointF | None = None
        self._has_clipboard = False  # updated by MaskEditDialog whenever clipboard changes
        # Blur-rect mode
        self._blur_mode = False
        self._blur_drag_start: QPointF | None = None
        self._blur_rect_item = None

    # ── drawing mode ───────────────────────────────────────────────────────
    def start_drawing(self):
        self._drawing = True
        self._draw_pts.clear()
        self._draw_items.clear()
        self._preview_line = None
        self._first_handle = None
        # Deactivate any selected mask so handles don't interfere
        if self._active_mask:
            self._active_mask.set_active(False)

    def cancel_drawing(self):
        for item in self._draw_items:
            self.removeItem(item)
        self._draw_items.clear()
        if self._preview_line:
            self.removeItem(self._preview_line)
            self._preview_line = None
        self._draw_pts.clear()
        self._drawing = False
        self._first_handle = None
        self.drawing_cancelled.emit()

    def _close_polygon(self):
        pts = list(self._draw_pts)
        self.cancel_drawing()   # cleans up, resets _drawing
        if len(pts) >= 3:
            self.polygon_finished.emit(pts)

    # ── active mask ────────────────────────────────────────────────────────
    def set_active_mask(self, mask: MaskItem | None):
        if self._active_mask and self._active_mask is not mask:
            self._active_mask.set_active(False)
        self._active_mask = mask
        if mask:
            mask.set_active(True)

    # ── events ─────────────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        pos = event.scenePos()

        # Blur-rect mode: start drag
        if self._blur_mode and event.button() == Qt.LeftButton:
            self._blur_drag_start = pos
            self._blur_rect_item = self.addRect(
                pos.x(), pos.y(), 0, 0,
                QPen(QColor(255, 80, 80), 2, Qt.DashLine),
                QBrush(QColor(255, 80, 80, 40)),
            )
            self._blur_rect_item.setZValue(10)
            event.accept()
            return

        # Right-click: handle-delete or context menu
        if event.button() == Qt.RightButton:
            if self._active_mask:
                clicked = self.itemAt(pos, QTransform())
                if isinstance(clicked, PointHandle) and clicked._mask is self._active_mask:
                    self._active_mask.remove_handle(clicked)
                    return

                # Inside the active mask polygon: Copy / Remove (+ optional Paste)
                if self._active_mask._poly is not None and self._active_mask._poly.contains(pos):
                    menu = QMenu()
                    copy_act   = menu.addAction("Copy Mask")
                    remove_act = menu.addAction("Remove Mask")
                    paste_act  = menu.addAction("Paste Mask Here") if self._has_clipboard else None
                    chosen = menu.exec_(QCursor.pos())
                    if chosen == copy_act:
                        self.copy_mask_requested.emit()
                    elif chosen == remove_act:
                        self.delete_active_mask.emit()
                    elif paste_act and chosen == paste_act:
                        self.paste_mask_at_requested.emit(pos.x(), pos.y())
                    return

            # Outside any active-mask area: Paste Here (if clipboard)
            if self._has_clipboard:
                menu = QMenu()
                paste_act = menu.addAction("Paste Mask Here")
                chosen = menu.exec_(QCursor.pos())
                if chosen == paste_act:
                    self.paste_mask_at_requested.emit(pos.x(), pos.y())
                return

            super().mousePressEvent(event)
            return

        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        x = max(0.0, min(float(self._img_w), pos.x()))
        y = max(0.0, min(float(self._img_h), pos.y()))

        if self._drawing:
            # Snap-close when >= 3 points and near first
            if len(self._draw_pts) >= 3:
                x0, y0 = self._draw_pts[0]
                if ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 <= SNAP_R:
                    self._close_polygon()
                    return
            # Add point
            self._draw_pts.append((x, y))
            r = HANDLE_R
            h = self.addEllipse(x - r, y - r, 2 * r, 2 * r,
                                 QPen(Qt.black, 1), QBrush(QColor(255, 255, 0)))
            h.setZValue(5)
            self._draw_items.append(h)
            if len(self._draw_pts) == 1:
                self._first_handle = h
            elif len(self._draw_pts) >= 2:
                px2, py2 = self._draw_pts[-2]
                ln = self.addLine(px2, py2, x, y, QPen(QColor(255, 255, 0), 2))
                ln.setZValue(4)
                self._draw_items.append(ln)
            return

        # Edit mode: edge insertion on active mask (only if not clicking on a handle)
        if self._active_mask:
            clicked = self.itemAt(pos, QTransform())
            if not (isinstance(clicked, PointHandle) and clicked._mask is self._active_mask):
                hit = self._active_mask.edge_hit(pos)
                if hit:
                    edge_idx, ix, iy = hit
                    self._active_mask.insert_on_edge(edge_idx, ix, iy)
                    return

        # Handle priority: if the click lands on/near a handle of the active mask,
        # let Qt process it as a drag — don't switch to another mask's polygon.
        if self._active_mask and self._active_mask.near_handle(pos):
            super().mousePressEvent(event)
            return

        # Click inside the active mask polygon (not on a handle/edge) → drag whole mask
        if (self._active_mask and self._active_mask._poly is not None
                and self._active_mask._poly.contains(pos)):
            self._mask_drag_active = True
            self._mask_drag_last = pos
            if self.views():
                self.views()[0].viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        # Click on a non-active polygon → select that mask
        for mi_idx, mi in enumerate(self._all_masks):
            if mi is self._active_mask:
                continue
            if mi._poly is not None and mi._poly.contains(pos):
                self.mask_selection_requested.emit(mi_idx)
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.scenePos()

        if self._blur_mode and self._blur_drag_start is not None and self._blur_rect_item is not None:
            x1 = min(self._blur_drag_start.x(), pos.x())
            y1 = min(self._blur_drag_start.y(), pos.y())
            x2 = max(self._blur_drag_start.x(), pos.x())
            y2 = max(self._blur_drag_start.y(), pos.y())
            self._blur_rect_item.setRect(x1, y1, x2 - x1, y2 - y1)
            event.accept()
            return

        if self._mask_drag_active and self._active_mask and self._mask_drag_last is not None:
            delta = pos - self._mask_drag_last
            self._mask_drag_last = pos
            self._active_mask.move_by(delta.x(), delta.y())
            event.accept()
            return

        if self._drawing and self._draw_pts:
            # Update preview line
            if self._preview_line:
                self.removeItem(self._preview_line)
                self._preview_line = None
            px2, py2 = self._draw_pts[-1]
            self._preview_line = self.addLine(
                px2, py2, pos.x(), pos.y(),
                QPen(QColor(255, 220, 0, 160), 1, Qt.DashLine),
            )
            self._preview_line.setZValue(4)
            # Snap indicator: highlight first handle and change cursor
            if len(self._draw_pts) >= 3 and self._first_handle:
                x0, y0 = self._draw_pts[0]
                near = ((pos.x() - x0) ** 2 + (pos.y() - y0) ** 2) ** 0.5 <= SNAP_R
                self._first_handle.setBrush(
                    QBrush(QColor(0, 255, 0) if near else QColor(255, 255, 0))
                )
                if self.views():
                    self.views()[0].viewport().setCursor(
                        Qt.PointingHandCursor if near else Qt.CrossCursor
                    )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._blur_mode and event.button() == Qt.LeftButton and self._blur_drag_start is not None:
            pos = event.scenePos()
            if self._blur_rect_item:
                self.removeItem(self._blur_rect_item)
                self._blur_rect_item = None
            x1 = min(self._blur_drag_start.x(), pos.x())
            y1 = min(self._blur_drag_start.y(), pos.y())
            x2 = max(self._blur_drag_start.x(), pos.x())
            y2 = max(self._blur_drag_start.y(), pos.y())
            self._blur_drag_start = None
            if x2 - x1 > 5 and y2 - y1 > 5:
                self.blur_region_requested.emit(x1, y1, x2, y2)
            event.accept()
            return

        if self._mask_drag_active:
            self._mask_drag_active = False
            self._mask_drag_last = None
            if self.views():
                self.views()[0].viewport().setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self._blur_mode:
                # Cancel any in-progress drag and exit blur mode
                if self._blur_rect_item:
                    self.removeItem(self._blur_rect_item)
                    self._blur_rect_item = None
                self._blur_drag_start = None
                self.blur_mode_exit_requested.emit()
                return
            if self._drawing:
                self.cancel_drawing()
                return
        if event.key() == Qt.Key_Delete and self._active_mask:
            selected = [
                item for item in self.selectedItems()
                if isinstance(item, PointHandle) and item._mask is self._active_mask
            ]
            if selected:
                for h in selected:
                    self._active_mask.remove_handle(h)
            else:
                # No handle selected — delete the whole active mask
                self.delete_active_mask.emit()
            return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Zoomable view
# ---------------------------------------------------------------------------
class _ZoomView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event):
        # Forward key events to scene
        self.scene().keyPressEvent(event)
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# MaskEditDialog — main dialog
# ---------------------------------------------------------------------------
class MaskEditDialog(QDialog):
    """
    Dialog for interactive segmentation mask editing.
    Emits objects_generated(list[dict]) when "Generate 3D BBs" completes.
    """
    objects_generated = pyqtSignal(list)
    masks_saved = pyqtSignal(list)   # list of {class_name, bbox} for each mask

    def __init__(self, frame_id: str, rgb_path: str, txt_path: str,
                 seg_labels_dir: str,
                 depth_dir: str | None = None,
                 camera_params_dir: str | None = None,
                 hdf_params: dict | None = None,
                 z_backward: bool = False,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit Masks — {frame_id}")
        self.resize(1280, 820)

        self._frame_id      = frame_id
        self._rgb_path      = rgb_path
        self._txt_path      = txt_path
        self._seg_labels_dir = seg_labels_dir
        self._depth_dir     = depth_dir
        self._cam_dir       = camera_params_dir
        self._hdf_params    = hdf_params or {"resolution": 1, "max_height_percent": 5}
        self._z_backward    = z_backward
        self._dirty         = False
        self._image_modified = False   # True when blur regions have been painted
        self._clipboard: list[tuple[float, float]] | None = None
        self._clipboard_class: int = 0

        # Load image
        _pil = _PIL_Image.open(rgb_path).convert("RGB")
        self._img_w, self._img_h = _pil.size
        _arr = np.array(_pil)
        _qimg = QImage(_arr.tobytes(), self._img_w, self._img_h,
                       self._img_w * 3, QImage.Format_RGB888)
        self._bg_pixmap = QPixmap.fromImage(_qimg)
        # Keep the numpy array so blur can operate on it without re-decoding
        self._img_arr: np.ndarray = np.array(_pil)  # H×W×3 RGB uint8

        # Classes
        self._classes = _load_classes(seg_labels_dir)
        if not self._classes:
            self._classes = ["object"]

        # Scene
        self._scene = MaskScene(self._img_w, self._img_h)
        self._bg_item = self._scene.addPixmap(self._bg_pixmap)
        self._bg_item.setZValue(0)
        self._scene.polygon_finished.connect(self._on_polygon_finished)
        self._scene.drawing_cancelled.connect(self._on_drawing_cancelled)
        self._scene.copy_mask_requested.connect(self._copy_active_mask)
        self._scene.paste_mask_at_requested.connect(self._paste_mask_at)
        self._scene.blur_region_requested.connect(self._on_blur_region)
        self._scene.blur_mode_exit_requested.connect(self._exit_blur_mode)

        # Build mask items from TXT
        self._masks: list[MaskItem] = []
        for i, md in enumerate(_load_txt(txt_path)):
            pts_px = [(x * self._img_w, y * self._img_h) for x, y in md["pts"]]
            mi = MaskItem(self._scene, pts_px, md["class_id"],
                          _PALETTE[i % len(_PALETTE)],
                          on_modified=self._mark_dirty)
            self._masks.append(mi)

        # Share the masks list with the scene for click-to-select
        self._scene._all_masks = self._masks

        # View
        self._view = _ZoomView(self._scene)

        # ── Right panel ───────────────────────────────────────────────────
        right = QFrame()
        right.setFixedWidth(210)
        right.setFrameShape(QFrame.StyledPanel)
        rl = QVBoxLayout(right)
        rl.setSpacing(6)
        rl.setContentsMargins(6, 6, 6, 6)

        rl.addWidget(QLabel("Masks:"))
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_mask_selected)
        self._list.installEventFilter(self)
        self._scene.mask_selection_requested.connect(self._list.setCurrentRow)
        self._scene.delete_active_mask.connect(self._on_delete_mask)
        rl.addWidget(self._list, 1)

        rl.addWidget(QLabel("Class:"))
        self._class_box = QComboBox()
        self._class_box.addItems(self._classes)
        self._class_box.currentIndexChanged.connect(self._on_class_changed)
        rl.addWidget(self._class_box)

        self._new_btn = QPushButton("New Mask")
        self._new_btn.clicked.connect(self._toggle_drawing)
        rl.addWidget(self._new_btn)

        del_mask_btn = QPushButton("Delete Mask")
        del_mask_btn.clicked.connect(self._on_delete_mask)
        rl.addWidget(del_mask_btn)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.HLine); sep1.setFrameShadow(QFrame.Sunken)
        rl.addWidget(sep1)

        self._blur_btn = QPushButton("Blur Region")
        self._blur_btn.setCheckable(True)
        self._blur_btn.setToolTip("Draw a rectangle to blur a face/area manually")
        self._blur_btn.clicked.connect(self._toggle_blur_mode)
        rl.addWidget(self._blur_btn)

        gen_btn = QPushButton("Generate 3D BBs")
        gen_btn.clicked.connect(self._on_generate)
        rl.addWidget(gen_btn)

        rl.addStretch(1)

        hint = QLabel("Drag handles to move.\nDrag polygon to reposition.\nClick edge to add point.\nRight-click handle to delete.\nDel: remove mask/point.\nCtrl+C / Ctrl+V: copy/paste mask.\nBlur Region: drag a rect to permanently blur that area.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 10px;")
        rl.addWidget(hint)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine); sep2.setFrameShadow(QFrame.Sunken)
        rl.addWidget(sep2)

        accept_btn = QPushButton("Accept")
        accept_btn.clicked.connect(self._on_accept_changes)
        rl.addWidget(accept_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        rl.addWidget(cancel_btn)

        # ── Main layout ───────────────────────────────────────────────────
        main = QHBoxLayout(self)
        main.setContentsMargins(6, 6, 6, 6)
        main.setSpacing(6)
        main.addWidget(self._view, 1)
        main.addWidget(right, 0)

        # Dialog-level Esc shortcut so it works regardless of which widget has focus
        esc = QShortcut(QKeySequence(Qt.Key_Escape), self)
        esc.activated.connect(self._on_esc)

        self._populate_list()

    def showEvent(self, event):
        super().showEvent(event)
        QApplication.processEvents()
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    # ── list management ────────────────────────────────────────────────────
    def _populate_list(self):
        self._list.blockSignals(True)
        self._list.clear()
        for i, mi in enumerate(self._masks):
            cid = mi.class_id
            name = self._classes[cid] if cid < len(self._classes) else f"class_{cid}"
            self._list.addItem(f"[{i}]  {name}")
        self._list.blockSignals(False)

    def _on_mask_selected(self, row: int):
        self._scene.set_active_mask(self._masks[row] if row >= 0 else None)
        if row >= 0:
            cid = self._masks[row].class_id
            self._class_box.blockSignals(True)
            self._class_box.setCurrentIndex(min(cid, self._class_box.count() - 1))
            self._class_box.blockSignals(False)

    def _on_class_changed(self, idx: int):
        row = self._list.currentRow()
        if row < 0:
            return
        self._masks[row].class_id = idx
        self._dirty = True
        self._populate_list()
        self._list.blockSignals(True)
        self._list.setCurrentRow(row)
        self._list.blockSignals(False)

    def _on_delete_mask(self):
        row = self._list.currentRow()
        if row < 0:
            return
        self._masks[row].remove_from_scene()
        del self._masks[row]
        self._scene._active_mask = None
        self._dirty = True
        self._populate_list()
        if self._masks:
            self._list.setCurrentRow(min(row, len(self._masks) - 1))

    # ── drawing mode ────────────────────────────────────────────────────────
    def _toggle_drawing(self):
        if self._scene._drawing:
            self._scene.cancel_drawing()
        else:
            self._scene.start_drawing()
            self._new_btn.setText("Cancel Drawing  (Esc)")
            self._view.setDragMode(QGraphicsView.NoDrag)
            self._view.viewport().setCursor(Qt.CrossCursor)

    def _toggle_blur_mode(self, checked: bool):
        self._scene._blur_mode = checked
        if checked:
            # Disable polygon drawing and switch view to NoDrag so the rect drag works
            if self._scene._drawing:
                self._scene.cancel_drawing()
                self._new_btn.setText("New Mask")
            self._view.setDragMode(self._view.NoDrag)
            self._view.viewport().setCursor(Qt.CrossCursor)
        else:
            self._view.setDragMode(self._view.ScrollHandDrag)
            self._view.viewport().unsetCursor()

    def _exit_blur_mode(self):
        """Deactivate blur mode and restore normal interaction."""
        self._blur_btn.setChecked(False)
        self._toggle_blur_mode(False)

    def _on_blur_region(self, x1: float, y1: float, x2: float, y2: float):
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(self._img_w, int(x2)), min(self._img_h, int(y2))
        if ix2 <= ix1 or iy2 <= iy1:
            return
        roi = self._img_arr[iy1:iy2, ix1:ix2]
        k = max(51, (max(ix2 - ix1, iy2 - iy1) // 5) | 1)  # odd, proportional to region
        self._img_arr[iy1:iy2, ix1:ix2] = cv2.GaussianBlur(roi, (k, k), 0)
        # Update scene preview only — file is written on Accept, not here
        qimg = QImage(self._img_arr.tobytes(), self._img_w, self._img_h,
                      self._img_w * 3, QImage.Format_RGB888)
        self._bg_pixmap = QPixmap.fromImage(qimg)
        self._bg_item.setPixmap(self._bg_pixmap)
        self._dirty = True
        self._image_modified = True
        self._exit_blur_mode()

    def _on_esc(self):
        """Dialog-level Esc handler — works regardless of focus."""
        if self._scene._blur_mode:
            # Cancel any in-progress blur drag then exit blur mode
            if self._scene._blur_rect_item:
                self._scene.removeItem(self._scene._blur_rect_item)
                self._scene._blur_rect_item = None
            self._scene._blur_drag_start = None
            self._exit_blur_mode()
        elif self._scene._drawing:
            self._scene.cancel_drawing()   # emits drawing_cancelled → _on_drawing_cancelled

    def _on_drawing_cancelled(self):
        self._new_btn.setText("New Mask")
        self._view.setDragMode(QGraphicsView.ScrollHandDrag)
        # unsetCursor lets Qt's ScrollHandDrag manage the cursor naturally;
        # setting an explicit cursor here gets overridden on the next mouse-move event
        self._view.viewport().unsetCursor()
        # Re-activate the previously selected mask
        row = self._list.currentRow()
        if row >= 0:
            self._scene.set_active_mask(self._masks[row])

    def _on_polygon_finished(self, pts: list):
        self._on_drawing_cancelled()   # resets button/cursor
        if len(pts) < 3:
            return
        c = _PALETTE[len(self._masks) % len(_PALETTE)]
        current_class = max(0, self._class_box.currentIndex())
        mi = MaskItem(self._scene, pts, current_class, c, on_modified=self._mark_dirty)
        self._masks.append(mi)
        self._dirty = True
        self._populate_list()
        self._list.setCurrentRow(len(self._masks) - 1)

    # ── save ────────────────────────────────────────────────────────────────
    def _collect_mask_data(self) -> list[dict]:
        return [
            {"class_id": mi.class_id,
             "pts": mi.get_normalized(self._img_w, self._img_h)}
            for mi in self._masks
        ]

    def _mask_class_info(self) -> list:
        """Return [{class_name, bbox:[x1,y1,x2,y2]}] for each mask."""
        info = []
        for mi in self._masks:
            if not mi.pts_px:
                continue
            xs = [p[0] for p in mi.pts_px]
            ys = [p[1] for p in mi.pts_px]
            cid = mi.class_id
            info.append({
                "class_name": self._classes[cid] if cid < len(self._classes) else "object",
                "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
            })
        return info

    def _mark_dirty(self):
        self._dirty = True

    def _save_txt(self):
        _save_txt(self._txt_path, self._collect_mask_data())
        if self._image_modified:
            _PIL_Image.fromarray(self._img_arr).save(self._rgb_path)
            self._image_modified = False
        self._dirty = False

    def _on_accept_changes(self):
        from PyQt5.QtWidgets import QDialogButtonBox as _DBB
        dlg = QDialog(self)
        dlg.setWindowTitle("Accept changes")
        vl = QVBoxLayout(dlg)
        vl.addWidget(QLabel("Masks have been modified. What would you like to do?"))
        btn_row = QHBoxLayout()
        btn_save = QPushButton("Save masks only")
        btn_gen  = QPushButton("Save + estimate 3D BBs")
        btn_cancel = QPushButton("Cancel")
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_gen)
        btn_row.addWidget(btn_cancel)
        vl.addLayout(btn_row)
        choice = [None]
        btn_save.clicked.connect(lambda: (choice.__setitem__(0, "save"), dlg.accept()))
        btn_gen.clicked.connect(lambda: (choice.__setitem__(0, "gen"), dlg.accept()))
        btn_cancel.clicked.connect(dlg.reject)
        dlg.exec_()
        if choice[0] is None:
            return
        self._save_txt()
        if choice[0] == "gen":
            self._on_generate()
        else:
            self.masks_saved.emit(self._mask_class_info())
            self.accept()

    def _copy_active_mask(self):
        active = self._scene._active_mask
        if active is None:
            return
        self._clipboard = list(active.pts_px)
        self._clipboard_class = active.class_id
        self._scene._has_clipboard = True

    def _paste_mask(self):
        if not self._clipboard:
            return
        view_pos = self._view.mapFromGlobal(QCursor.pos())
        scene_pos = self._view.mapToScene(view_pos)
        self._paste_mask_at(scene_pos.x(), scene_pos.y())

    def _paste_mask_at(self, cx: float, cy: float):
        """Paste clipboard mask centred at the given scene position."""
        if not self._clipboard:
            return
        xs = [p[0] for p in self._clipboard]
        ys = [p[1] for p in self._clipboard]
        src_cx = sum(xs) / len(xs)
        src_cy = sum(ys) / len(ys)
        dx, dy = cx - src_cx, cy - src_cy
        pts = [(x + dx, y + dy) for x, y in self._clipboard]
        self._do_paste(pts)

    def _do_paste(self, pts: list[tuple[float, float]]):
        color = _PALETTE[len(self._masks) % len(_PALETTE)]
        mi = MaskItem(self._scene, pts, self._clipboard_class, color,
                      on_modified=self._mark_dirty)
        self._masks.append(mi)
        self._scene._all_masks = self._masks
        self._populate_list()
        self._list.setCurrentRow(len(self._masks) - 1)
        self._mark_dirty()

    def eventFilter(self, obj, event):
        if obj is self._list and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Delete:
                self._on_delete_mask()
                return True
            if event.key() == Qt.Key_C and (event.modifiers() & Qt.ControlModifier):
                self._copy_active_mask()
                return True
            if event.key() == Qt.Key_V and (event.modifiers() & Qt.ControlModifier):
                self._paste_mask()
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self._on_delete_mask()
            return
        if event.key() == Qt.Key_C and (event.modifiers() & Qt.ControlModifier):
            self._copy_active_mask()
            return
        if event.key() == Qt.Key_V and (event.modifiers() & Qt.ControlModifier):
            self._paste_mask()
            return
        # Intercept Escape so it goes through reject() with the unsaved-changes prompt
        if event.key() == Qt.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def reject(self):
        if self._masks_dirty():
            reply = QMessageBox.question(
                self, "Discard changes?",
                "You have unsaved mask changes. Discard them and close?",
                QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Discard:
                return
        super().reject()

    def closeEvent(self, event):
        if self._masks_dirty():
            reply = QMessageBox.question(
                self, "Discard changes?",
                "You have unsaved mask changes. Discard them and close?",
                QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Discard:
                event.ignore()
                return
        event.accept()

    def _masks_dirty(self) -> bool:
        return self._dirty

    # ── generate 3D BBs ────────────────────────────────────────────────────
    def _on_generate(self):
        if not self._masks:
            QMessageBox.warning(self, "No masks", "Draw at least one mask first.")
            return

        reply = QMessageBox.warning(
            self, "Replace existing results",
            "This will replace all existing 3D bounding boxes for this frame.\n\nContinue?",
            QMessageBox.Ok | QMessageBox.Cancel,
        )
        if reply == QMessageBox.Cancel:
            return

        self._save_txt()

        if not self._depth_dir or not self._cam_dir:
            QMessageBox.critical(self, "Error",
                                 "Depth or camera-parameters folder not configured.")
            return

        import cv2
        from pose_estimation_pipeline import (
            find_depth_file, load_depth_file, find_camera_params_file,
            load_intrinsics, align_depth_to_color,
        )

        dep_path = find_depth_file(self._depth_dir, self._frame_id)
        if dep_path is None:
            QMessageBox.critical(self, "Error",
                                 f"No depth file found for frame '{self._frame_id}'.")
            return
        p_path = find_camera_params_file(self._cam_dir, self._frame_id)

        depth_raw = load_depth_file(dep_path)
        if p_path and p_path.lower().endswith(".json"):
            depth, fx, fy, cx, cy = align_depth_to_color(depth_raw, p_path)
        else:
            depth = depth_raw
            fx, fy, cx, cy = (load_intrinsics(p_path) if p_path
                              else (644.145, 644.145, 640., 360.))

        rgb_arr = np.array(_PIL_Image.open(self._rgb_path).convert("RGB"))

        def _load_fn(idx):
            return rgb_arr, depth, fx, fy, cx, cy

        # Rasterize each polygon into a full-resolution binary mask
        precomputed_list = []
        for mi in self._masks:
            if len(mi.pts_px) < 3:
                continue
            pts_arr = np.array(mi.pts_px, dtype=np.int32)
            full_mask = np.zeros((self._img_h, self._img_w), dtype=np.uint8)
            cv2.fillPoly(full_mask, [pts_arr], 255)
            x1 = max(0, int(pts_arr[:, 0].min()))
            y1 = max(0, int(pts_arr[:, 1].min()))
            x2 = min(self._img_w, int(pts_arr[:, 0].max()))
            y2 = min(self._img_h, int(pts_arr[:, 1].max()))
            cid = mi.class_id
            cls_name = self._classes[cid] if cid < len(self._classes) else "object"
            precomputed_list.append({
                "bbox": [x1, y1, x2, y2],
                "mask": full_mask,
                "class_name": cls_name,
            })

        if not precomputed_list:
            QMessageBox.warning(self, "No valid masks",
                                "No masks with 3 or more points to process.")
            return

        from auto_bbox_dialog import AutoBBoxValidationDialog
        from PyQt5.QtWidgets import QDialog
        dlg = AutoBBoxValidationDialog(
            _load_fn, n_frames=1,
            precomputed=precomputed_list,
            parent=self,
        )
        dlg.setStyleSheet(self.styleSheet())
        if dlg.exec_() != QDialog.Accepted or not dlg.result:
            return

        objs = dlg.result if isinstance(dlg.result, list) else [dlg.result]

        if self._z_backward:
            for obj in objs:
                obj["centroid"]["y"] = -obj["centroid"]["y"]
                obj["centroid"]["z"] = -obj["centroid"]["z"]
                obj["rotations"]["y"] = -obj["rotations"]["y"]

        self.objects_generated.emit(objs)
        self.accept()

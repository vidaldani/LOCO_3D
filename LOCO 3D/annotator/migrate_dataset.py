#!/usr/bin/env python3
"""
Migrate 01/ dataset from per-frame files to COCO-style per-frame JSON format.

New layout after migration:
  01/
  ├── dataset_info.json
  ├── calib/
  │   └── d435_intrinsics.json
  ├── annotations/
  │   ├── categories.json     (shared category list)
  │   ├── 010004.json         (per-frame annotation files)
  │   ├── 010005.json
  │   └── ...
  ├── images/            (was: color/)
  ├── depth/             (unchanged)
  ├── depth_files/       (unchanged)
  └── point_clouds/      (was: point_cloud/)

Old directories (labels_3d/, seg_labels/, intrinsics/, color/, point_cloud/)
are left in place so you can verify and delete them manually.

Usage:
  python3 migrate_dataset.py [root]                   # full migration
  python3 migrate_dataset.py [root] --dry-run         # preview only
  python3 migrate_dataset.py [root] --split-coco      # split existing instances_all.json
"""
import os
import json
import math
import shutil
import argparse
from PIL import Image

CATEGORIES = [
    {"id": 1, "name": "stillage",     "supercategory": "load_carrier"},
    {"id": 2, "name": "pallet_truck", "supercategory": "vehicle"},
    {"id": 3, "name": "pallet",       "supercategory": "load_carrier"},
    {"id": 4, "name": "forklift",     "supercategory": "vehicle"},
    {"id": 5, "name": "small_load_carrier", "supercategory": "container"},
]
NAME_TO_CAT = {c["name"]: c["id"] for c in CATEGORIES}
OLD_CLASS_IDX_TO_NAME = {0: "stillage", 1: "pallet", 2: "pallet_truck"}


def _iou(b1, b2):
    """b1, b2 are [x1, y1, x2, y2]"""
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def _polygon_area(pts):
    """Shoelace area for [(x,y), ...] pixel coords."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def load_seg_txt(txt_path):
    """Return list of {class_id, pts_norm} from YOLO polygon TXT."""
    out = []
    if not os.path.exists(txt_path):
        return out
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            vals = [float(x) for x in parts[1:]]
            pts = [(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
            out.append({"class_id": cid, "pts_norm": pts})
    return out


def _frame_annotations(fid, img_w, img_h, old_labels3d_dir, old_seg_dir):
    """Build the annotations list for one frame from old per-frame files."""
    lbl3d_path = os.path.join(old_labels3d_dir, f"{fid}.json")
    objs_3d = []
    if os.path.exists(lbl3d_path):
        with open(lbl3d_path) as f:
            objs_3d = json.load(f).get("objects", [])

    seg_path = os.path.join(old_seg_dir, f"{fid}.txt")
    segs = load_seg_txt(seg_path)

    annotations = []
    ann_id = 1
    seg_used = [False] * len(segs)

    for obj in objs_3d:
        b2d = obj.get("bbox_2d", [0, 0, 0, 0])  # [x1, y1, x2, y2]
        name = obj.get("name", "object")
        cat_id = NAME_TO_CAT.get(name, 0)

        best_iou, best_si = 0.0, -1
        for si, seg in enumerate(segs):
            if seg_used[si]:
                continue
            pts_px = [(p[0] * img_w, p[1] * img_h) for p in seg["pts_norm"]]
            xs, ys = [p[0] for p in pts_px], [p[1] for p in pts_px]
            sb = [min(xs), min(ys), max(xs), max(ys)]
            iou = _iou(b2d, sb)
            if iou > best_iou:
                best_iou, best_si = iou, si

        segmentation, area = [], 0
        if best_si >= 0 and best_iou > 0.2:
            seg_used[best_si] = True
            pts_px = [(p[0] * img_w, p[1] * img_h) for p in segs[best_si]["pts_norm"]]
            segmentation = [[c for pt in pts_px for c in pt]]
            area = int(_polygon_area(pts_px))

        if area == 0:
            area = int((b2d[2] - b2d[0]) * (b2d[3] - b2d[1]))

        cen  = obj.get("centroid", {})
        dims = obj.get("dimensions", {})
        rots = obj.get("rotations", {})
        yaw_rad = rots.get("y", 0.0) * math.pi / 180.0

        annotations.append({
            "id": ann_id,
            "category_id": cat_id,
            "bbox": [int(b2d[0]), int(b2d[1]),
                     int(b2d[2] - b2d[0]), int(b2d[3] - b2d[1])],
            "segmentation": segmentation,
            "area": area,
            "iscrowd": 0,
            "bbox_3d": {
                "center":     {"x": cen.get("x", 0.0), "y": cen.get("y", 0.0), "z": cen.get("z", 0.0)},
                "dimensions": {"height": dims.get("height", 0.0),
                               "width":  dims.get("width",  0.0),
                               "length": dims.get("length", 0.0)},
                "yaw": yaw_rad,
            },
        })
        ann_id += 1

    # Unmatched seg polygons (no 3D label)
    for si, seg in enumerate(segs):
        if seg_used[si]:
            continue
        class_name = OLD_CLASS_IDX_TO_NAME.get(seg["class_id"], f"class_{seg['class_id']}")
        cat_id = NAME_TO_CAT.get(class_name, 0)
        pts_px = [(p[0] * img_w, p[1] * img_h) for p in seg["pts_norm"]]
        xs, ys = [p[0] for p in pts_px], [p[1] for p in pts_px]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        annotations.append({
            "id": ann_id,
            "category_id": cat_id,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "segmentation": [[c for pt in pts_px for c in pt]],
            "area": int(_polygon_area(pts_px)),
            "iscrowd": 0,
            "bbox_3d": None,
        })
        ann_id += 1

    return annotations


def split_coco_to_per_frame(root: str, delete_unified: bool = False):
    """Split an existing instances_all.json into per-frame JSON files."""
    ann_dir  = os.path.join(root, "annotations")
    unified  = os.path.join(ann_dir, "instances_all.json")
    cat_file = os.path.join(ann_dir, "categories.json")

    if not os.path.isfile(unified):
        print(f"Not found: {unified}")
        return

    with open(unified) as f:
        db = json.load(f)

    cats = db.get("categories", CATEGORIES)
    with open(cat_file, "w") as f:
        json.dump(cats, f, indent=2)
    print(f"Wrote {cat_file}")

    imgid_to_entry = {im["id"]: im for im in db.get("images", [])}
    imgid_to_anns: dict = {}
    for ann in db.get("annotations", []):
        imgid_to_anns.setdefault(ann["image_id"], []).append(ann)

    written = 0
    for img_id, img_entry in imgid_to_entry.items():
        fid = os.path.splitext(img_entry["file_name"])[0]
        anns = imgid_to_anns.get(img_id, [])
        # Strip image_id from each annotation (redundant in per-frame files)
        clean_anns = [{k: v for k, v in a.items() if k != "image_id"} for a in anns]
        frame_doc = {
            "file_name": img_entry["file_name"],
            "width":     img_entry.get("width", 1920),
            "height":    img_entry.get("height", 1080),
            "annotations": clean_anns,
        }
        out_path = os.path.join(ann_dir, f"{fid}.json")
        with open(out_path, "w") as f:
            json.dump(frame_doc, f, indent=2)
        written += 1

    print(f"Wrote {written} per-frame annotation files to {ann_dir}/")

    if delete_unified:
        os.remove(unified)
        print(f"Deleted {unified}")


def migrate(root: str, dry_run: bool = False):
    old_color    = os.path.join(root, "color")
    old_pcd      = os.path.join(root, "point_cloud")
    old_labels3d = os.path.join(root, "labels_3d")
    old_seg      = os.path.join(root, "seg_labels")
    old_intrin   = os.path.join(root, "intrinsics")

    new_images   = os.path.join(root, "images")
    new_pcd      = os.path.join(root, "point_clouds")
    new_calib    = os.path.join(root, "calib")
    new_annot    = os.path.join(root, "annotations")

    if dry_run:
        print("[DRY RUN] No files will be moved or created.")

    for d in [new_images, new_pcd, new_calib, new_annot]:
        if not dry_run:
            os.makedirs(d, exist_ok=True)
        print(f"  mkdir {d}")

    # Move color/ → images/
    if os.path.isdir(old_color):
        files = sorted(os.listdir(old_color))
        print(f"Moving {len(files)} images: color/ → images/")
        for fname in files:
            src = os.path.join(old_color, fname)
            dst = os.path.join(new_images, fname)
            if not os.path.exists(dst):
                if not dry_run:
                    shutil.move(src, dst)
            else:
                print(f"  SKIP {fname} (already in images/)")

    # Move point_cloud/ → point_clouds/
    if os.path.isdir(old_pcd):
        files = sorted(os.listdir(old_pcd))
        print(f"Moving {len(files)} point clouds: point_cloud/ → point_clouds/")
        for fname in files:
            src = os.path.join(old_pcd, fname)
            dst = os.path.join(new_pcd, fname)
            if not os.path.exists(dst):
                if not dry_run:
                    shutil.move(src, dst)

    # Create calib/d435_intrinsics.json
    calib_out = os.path.join(new_calib, "d435_intrinsics.json")
    if os.path.isdir(old_intrin) and not os.path.exists(calib_out):
        intrin_files = sorted(f for f in os.listdir(old_intrin) if f.endswith(".json"))
        if intrin_files:
            src = os.path.join(old_intrin, intrin_files[0])
            print(f"Creating calib/d435_intrinsics.json from {intrin_files[0]}")
            with open(src) as f:
                data = json.load(f)
            lc = data.get("left_camera", {})
            rc = data.get("right_camera", {})
            ext = data.get("extrinsics", {})
            calib = {
                "depth_camera": {
                    "fx": lc.get("fx"), "fy": lc.get("fy"),
                    "cx": lc.get("cx"), "cy": lc.get("cy"),
                    "width":  lc.get("resolution", {}).get("width"),
                    "height": lc.get("resolution", {}).get("height"),
                },
                "color_camera": {
                    "fx": rc.get("fx"), "fy": rc.get("fy"),
                    "cx": rc.get("cx"), "cy": rc.get("cy"),
                    "width":  rc.get("resolution", {}).get("width"),
                    "height": rc.get("resolution", {}).get("height"),
                },
                "extrinsics": {
                    "baseline_m": ext.get("translation", {}).get("x", 0.0),
                },
            }
            if not dry_run:
                with open(calib_out, "w") as f:
                    json.dump(calib, f, indent=2)

    # Discover frame IDs from images/ (or color/ as fallback)
    img_dir = new_images if os.path.isdir(new_images) else old_color
    frame_ids = sorted(
        os.path.splitext(fn)[0]
        for fn in os.listdir(img_dir)
        if fn.lower().endswith(".png")
    )
    print(f"Found {len(frame_ids)} frames.")

    # Write categories.json
    cat_file = os.path.join(new_annot, "categories.json")
    print(f"Writing annotations/categories.json")
    if not dry_run:
        with open(cat_file, "w") as f:
            json.dump(CATEGORIES, f, indent=2)

    # Write per-frame annotation JSON files
    pcd_dir = new_pcd if os.path.isdir(new_pcd) else old_pcd
    ann_count = 0
    for fid in frame_ids:
        img_path = os.path.join(img_dir, f"{fid}.png")
        try:
            w, h = Image.open(img_path).size
        except Exception:
            w, h = 1920, 1080

        has_labels = (os.path.exists(os.path.join(old_labels3d, f"{fid}.json"))
                      or os.path.exists(os.path.join(old_seg, f"{fid}.txt")))
        if not has_labels:
            continue

        anns = _frame_annotations(fid, w, h, old_labels3d, old_seg)
        frame_doc = {
            "file_name": f"{fid}.png",
            "width":     w,
            "height":    h,
            "annotations": anns,
        }
        out_path = os.path.join(new_annot, f"{fid}.json")
        if not dry_run:
            with open(out_path, "w") as f:
                json.dump(frame_doc, f, indent=2)
        ann_count += len(anns)

    ann_frames = len([fid for fid in frame_ids
                      if os.path.exists(os.path.join(old_labels3d, f"{fid}.json"))
                      or os.path.exists(os.path.join(old_seg, f"{fid}.txt"))])
    print(f"Wrote {ann_frames} per-frame annotation files "
          f"({ann_count} total annotations) to annotations/")

    # seg_labels working dir + classes.txt (for mask editor)
    seg_work_dir = os.path.join(new_annot, "seg_labels")
    if not dry_run:
        os.makedirs(seg_work_dir, exist_ok=True)
        with open(os.path.join(seg_work_dir, "classes.txt"), "w") as f:
            f.write("\n".join(c["name"] for c in CATEGORIES))

    # dataset_info.json
    info_out = os.path.join(root, "dataset_info.json")
    print("Writing dataset_info.json")
    if not dry_run:
        with open(info_out, "w") as f:
            json.dump({
                "info": {
                    "description":          "Logistics Dataset",
                    "version":              "1.0",
                    "year":                 2026,
                    "sensor":               "RealSense D435",
                    "calibration_file":     "calib/d435_intrinsics.json",
                    "annotations_dir":      "annotations/",
                    "categories_file":      "annotations/categories.json",
                    "bbox_2d_format":       "xywh_pixels",
                    "segmentation_format":  "polygon_pixels",
                    "bbox_3d_units":        "meters",
                    "bbox_3d_frame":        "camera",
                    "bbox_3d_axes":         {"x": "right", "y": "down", "z": "forward"},
                    "bbox_3d_center":       "geometric_center",
                    "bbox_3d_rotation":     "yaw_radians",
                    "bbox_3d_yaw_axis":     "y",
                },
            }, f, indent=2)

    print("\nDone. New structure:")
    print(f"  {root}/")
    print(f"  ├── dataset_info.json")
    print(f"  ├── calib/d435_intrinsics.json")
    print(f"  ├── annotations/categories.json")
    print(f"  ├── annotations/{ann_frames} per-frame .json files  ({ann_count} annotations)")
    print(f"  ├── annotations/seg_labels/  (mask editor working dir)")
    print(f"  ├── images/  ({len(frame_ids)} frames)")
    print(f"  ├── depth/   (unchanged)")
    print(f"  ├── depth_files/  (unchanged)")
    print(f"  └── point_clouds/  ({len(frame_ids)} frames)")
    print()
    print("Old directories kept for reference (delete manually after verifying):")
    for d in ["color", "point_cloud", "intrinsics", "labels_3d", "seg_labels"]:
        if os.path.isdir(os.path.join(root, d)):
            print(f"  rm -rf '{os.path.join(root, d)}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate dataset to per-frame COCO JSON format")
    parser.add_argument("root", nargs="?",
                        default="/home/tumwfml-ubunt6/LOCO 3D/3D dataset/Realsense_D435/01",
                        help="Dataset root directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without writing files")
    parser.add_argument("--split-coco", action="store_true",
                        help="Split existing instances_all.json into per-frame files and delete it")
    args = parser.parse_args()

    if args.split_coco:
        split_coco_to_per_frame(args.root, delete_unified=True)
    else:
        migrate(args.root, dry_run=args.dry_run)

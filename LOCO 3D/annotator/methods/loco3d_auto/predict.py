"""
LOCO3D Auto prediction script — uses the built-in YOLO + depth + pose pipeline.

Usage (called by the eval window):
    python predict.py --input /tmp/eval_input.json --output /tmp/eval_output.json

Input JSON schema:
    {
      "frames": [
        {
          "frame_id": "010001",
          "pcd_path": "/abs/path/010001.pcd",
          "image_path": "/abs/path/010001.png",
          "dataset_root": "/abs/path/LOCO_3D/Kinect_V2"
        }
      ]
    }

Output JSON schema:
    {
      "010001": [
        {
          "category_id": 3,
          "score": 0.90,
          "bbox_3d": {
            "center": {"x": 1.2, "y": 0.0, "z": 2.5},
            "dimensions": {"length": 1.2, "width": 0.8, "height": 0.144},
            "yaw": 0.78
          }
        }
      ]
    }
"""

import sys, os, json, math, time, argparse
import numpy as np

# Allow importing from the annotator package
_HERE = os.path.dirname(os.path.abspath(__file__))
_ANNOTATOR_DIR = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ANNOTATOR_DIR)

from pose_estimation_pipeline import (
    apply_hist_depth_filter,
    estimate_3d_pose,
    find_depth_file,
    load_depth_file,
    find_camera_params_file,
    load_intrinsics,
    make_label_object,
)

try:
    from ultralytics import YOLO
    import cv2
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False

# COCO category IDs used in this project
_NAME_TO_CAT = {
    "pallet_truck":       2,
    "pallet truck":       2,
    "pallet":             3,
    "stillage":           4,
    "forklift":           5,
    "small_load_carrier": 6,
    "small load carrier": 6,
}

_MODEL_PATH = os.path.join(_ANNOTATOR_DIR, "models", "best.pt")


def _run_frame(frame_info: dict, model) -> list[dict]:
    """Run the pipeline on one frame and return a list of predictions."""
    dataset_root = frame_info["dataset_root"]
    frame_id     = frame_info["frame_id"]
    image_path   = frame_info.get("image_path", "")

    depth_dir  = os.path.join(dataset_root, "depth_files")
    calib_dir  = os.path.join(dataset_root, "calib")

    depth_path = find_depth_file(depth_dir, frame_id)
    calib_path = find_camera_params_file(calib_dir, frame_id)
    if depth_path is None or calib_path is None or not os.path.exists(image_path):
        return []

    depth_raw = load_depth_file(depth_path)   # uint16 mm
    depth_m   = depth_raw.astype(np.float32) / 1000.0

    fx, fy, cx_cam, cy_cam = load_intrinsics(calib_path)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return []

    results_yolo = model(image_path, verbose=False)
    if not results_yolo or results_yolo[0].masks is None:
        return []

    det = results_yolo[0]
    preds = []
    for i in range(len(det.boxes)):
        cls_name = det.names[int(det.boxes.cls[i].item())]
        score    = float(det.boxes.conf[i].item())
        x1, y1, x2, y2 = (int(v) for v in det.boxes.xyxy[i].tolist())
        mask_full = det.masks.data[i].cpu().numpy()  # 0/1 float, model resolution
        # Resize mask to image size
        mask_img = cv2.resize(mask_full, (img_bgr.shape[1], img_bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        mask_crop = mask_img[y1:y2, x1:x2].astype(bool)
        depth_crop = depth_m[y1:y2, x1:x2]
        try:
            result = estimate_3d_pose(
                depth_crop, mask_crop, x1, y1, fx, fy, cx_cam, cy_cam,
                resolution=1, max_height_percent=5,
            )
        except Exception:
            continue
        if result is None:
            continue
        center, dims, yaw_deg = result
        obj = make_label_object(cls_name, center, dims, yaw_deg)
        cat_id = _NAME_TO_CAT.get(cls_name, 0)
        preds.append({
            "category_id": cat_id,
            "score": score,
            "bbox_3d": {
                "center": obj["centroid"],
                "dimensions": obj["dimensions"],
                "yaw": math.radians(obj["rotations"]["y"]),
            },
        })
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        payload = json.load(f)

    if not _HAS_YOLO:
        print("ERROR: ultralytics not installed", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(_MODEL_PATH):
        print(f"ERROR: model not found at {_MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    model = YOLO(_MODEL_PATH)

    output = {}
    for frame_info in payload["frames"]:
        frame_id = frame_info["frame_id"]
        try:
            preds = _run_frame(frame_info, model)
        except Exception as e:
            print(f"WARN: frame {frame_id} failed: {e}", file=sys.stderr)
            preds = []
        output[frame_id] = preds

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()

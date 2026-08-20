"""
pose_estimation_pipeline.py

All pipeline functions for autonomous 3D bounding box generation.
Core algorithm functions are extracted verbatim from
script/AzureKinect_pose_comparison.ipynb.
"""

import os
import re
import json

import numpy as np
from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------------
# Histogram depth filtering
# ---------------------------------------------------------------------------

def calculate_peak_region(peak_index, hist, h_threshold):
    left_edge = peak_index
    right_edge = peak_index

    while left_edge > 0 and hist[left_edge] > h_threshold:
        left_edge -= 1
    while right_edge < len(hist) - 1 and hist[right_edge] > h_threshold:
        right_edge += 1

    element_count = np.sum(hist[left_edge:right_edge + 1])
    return left_edge, right_edge, element_count


def apply_hist_depth_filter(depth_image, ignore_background=False, resolution=1, max_height_percent=5):
    depth_values = depth_image.flatten()
    depth_values = depth_values[np.isfinite(depth_values) & (depth_values != 0)]

    if len(depth_values) == 0:
        return depth_image, np.zeros(1), np.array([0, 1]), 0, 0, 0, 0

    bins = round(max(val for val in depth_values if np.isfinite(val)) * resolution)
    bins = max(1, bins)

    hist, bin_edges = np.histogram(depth_values, bins=bins)
    max_peak_index = np.argmax(hist)
    min_height = hist[max_peak_index] * max_height_percent / 100

    threshold = max(np.average(hist[:]), min_height)
    peaks = np.where(hist > threshold)[0]

    min_height = hist[max_peak_index] * max_height_percent / 100

    if len(peaks) > 0:
        results_dict = []
        for peak in peaks:
            left_edge, right_edge, count = calculate_peak_region(peak, hist, min_height)
            results_dict.append({
                'left_edge': left_edge,
                'right_edge': right_edge,
                'count': count,
            })

        sorted_results = sorted(results_dict, key=lambda x: x['count'], reverse=True)

        unique_results = {(item['left_edge'], item['right_edge']): item for item in sorted_results}
        sorted_results = list(unique_results.values())

        if ignore_background:
            dominant_peak = next(
                (elem for elem in sorted_results if elem['right_edge'] != len(hist) - 1),
                sorted_results[0],
            )
        else:
            dominant_peak = sorted_results[0]

        left_edge  = dominant_peak['left_edge']
        right_edge = dominant_peak['right_edge']
        lower_bound = bin_edges[left_edge]
        upper_bound = bin_edges[right_edge]

        mask = (depth_image >= lower_bound) & (depth_image <= upper_bound)
        filtered_image = np.where(mask, depth_image, 0)

        return filtered_image, hist, bin_edges, lower_bound, upper_bound, threshold, min_height

    print("No peaks found in image")
    return depth_image, hist, bin_edges, np.min(depth_values), np.max(depth_values), threshold, min_height


# ---------------------------------------------------------------------------
# Yaw + aligned bounding box
# ---------------------------------------------------------------------------

def estimate_yaw_and_aligned_bbox_from_top4_front_hull_segments(
    depth_crop,
    mask_crop,
    x1, y1,
    cx, cy, fx, fy
):
    """
    Returns yaw + geometry for visualization. No plotting inside.
    """
    ys, xs = np.where(mask_crop > 0)
    if len(xs) < 30:
        return None

    global_xs = xs + x1
    zs = depth_crop[ys, xs].astype(np.float32) / 1000.0
    valid = zs > 0

    global_xs = global_xs[valid]
    zs = zs[valid]

    X = (global_xs - cx) * zs / fx
    Z = zs
    points_xz = np.stack([X, Z], axis=1)

    if len(points_xz) < 30:
        return None

    # QJ joggle: prevents "flat simplex" errors when all points share the same Z value
    hull = ConvexHull(points_xz, qhull_options='QJ')
    hull_pts = points_xz[hull.vertices]
    N = len(hull_pts)

    segments = []
    for i in range(N):
        p1 = hull_pts[i]
        p2 = hull_pts[(i + 1) % N]
        d = p2 - p1
        L = np.linalg.norm(d)
        mid_z = 0.5 * (p1[1] + p2[1])
        segments.append({"p1": p1, "p2": p2, "length": L, "mid_z": mid_z})

    top4 = sorted(segments, key=lambda s: s["length"], reverse=True)[:4]
    front2 = sorted(top4, key=lambda s: s["mid_z"])[:2]
    best = max(front2, key=lambda s: s["length"])

    p1, p2 = best["p1"], best["p2"]
    edge_dir = p2 - p1
    edge_dir /= np.linalg.norm(edge_dir)

    # arctan2(-Z, X): angle that rotates local X onto the edge direction in the
    # renderer's _rotation_y convention (R maps local X → [cosθ, 0, -sinθ]).
    yaw = np.arctan2(-edge_dir[1], edge_dir[0])
    yaw_deg = np.rad2deg(yaw)

    if yaw_deg > 90:
        yaw_deg -= 180
    elif yaw_deg < -90:
        yaw_deg += 180

    u = edge_dir
    v = np.array([-u[1], u[0]])

    proj_u = points_xz @ u
    proj_v = points_xz @ v

    u_min, u_max = proj_u.min(), proj_u.max()
    v_min, v_max = proj_v.min(), proj_v.max()

    bbox = np.array([
        u_min * u + v_min * v,
        u_max * u + v_min * v,
        u_max * u + v_max * v,
        u_min * u + v_max * v,
    ])

    return {
        "yaw_deg":   float(yaw_deg),
        "points_xz": points_xz,
        "hull_pts":  hull_pts,
        "edge":      (p1, p2),
        "bbox":      bbox,
        "direction": u,
    }


# ---------------------------------------------------------------------------
# Full 3D pose — accepts intrinsics directly instead of a file path
# ---------------------------------------------------------------------------

def estimate_3d_pose(depth_crop, mask_crop, x1, y1, fx, fy, cx, cy,
                     class_dims=None, class_dims_range=None):
    """
    3D pose estimation using aligned X-Z bounding box + axis-aligned Y extent.

    Args:
        class_dims: optional (L, W, H) tuple in metres.  When provided the BB
                    dimensions are forced to the class spec: the corner of the
                    oriented footprint that is closest to the sensor is used as
                    anchor and the full L×W rectangle is extended from there.
                    Height is set to H and anchored to the topmost (min-Y)
                    measured point.
        class_dims_range: optional ((L_min, L_max), (W_min, W_max), H) tuple.
                    Like class_dims but clamps the measured footprint to the
                    valid range instead of snapping to a fixed size.  Use for
                    object classes that come in multiple sizes (e.g. KLT
                    containers).  Takes precedence over class_dims.

    Returns:
        distance (float),
        center   (np.ndarray, shape (3,)),
        dimensions (np.ndarray, shape (3,))  — [width_perp, height_y, depth_along],
        yaw_deg  (float),
        yaw_bbox_result (dict | None)
    """
    ys, xs = np.where(mask_crop > 0)
    if len(xs) == 0:
        raise ValueError("Empty mask")

    global_xs = xs + x1
    global_ys = ys + y1

    zs = depth_crop[ys, xs].astype(np.float32) / 1000.0
    valid = zs > 0

    global_xs = global_xs[valid]
    global_ys = global_ys[valid]
    zs = zs[valid]

    Xs = (global_xs - cx) * zs / fx
    Ys = (global_ys - cy) * zs / fy
    Zs = zs

    if len(Zs) == 0 or float(np.median(Zs)) >= 6.0:
        raise ValueError("Object median depth >= 6 m — depth data unreliable at this range")

    yaw_bbox_result = estimate_yaw_and_aligned_bbox_from_top4_front_hull_segments(
        depth_crop, mask_crop, x1, y1, cx, cy, fx, fy
    )

    if yaw_bbox_result is None:
        yaw = 0.0
        bbox_xz = None
    else:
        yaw = yaw_bbox_result["yaw_deg"]
        bbox_xz = yaw_bbox_result["bbox"]

    min_y = Ys.min()
    max_y = Ys.max()

    # Resolve range-based dims.
    # side_a = bbox u-extent (along reference edge direction)
    # side_b = bbox v-extent (perpendicular to reference edge)
    # Clamping is deferred until after face classification so each measured side
    # is clamped against the range that matches its physical dimension (long vs short).
    _range_sides = None
    if class_dims_range is not None and bbox_xz is not None:
        (L_min, L_max), (W_min, W_max), H_spec = class_dims_range
        side_a = float(np.linalg.norm(bbox_xz[1] - bbox_xz[0]))
        side_b = float(np.linalg.norm(bbox_xz[2] - bbox_xz[1]))
        m_long  = max(side_a, side_b)
        m_short = min(side_a, side_b)
        # Midpoint estimates are used only for the volume check below.
        clamped_long  = float(np.clip(m_long,  L_min, L_max))
        clamped_short = float(np.clip(m_short, W_min, W_max))
        class_dims = (clamped_long, clamped_short, H_spec)
        _range_sides = (side_a, side_b, L_min, L_max, W_min, W_max)

    if bbox_xz is None:
        long_dim = short_dim = 0.0
        center_x = center_z = 0.0
    elif class_dims is not None:
        L, W, _H = class_dims
        # _H may be a (H_min, H_max) range tuple — use the midpoint for volume check.
        _H_scalar = float(sum(_H) / len(_H)) if isinstance(_H, (tuple, list)) else float(_H)
        # Reject if the measured volume is less than 10 % of the class volume.
        measured_long = max(
            np.linalg.norm(bbox_xz[1] - bbox_xz[0]),
            np.linalg.norm(bbox_xz[2] - bbox_xz[1]),
        )
        measured_short = min(
            np.linalg.norm(bbox_xz[1] - bbox_xz[0]),
            np.linalg.norm(bbox_xz[2] - bbox_xz[1]),
        )
        measured_h = max_y - min_y
        measured_vol = measured_long * measured_short * measured_h
        class_vol    = L * W * _H_scalar
        if measured_vol < 0.10 * class_vol:
            raise ValueError(
                f"Measured volume ({measured_vol:.3f} m³) < 10 % of "
                f"class volume ({class_vol:.3f} m³) — too little data to estimate pose"
            )
        long_class  = max(L, W)
        short_class = min(L, W)

        # Decide which face the reference edge belongs to.
        # If ref_edge_len > 110 % of the expected short dimension → long face visible.
        # For range-based dims use W_min (the smallest possible short side) as the
        # threshold, so a genuinely long face is never mis-classified when depth
        # outliers inflate the measured short-side extent and raise short_class.
        u_ref = yaw_bbox_result["direction"]
        p1_edge, p2_edge = yaw_bbox_result["edge"]
        ref_edge_len = float(np.linalg.norm(p2_edge - p1_edge))
        # Use the full hull u-extent for face classification rather than a single
        # hull segment.  When the visible face has multiple hull vertices,
        # individual segments are shorter than the full face width and can fall
        # below the threshold, causing a wrong short-face classification.
        # The u-extent is the projection of ALL hull points onto u, so it always
        # reflects the complete observed width along the reference direction.
        face_extent = float(np.linalg.norm(bbox_xz[1] - bbox_xz[0]))  # full u-extent
        # For range-based dims: face_extent > W_max means it cannot be the short face.
        # For fixed dims: use 110 % of the nominal short class dimension.
        if _range_sides is not None:
            threshold_short = _range_sides[5]   # W_max
        else:
            threshold_short = 1.10 * short_class
        edge_is_long_face = face_extent > threshold_short

        # Near corner = corner of the oriented footprint closest to sensor (origin)
        dists    = np.linalg.norm(bbox_xz, axis=1)
        near_idx = int(np.argmin(dists))
        near     = bbox_xz[near_idx]
        adj1     = bbox_xz[(near_idx + 1) % 4]
        adj2     = bbox_xz[(near_idx - 1) % 4]
        e1 = adj1 - near;  len1 = np.linalg.norm(e1)
        e2 = adj2 - near;  len2 = np.linalg.norm(e2)
        dir1 = e1 / max(len1, 1e-9)
        dir2 = e2 / max(len2, 1e-9)

        # dir_u aligns with the reference edge; dir_v is perpendicular.
        if abs(float(np.dot(dir1, u_ref))) >= abs(float(np.dot(dir2, u_ref))):
            dir_u, dir_v = dir1, dir2
        else:
            dir_u, dir_v = dir2, dir1

        # long_dir is the direction of the object's long physical axis.
        if edge_is_long_face:
            long_dir, short_dir = dir_u, dir_v
        else:
            long_dir, short_dir = dir_v, dir_u

        # Physical yaw = angle of long axis in XZ plane.
        # Convention (from dataset_info.json: x=right, z=forward):
        #   yaw=0°  → long axis lateral (along X)
        #   yaw=90° → long axis forward (along Z)
        # arctan2(long_dir[1], long_dir[0]) gives the standard counter-clockwise
        # angle from +X toward +Z.  Renderer convention L→local X is consistent:
        # Ry(θ) applied to L corners spans the correct world axis for any θ.
        yaw = float(np.rad2deg(np.arctan2(long_dir[1], long_dir[0])))
        if yaw > 90.0:
            yaw -= 180.0
        elif yaw < -90.0:
            yaw += 180.0

        # For range-based dims: measure the visible face, infer the hidden one
        # proportionally. A long face measured at L_min maps to W_min; at L_max
        # to W_max; values in between interpolate linearly. This avoids using the
        # noisy perpendicular depth extent (unreliable for front-facing flat objects).
        if _range_sides is not None:
            sa, sb, L_min, L_max, W_min, W_max = _range_sides
            if edge_is_long_face:
                long_dim  = float(np.clip(sa, L_min, L_max))
                t         = (long_dim - L_min) / max(L_max - L_min, 1e-9)
                short_dim = float(W_min + t * (W_max - W_min))
            else:
                short_dim = float(np.clip(sa, W_min, W_max))
                t         = (short_dim - W_min) / max(W_max - W_min, 1e-9)
                long_dim  = float(L_min + t * (L_max - L_min))
        else:
            long_dim  = long_class
            short_dim = short_class

        center_x = near[0] + long_dir[0] * long_dim / 2 + short_dir[0] * short_dim / 2
        center_z = near[1] + long_dir[1] * long_dim / 2 + short_dir[1] * short_dim / 2
        c_xz = np.array([center_x, center_z])

        # u_dim / v_dim are the extents along dir_u / dir_v for the preview bbox.
        u_dim = long_dim if edge_is_long_face else short_dim
        v_dim = short_dim if edge_is_long_face else long_dim
        fitted_corners = np.array([
            c_xz - dir_u * u_dim / 2 - dir_v * v_dim / 2,
            c_xz + dir_u * u_dim / 2 - dir_v * v_dim / 2,
            c_xz + dir_u * u_dim / 2 + dir_v * v_dim / 2,
            c_xz - dir_u * u_dim / 2 + dir_v * v_dim / 2,
        ])
        if yaw_bbox_result is not None:
            yaw_bbox_result = dict(yaw_bbox_result)
            yaw_bbox_result["bbox"] = fitted_corners
    else:
        # No class prior: use raw measured extents.
        long_dim  = float(np.linalg.norm(bbox_xz[1] - bbox_xz[0]))
        short_dim = float(np.linalg.norm(bbox_xz[2] - bbox_xz[1]))
        center_x = bbox_xz[:, 0].mean()
        center_z = bbox_xz[:, 1].mean()

    if class_dims is not None:
        measured_h = max_y - min_y
        H_spec = class_dims[2]
        if isinstance(H_spec, (tuple, list)):
            height = float(np.clip(measured_h, H_spec[0], H_spec[1]))
        else:
            height = H_spec if measured_h > H_spec else measured_h
        center_y = max_y - height / 2
    else:
        height   = max_y - min_y
        center_y = 0.5 * (min_y + max_y)

    # dimensions[0] = long side (length), dimensions[1] = height, dimensions[2] = short side (width)
    dimensions = np.array([long_dim, height, short_dim])
    center     = np.array([center_x, center_y, center_z])
    distance   = float(np.linalg.norm(center))

    return distance, center, dimensions, yaw, yaw_bbox_result


# ---------------------------------------------------------------------------
# Depth-to-color alignment
# (mirrors k4a_transformation_depth_image_to_color_camera from the Kinect SDK)
# ---------------------------------------------------------------------------

def transform_depth_to_color(depth_m,
                              d_fx, d_fy, d_cx, d_cy,
                              c_fx, c_fy, c_cx, c_cy,
                              tx, out_w, out_h):
    """
    Reproject a depth image from depth-camera space into color-camera space.

    Each depth pixel is un-projected to 3-D using depth intrinsics, the
    extrinsic translation (tx, 0, 0) is applied, and the point is projected
    into the color image using color intrinsics.  The result is a float32
    depth image at (out_h × out_w) whose values are in the same units as the
    input (metres).  Pixels with no depth coverage are left as 0.
    """
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64),
                        np.arange(h, dtype=np.float64))
    Z = depth_m.astype(np.float64)
    valid = Z > 0

    X = np.where(valid, (u - d_cx) * Z / d_fx, 0.0)
    Y = np.where(valid, (v - d_cy) * Z / d_fy, 0.0)

    # Apply extrinsic (rotation ≈ identity for D435; translation only)
    X_c = X + tx

    with np.errstate(divide="ignore", invalid="ignore"):
        u_c = np.where(valid, np.round(c_fx * X_c / Z + c_cx), -1).astype(np.int32)
        v_c = np.where(valid, np.round(c_fy * Y   / Z + c_cy), -1).astype(np.int32)

    in_bounds = valid & (u_c >= 0) & (u_c < out_w) & (v_c >= 0) & (v_c < out_h)

    # Use minimum depth at each color pixel to resolve occlusions
    aligned = np.full(out_h * out_w, np.inf, dtype=np.float64)
    np.minimum.at(aligned, v_c[in_bounds] * out_w + u_c[in_bounds], Z[in_bounds])
    aligned[aligned == np.inf] = 0.0
    return aligned.reshape(out_h, out_w).astype(np.float32)


def align_depth_to_color(depth_m: np.ndarray, json_path: str) -> tuple:
    """
    Load intrinsics from a JSON calibration file and reproject depth into color space.

    Supports two JSON layouts:
      - Old ZED2-style:  {"left_camera": {depth}, "right_camera": {color}}
      - New COCO-style:  {"depth_camera": {...},   "color_camera": {...}}

    Returns (aligned_depth, fx, fy, cx, cy) where the intrinsics belong to
    whichever camera frame the returned depth is in:
    - If JSON has both cameras with different resolutions → reprojected depth
      at color resolution, color intrinsics.
    - If depth already matches color resolution (e.g. pre-aligned Kinect) →
      depth returned unchanged, color intrinsics.
    - If JSON has only one camera → depth unchanged, that camera's intrinsics.
    """
    with open(json_path) as f:
        data = json.load(f)

    # Support old ZED2 keys and new COCO-style keys
    dep_cam = data.get("depth_camera") or data.get("left_camera")
    col_cam = data.get("color_camera") or data.get("right_camera")
    ext     = data.get("extrinsics", {}).get("translation", {})

    if dep_cam is None:
        cam = data
        return depth_m, float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])

    d_fx, d_fy = float(dep_cam["fx"]), float(dep_cam["fy"])
    d_cx, d_cy = float(dep_cam["cx"]), float(dep_cam["cy"])

    if col_cam is None:
        return depth_m, d_fx, d_fy, d_cx, d_cy

    c_fx, c_fy = float(col_cam["fx"]), float(col_cam["fy"])
    c_cx, c_cy = float(col_cam["cx"]), float(col_cam["cy"])
    # Resolution may be stored flat (width/height) or nested (resolution.width/height)
    if "resolution" in col_cam:
        out_w = int(col_cam["resolution"]["width"])
        out_h = int(col_cam["resolution"]["height"])
    else:
        out_w = int(col_cam["width"])
        out_h = int(col_cam["height"])
    tx    = float(ext.get("x", 0.0))

    dep_h, dep_w = depth_m.shape[:2]
    if dep_h == out_h and dep_w == out_w:
        # Already in color space (e.g. Kinect _transformed_depth)
        return depth_m, c_fx, c_fy, c_cx, c_cy

    aligned = transform_depth_to_color(
        depth_m, d_fx, d_fy, d_cx, d_cy,
        c_fx, c_fy, c_cx, c_cy,
        tx, out_w, out_h,
    )
    return aligned, c_fx, c_fy, c_cx, c_cy


# ---------------------------------------------------------------------------
# File discovery helpers — same numeric-ID approach as _find_rgb_image
# ---------------------------------------------------------------------------

def _last_numeric_id(name: str) -> int | None:
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else None


def find_depth_file(depth_dir: str, frame_id: str) -> str | None:
    """Find the depth file (.npy or 16-bit PNG) for frame_id in depth_dir."""
    if not os.path.isdir(depth_dir):
        return None

    # Exact-name fast paths — prefer .npy, fall back to .png
    for candidate in (f"{frame_id}_depth.npy", f"{frame_id}.npy",
                      f"{frame_id}_depth.png", f"{frame_id}.png"):
        path = os.path.join(depth_dir, candidate)
        if os.path.exists(path):
            return path

    # Numeric-ID fallback
    target = _last_numeric_id(frame_id)
    if target is None:
        return None

    for fname in sorted(os.listdir(depth_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".npy", ".png"):
            continue
        if _last_numeric_id(fname) == target:
            return os.path.join(depth_dir, fname)

    return None


def load_depth_file(path: str) -> "np.ndarray":
    """Load a depth file (.npy or 16-bit PNG) and return a uint16 H×W array in mm."""
    import numpy as _np
    if path.lower().endswith(".npy"):
        return _np.load(path)
    # 16-bit grayscale PNG — values already in mm
    import cv2 as _cv2
    arr = _cv2.imread(path, _cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"Could not read depth PNG: {path}")
    return arr.astype(_np.uint16)


def find_camera_params_file(params_dir: str, frame_id: str) -> str | None:
    """Find camera intrinsics file (.npz or .json) for frame_id in params_dir.

    Also accepts a direct file path (new COCO format where calib is a single shared file).
    """
    if os.path.isfile(params_dir):
        return params_dir
    if not os.path.isdir(params_dir):
        return None

    # If there is exactly one file in the folder, treat it as a shared intrinsics file
    all_files = [f for f in os.listdir(params_dir)
                 if f.lower().endswith(".npz") or f.lower().endswith(".json")]
    if len(all_files) == 1:
        return os.path.join(params_dir, all_files[0])

    # Exact-name fast paths (prefer .npz)
    for ext in (".npz", ".json"):
        for candidate in (f"{frame_id}_camera_parameters{ext}", f"{frame_id}{ext}"):
            path = os.path.join(params_dir, candidate)
            if os.path.exists(path):
                return path

    # Numeric-ID fallback
    target = _last_numeric_id(frame_id)
    if target is None:
        return None

    for ext_order in (".npz", ".json"):
        for fname in sorted(os.listdir(params_dir)):
            if not fname.lower().endswith(ext_order):
                continue
            if _last_numeric_id(fname) == target:
                return os.path.join(params_dir, fname)

    return None


# ---------------------------------------------------------------------------
# Camera intrinsics loaders
# ---------------------------------------------------------------------------

def load_intrinsics_npz(path: str) -> tuple:
    """Load fx, fy, cx, cy from an Azure Kinect-style .npz file."""
    params = np.load(path)
    intr = params["rgb_intrinsics"]
    return float(intr[0, 0]), float(intr[1, 1]), float(intr[0, 2]), float(intr[1, 2])


def load_intrinsics_json(path: str) -> tuple:
    """Load fx, fy, cx, cy from a calibration JSON.

    Supports old ZED2-style (left_camera/right_camera) and new COCO-style
    (depth_camera/color_camera) layouts.  Always returns the COLOR camera
    intrinsics when both cameras are present.
    """
    with open(path, "r") as f:
        data = json.load(f)
    # Prefer color camera (right_camera = color in ZED2; color_camera in new format)
    cam = (data.get("color_camera") or data.get("right_camera")
           or data.get("left_camera") or data)
    return float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])


def load_intrinsics(path: str) -> tuple:
    """Dispatch to the correct loader based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        return load_intrinsics_npz(path)
    if ext == ".json":
        return load_intrinsics_json(path)
    raise ValueError(f"Unsupported intrinsics file format: {path}")


# ---------------------------------------------------------------------------
# Output conversion
# ---------------------------------------------------------------------------

def make_label_object(class_name: str, center: np.ndarray,
                      dimensions: np.ndarray, yaw_deg: float) -> dict:
    """
    Convert pipeline output to the GUI's label JSON format.

    dimensions layout: [long_side, height_y, short_side]
    label JSON layout:
        length = dimensions[0]  (long side from top-down view)
        width  = dimensions[2]  (short side from top-down view)
        height = dimensions[1]  (Y extent)
    """
    return {
        "name": class_name,
        "centroid": {
            "x": float(center[0]),
            "y": float(center[1]),
            "z": float(center[2]),
        },
        "dimensions": {
            "length": float(dimensions[0]),
            "width":  float(dimensions[2]),
            "height": float(dimensions[1]),
        },
        "rotations": {
            "x": 0.0,
            "y": float(yaw_deg),
            "z": 0.0,
        },
    }

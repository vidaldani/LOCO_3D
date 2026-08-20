import sys
import os
import json
import gc
import numpy as np
import open3d as o3d

bbox_lines = [
    [0,1], [1,2], [2,3], [3,0],
    [4,5], [5,6], [6,7], [7,4],
    [0,4], [1,5], [2,6], [3,7],
]

CLASS_COLORS = {
    "forklift":           [0.0, 1.0, 0.0],
    "pallet_truck":       [1.0, 0.5, 0.0],
    "pallet":             [0.0, 1.0, 1.0],
    "small_load_carrier": [1.0, 0.0, 1.0],
    "stillage":           [1.0, 1.0, 0.0],
    "person":             [1.0, 1.0, 0.0],
}
DEFAULT_COLOR = [0.0, 0.5, 1.0]


def build_bbox_geometry(obj):
    cx = obj["centroid"]["x"]
    cy = obj["centroid"]["y"]
    cz = obj["centroid"]["z"]
    center = np.array([cx, cy, cz])

    length = obj["dimensions"]["length"]
    width  = obj["dimensions"]["width"]
    height = obj["dimensions"]["height"]

    yaw_deg = obj["rotations"]["y"]

    L, W, H = length, width, height
    local_corners = np.array([
        [-L/2, -H/2, -W/2],
        [ L/2, -H/2, -W/2],
        [ L/2,  H/2, -W/2],
        [-L/2,  H/2, -W/2],
        [-L/2, -H/2,  W/2],
        [ L/2, -H/2,  W/2],
        [ L/2,  H/2,  W/2],
        [-L/2,  H/2,  W/2],
    ])

    yaw_rad = np.deg2rad(yaw_deg)
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, yaw_rad, 0))

    rotated_corners = (R @ local_corners.T).T
    bbox_points = rotated_corners + center

    lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(bbox_points),
        lines=o3d.utility.Vector2iVector(bbox_lines),
    )
    normalized = obj["name"].replace(" ", "_")
    color = CLASS_COLORS.get(normalized, DEFAULT_COLOR)
    lineset.colors = o3d.utility.Vector3dVector([color] * len(bbox_lines))

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    frame.rotate(R, center=[0, 0, 0])
    frame.translate(center)

    return lineset, frame


def run_viewer(tmp_json_path):
    with open(tmp_json_path, "r") as f:
        data = json.load(f)

    pcd_path = data["pcd_path"]
    objects  = data["objects"]

    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_points():
        print(f"[worker] WARNING: empty or unreadable point cloud: {pcd_path}")

    geometries = [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)]
    for obj in objects:
        lineset, coord_frame = build_bbox_geometry(obj)
        geometries.append(lineset)
        geometries.append(coord_frame)

    frame_name = os.path.splitext(os.path.basename(pcd_path))[0]
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Label Editor — {frame_name}", width=1600, height=900)

    for g in geometries:
        vis.add_geometry(g)

    vis.poll_events()
    vis.update_renderer()

    ctr = vis.get_view_control()
    eye    = np.array([0.0,  0.0,  0.0])
    lookat = np.array([0.0,  0.0, -1.0])
    up     = np.array([0.0, -1.0,  0.0])
    front  = (lookat - eye) / np.linalg.norm(lookat - eye)
    ctr.set_front(front)
    ctr.set_lookat(lookat)
    ctr.set_up(up)
    ctr.set_zoom(0.01)

    vis.run()
    vis.destroy_window()
    del vis, pcd, geometries
    gc.collect()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: _o3d_viewer_worker.py <tmp_json_path>")
        sys.exit(1)
    run_viewer(sys.argv[1])

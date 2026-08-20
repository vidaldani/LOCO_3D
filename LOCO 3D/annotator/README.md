# 3D Bounding Box Annotator

A PyQt5 desktop GUI for browsing, visualizing, and correcting 3D bounding box labels on Azure Kinect point cloud data.

![GUI Screenshot](images/gui.png)

## Features

- Embedded 3D viewer (PyVista) — no separate window needed
- Browse and navigate point cloud frames with ← / → buttons or random selection
- Edit bounding box centroids, dimensions, and yaw rotation via spinboxes with live 3D preview
- Add new objects with preset dimensions per class
- Swap dimension axes (L↔W, L↔H, W↔H) per object or all at once
- Unsaved-changes prompt on navigation
- Dark theme UI

## Requirements

- Python 3.8+
- Open3D
- PyVista
- PyVistaQt
- PyQt5
- NumPy

Install all dependencies (outside a virtual environment is recommended):

```bash
pip install open3d pyvista pyvistaqt PyQt5 numpy
```

## Running

```bash
cd 3D_bb_annotator
python3 label_editor_gui.py
```

On first use the **Browse PCD folder** and **Browse Labels folder** buttons in the left panel to load the dataset folders.

## Label Format

Labels are JSON files with one file per point cloud frame, named `frame_XXXX.json`:

```json
{
    "folder": "pcd_colored",
    "filename": "frame_0011.pcd",
    "path": "...",
    "objects": [
        {
            "name": "forklift",
            "centroid": { "x": -0.668, "y": -0.061, "z": 2.460 },
            "dimensions": { "length": 1.2, "width": 2.32, "height": 2.19 },
            "rotations": { "x": 0, "y": 298.0, "z": 0 }
        }
    ]
}
```

Supported class names: `forklift`, `pallet truck`, `pallet`, `klt`, `stillage`, `small_load_carrier`

Custom object classes can also be added.

## Controls

| Action | How |
|---|---|
| Navigate frames | ← / → buttons, or click a file in the left list |
| Random frame | **Random Frame** button |
| Select object | Click an entry in the Objects list |
| Edit values | Adjust spinboxes — 3D view updates live |
| Swap dimensions | **L↔W**, **L↔H**, **W↔H** buttons next to spinboxes |
| Swap W↔H all | **Swap All W↔H** button (applies to every object in the frame) |
| Add object | **+ Add Object** button — choose type, dimensions and pose are pre-filled |
| Save | **Save JSON** button (also prompted on navigation if unsaved changes exist) |

## Files

| File | Description |
|---|---|
| `label_editor_gui.py` | Main application — run this |
| `_o3d_viewer_worker.py` | Legacy Open3D subprocess worker (not used by the GUI) |

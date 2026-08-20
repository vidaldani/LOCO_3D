# LOCO 3D — Logistics Objects with 3D Annotations

Dataset and annotation tooling for 3D bounding box labelling of logistics objects (pallets, KLTs, pallet trucks, forklifts, stillages) captured with an **Intel RealSense D435** and a **Microsoft Kinect V2** in real warehouse environments.

---

## Repository structure

```
LOCO_3D/
├── annotator/                  # 3D label editor application (run this)
│   ├── label_editor_gui.py     # main entry point
│   ├── auto_bbox_dialog.py     # automatic 3D BB generation dialog
│   ├── pose_estimation_pipeline.py
│   ├── mask_edit_dialog.py
│   ├── migrate_dataset.py
│   └── models/
│       └── best.pt             # YOLOv11-Seg weights for auto-detection
│
└── LOCO_3D/
    ├── Kinect_V2/
    │   ├── images/             # RGB frames (.png, 1920×1080)          ← download separately
    │   ├── depth_files/        # Raw depth arrays (.npy, uint16 mm)    ← download separately
    │   ├── depth_images/       # Depth visualisations (.png)           ← download separately
    │   ├── point_clouds/       # Coloured point clouds (.pcd)          ← download separately
    │   ├── annotations/        # COCO-style JSON labels  ✓ in git
    │   ├── calib/              # Camera intrinsics JSON  ✓ in git
    │   └── dataset_info.json                             ✓ in git
    └── Realsense_D435/
        └── ...                 # same structure as Kinect_V2
```

> **Raw footage** (the original `.zip` video archives, ~39 GB) and the unprocessed `3D_dataset_raw/` folder are excluded from this repository — contact the maintainer if you need them.

---

## Dataset download

The binary sensor data (~10 GB) is not stored in this git repository. Download it from LRZ Sync+Share and place it inside `LOCO_3D/`:

**[⬇ Download LOCO\_3D binary dataset — LRZ Sync+Share](#)**
*(link to be provided by the project maintainer — contact [danielvidalsoroa@gmail.com](mailto:danielvidalsoroa@gmail.com))*

After downloading, extract so that the folder structure matches the tree above. For example:

```
loco-3d/
└── LOCO_3D/
    ├── Kinect_V2/
    │   ├── images/        ← extracted here
    │   ├── depth_files/   ← extracted here
    │   ├── depth_images/  ← extracted here
    │   └── point_clouds/  ← extracted here (or regenerate, see below)
    └── Realsense_D435/
        ├── images/
        ├── depth_files/
        └── point_clouds/
```

> **Regenerating point clouds** — point clouds can be recreated from the depth files if needed (saves ~5 GB):
> ```bash
> python3 annotator/migrate_dataset.py --generate-pcd LOCO_3D/Kinect_V2
> python3 annotator/migrate_dataset.py --generate-pcd LOCO_3D/Realsense_D435
> ```

---

## Installation

Python **3.10 or 3.11** is recommended. A dedicated virtual environment or conda environment avoids conflicts with system packages.

### 1 — Create an environment

```bash
conda create -n loco3d python=3.11
conda activate loco3d
```

or with venv:

```bash
python3 -m venv loco3d_env
source loco3d_env/bin/activate
```

### 2 — Install dependencies

```bash
pip install \
    pyqt5 \
    pyvista \
    pyvistaqt \
    open3d \
    numpy \
    scipy \
    opencv-python \
    pillow \
    matplotlib \
    ultralytics \
    supervision
```

| Package | Purpose |
|---|---|
| `pyqt5` | GUI framework |
| `pyvista` + `pyvistaqt` | Embedded 3D viewer |
| `open3d` | Point cloud loading |
| `numpy` / `scipy` | Numerical routines, convex hull |
| `opencv-python` | Image processing |
| `pillow` | Image I/O |
| `matplotlib` | Depth histogram plots |
| `ultralytics` | YOLOv11 inference for auto-detection |
| `supervision` | Detection post-processing |

> On Linux you may also need `apt install libgl1 libxcb-xinerama0` if PyQt5 complains about missing display libraries.

### 3 — Clone the repo

```bash
git clone git@gitlab.lrz.de:00000000014B7825/loco-3d.git
cd loco-3d
```

### 4 — Download the binary dataset

The git repo contains annotations and code only. Download the sensor images and depth files from the link in the [Dataset download](#dataset-download) section above and extract them so that `LOCO_3D/Kinect_V2/images/`, `LOCO_3D/Kinect_V2/depth_files/`, etc. exist inside the cloned folder.

---

## Running the annotator

```bash
cd annotator
python3 label_editor_gui.py
```

The application opens a project manager. On first launch you will be asked to create or open a project — point it at the dataset sensor folder you want to annotate (e.g. `LOCO_3D/Kinect_V2`).

---

## Annotation workflow

### Opening a project

1. Launch the app (`python3 label_editor_gui.py`).
2. Click **New Project** (or **Open Project** if you already have one).
3. Set the **Sensor folder** to the dataset directory, e.g.:
   ```
   /path/to/loco-3d/LOCO_3D/Kinect_V2
   ```
   The app will find `images/`, `depth_files/`, `point_clouds/`, and `annotations/` automatically.
4. The frame list on the left populates. Frames with existing annotations show a status indicator.

### Navigating frames

| Action | Control |
|---|---|
| Next / previous frame | **← →** buttons or arrow keys |
| Jump to a specific frame | Click the frame in the list |
| Filter by annotation status | Status dropdown at the top of the list |

### Manual 3D bounding box annotation

1. Select a frame from the list. The RGB image and the 3D point cloud are displayed side by side.
2. Click **+ Add Object** and choose the object class from the dropdown. A default bounding box is placed at the scene centre.
3. In the **Object properties** panel on the right, adjust:
   - **Centroid** (x, y, z in metres)
   - **Dimensions** — length (long side), width (short side), height (vertical extent)
   - **Rot Y** — yaw in degrees. `0°` = long axis pointing right (+X); `90°` = long axis pointing forward (+Z, into the scene)
4. Use the **+90°** / **−90°** buttons for quick 90° snaps.
5. Use the **L↔W**, **L↔H**, **W↔H** swap buttons if you need to rotate the box axes.
6. The 3D viewer updates live. Orbit with left-click drag, zoom with scroll wheel.
7. Click **Save** (or press **Ctrl+S**) when you are happy with the frame.

> **Dimension convention (important)**
> - `length` = longest horizontal side (from a top-down view)
> - `width`  = shortest horizontal side
> - `height` = vertical extent
>
> Typical dimensions per class:
>
> | Class | Length (m) | Width (m) | Height (m) |
> |---|---|---|---|
> | Pallet | 1.200 | 0.800 | 0.144 |
> | Small load carrier | 0.400–0.600 | 0.300–0.400 | 0.147 |
> | Stillage | 1.200 | 0.800 | 0.970 |
> | Forklift | 2.800 | 1.300 | 2.150 |
> | Pallet truck | 1.800 | 0.550 | 1.200 |

### Automatic 3D bounding box generation

The app can propose a 3D bounding box automatically using the YOLOv11 segmentation model and depth data.

1. Open a frame and click **Generate 3D BB**.
2. **Step 1 — Detection**: the YOLO model runs on the RGB image. Select the detection(s) you want to annotate from the list, then click **Next →**.
3. **Step 2 — Depth filtering (HDF)**: the histogram depth filter isolates the object's depth cluster. The overlay image shows which depth pixels were kept (yellow mask). Adjust the **Resolution** and **Max height %** sliders if the mask is noisy, then click **Next →**.
4. **Step 3 — Pose estimation**: the pipeline fits an oriented bounding box to the filtered point cloud. The top-down view shows the fitted box and its yaw. Click **Accept** to insert it into the annotation, or **Back** to adjust the HDF parameters.
5. Fine-tune centroid, dimensions, or yaw in the properties panel if needed, then **Save**.

> If auto-generation fails (too few depth points, object at extreme range), the mask is still saved and a note is added — add the 3D box manually.

### Deleting or editing existing objects

- Click an object in the **Objects** list to select it and load its properties.
- Click **Delete Object** to remove it.
- All changes are local until you click **Save**.

### Saving and sharing annotations

Annotations are saved as JSON files in `LOCO_3D/<sensor>/annotations/`, one file per frame. After annotating a session, commit and push:

```bash
git add LOCO_3D/Kinect_V2/annotations/
git commit -m "annotate frames 010002–010050 (Kinect_V2)"
git push
```

Pull the latest annotations from your teammates before starting a session:

```bash
git pull
```

---

## Annotation file format

Each annotation file is named after its frame (e.g. `010002.json`) and follows a COCO-compatible structure extended with a `bbox_3d` field:

```json
{
  "file_name": "010002.png",
  "width": 1920,
  "height": 1080,
  "annotations": [
    {
      "id": 1,
      "category_id": 3,
      "bbox": [639, 627, 432, 159],
      "segmentation": [[...]],
      "area": 44225,
      "iscrowd": 0,
      "bbox_3d": {
        "center":     { "x": -0.423, "y": 0.318, "z": 2.228 },
        "dimensions": { "length": 1.2, "width": 0.8, "height": 0.144 },
        "yaw": -1.249,
        "rx": 0.0,
        "rz": 0.0
      }
    }
  ]
}
```

| Field | Description |
|---|---|
| `center` | Centroid in camera coordinates (metres); x=right, y=down, z=forward |
| `dimensions.length` | Long horizontal side (metres) |
| `dimensions.width` | Short horizontal side (metres) |
| `dimensions.height` | Vertical extent (metres) |
| `yaw` | Rotation around Y axis in **degrees**; 0° = long axis along X, 90° = long axis along Z |

---

## Coordinate system

All 3D quantities are expressed in the **camera frame** of the respective sensor:

```
x → right
y → down
z → forward (into the scene)
```

See each sensor's `dataset_info.json` for depth units and image resolution.

---

## Contact

Daniel Vidal · [danielvidalsoroa@gmail.com](mailto:danielvidalsoroa@gmail.com)

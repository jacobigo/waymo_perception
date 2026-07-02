# 3D Semantic Segmentation & LiDAR-Camera Fusion

**by Jacob Igo**

> Learning how self-driving perception actually works — building 3D semantic segmentation and sensor fusion from scratch on the Waymo Open Dataset, without the Waymo Python package and with minimal LLM help.

<img src="media/sensor_fusion_no_labels.gif" alt="LiDAR projected onto camera, colored by depth" width="600">

---

## Why I'm doing this

I grew up in Phoenix watching Waymo go from a novelty to something I ride whenever I get the chance, and I'm convinced autonomous vehicles are one of the more important things being built right now — safer roads, and genuinely fascinating tech. I want to work in this field, and the best way I know to understand something is to rebuild it myself.

So this is me taking raw sensor data and turning it into a labeled 3D scene, one piece at a time, learning the geometry and the gotchas along the way. It's a learning project, and I'm having a great time with it.

---

## The data

**Waymo Open Dataset v2.0**, read live from the `waymo_open_dataset_v_2_0_0` GCS bucket — no local copies, and no Waymo package. I parse the parquet files directly with PyArrow.

Each file is one 20-second driving segment with 5 LiDARs and 5 cameras per frame. LiDAR comes as **range images** (azimuth / inclination / range), not raw XYZ, so it needs a spherical-to-Cartesian conversion first. Labels are 23 semantic classes, and only laser 1 is labeled.

| Folder | What it holds |
|---|---|
| `lidar/` | Range-image 3D points |
| `lidar_segmentation/` | Per-point labels (laser 1 only) |
| `lidar_calibration/` | Extrinsics + beam inclinations per laser |
| `camera_image/` | JPEG frames per timestamp |
| `camera_calibration/` | Camera intrinsics + extrinsics |

---

## What works so far

- [x] Range-image decode (spherical → Cartesian)
- [x] Extrinsic transform to a global frame
- [x] Multi-laser fusion into one point cloud
- [x] Segmentation labels decoded & points colored by class
- [x] Memory-safe, timestamp-aligned data loading
- [x] Bird's-eye and interactive 3D (Plotly) visualization
- [x] Scene animations (matplotlib / ffmpeg)
- [x] LiDAR → camera projection
- [x] Fused overlay video (colored by depth)
- [ ] Labeled sensor-fusion render — *in progress*

---

## A few things I learned

**Range images to 3D points.** The LiDAR isn't XYZ; it's spherical (phi, theta, rho). Converting it back to Cartesian was a nice callback to Calc III, and the first time I've applied that math to a real problem.

**Beam inclination & azimuth yaw (, or, the cricket chirping in a dark room).** My point clouds kept coming out wrong and the fused overlay sat too high on the image. I assumed the beam inclinations were in descending order (they weren't), which capped the vertical spread; and I didn't know the sensor has a mounting yaw that has to be subtracted from the azimuth, which was swinging the whole cloud onto the wrong bearing. Finding these taught me to trust the geometry and check my assumptions about how the data is stored.

**Extrinsic transforms.** The 4×4 extrinsic relates a sensor's measurements to the vehicle/global frame. You stack XYZ, add a row of ones to go homogeneous, multiply by the matrix, and drop the extra row.

**LiDAR with camera projection.** Multiply the global points by the inverse camera extrinsic to get camera-local coords, divide by depth, scale by the intrinsics (focal length plus principal point), then keep only the points that land in front of the camera and inside the image. Hard to implement labels in this process due to sparsity in the datset.

---

## Sensor fusion

I'm doing **early fusion**: projecting LiDAR onto the camera image and (as of now) coloring the beams by depth. The current render is verified aligned (after the beam/yaw fixes above). Next up is coloring by semantic class on the 30 (give or take per scene) labeled frames and falling back to depth on the rest, since labels are too sparse to interpolate across frames honestly. Late fusion is a later experiment.

---

## Model training

Now I'm training a model to predict the labels that I previously only read.

**PointNet baseline (in progress).** Each frame is subsampled to 4,096 points and every point is classified into one of the 23 classes. PointNet treats the cloud as an *unordered* set: per-point convolution layers describe each point, a global max-pool summarizes the whole scene, and that is then fed back to every point so it labels itself knowing both itself (local) and the scene (global). Used weighted cross-entropy (with class 0 ignored) to beat the heavy class imbalance, as well as per-class **mIoU** rather than accuracy as the base metric, and disk-caching processed frames so epochs stop re-streaming from GCS. Rare classes are still near zero; next step is PointNet++.

**Sensor-fusion model (later).** Give each point its camera color too — widen the input from XYZ to XYZ + RGB (plus a validity flag for points no camera sees) and measure the **mIoU gain** from adding appearance to geometry.

---

## Project layout

| File | Role |
|---|---|
| `semseg.ipynb` | LiDAR-only segmentation pipeline |
| `sensor_fusion.ipynb` | LiDAR-camera fusion |
| `semseg_modeling.ipynb` | PointNet training |
| `semseg_functions.py` | Shared helpers |
| `media/` | Generated videos and plots |

---

## Roadmap

- **Now:** tuning the PointNet baseline; then upgrading to PointNet++
- **Next:** semantic labels in the fused render (dual-coloring), then a fusion segmentation model (geometry + camera RGB)
- **Eventually:** run these perception pieces in a CARLA sim to evaluate and iterate

---

## Running it

Python 3.10, notebook-driven, data read live from GCS (no local copies).

**1. Install the dependencies** (plus `ffmpeg` on your PATH for the animations):

```bash
pip install -r requirements.txt
sudo apt install ffmpeg   # or: brew install ffmpeg
```

For GPU training, install the CUDA build of PyTorch first (see the note in `requirements.txt`); a plain install pulls the CPU-only build.

**2. Authenticate with Google Cloud** — the helpers shell out to `gcloud`, so make sure it's installed and logged in:

```bash
gcloud auth login
gcloud auth print-access-token   # sanity check: should print a token
```

The token **expires after an hour**, so re-run the notebook's auth cell (or re-import the helper module) for long sessions. Paths to `gcloud` and its config are set at the top of `semseg_functions.py` — adjust them for your machine.

**3. Launch Jupyter and run a notebook top to bottom:**

```bash
jupyter lab   # then open semseg.ipynb or sensor_fusion.ipynb
```

- `semseg.ipynb` — LiDAR-only segmentation pipeline
- `sensor_fusion.ipynb` — LiDAR-camera fusion

Generated videos and plots land in `media/`.

---

## References

- [Waymo Open Dataset](https://waymo.com/open/) · [v2.0 docs](https://waymo.com/open/data/perception/) · Sun et al., *Waymo Open Dataset*, CVPR 2020 ([arXiv:1912.04838](https://arxiv.org/abs/1912.04838))
- [PyArrow: Parquet](https://arrow.apache.org/docs/python/parquet.html) · [PyArrow: GCS filesystem](https://arrow.apache.org/docs/python/filesystems.html#google-cloud-storage-file-system)
- [OpenCV pinhole model / calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) · [Spherical coordinates](https://en.wikipedia.org/wiki/Spherical_coordinate_system) · [Homogeneous coordinates](https://en.wikipedia.org/wiki/Homogeneous_coordinates)

# 3D Semantic Segmentation & LiDAR-Camera Sensor Fusion - by Jacob Igo



> _Learning how perception works with the Waymo Open Perception Dataset, by performing 3D Semantic Segmentation and Sensor Fusion from scratch, without the Waymo Python package and extremely minimal LLM assistance._
> <!-- TODO: e.g. "A learning-focused exploration of 3D semantic segmentation on the Waymo Open Dataset, extended into early LiDAR-camera sensor fusion." -->

<img src="media/sensor_fusion_no_labels.gif" alt="demo" width="600">
<!-- TODO: swap in your best still frame or GIF once the labeled-fusion render is done. -->

---

## Table of Contents

- [Motivation](#motivation)
- [Dataset](#dataset)
- [What I've Built So Far](#what-ive-built-so-far)
- [Technical Deep-Dives](#technical-deep-dives)
- [Pipeline & Project Structure](#pipeline--project-structure)
- [Sensor Fusion](#sensor-fusion)
- [Challenges & Lessons Learned](#challenges--lessons-learned)
- [Roadmap](#roadmap)
- [Setup & Running](#setup--running)
- [References](#references)

---

## Motivation

<!-- TODO: Why did you start this? What did you want to learn?
     A few sentences on your goals: understanding how raw sensor data becomes
     a labeled 3D scene, learning the geometry/math behind it, etc.
     This is a learning project — say so, and say what "done" looks like to you. -->

I am very interested in autonomous vehicles and want to pursue it for a career, as I believe there is a strong potential market for them one day due to their game-changing safety and convenience 

(, and because I think the tech is beyond fascinating. Having grown up in Phoenix, AZ, I've seen Waymo grow tremendously and I always use their services when I have the chance to. This is the future).

Therefore, I want to learn how self-driving cars work, and what better way to do it than through recreating their functions. 

This is one of the first steps in my journey of learning the ins and outs of self-driving cars, and I'm having a great time.

---

## Dataset

**Waymo Open Dataset v2.0** (`waymo_open_dataset_v_2_0_0` GCS bucket).

<!-- TODO: In your own words, describe the data you're working with. Cover:
     - What one parquet file represents (~20s driving segment)
     - The sensors: 5 LiDARs, 5 cameras
     - The range-image format (not raw x/y/z)
     - The 23 semantic classes
     - That you access it directly via pyarrow + GcsFileSystem (no waymo package) -->

_This holds the lidar points along with their labels, calibration data, and the corresponding image frames. These are the folders I am working with (for now, I will be using more throughout this project)._

| Folder | What it holds |
|---|---|
| `lidar/` | Range Image 3D points |
| `lidar_segmentation/` | Lidar Labels |
| `lidar_calibration/` | Extrinsic Matrices per Laser |
| `camera_image/` | Images per Timestamp |
| `camera_calibration/` | Extrinsic / Intrinsic Matrices |


---

## What I've Built So Far

<!-- TODO: Turn this into an honest checklist of what's working. Check the boxes
     you've actually completed. This is the "what I've done" half of the story. -->

- [ ] GCS authentication & parquet access
- [ ] Range-image decode (spherical → Cartesian)
- [ ] Extrinsic transform to a global frame
- [ ] Multi-laser fusion into one point cloud
- [ ] Segmentation-label decoding & per-point coloring
- [ ] Memory-safe, timestamp-aligned data loading
- [ ] Bird's-eye + 3D (Plotly) visualization
- [ ] Scene animation (matplotlib / ffmpeg)
- [ ] LiDAR → camera projection
- [ ] LiDAR-camera fused overlay video
- [ ] Labeled sensor-fusion render
- [ ] _..._

---

## Technical Deep-Dives

<!-- This is the heart of the README — where you show you actually understand
     the machinery, not just that it runs. For each topic, explain the concept
     in your own words and note what tripped you up. Short is fine; clarity matters. -->

### Range Images → 3D Points (Spherical → Cartesian)

<!-- TODO: What is a range image? What are azimuth (phi), inclination (theta),
     and range (rho)? How do you convert them to X/Y/Z? -->

Converting spherical coordinates (phi, theta, rho: range image format) to cartesian (x, y, z) was a refresher from Calculus III, and a welcome one since I found a worthy application of it. This is necessary for plotting in a 3D space, as well as for future model training.

### Beam Inclination & Azimuth Correction

<!-- TODO: Why beam inclinations need reversing for laser 1, and why the azimuth
     grid must be corrected by the sensor's mounting yaw. This was a real bug —
     describe what went wrong and how you found it. -->

This was a challenge that I realized deep into development, as I didn't know that the sensor had a mounting yaw, and had to apply this to the azimuth (phi) calculation. This made the everything swing on the wrong bearing, which ruined the segmentation plotting. I realized it was necessary to apply this transformation to get the correct image.

For beam inclination, I assumed the beam values were in descending order, but were actually ascending, so this capped my height to a wrong value when plotting.

### Extrinsic Transform (Sensor → Vehicle/Global Frame)

<!-- TODO: What the 4x4 extrinsic matrix does, and how homogeneous coordinates
     let you apply it as a single matrix multiply. -->

The extrinsic matrix is for the camera and lidar sensors, as it relates the position of these sensors so that their measured points can be represented relative to them (or global, not relative to them). 

To use these with the points, we must stack the X, Y, and Z coordinates in a numpy array, then add a 1's column to the right to make it 4xN (homogeneous) after transposing, then do a matrix multiple by the extrinsic matrix, and finally get rid of the 4th added column. 

### Segmentation Labels

<!-- TODO: Label tensor layout (instance vs semantic channel), the fact that
     only laser 1 is labeled, and how labels stay aligned to points through masking. -->

After inspecting the data with pandas, I noticed that the segmentation labels are pretty sparse: only about 30 timestamps compared to 198 for lidar, as well as only laser 1 containing labels. 

To get these labels, there is a Masking that has to be done to get only true values (values that are actually visible, non negative) after converting. 

### LiDAR → Camera Projection

<!-- TODO: world → camera-local (inverse extrinsic), perspective divide by depth,
     intrinsic scaling (f_u/f_v/c_u/c_v), behind-camera and in-bounds masking.
     This is the core of the fusion work. -->

To get the already-processed 3D global coordinates relative to the camera we are taking frames from, we must:

1. Multiply the 3D global coords by the inverse of the extrinsic matrix for the camera.

2. Divide by the depth (X axis in this case) to get a normalized set of 2D coords (u, v)

3. Scale by the intrinsic values of the camera (focal length, lens centerpoint)

Finally, you take these (u, v) coordinates and do a masking that only takes points within the bounds of the image dimensions. 

---

## Pipeline & Project Structure

| File | Role |
|---|---|
| `semseg.ipynb` | Semantic Segmentation |
| `semseg_functions.py` | SemSeg functions |
| `sensor_fusion.ipynb` | Sensor Fusion learning |
| `sensor_fusion_functions.py` | Fusion functions |
| `media/` | videos/plots generated |

<!-- TODO (optional): a small diagram or bullet flow of how data moves:
     parquet row group → decode → transform → fuse → project → render. -->


---

## Sensor Fusion

<!-- TODO: Describe the early-fusion approach: projecting labeled 3D points onto
     the 2D camera image. Note the dual-coloring strategy for sparse labels
     (semantic color where labels exist, depth fallback elsewhere) and why label
     interpolation across frames isn't viable. -->

This implementation does an Early Fusion approach, but I will try Late Fusion in the future. We are using depth to measure and color the Lidar beams overlayed on top of the image (working on segmentation labels actively, TBD)

---

## Challenges & Lessons Learned

<!-- TODO: This section is gold for showing learning. Be specific and honest.
     Candidates from your journey:
     - The swapped azimuth/elevation meshgrid bug (wrong cloud shape)
     - The "overlay too high" bug (beam order + azimuth yaw)
     - Kernel crashes from reloading huge LiDAR row groups
     - Dropped animation frames hitting matplotlib's embed limit
     - Memory-safe row-group batching
     For each: what broke, how you diagnosed it, what you learned. -->

1. Memory Usage

- My first implementations of the data retrieval and processing algorithms were very sub-optimal, and it led to my kernel crashing quite often, so I tried to think of ways to minimize my data usage while still getting demonstrative results.

- Issues included retrieving large files multiple times for only a small portion of their data, holding large dataframes in memory for too long, and loading unnecessary columns that went unused.

- FIX: being memory efficient and doing processing/projecting immediately after loading to not hold too much data in memory. Also using the "del" keyword and the garbage collector to delete data that that wasn't necesarry in the loop.


2. Unaligned LiDAR and Camera for Fusion video

- The lidar points were too high up on the image, and it took me a while to figure out that there was a root issue in my lidar processing function, which had to do with height correction along with azimuth.

- FIX: I had to reverse the theta series array because I assumed it would be in descending order, but was actually in ascending, which changed my point cloud direction change when iterating over timestamps. 

- FIX: I had to do a small transformation to the azimuth calculation to factor in the yaw of the sensor, which translated it to be visualized at the correct angle relative to the camera.

---

## Roadmap

<!-- TODO: The "what I plan to do" half. Order by what's next. Candidates:
     - Integrate seg labels into the fused render (dual-coloring)
     - 3D bounding-box visualization
     - Aggregate labeled frames across many segments
     - Train a 3D semantic-segmentation model
     - Finish modularizing helpers into the *_functions.py files -->

- [ ] _Next: Implementing predefined segmentation labels into the sensor fusion pipeline._
- [ ] _Later: Creating a model to detect labels from each relevant sensor output._
- [ ] _Eventually: Running optimized versions of these perception functions in a CARLA simulator to evaluate my progress, and iterate from there._

---

## Setup & Running

<!-- TODO: How someone (including future-you) gets this running:
     - Python / environment / key dependencies (pyarrow, pandas, numpy, etc.)
     - gcloud auth requirement (token expires after 1 hour)
     - Which notebook to open and run -->

**Environment:** Python 3.10, run inside Jupyter (these are notebook-driven).

**Python dependencies:** `pyarrow` (parquet + GCS filesystem access), `pandas`,
`numpy`, `matplotlib`, `Pillow` (JPEG decode), `plotly` (interactive 3D),
`tensorflow` and `open3d` (imported by the helper module), plus `gcsfs` and
`google-cloud-storage`. Install them into a Python 3.10 environment with your
package manager of choice.

**System dependency:** `ffmpeg` must be on your PATH — the scene/fusion
animations are written to disk with matplotlib's `FFMpegWriter`.

**Google Cloud authentication:** The data is read live from the public GCS
bucket `waymo_open_dataset_v_2_0_0` — there are no local copies. Authentication
goes through the Google Cloud SDK: sign in once with the `gcloud auth login` flow,
and make sure `gcloud` is installed (this project expects it at `/usr/bin/gcloud`,
with config under `/home/jacob/.config/gcloud` — adjust the two paths at the top
of `semseg_functions.py` for your machine). On import, the helper module shells
out to `gcloud auth print-access-token` and builds the `GcsFileSystem` from that
token. **The token expires after one hour**, so for long sessions you'll need to
re-import the module (or re-run its first cell) to refresh it.

**Running it:** Open `semseg.ipynb` for the LiDAR-only segmentation pipeline, or
`sensor_fusion.ipynb` for the LiDAR-camera fusion work, and run the cells top to
bottom. Generated videos and plots land in `media/`.

---

## References

<!-- TODO: Links and resources that helped you. Candidates:
     - Waymo Open Dataset docs / paper
     - Camera intrinsics / extrinsics / pinhole model references
     - Spherical-to-Cartesian / range-image explanations -->

**Dataset**
- [Waymo Open Dataset — official site](https://waymo.com/open/)
- [Waymo Open Dataset v2.0 documentation](https://waymo.com/open/data/perception/)
- Sun et al., *Scalability in Perception for Autonomous Driving: Waymo Open Dataset*, CVPR 2020 — [arXiv:1912.04838](https://arxiv.org/abs/1912.04838)

**Data access**
- [Apache Arrow / PyArrow — reading Parquet](https://arrow.apache.org/docs/python/parquet.html)
- [PyArrow filesystems — Google Cloud Storage](https://arrow.apache.org/docs/python/filesystems.html#google-cloud-storage-file-system)
- [gcloud auth print-access-token reference](https://cloud.google.com/sdk/gcloud/reference/auth/print-access-token)

**Geometry & projection**
- [OpenCV camera calibration & 3D reconstruction (pinhole model, intrinsics/extrinsics)](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [Spherical coordinate system (azimuth / inclination / radius)](https://en.wikipedia.org/wiki/Spherical_coordinate_system)
- [Homogeneous coordinates](https://en.wikipedia.org/wiki/Homogeneous_coordinates)

<!-- TODO: add any blog posts, videos, or docs that personally helped you click. -->
- _..._

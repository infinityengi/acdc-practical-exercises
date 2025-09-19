# Module 01 — Sensor Data Processing I

> README for Module 01 of the ACDC Student Portfolio

---

## TL;DR
This module demonstrates sensor-data processing and perception skills for image and LiDAR modalities. Deliverables include Jupyter notebooks, model training scripts, trained model artifacts, ROS/ROS2 demo packages, visualizations (PNGs/GIFs), and a metrics report (MIoU, per-class IoU). The repository is structured so each subtask maps directly to GitHub Issues and project board cards for progress tracking.

---

## Badges (placeholders)

[![Build Status](https://img.shields.io/badge/build-pending-lightgrey)](#)
[![Notebook Coverage](https://img.shields.io/badge/notebooks-complete-yellowgreen)](#)
[![License](https://img.shields.io/badge/license-MIT-blue)](#)

---

## Table of contents
- [Overview](#overview)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Notebooks & Scripts (what to run)](#notebooks--scripts-what-to-run)
- [ROS packages & demos](#ros-packages--demos)
- [Artifacts & Results](#artifacts--results)
- [Progress tracking & converting to GitHub Issues](#progress-tracking--converting-to-github-issues)
- [Development notes & tips](#development-notes--tips)
- [Optional improvements / future work](#optional-improvements--future-work)
- [Contacts & acknowledgements](#contacts--acknowledgements)

---

## Overview
**Module goal:** Implement end‑to‑end pipelines for semantic segmentation on both image and LiDAR data, using Jupyter notebooks for experimentation and ROS/ROS2 nodes for deployment and visualization.

**Key skills demonstrated:**
- Dataset parsing & preprocessing for images and projected LiDAR range-images
- Data augmentation policies and class-imbalance techniques (focal loss)
- Building and training segmentation networks (U-Net, SqueezeSegV2)
- Evaluation (MIoU, confusion matrix, per-class IoU)
- ROS / ROS2 integration: nodes, launch files, bag playback, RViz visualization
- Packaging deliverables and reproducible experiments

---

## Quick start
### Prerequisites
- Linux (Ubuntu recommended)
- Python 3.8+ (virtualenv or conda recommended)
- GPU + CUDA (for training, optional for inference)
- ROS / ROS2 (match the version used in your course; see `ros/README.md`)

### Create environment (example)
```bash
# create and activate virtualenv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Quick run: inference demo (example)
```bash
# Run a minimal inference notebook (Jupyter)
jupyter notebook notebooks/pcl_inference.ipynb
# or run a one-off script
python scripts/run_pcl_inference.py --model artifacts/pcl/saved_model --input data/samples/sample.npy --out results/sample_seg.png
```

> Replace `artifacts/...`, `data/...` with actual paths or links in this repository.

---

## Repository layout (recommended)
```
module01/
├─ notebooks/
│  ├─ pcl_segmentation.ipynb
│  ├─ pcl_inference.ipynb
│  ├─ image_segmentation_unet.ipynb
│  └─ image_seg_inference.ipynb
├─ data/
│  ├─ pcl/            # .npy projected scans (train/val)
│  └─ image/          # images + segmentation masks
├─ models/
│  ├─ squeezesegv2.py
│  └─ unet.py
├─ scripts/
│  ├─ train_pcl.py
│  ├─ train_image_seg.py
│  └─ run_pcl_inference.py
├─ ros_ws/
│  └─ src/pcl_segmentation/    # ROS1 package
├─ ros2_ws/
│  └─ src/image_segmentation_r2/  # ROS2 package
├─ artifacts/
│  ├─ pcl/saved_model/
│  └─ image/saved_model/
├─ reports/
│  └─ module01_metrics.md
├─ requirements.txt
└─ Module01_README.md
```

---

## Notebooks & Scripts (what to run)
Each bullet is ready to convert to a GitHub Issue; use the listed subtasks as checklist items.

### 1) `pcl_segmentation.ipynb` — Semantic Point Cloud Segmentation
- **Purpose:** train & evaluate SqueezeSegV2 on projected LiDAR (range images).
- **Run / commands:**
```bash
# train
python scripts/train_pcl.py --data_dir data/pcl --epochs 10 --batch_size 4 --out_dir artifacts/pcl
# inference
python scripts/run_pcl_inference.py --model artifacts/pcl/saved_model --input data/pcl/val/sample.npy --out results/pcl/sample_seg.png
```
- **Deliverables:** `artifacts/pcl/checkpoint-*`, `results/pcl/*.png`, training logs, `reports/module01_metrics.md`
- **Placeholders:** [Notebook link](NOTEBOOK_LINK_HERE) | [Model weights](MODEL_WEIGHTS_LINK_HERE)

### 2) `image_segmentation_unet.ipynb` — Image segmentation (U-Net)
- **Purpose:** train U-Net, compare augmentation policies, report MIoU.
- **Run / commands:**
```bash
python scripts/train_image_seg.py --data_dir data/image --epochs 30 --batch_size 4 --augment True --out_dir artifacts/image
```
- **Deliverables:** saved model, inference PNGs, loss/MIoU plots.
- **Placeholders:** [Notebook link](NOTEBOOK_LINK_HERE) | [Model weights](MODEL_WEIGHTS_LINK_HERE)

---

## ROS packages & demos
Two ROS demos are included: a point-cloud segmentation ROS1 package and an image segmentation ROS2 pipeline. Each package contains a node, a launch file, and demo bag playback instructions.

### ROS (Point Cloud segmentation)
- **Package:** `ros_ws/src/pcl_segmentation/`
- **Run (example):**
```bash
# in a catkin workspace
after building:
roslaunch pcl_segmentation demo.launch bag:=data/bags/pcl_demo.bag model:=artifacts/pcl/saved_model
```
- **Outcome:** publishes `/sensors/lidar/segmented` (PointCloud2) viewable in RViz.
- **Placeholders:** [ROS package link](ROS_PACKAGE_LINK_HERE) | [Demo bag](BAG_LINK_HERE)

### ROS2 (Image segmentation)
- **Package:** `ros2_ws/src/image_segmentation_r2/`
- **Run (example):**
```bash
# play bag
ros2 bag play data/bags/image_demo
# launch segmentation nodes
ros2 launch image_segmentation_r2 image_segmentation_r2.launch.py model:=artifacts/image/saved_model
```
- **Outcome:** publishes `/image_segmented`, can be recorded to disk or shown in rqt_image_view.

---

## Artifacts & Results
These are the deliverables that should be committed to the `artifacts/` folder or hosted as release assets:
- `artifacts/pcl/saved_model/` — SqueezeSegV2 SavedModel or checkpoint
- `artifacts/image/saved_model/` — U-Net SavedModel
- `reports/module01_metrics.md` — training curves, MIoU, per-class IoU, confusion matrices
- `results/` — example PNGs, rotating 3D point-cloud GIFs, demo videos

**Placeholders for visual media:**

![Demo GIF placeholder](GIF_PLACEHOLDER)

---

## Progress tracking & converting to GitHub Issues
Suggested labels: `status/To Do`, `status/In Progress`, `status/Blocked`, `status/Done`, `type/notebook`, `type/ros`, `priority/high`.

Example issue checklist (paste into new Issue body):
```markdown
### Title: pcl: implement data loader and preprocessing
#### Goal
Implement `parse_sample` to read .npy point-cloud range images, normalize, and return (lidar, mask, label).

#### Acceptance criteria
- [ ] Loads a sample and prints shapes `[32,240,6]`
- [ ] Normalizes using config mean/std
- [ ] Binary mask for depth>0
- [ ] Unit tests for shape and dtype

Files: `data/parse_pcl.py`, `tests/test_parse_pcl.py`
```

### Suggested issue table (paste into project board CSV)
| Issue | Owner | Status | Artifact |
|---|---:|---|---|
| `pcl: data loader` | @you | To Do | `data/parse_pcl.py` |
| `pcl: augmentations` | @you | To Do | `utils/augmentations.py` |
| `pcl: SqueezeSegV2 model` | @you | To Do | `models/squeezesegv2.py` |
| `pcl: ROS node` | @you | To Do | `ros_ws/src/pcl_segmentation` |

---

## Development notes & tips
- **Reproducibility:** pin dependencies in `requirements.txt` and add a `conda` environment yaml if needed.
- **Data handling:** keep raw datasets out of `git`; add small sample data for demos in `data/samples/`. Use `.gitignore` for large artifacts.
- **Model exports:** provide both `SavedModel` and a lightweight `TFLite` (optional) for edge demos.
- **Performance:** measure per-frame inference latency and add it to `reports/module01_metrics.md`.
- **Testing:** include unit tests for data parsers and loss functions. Keep model training out of CI unless using small smoke tests.

---

## Optional improvements / future work
- Add an interactive web demo (Streamlit/Gradio) to toggle GT / preds for images.
- Add unit tests and CI (GitHub Actions) to run linting and lightweight notebooks with `nbconvert` smoke tests.
- Add TFLite/ONNX conversion for edge deployment and CPU-only demos.
- Add README GIFs and short demo videos for recruiter-friendly presentation.

---

## Contacts & acknowledgements
- **Author / Owner:** Your Name — replace with your GitHub handle and contact email.
- **Course:** ACDC — Sensor Data Processing I

---

*This README is generated to be recruiter-friendly and directly convertible into GitHub Issues / project board cards. Replace the `PLACEHOLDER` links with actual notebook/model/bag links before publishing.*


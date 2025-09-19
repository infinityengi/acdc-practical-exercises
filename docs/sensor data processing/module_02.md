# Module 02 — Sensor Data Processing II
**Repository path suggestion:** `modules/module_02_sensor_data_processing_ii/`

---

## Module Overview / Summary
- **Goal:** Process and fuse multi-sensor data (GNSS, LiDAR, multi-camera, point clouds) to build robust localization and occupancy representations for automated driving.  
- **Learning outcomes:** convert geodetic → local frames (WGS84 → UTM → map), implement LiDAR odometry, fuse GNSS + odometry for high-rate pose prediction, construct BEV semantic grid maps from multi-camera input (IPM), implement classical geometric occupancy-grid mapping (Inverse Sensor Model + Binary Bayes) and a deep-learning-based occupancy model (PointPillars → Evidential OGM), and evaluate performance quantitatively in notebooks.  
- **Key skills demonstrated:** ROS2 (nodes, TF2, message_sync), point-cloud processing (PCL), LiDAR odometry/ICP, camera geometry (intrinsics/extrinsics, homography/IPM), occupancy-grid algorithms, TensorFlow model training & deployment, evaluation with pandas and visualization.

---

## Quick TL;DR
Module 02 implements end-to-end sensor processing: GNSS-to-UTM conversion and yaw estimation, LiDAR odometry (KISS-ICP) and pose prediction, multi-camera synchronization and IPM-based BEV fusion, geometric Inverse Sensor Model with Binary Bayes for occupancy, and a PointPillars-based evidential OGM solution (training + ROS nodelet). Deliverables: runnable ROS packages, Jupyter notebooks for evaluation, trained model weights, RViz configs, and result visualizations (PNGs / GIFs). This is structured so each subtask can be converted directly into a GitHub Issue / project card.

---

## Table of Contents
- [A — Localization: GNSS + LiDAR Odometry](#a--localization-gnss--lidar-odometry)
- [B — Camera-based Semantic Grid Mapping (IPM / BEV fusion)](#b--camera-based-semantic-grid-mapping-ipm--bev-fusion)
- [C — Point Cloud Occupancy Grid Mapping (Geometric & Deep ISM)](#c--point-cloud-occupancy-grid-mapping-geometric--deep-ism)
- [Artifacts / Deliverables](#artifacts--deliverables)
- [Progress Tracking / Status Notes](#progress-tracking--status-notes)
- [How to convert to GitHub Issues](#how-to-convert-to-github-issues)
- [Optional Improvements / Nice-to-haves](#optional-improvements--nice-to-haves)
- [Contact / Next steps](#contact--next-steps)

---

## A — Localization: GNSS + LiDAR Odometry (ROS + Jupyter evaluation)

### Summary
Fuse low-frequency GNSS with high-frequency LiDAR odometry to produce a robust, high-rate predicted pose stream in a local map frame. Evaluate predicted pose vs ground-truth in a Jupyter notebook.

### A.1 Prepare dataset & environment
- **Objective:** Load rosbag(s) and prepare a reproducible ROS2 workspace.
- **Inputs / Requirements:**
  - `localization.db3` (rosbag2) placed in `data/rosbags/`
  - ROS2 distribution (e.g., Foxy / Humble) installed
  - `colcon` / workspace (`src/` with packages)
- **Expected output / artifact:**
  - Confirmed rosbag metadata; `data/rosbags/localization.db3`
- **Tools / Technologies:**
  - `ros2`, `ros2 bag`, `colcon`, Python (pandas, numpy)
- **Commands / Snippets**
  ```bash
  # Inspect bag
  ros2 bag info data/rosbags/localization.db3

  # Build workspace
  colcon build --packages-select localization_node
  source install/setup.bash

* **Notes / Tips:** Copy the bag into repo `data/` to make experiment reproducible.

---

### A.2 GNSS → UTM → local map transform

* **Objective:** Convert GNSS lat/lon → UTM coordinates and align to a `carla_map` or local `map` origin.
* **Inputs / Requirements:**

  * `/gnss/navsatfix` messages or CSV of lat/lon
  * `GeographicLib` (C++) or `pyproj` (Python)
* **Expected output / artifact:**

  * `utils/project_to_utm.py` or `projectToUTM()` function that returns (utm\_x, utm\_y, zone)
  * A chosen local origin and transform to `map` frame
* **Tools / Technologies:** `pyproj` or `GeographicLib`; TF2 for publishing transforms
* **Notes / Tips:** Use the first GNSS sample as a local origin to keep coordinates small and numerical stable; handle zone transitions robustly.

---

### A.3 Estimate yaw from GNSS track

* **Objective:** Compute heading (yaw) from consecutive GNSS (UTM) points.
* **Inputs / Requirements:** consecutive UTM points (x,y) with timestamps
* **Expected output / artifact:** `gnss_heading_node` or function that publishes `/localization/gnss_pose` (`geometry_msgs/PoseStamped`)
* **Tools / Technologies:** numpy, TF2
* **Implementation hint (pseudo):**

  ```python
  def estimate_yaw(last_point, current_point):
      dx = current_point.x - last_point.x
      dy = current_point.y - last_point.y
      yaw = np.arctan2(dy, dx)
      return yaw
  ```

---

### A.4 LiDAR odometry (KISS-ICP)

* **Objective:** Create a LiDAR odometry node that publishes incremental odometry between scans.
* **Inputs / Requirements:** `/lidar/points` or `/points2` topic
* **Expected output / artifact:** `/lidar/odometry` or TF transforms at LiDAR rate
* **Tools / Technologies:** KISS-ICP or other ICP variant, PCL, ROS2 node
* **Notes / Tips:** Tune `min_motion_thresh` and distance filters to reduce false motion; publish both odom and TF for downstream use.

---

### A.5 Fuse GNSS + LiDAR odometry → Predicted pose

* **Objective:** Apply LiDAR incremental deltas to last GNSS pose to produce high-rate predicted poses in the `map` frame.
* **Inputs / Requirements:** GNSS pose (low frequency), LiDAR odometry deltas (high frequency)
* **Expected output / artifact:** `/localization/predicted_pose` publishing predicted poses
* **Tools / Technologies:** TF2, Python/C++
* **Implementation outline:**

  1. When a new GNSS pose arrives, store as `pose_map`.
  2. For each LiDAR odometry delta, compute new pose:

     * Extract translation delta `(dx, dy)` and rotation delta `dpsi`.
     * Rotate and translate deltas according to current `pose_map` yaw (use quaternion multiplication or 2D transform).
  3. Publish predicted pose with LiDAR timestamp.
* **Notes / Tips:** For very small deltas, linearize rotations to avoid rounding issues.

---

### A.6 Visualization & Recording

* **Objective:** Visualize trajectories (ground truth, GNSS, predicted) in RViz and record evaluation bag.
* **Expected output / artifact:** RViz config file `rviz/localization.rviz`, recorded bag `data/localization_evaluation.db3`
* **Commands / Snippets:**

  ```bash
  ros2 run rviz2 rviz2 -d rviz/localization.rviz
  ros2 bag record /ground_truth/pose /localization/predicted_pose /localization/gnss_pose -o data/localization_evaluation
  ```
* **Notes / Tips:** Color-code topics (e.g., GT = red, GNSS = purple, Predicted = green).

---

### A.7 Jupyter evaluation notebook

* **Objective:** Compute position & yaw errors and produce evaluation plots & metrics.
* **Inputs / Requirements:** `data/localization_evaluation.db3` (or original bag); Python packages: `rosbags`/`rosbag2`, `pandas`, `numpy`, `matplotlib`
* **Expected output / artifact:** `notebooks/localization_evaluation.ipynb`
* **Subtasks / Steps:**

  1. Extract poses for ground-truth and predicted into DataFrames with timestamps.
  2. Align time series using `pd.merge_asof` (tolerance = e.g. 50 ms).
  3. Compute error columns:

     * `dx = x_pred - x_gt`, `dy = y_pred - y_gt`
     * vehicle-centric errors:

       ```python
       df['dlon'] = df['dx']*np.cos(df['psi_gt']) + df['dy']*np.sin(df['psi_gt'])
       df['dlat'] = -df['dx']*np.sin(df['psi_gt']) + df['dy']*np.cos(df['psi_gt'])
       ```
     * yaw error `dpsi = wrap_angle(yaw_pred - yaw_gt)`
  4. Compute metrics: RMSE (position), lateral RMSE, longitudinal RMSE, yaw MAE.
  5. Plots: trajectory overlay, per-axis error over time, boxplots of errors.
* **Notes / Tips:** Export metrics as `results/metrics_localization.json` for CI.

---

## B — Camera-based Semantic Grid Mapping (IPM → BEV fusion)

### Summary

Synchronize multiple camera streams, compute intrinsic/extrinsic matrices, apply inverse perspective mapping (IPM) to generate BEV tiles from each camera, and fuse them into a semantic BEV / grid map.

### B.1 Build & run the package

* **Objective:** Build ROS package and confirm node runs.
* **Commands:**

  ```bash
  cd colcon_ws
  colcon build --packages-select camera_based_semantic_grid_mapping_r2
  source install/setup.bash
  ros2 run camera_based_semantic_grid_mapping_r2 camera_bev_node
  ```

---

### B.2 Synchronize multi-camera topics

* **Objective:** Use `ApproximateTimeSynchronizer` to obtain temporally aligned images + CameraInfo from N cameras.
* **Inputs / Requirements:** `/camX/image_raw` and `/camX/camera_info` for X ∈ {0..7}
* **Expected output / artifact:** Synchronized callback that receives `List[Image]` and `List[CameraInfo]`
* **Tools / Technologies:** ROS2 Python / `message_filters`
* **Code snippet:**

  ```python
  import message_filters
  subs = []
  for cam in cams:
      subs.append(message_filters.Subscriber(node, Image, f"/cam{cam}/image_raw"))
      subs.append(message_filters.Subscriber(node, CameraInfo, f"/cam{cam}/camera_info"))
  ats = message_filters.ApproximateTimeSynchronizer(subs, queue_size=5, slop=0.01)
  ats.registerCallback(self.multi_cam_callback)
  ```
* **Notes / Tips:** Start with `slop=0.01` (10 ms). Increase if frames are dropped.

---

### B.3 Extract intrinsics & handle distortion

* **Objective:** Read `CameraInfo` message to build intrinsics matrix `K` and optionally undistort.
* **Inputs / Requirements:** `camera_info_msg.k` and `camera_info_msg.distortion_model`
* **Expected output / artifact:** per-camera `K` and `distCoeffs`, undistortion routine if needed
* **Code snippet:**

  ```python
  K = np.reshape(camera_info_msg.k, (3,3))
  dist = np.array(camera_info_msg.d)
  # optionally use cv2.undistort
  ```

---

### B.4 Compute extrinsics → homogeneous matrix

* **Objective:** Obtain the extrinsic transform from camera frame → vehicle base and build a 4×4 homogeneous matrix.
* **Inputs / Requirements:** TF2 tree with camera frames and base\_link
* **Expected output / artifact:** `E_cam_to_base` 4×4 matrix
* **Implementation hint:**

  ```python
  trans = tf_lookup.translation
  rot = tf_lookup.rotation  # quaternion
  R = quaternion_to_rotation_matrix(rot)
  E = np.eye(4)
  E[:3,:3] = R
  E[:3,3] = np.array([trans.x, trans.y, trans.z])
  ```

---

### B.5 Inverse Perspective Mapping (IPM) → BEV semantic grid

* **Objective:** Compute homography from ground plane to image plane and generate BEV tiles; fuse tiles from all cameras.
* **Inputs / Requirements:** images, `K`, `E`, chosen ground plane parameters (e.g., z=0)
* **Expected output / artifact:** Published BEV image (`sensor_msgs/Image`) and optionally `grid_map::GridMap` layers
* **Steps / Subtasks:**

  * Compute homography `H` using `K`, camera extrinsic `E`, and plane equation.
  * For each camera: transform image to BEV using OpenCV:

    ```python
    bev = cv2.warpPerspective(image, H, (bev_width, bev_height))
    ```
  * Fuse BEV tiles (e.g., weighted averaging or semantic class merge).
* **Tools / Technologies:** OpenCV (`cv2`), NumPy, ROS2 image\_transport
* **Notes / Tips:** Use per-camera weights based on distance / overlap. Crop far field or use confidence masks.

---

### B.6 RViz visualization

* **Objective:** Provide RViz config to visualize camera frames, camera images, BEV overlays, and fused grid map.
* **Expected output / artifact:** `rviz/camera_bev.rviz`

---

## C — Point Cloud Occupancy Grid Mapping

Module split into **C.1 Geometric ISM** and **C.2 DeepISM (PointPillars + evidential head)**.

---

### C.1 Geometric Inverse Sensor Model (classical)

#### C1.1 Ground point extraction (PCL PassThrough)

* **Objective:** Filter out ground / non-ground using PCL filters to isolate obstacles.
* **Inputs / Requirements:** `/points2`
* **Expected output / artifact:** topic `/points2_obstacles` or filtered pointcloud saved in `data/pointclouds/`
* **Tools / Technologies:** PCL, ROS nodelet (PassThrough)
* **Sample launch snippet:**

  ```xml
  <node pkg="nodelet" type="nodelet" name="GroundExtraction" args="load pcl/PassThrough $(arg nodelet_manager)">
    <remap from="~input" to="/points2" />
    <remap from="~output" to="/points2_obstacles" />
    <rosparam>
      filter_field_name: z
      filter_limit_min: -2.0
      filter_limit_max: 2.0
    </rosparam>
  </node>
  ```

#### C1.2 Implement Geometric ISM + Binary Bayes update

* **Objective:** Convert LiDAR reflections into occupancy probability grid via inverse sensor model + Bayesian update.
* **Inputs / Requirements:** filtered point cloud, sensor origin, grid parameters (cell size, extents)
* **Expected output / artifact:** published `grid_map` with `occupancy_probability` layer
* **Algorithm outline:**

  1. Initialize grid with prior `p = 0.5`.
  2. For each LiDAR hit:

     * Raytrace from sensor cell → reflected cell using `LineIterator`.
     * Assign `p_ism` values (e.g., reflection=0.9, near cells=0.8, free cells=0.1).
     * Update cell probability via Binary Bayes:

       ```cpp
       p_new = (p_ism * p_old) / (p_ism * p_old + (1-p_ism) * (1-p_old))
       ```
* **Tools / Technologies:** `grid_map` ROS package, C++ or Python
* **Notes / Tips:** Visualize intermediate layers (belief, hit-count) to debug.

#### C1.3 Compile & visualize

* **Commands:**

  ```bash
  # build
  colcon build --packages-select pointcloud_ogm
  source install/setup.bash

  # run
  ros2 launch pointcloud_ogm GeometricISM.launch
  ```
* **Artifacts:** `rviz/pointcloud_ogm.rviz`, `results/geometric_ogm/*.png`, demo GIF `results/geometric_ogm/demo.gif`

---

### C.2 Deep Learning-based ISM (PointPillars + Evidential OGM)

#### C2.1 Dataset & preprocessing

* **Objective:** Convert point clouds + labels into pillar tensors suitable for PointPillars training and generate OGM labels.
* **Inputs / Requirements:** raw pointcloud files + ground-truth grid maps (synthetic & real)
* **Expected output / artifact:** `data/tf_datasets/` or `data/npy/` with `(pillars, labels)` arrays
* **Subtasks:**

  * Implement `preprocess_sample()` that:

    * voxelizes points into pillars
    * computes pillar features (x,y,z,intensity)
    * produces target grid map per sample
  * Implement augmentation: random yaw rotation applied equally to points and grid labels
* **Tools / Technologies:** NumPy, TensorFlow data pipelines

#### C2.2 Build PointPillars backbone + evidential head

* **Objective:** Implement a PointPillars backbone and an evidential prediction head that outputs class evidences for `occupied` and `free`.
* **Inputs / Requirements:** pillar tensors
* **Expected output / artifact:** `models/pointpillars_evidential.py` + Keras model saving to `models/deepism_best.h5`
* **Architecture notes:**

  * Final conv: `Conv2D(filters=2, kernel_size=(3,3), activation='relu')` → evidences for `occupied` and `free`
  * Convert evidences → Dirichlet `alpha = evidence + 1`, compute belief masses `m_k = (alpha_k - 1) / S` with `S = sum(alpha)`
* **Loss & metrics:**

  * Custom `ExpectedMeanSquaredError` that penalizes incorrect beliefs while accounting for uncertainty
  * Evaluate IoU for occupied class and calibration metrics

#### C2.3 Training & evaluation

* **Objective:** Train the model and evaluate on held-out datasets.
* **Inputs / Requirements:** preprocessed datasets, training hyperparameters file `models/config.json`
* **Expected output / artifact:** training logs, `models/deepism_best.h5`, validation results and visual examples
* **Subtasks:**

  * Implement training script `train_deepism.py`
  * Save best checkpoint and training curves to `results/training/`
  * Evaluate on `dataset_test` and `dataset_real`, produce per-cell IoU, accuracy, and calibration plots

#### C2.4 Deployment (nodelet)

* **Objective:** Deploy trained model as a ROS nodelet that consumes pointclouds and publishes evidential grid maps.
* **Inputs / Requirements:** saved model weights, live `/points2`
* **Expected output / artifact:** `src/pointcloud_ogm/DeepISM_nodelet` publishing grid-map layers: `m_occupied`, `m_free`, and `uncertainty`
* **Implementation details:**

  * Implement `tensor_to_grid_map(prediction, grid_map)` mapping tensor outputs back onto the same coordinate system used by Geometric ISM
  * Ensure correct scaling and alignment
* **Notes / Tips:** Validate DeepISM side-by-side with GeometricISM on the same input to inspect differences.

---

## Artifacts / Deliverables

* **Notebooks**

  * `notebooks/localization_evaluation.ipynb` — extraction, alignment, error metrics, plots
  * `notebooks/camera_bev_demo.ipynb` — IPM + multi-camera BEV fusion walkthrough
  * `notebooks/deepism_training.ipynb` — data prep and training summary
* **ROS packages / source**

  * `src/localization_node/` — GNSS → UTM, yaw estimator, posePrediction
  * `src/camera_based_semantic_grid_mapping_r2/` — multi-camera sync, IPM, BEV fusion node
  * `src/pointcloud_ogm/` — `GeometricISM` and `DeepISM` nodelets
* **Model artifacts**

  * `models/deepism_best.h5`
  * `models/config.json`
* **Visuals & configs**

  * `rviz/localization.rviz`, `rviz/camera_bev.rviz`, `rviz/pointcloud_ogm.rviz`
  * `results/*.png`, `results/*.gif`
* **Data**

  * `data/rosbags/localization.db3`
  * Prepared datasets: `data/tf_datasets/` or `data/npy/`
* **Docs**

  * `README.md` per package (run instructions)
  * `docs/IPM_explanation.md` (homography formulas + examples)
  * `docs/DeepISM_model_card.md` (architectural choices, metrics)

> Placeholder links (replace with real links in repo):
>
> * Notebook: `[localization_evaluation.ipynb](./notebooks/localization_evaluation.ipynb)`
> * Model: `[deepism_best.h5](./models/deepism_best.h5)`
> * ROS package: `[src/localization_node](./src/localization_node)`

---

## Progress Tracking / Status Notes

Use this table to quickly convert into GitHub Projects or Issues.

| Task ID | Task          | Subtasks (high-level)            |           Status | Owner | Issue / PR    |
| ------- | ------------- | -------------------------------- | ---------------: | ----- | ------------- |
| M02-A1  | Dataset & env | Inspect bag, build workspace     |            To Do | @you  | `[ISSUE-###]` |
| M02-A2  | GNSS → UTM    | `projectToUTM()` + origin        |      In Progress | @you  | `[ISSUE-###]` |
| M02-A3  | GNSS yaw      | implement & publish `gnss_pose`  |            To Do | @you  | `[ISSUE-###]` |
| M02-A4  | LiDAR odom    | integrate KISS-ICP               |            To Do | @you  | `[ISSUE-###]` |
| M02-A5  | Pose fusion   | posePrediction node              |            To Do | @you  | `[ISSUE-###]` |
| M02-A7  | Evaluation    | Notebook & metrics export        |            To Do | @you  | `[ISSUE-###]` |
| M02-B2  | Camera sync   | ApproxTimeSynchronizer           |            To Do | @you  | `[ISSUE-###]` |
| M02-B5  | IPM + BEV     | compute homographies, warp, fuse |            To Do | @you  | `[ISSUE-###]` |
| M02-C1  | Geometric ISM | PassThrough, raytrace, bayes     | Done / To verify | @you  | `[PR-###]`    |
| M02-C2  | DeepISM       | data prep, train, nodelet deploy |      In Progress | @you  | `[ISSUE-###]` |

**Suggested Issue checklist (copy to any Issue body)**

* [ ] Implement core function(s)
* [ ] Unit / smoke test
* [ ] Visual verification (RViz / notebook)
* [ ] Documentation (`README.md`, usage)
* [ ] Create PR & request review

---

## How to convert each subtask into a GitHub Issue (recommended format)

**Issue title:** `M02-A5: Fuse GNSS + LiDAR odometry → predicted_pose node`
**Issue body template:**

* **Description:** Brief explanation of the goal.
* **Acceptance criteria:**

  * [ ] Node consumes `/lidar/odometry` and `/localization/gnss_pose`
  * [ ] Publishes `/localization/predicted_pose` at LiDAR rate
  * [ ] Unit test for `getIncrementalMovement()` with synthetic input
  * [ ] Notebook `notebooks/localization_evaluation.ipynb` shows reduced RMSE compared to baseline
* **Files to change:** `src/localization_node/*`
* **Tests / checks:** `nbval` for notebook example; smoke-run script `scripts/run_localization_eval.sh`

---

## Optional Improvements / Nice-to-haves

* Auto-export evaluation metrics as JSON (`results/metrics_localization.json`) for CI regression checks
* Add GIF demos for each major output (localization overlay, BEV fusion, OGM comparisons)
* Add unit tests for math utilities: `estimateGNSSYawAngle`, `posePrediction`
* Dockerfile / devcontainer for reproducing experiments
* Small static web page summarizing each module's demo and key metrics

---

## Summary / Final Notes

This document is intentionally structured to be recruiter-friendly and directly actionable. Each major assignment is broken into small, testable subtasks that can be copied into GitHub Issues or a project board. Deliverables include notebooks for evaluation, ROS packages for runtime, model artifacts, and clear visualization configs.


## Module 03 — Object Prediction, Association & Fusion (Multi-Object Tracking with Kalman Filter)

> Short name you can use for GitHub boards / issues: **`M03-MOT-KF`**

---

## 1) Module overview / summary

* **Goal:** Implement a full Multi-Object Tracking (MOT) pipeline in ROS using a Multi-Instance Kalman Filter (MIKF):

  1. predict global tracks to current timestamp,
  2. associate sensor detections (camera, radar) to global tracks (IoU / Mahalanobis),
  3. perform measurement update (Kalman measurement update / fusion).
* **Learning outcomes:**

  * Correctly implement Kalman prediction & update equations in C++ (ROS nodes).
  * Implement robust association (IoU & Mahalanobis) and tune thresholds.
  * Visualize and evaluate fused tracks in RViz and quantitatively compare parameter choices.
* **Key skills demonstrated:** Kalman filtering (prediction & update), uncertainty handling (covariances), association methods, ROS (rosbag, nodes, topics), Eigen linear algebra use in C++, RViz visualization, experimental parameter tuning and evaluation, producing polished artifacts (notebooks, scripts, RViz screenshots/GIFs).

---

## 2) Detailed task breakdown (organized so each can be a GitHub Issue / Project card)

> For each assignment below I list: **Objective → Inputs/Requirements → Expected output / artifact → Tools/technologies → Step-by-step subtasks → Implementation notes / tips → Optional improvements**.

---

### A — Task A1: Prediction step (StatePredictor.cpp)

**Objective:** implement Kalman filter prediction for each `GlobalObject` so global tracks are time-aligned with incoming measurements.

* **Inputs / requirements**

  * `data_->object_list_fused.objects` (global tracks)
  * Ego motion topic (`/sensors/vehicleCAN/ikaEgoMotion`) optionally for coordinate updates
  * `Δt` (time difference) computed from timestamps
  * `F_const`, `F_timevar`, `Q_timevar` entries from `kalman_filter.yaml` (or fusion config)

* **Expected output / artifact**

  * Modified `data_->object_list_fused.objects` with predicted state `x_hat_G` and covariance `P` updated; updated timestamps.
  * Unit test / notebook showing before/after predicted positions for a sample frame.
  * RViz screenshot/GIF of predicted tracks vs detections.

* **Tools / technologies**

  * ROS (catkin / colcon), C++ (StatePredictor.cpp), Eigen.
  * rosbag file (≈77.4 MB, \~125,992 messages — used in lab).
  * RViz for visualization.

* **Step-by-step subtasks**

  1. Add scaffolding: open `StatePredictor.cpp` and load config params (`F_const`, `F_timevar`, `Q_timevar`).
  2. For each `globalObject` do:

     * Compute `Δt = globalObject.header.stamp - previous_stamp` (guard `Δt>0`)
     * Build `F = F_const + Δt * F_timevar`
     * Build `Q = Δt * Q_timevar` (or use provided time-variant scaling)
     * Convert `globalObject` state to Eigen vector (`IkaUtilities::getEigenStateVec`)
     * Compute `x_hat_G_new = F * x_hat_G_old`
     * Compute `P_new = F * P_old * F^T + Q`
     * Write back `x_hat_G` & `P` to `globalObject` and update timestamps.
  3. Add logging for large `Δt` and for negative/zero `Δt`.
  4. Run on a short rosbag segment and display predicted tracks in RViz.

* **Notes / tips**

  * Guard numerical stability when inverting matrices elsewhere (use `ldlt()` or `llt()` if symmetric positive definite).
  * Keep state dimension consistent with `x = [x,y,z,vx,vy,ax,ay,l,w,h]^T`; use only relevant subblocks when necessary.
  * Update both `data_->object_list_fused.header.stamp` and each object's header.

* **Optional improvements**

  * Add unit tests for small synthetic trajectories (constant velocity).
  * Add CI step: compile & run a minimal rosbag test on GitHub Actions for PRs.

---

### B — Task A2: Object association (IoU & Mahalanobis)

**Objective:** implement association routines to match sensor detections to global tracks using IoU and Mahalanobis distance; expose config options to choose method and thresholds.

* **Inputs / requirements**

  * Sensor topics: `/sensors/camera_front/ikaObjectList`, `/sensors/radar_front/ikaObjectList`.
  * Global tracks list.
  * `fusion.yaml` config: `association_method` (IoU / Mahalanobis), `mahalanobis_threshold`, `iou_overlap_threshold`, `dim_red_mat`.

* **Expected output / artifact**

  * Functions: `IouCalculator::computeIoU(a,b)` and `MahalanobisCalculator::distance(meas, global, dim_red_mat)` implemented and tested.
  * Association matrix/assignment result per frame (CSV or notebook visual).
  * Confusion / association statistics (TP, FP, FN) vs ground truth (if available).

* **Tools / technologies**

  * C++ association module, Eigen for matrix ops, small Jupyter notebook for analysis/plots.

* **Step-by-step subtasks**

  1. Implement IoU:

     * Accept axis-aligned bounding boxes (center + l,w,h) → compute corner coords → compute intersection area and union area → return IoU (handle 0 division).
  2. Implement Mahalanobis:

     * Use `dim_red_mat` to project sensor and global state to relevant dims (e.g., x,y).
     * Compute innovation `v = H (x_sensor − x_global)` (H from dim\_red\_mat).
     * Compute innovation covariance `S = H P_S H^T + H P_G H^T` (use sensor covariances).
     * Compute `d = sqrt( v^T S^{-1} v )`.
  3. For each measured object, compute distance to all global tracks.
  4. Use gating thresholds (`mahal_thresh`, `iou_thresh`) to filter candidate matches.
  5. Solve assignment: greedily or using Hungarian algorithm (optional but recommended for global optimum).
  6. Produce per-frame association visualization (lines linking measurement → global object in RViz or a 2D plot).
  7. Log associations for offline analysis.

* **Notes / tips**

  * When using Mahalanobis, ensure `S` is SPD; regularize by adding small diagonal `εI` when necessary.
  * For crowded scenes IoU can fail; Mahalanobis is better when covariances large.
  * Default example thresholds to try: Mahalanobis: `2.0, 3.0, 4.0, 5.0`; IoU: `0.1, 0.2, 0.5`.

* **Optional improvements**

  * Implement Hungarian algorithm for optimal matching (use `Eigen` + `lapjv` library or implement simple O(n^3) solver).
  * Export association matrix per frame to CSV for offline metrics.

---

### C — Task A3: Measurement update / object fusion (StateFuser::runSingleSensor)

**Objective:** implement the Kalman measurement update (fusion) for each associated pair; tune process/measurement noise and association thresholds; visualize effect.

* **Inputs / requirements**

  * Associated measurement indices per global track (`data_->associated_measured`).
  * Measurement matrix `C` (mapping state to measurement), measurement noise `R`, global `P_G`.
  * Code skeleton: `StateFuser::runSingleSensor`.

* **Expected output / artifact**

  * Correct implementation of Kalman update: innovation, `S`, Kalman gain `K`, state & covariance updates.
  * RViz visualization showing fused tracks vs raw sensors before/after.
  * Plots showing effect of Q and thresholds on trajectory drift/noise.
  * Short notebook analyzing stability metrics and counts of false associations.

* **Tools / technologies**

  * C++ (StateFuser), Eigen, ROS, RViz, Jupyter notebook for parameter sweeps.

* **Step-by-step subtasks**

  1. In `StateFuser::runSingleSensor`:

     * Build measurement matrix `C` for the measured state (e.g., selecting x,y,l,w,h).
     * For each global object:

       * If no associated measurement → skip.
       * Load `x_hat_G` and `P_G` (Eigen vectors / matrices).
       * Read measurement `z` and its covariance `R`.
       * Compute `v = z − C * x_hat_G`.
       * Compute `S = C * P_G * C^T + R`.
       * Compute `K = P_G * C^T * S^{-1}`.
       * Update `x_hat_G ← x_hat_G + K * v`.
       * Update `P_G ← (I − K*C) * P_G`.
       * Store updated states back in `data_->object_list_fused`.
  2. Add safe checks: if `S` is nearly singular, regularize before inverse.
  3. Run a sweep of `time_variant_process_noise_matrix` diagonal entries (`Q`) and `mahalanobis_threshold` & `iou_overlap_threshold`.
  4. Produce plots:

     * Trajectories under different Q values.
     * Number of associations vs threshold.
     * Track stability metric (variance, position RMSE w\.r.t. ground truth if available).
  5. Visualize in RViz and record GIFs.

* **Notes / tips**

  * Use `IkaUtilities::getEigenStateVec` / `getEigenVarianceVec` to simplify conversions.
  * Parameter comments from notes: small `Q` (0.001) → favors model (may drift on curves); large `Q` (100) → noisy, unstable tracks.
  * Tune measurement noise `R` per sensor (camera vs radar).

* **Optional improvements**

  * Add multi-sensor sequential fusion vs joint fusion comparison.
  * Add a short Jupyter notebook that automates parameter grid search and produces a table of metrics (stability, drift, association count).

---

### D — Task A4: ROS integration, bag playback & RViz demo

**Objective:** ensure the fusion node integrates with ROS topics, play the provided rosbag and visualize final fused tracks; produce demo assets.

* **Inputs / requirements**

  * `fusion.launch`, `bag_playback.launch`, rosbag file (provided: 9m59s).
  * RViz config for showing `ikaObjectList` topics and fused outputs.

* **Expected output / artifact**

  * Working `roslaunch` sequence that plays bag and runs fusion node.
  * RViz config file saved (`module3_rviz.rviz`).
  * Demo GIF (10–20 s) of fused tracks and detections.
  * README section with demo steps and commands.

* **Tools / technologies**

  * ROS, RViz, `rosbag play`, ffmpeg (to record GIFs), shell.

* **Step-by-step subtasks**

  1. Verify topics names: `/sensors/camera_front/ikaObjectList`, `/sensors/radar_front/ikaObjectList`, `/sensors/vehicleCAN/ikaEgoMotion`, `/ikaGPS`.
  2. Launch fusion node with `roslaunch fusion.launch`.
  3. Play bag:

     ```bash
     rosbag play my_bag_file.bag --clock --pause
     rosbag info my_bag_file.bag    # sanity check
     rosbag play my_bag_file.bag -r 0.5   # slower playback for demo
     ```
  4. Open RViz with prepared config and record a short video or convert to GIF:

     ```bash
     rosrun image_view video_recorder images:=/rviz/screenshot out.avi
     ffmpeg -i out.avi -vf "fps=15,scale=800:-1:flags=lanczos" demo.gif
     ```
  5. Save `module3_rviz.rviz` and include in repo.

* **Notes / tips**

  * Use `--clock` so nodes using ROS time sync properly with bag.
  * For reproducibility include exact rosbag file md5 / size and a small sample bag (if size limits).

* **Optional improvements**

  * Add a short demo script `demo/run_demo.sh` to automate launching and recording.

---

### E — Task A5: Evaluation & parameter study

**Objective:** quantitatively evaluate association & fusion choices and document best settings.

* **Inputs / requirements**

  * Logged associations and fused track outputs per frame.
  * Ground truth if available; if not, use heuristics: continuity, covariance growth, or manual inspection.

* **Expected output / artifact**

  * Notebook (`evaluation_module3.ipynb`) with:

    * Plots of trajectory stability vs `Q` values.
    * Association statistics vs thresholds.
    * Confusion matrix (if ground truth).
    * Table of recommended parameters with rationale.
  * Metric CSVs, README summary.

* **Tools / technologies**

  * Jupyter, pandas, matplotlib (no special colors), numpy.

* **Step-by-step subtasks**

  1. Export run logs: associations per frame + fused state per object → CSV.
  2. Load CSV in notebook and compute metrics:

     * Track jitter (mean positional variance per track)
     * Drift (difference between predicted-only vs fused positions)
     * False associations count per frame
  3. Grid search: try different `Q` diagonals and `mahalanobis_threshold`, record metrics.
  4. Produce summary table recommending final config.

* **Notes / tips**

  * When ground truth is not available rely on consistency metrics and visual inspection.
  * Show at least one visual comparing "before" and "after" fusion on a chosen target track.

* **Optional improvements**

  * Provide an automated script to run a parameter sweep and produce a Markdown report.

---

## 3) Artifacts / deliverables (what to include in repository)

* **C++ source & build**

  * `src/StatePredictor.cpp` — implemented prediction step (link placeholder: `./src/StatePredictor.cpp`)
  * `src/StateFuser.cpp` / `StateFuser::runSingleSensor` — measurement update implemented (`./src/StateFuser.cpp`)
  * `src/IouCalculator.cpp, MahalanobisCalculator.cpp` (`./src/`)
  * Updated `CMakeLists.txt` and package manifest.
* **Config**

  * `config/kalman_filter.yaml` (with `F_const`, `F_timevar`, `Q_timevar`).
  * `config/fusion.yaml` (association method, thresholds, `dim_red_mat`, `initial_global_variances`).
* **ROS launch & rviz**

  * `launch/fusion.launch`, `launch/bag_playback.launch`
  * `rviz/module3_rviz.rviz`
* **Data**

  * `data/` — small sample rosbag or link placeholder to full bag; metadata file with bag size & message counts (77.4 MB, 125,992 messages, 9m59s).
* **Notebooks & analysis**

  * `notebooks/evaluation_module3.ipynb` — parameter sweeps & plots
  * `notebooks/association_analysis.ipynb` — IoU vs Mahalanobis distributions
* **Scripts**

  * `demo/run_demo.sh` — launch, play bag, record demo
  * `tools/export_associations.py` — parse ROS messages and emit CSV
  * `tools/param_sweep.sh` — run parameter grid and collect metrics
* **Visual assets**

  * `media/rviz_screenshots/*.png`
  * `media/demo_short.gif` (10–20 s)
  * `media/plots/*.png`
* **Docs**

  * `README.md` (module TL;DR + run instructions)
  * `docs/parameter_recommendations.md`
* **Placeholders to fill**

  * `[LINK_TO_NOTEBOOK_MODULE3]`, `[LINK_TO_STATE_PREDICTOR_SRC]`, `[LINK_TO_RVIZ_CONFIG]`, `[LINK_TO_DEMO_GIF]` — keep these placeholders in README until assets are uploaded.

---

## 4) Progress tracking / status notes (ready to paste into GitHub Issues / Projects)

You can copy this table into a GitHub Project card or a Markdown file. Update **Status** to `To Do / In Progress / Done` and paste Issue/PR links.

| Task ID | Task (top level) | Subtask / deliverable                     | Owner | Status | Issue link | PR link | Notes                           |
| ------- | ---------------: | ----------------------------------------- | ----- | ------ | ---------- | ------- | ------------------------------- |
| M03-A1  |  Prediction step | Implement `StatePredictor.cpp` prediction | @you  | To Do  | `#`        | `#`     | uses `F` & `Q` from config      |
| M03-A2  |      Association | Implement IoU + Mahalanobis               | @you  | To Do  | `#`        | `#`     | add Hungarian optional          |
| M03-A3  |    Fusion update | Implement `StateFuser::runSingleSensor`   | @you  | To Do  | `#`        | `#`     | include gates & regularization  |
| M03-A4  |         ROS demo | Create `launch/` & rviz config & demo GIF | @you  | To Do  | `#`        | `#`     | include `rosbag info` metadata  |
| M03-A5  |       Evaluation | Notebooks: parameter sweep & metrics      | @you  | To Do  | `#`        | `#`     | export CSVs for reproducibility |

**Example commands to add to README (copy/paste):**

```bash
# build & run (catkin)
catkin_make
source devel/setup.bash
roslaunch module3 fusion.launch
rosbag play data/example.bag --clock
```

---

## 5) Summary / TL;DR for Module 03

Implemented a ROS-based Multi-Object Tracking pipeline that predicts object states using a Kalman filter, associates sensor detections with global tracks using IoU and Mahalanobis gating, and fuses measurements with a Kalman measurement update. Deliverables include C++ implementations (`StatePredictor`, `StateFuser`, association calculators), configuration files for `F` / `Q` / thresholds, RViz demo & GIF, and Jupyter notebooks with parameter sweeps & metrics. Key achievements: robust uncertainty-aware association, documented effects of `Q` and association thresholds on track stability (drift vs noise), and ready-to-run demo + evaluation artifacts for recruiters. Suggested metric to show on portfolio front page: **track stability score** (e.g., mean positional variance across tracked objects) before vs after fusion and a short 10–20s RViz GIF demonstrating fused tracks.

---

## Quick implementation checklist (copy to Issue template)

* [ ] Implement prediction (`StatePredictor.cpp`) and unit test with synthetic data.
* [ ] Implement IoU & Mahalanobis association functions; add Hungarian assignment.
* [ ] Implement measurement update in `StateFuser::runSingleSensor`.
* [ ] Add config files (`kalman_filter.yaml`, `fusion.yaml`) with sensible defaults and comments.
* [ ] Create `launch/` files and `rviz/` config; record demo GIF.
* [ ] Export associations to CSV and create `notebooks/` for evaluation.
* [ ] Write `README.md` with run instructions and placeholders for links.

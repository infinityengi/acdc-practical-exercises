## 1) Module Overview / Summary

* **Module goals & learning outcomes**

  * Implement a robust trajectory planner using **Direct Multiple Shooting (DMS)**, integrate it into a ROS node, and evaluate in closed-loop simulation.&#x20;
  * Implement vehicle stabilization: odometry updates, cascaded PID controllers for longitudinal/lateral control, feedforward + inverse single-track steering mapping.&#x20;
  * Implement route planning using **Lanelet2** + Dijkstra and convert lanelet sequences to centerline trajectories for the planner.&#x20;
* **Key skills demonstrated**

  * C++ ROS node development, optimization (DMS), kinematic vehicle modeling, control (PID), model-based steering mapping, lanelet map handling, closed-loop simulation, visualization (RViz), dataset replay (bag files), and metrics evaluation.

---

## 2) Detailed Task Breakdown

Below each assignment is decomposed into hierarchical subtasks (main task → subtasks → optional improvements). Each subtask includes objective, inputs/requirements, expected output, tools/technologies, and implementation tips.

---

### A — Trajectory Planner (Direct Multiple Shooting)

**Reference:** guidance level notes (Direct Multiple Shooting, system dynamics, cost terms).&#x20;

* **Main objective**

  * Implement a DMS trajectory optimizer in C++ (inside a ROS node) using the kinematic single-track model, appropriate cost terms (jerk, steering rate, velocity deviation, collision avoidance), and constraints.

* **Subtasks**

  1. **Project scaffolding**

     * **Objective:** Add `trajectory_planner` package to ROS workspace.
     * **Inputs:** ROS workspace, CMakeLists, package.xml, example `.osm` map for later use.
     * **Output:** Buildable ROS package skeleton (source files + CMake).
     * **Tools:** ROS (1 or 2 depending on your stack), CMake, colcon / catkin\_make.
     * **Tips:** Start with a minimal node that prints and accepts parameters (launch/params YAML).
     * **Commands (examples):**

       ```bash
       # ROS1
       catkin_make
       source devel/setup.bash
       roslaunch trajectory_planner vehicle_guidance.launch

       # ROS2
       colcon build
       source install/setup.bash
       ros2 launch trajectory_planner vehicle_guidance.launch.py
       ```
  2. **Implement system dynamics (kinematic single-track)**

     * **Objective:** Code the ODEs:

       * `dx/dt = v_x * cos(theta)`
       * `dy/dt = v_x * sin(theta)`
       * `ds/dt = v_x`
       * `dvx/dt = a_x`
       * `dax/dt = j` (jerk)
       * `dtheta/dt = v_x / L * tan(delta)`
       * `ddelta/dt = alpha` (steering rate)
     * **Inputs:** State vector, control vector, wheelbase L.
     * **Output:** Dynamics functor for the optimizer.
     * **Tools:** C++, Eigen, CppAD (if using CppAD for auto-diff), your chosen optimizer (IPOPT/ACADO/ACADOS).
     * **Tips:** Encapsulate dynamics in a function so it’s reusable for integration and constraints. Use unit tests for single step integration.
  3. **Formulate cost terms**

     * **Objective:** Implement cost components: longitudinal/lateral jerk, steering rate, velocity deviation, collision avoidance barrier term.
     * **Inputs:** state & control trajectories, parameters (jerkRef, alphaRef, vref, weights).
     * **Output:** Total cost function assembled for the optimizer.
     * **Tools:** CppAD or chosen auto-diff, C++.
     * **Tips:** Normalize terms (scales like `vScale`) to balance costs; test each term independently.
  4. **Add constraints & bounds**

     * **Objective:** Enforce `vx <= v_max`, `|delta| <= delta_max`, `|ax| <= a_max`, control bounds for jerk and alpha.
     * **Inputs:** vehicle limits.
     * **Output:** Constraint definitions for optimizer.
  5. **Direct multiple shooting implementation**

     * **Objective:** Implement DMS: state segments + collocation constraints or continuity via integrator.
     * **Inputs:** horizon length, N segments, integrator (RK4/simple Euler).
     * **Output:** Optimizer setup ready to solve for states & controls.
     * **Tips:** Start with small horizon & coarse discretization. Verify continuity residuals.
  6. **Collision avoidance (dynamic objects)**

     * **Objective:** Integrate dynamic object positions (from bag or simulated) into the cost using a barrier function.
     * **Inputs:** dynamic object list (x,y), reference distance `d_ref`, weight.
     * **Output:** Collision terms in objective.
     * **Tips:** Use conditional expressions or smooth approximations to avoid discontinuities in optimizer.
  7. **ROS interface & parameterization**

     * **Objective:** Subscribe to odometry, dynamic objects, and publish planned trajectory.
     * **Outputs:** `/planned_trajectory` topic, parameter YAML for tuning.
     * **Tips:** Keep frequency decoupled: planner at lower rate than controller.
  8. **Closed-loop simulation integration**

     * **Objective:** Use planner output as reference for controller (see stabilization tasks). Run in closed-loop with controller node; verify behavior in RViz.
     * **Inputs:** planner output topics, controller input interface.
     * **Output:** Live closed-loop simulation in RViz.
  9. **Testing & debugging**

     * **Objective:** Unit tests for dynamics / cost; integration tests with simple scenarios; visualize reference vs planned (green vs blue lines).
     * **Tools:** rostest / gtest or custom python scripts to replay sample scenarios.
     * **Tips:** Start with static obstacles to validate collision term before adding dynamic ones.

* **Expected artifacts**

  * `trajectory_planner/` C++ source, `trajectory_planner.cpp`, tests, `vehicle_guidance.launch`, param YAML, sample bag for dynamic objects, RViz configuration, plots of cost components.

---

### B — Vehicle Stabilization & Control (Odometry, PID, Longitudinal & Lateral Controllers)

**Reference:** stabilization notes (odometry equations, PID, feedforward/feedback, inverse single-track).&#x20;

* **Main objective**

  * Implement odometry updates, compute control deviations, and implement discrete PID controllers for velocity and cascaded lateral control (dy → desired yaw rate → steering angle via inverse single-track).

* **Subtasks**

  1. **Odometry update function**

     * **Objective:** Implement discrete odometry update:

       ```
       odom_dy += sin(odom_dpsi + yawRate * 0.5 * dt) * velocity * dt;
       odom_dpsi += yawRate * dt;
       ```
     * **Inputs:** `cur_vehicle_state_.yaw_rate`, `cur_vehicle_state_.velocity`, `dt`.
     * **Output:** `odom_dy`, `odom_dpsi`.
     * **Tools:** C++ within a controller node module.
     * **Tips:** Guard for small dt, and test with synthetic yawRate / velocity signals.
  2. **Deviation calculation**

     * **Objective:** Compute lateral (`dy = odom_dy - y_tgt`) and heading dev (`dpsi = odom_dpsi - psi_tgt`).
     * **Inputs:** `odom_` values, trajectory target `y_tgt`, `psi_tgt`.
     * **Output:** `dy`, `dpsi`.
  3. **Discrete PID implementation**

     * **Objective:** Implement PID output:

       ```
       u = Kp*e + Ki*i_val + Kd*d_val
       ```

       with anti-windup and derivative filtering.
     * **Inputs:** error `e`, dt, previous integral & derivative states.
     * **Output:** control output `u`.
     * **Tools:** C++ PID class (`PID.cpp`), parameter YAML to tune `Kp,Ki,Kd`.
     * **Tips:** Implement integral anti-windup (clamping), and derivative on measurement or filtered derivative to reduce noise.
  4. **Longitudinal controller**

     * **Objective:** Combined feedforward (trajectory acceleration) + PID feedback for velocity.
     * **Inputs:** `v_tgt`, `cur_velocity`, `a_ff` (feedforward acceleration).
     * **Output:** throttle/accel command `a = a_ff + a_fb_v`.
     * **Tips:** Limit applied acceleration; use saturation to model actuator limits.
  5. **Cascaded lateral controller**

     * **Objective:** Two-stage controller: lateral deviation PID → desired yaw rate `psi_dot_des`. Then pid on heading dev → `psi_dot` control output.
     * **Formulas & flow:** `e_y -> yaw_rate_ref` via PID; `e_psi -> psi_dot_des` via second PID.
     * **Output:** `psi_dot_des` (desired yaw rate).
  6. **Inverse single-track model**

     * **Objective:** Convert `psi_dot_des` to steering angle:

       $$
       \delta = \dfrac{\dot{\psi}(l + EG \cdot v^2)}{v}
       $$
     * **Inputs:** `psi_dot_des`, `wheelbase`, `self_st_gradient` (EG), `velocity`.
     * **Output:** `st_ang_pid` (desired steering angle).
     * **Tips:** Handle low speed `v -> 0` (avoid division by zero) — fallback to previous steering or clamp.
  7. **ROS controller node & topics**

     * **Objective:** Implement `stabilization_ctrl` node that subscribes to planner `/planned_trajectory` (or `cmd_*`) and to vehicle state; publishes actuator commands (steer, accel).
     * **Tools:** ROS topics, message types (e.g., `nav_msgs/Odometry`, custom control msg).
  8. **Tuning & validation**

     * **Objective:** Tune PID gains via step responses in simulation and log tracking error metrics (RMSE, max deviation).
     * **Tips:** Use sweep tests (parameter sweep) and logs to compute performance.

* **Optional improvements**

  * Implement Model Predictive Control (MPC) replacement for PID in lateral loop.
  * Add automatic tuning scripts (grid search) and a small PyTorch / scipy optimizer for gain selection.

* **Expected artifacts**

  * `stabilization_ctrl/` C++ sources: `TrajectoryCtrl.cpp`, `PID.cpp`, parameter YAML with gains, unit tests, log/playback scripts, plots of `dy`, `dpsi`, velocity tracking, steering output.

---

### C — Route Planning with Lanelet2 + Dijkstra

**Reference:** navigation notes (lanelet2 primitives, routing graph, Dijkstra).&#x20;

* **Main objective**

  * Load `.osm` lanelet maps, build a routing graph, compute shortest path between start/end, extract centerline path for the trajectory planner.

* **Subtasks**

  1. **Environment & dependencies**

     * **Objective:** Install/enable `lanelet2` (C++ or Python bindings) in your environment.
     * **Tools:** `lanelet2`, `pyproj`, ROS integration packages (e.g., `lanelet2_ros`).
     * **Tips:** Use the correct projector (UTM) for your origin.
  2. **Load map & projection**

     * **Objective:** Use `lanelet2.io.load()` with `UtmProjector` to load `.osm` into map frame.
     * **Inputs:** `.osm` file + origin coordinates.
     * **Output:** Lanelet map object.
  3. **Find nearest lanelets for start/end**

     * **Objective:** Convert lat/lon → UTM and find nearest lanelet satisfying traffic rules.
     * **Inputs:** start/end WGS84 coords.
     * **Output:** start\_lanelet\_id, end\_lanelet\_id.
  4. **Build routing graph & run Dijkstra**

     * **Objective:** Create `RoutingGraph` and call `getRoute()` / `shortestPath()`.
     * **Output:** sequence of lanelets representing route.
  5. **Extract centerline & convert to path**

     * **Objective:** Use lanelet centerlines to derive `[x,y]` waypoints for planner consumption.
     * **Output:** path array, optionally smoothed.
  6. **Visualization**

     * **Objective:** Plot map + start/end + path (matplotlib) and/or publish to RViz for verification.
     * **Tools:** Python notebook or ROS publisher.
  7. **Integration**

     * **Objective:** Provide produced path to DMS planner as reference (blue line reference in RViz).
     * **Tips:** Ensure consistent coordinate frames (map / odom).

* **Expected artifacts**

  * `notebooks/route_planning_lanelet2.ipynb` (or C++ example), `maps/*.osm`, `scripts/latlon2mapframe.py`, exported path `.csv`, RViz config.

---

### D — Closed-Loop Simulation, Logging & Evaluation

* **Main objective**

  * Run planner + controller in closed loop, replay dynamic object bag files, visualize in RViz and compute performance metrics.

* **Subtasks**

  1. **Create ROS launch for closed-loop**

     * Combine `trajectory_planner.launch`, `stabilization_ctrl.launch`, `lanelet_loader.launch`, `rviz`.
  2. **Replay bag files**

     * **Command:**

       ```bash
       rosbag play dynamic_objects.bag --loop -r 1.0
       ```
     * **Tip:** Use `--clock` and start nodes with simulated time.
  3. **Record logs**

     * **Tools:** `rosbag record` or ros2 bag; record `/odom`, `/planned_trajectory`, `/cmd_vel`, `/steer_cmd`.
  4. **Compute metrics**

     * **Metrics to compute:**

       * Trajectory tracking RMSE (lateral & yaw)
       * Max lateral deviation
       * Mean control effort (integral of |accel| + |steer\_rate|)
       * Planner solve time / real-time factor
     * **Python snippet (RMSE):**

       ```python
       import numpy as np
       def rmse(ref, actual):
           return np.sqrt(((ref - actual)**2).mean())
       ```
  5. **Visual artifacts**

     * Generate plots: `lateral_error.png`, `yaw_error.png`, control signals, planner solve time histogram.
     * Record short demo GIFs from RViz for README.

---

### E — Documentation, README & Portfolio Polish

* **Subtasks**

  1. **Module README (TL;DR)** — short description, how to run, example commands, expected outputs, performance numbers.
  2. **Issue / PR templates** — include `ISSUE_TEMPLATE.md` and `PULL_REQUEST_TEMPLATE.md` for the repo.
  3. **Demo assets** — RViz screenshots, GIFs, recorded video (`mp4`) under `docs/` or `assets/`.
  4. **Notebook to reproduce metrics** — `analysis/metrics.ipynb`.
  5. **License + contribution guide**.

---

## 3) Artifacts / Deliverables (files to produce)

* **Code & nodes**

  * `trajectory_planner/src/trajectory_planner.cpp`
  * `stabilization_ctrl/src/TrajectoryCtrl.cpp`, `stabilization_ctrl/src/PID.cpp`
  * `lanelet_planner/` (python notebook or C++ wrapper)
* **Build & launch**

  * `CMakeLists.txt`, `package.xml`, `launch/vehicle_guidance.launch` (or `launch.py`)
* **Data & configs**

  * `maps/your_map.osm`, `params/trajectory_params.yaml`, `params/controller_gains.yaml`, `rviz/vehicle_guidance.rviz`
* **Notebooks & analysis**

  * `notebooks/route_planning_lanelet2.ipynb`&#x20;
  * `notebooks/metrics.ipynb` (RMSE, plots)
* **Binaries / bags / media**

  * `bags/dynamic_objects.bag`, `assets/demo.gif`, `assets/demo.mp4`
* **Docs**

  * `README.md` (module TL;DR + run instructions)
  * `docs/` with plots and short technical writeups
* **Tests**

  * unit tests for dynamics, PID, and integration tests

Placeholders for links (replace with actual URLs):

* `NOTES: [Notebook link placeholder]()`
* `Code: trajectory_planner/src/trajectory_planner.cpp` → `[View on GitHub](<repo_link>/trajectory_planner/src/trajectory_planner.cpp)`

---

## 4) Progress Tracking / Status Notes

Use these templates directly as GitHub Issues / Project cards. Below is a sample markdown table to track each major task and its status.

**Project board / Issue table (copy into project README or Issues):**

|       Task ID | Task               | Subtask                  | Owner |      Status | GitHub Issue / PR |
| ------------: | ------------------ | ------------------------ | ----- | ----------: | ----------------- |
|     M04-TP-01 | Trajectory Planner | Project skeleton & build | @you  |       To Do | [issue#](link)    |
|     M04-TP-02 | Trajectory Planner | Dynamics & integrator    | @you  | In Progress | [issue#](link)    |
|   M04-STAB-01 | Stabilization      | Odometry & deviations    | @you  |       To Do | [issue#](link)    |
|   M04-STAB-02 | Stabilization      | PID controllers          | @you  |       To Do | [issue#](link)    |
|   M04-LANE-01 | Lanelet2           | Load maps & routing      | @you  |       To Do | [issue#](link)    |
| M04-CLOSED-01 | Integration        | Closed-loop simulation   | @you  |       To Do | [issue#](link)    |
|    M04-DOC-01 | Docs               | README + demo GIF        | @you  |       To Do | [PR#](link)       |

**Suggested Issue template (copy into `.github/ISSUE_TEMPLATE/bug.md`):**

```markdown
### Summary
Short description of the task / bug.

### Steps to reproduce
1. ...
2. ...

### Expected behavior
...

### Actual behavior
...

### Logs / Files
- node log: path/to/log
- bag: bags/xxx.bag

### Notes
- Related to M04-TP-02
```

---

## 5) Summary / TL;DR for Module 04

This module implements a full guidance pipeline: a Direct Multiple Shooting trajectory planner (kinematic single-track dynamics + costs for jerk, steering rate, velocity deviation, and collision avoidance), lanelet2-based route planning (Dijkstra shortest path → centerline), and a stabilization stack (odometry, cascaded PID for lateral & longitudinal control with an inverse single-track steering mapping). Integrated into ROS and validated in closed-loop simulation with RViz & bag playback, deliverables include planner & controller C++ nodes, notebooks for lanelet2 and metrics, RViz demo GIFs, and a professional README — with measurable metrics such as lateral RMSE and planner solve time to show performance improvements.  &#x20;

---

## Quick run / build cheat-sheet (copy into README)

```bash
# build (ROS1)
catkin_make
source devel/setup.bash

# build (ROS2)
colcon build --packages-select trajectory_planner stabilization_ctrl
source install/setup.bash

# run closed-loop (example)
roslaunch trajectory_planner vehicle_guidance.launch
rosbag play bags/dynamic_objects.bag --loop --clock

# record logs for metrics
rosbag record /odom /planned_trajectory /cmd_accel /cmd_steer -O logs/run1.bag
```
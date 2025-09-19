## 1) Module Overview / Summary

* **Short description:**
  Implement a V2X-enabled stack that receives ETSI-style messages (SPATEM, MAPEM, CAM) over MQTT, visualizes them in RViz, fuses infrastructure object lists from multiple stations in the cloud, and extends the vehicle trajectory planner so it reacts to traffic-light states.&#x20;

* **Learning outcomes / goals:**

  * Bridge MQTT ↔ ROS and configure topics for V2X messages.
  * Parse MAP & SPAT messages to identify intersections, signal groups, and timings.
  * Visualize intersections, signals, and cooperative vehicle states in RViz (ETSI visualization).
  * Implement rule-based traffic-light aware trajectory planner behavior (stop at red, go on green).
  * Fuse object lists from multiple infrastructure sensors (cloud fusion) using probabilistic fusion (e.g., Kalman filter).&#x20;

* **Key skills demonstrated:**

  * ROS package & launch management, MQTT bridging, RViz marker programming, message parsing (ETSI types), C++/ROS node integration, simple multi-sensor fusion, writing reproducible launch/config artifacts, producing demo visualizations and metrics.

---

## 2) Detailed Task Breakdown (main tasks → subtasks → optional improvements)

> Each main task below can be directly converted to a GitHub Issue / Project card. Citations indicate which note the task came from. &#x20;

---

### A — Task: Configure MQTT ↔ ROS Bridge (SPATEM / MAPEM / CAM)

**Source:** v2i notes (bridge config).&#x20;

* **Objective:** Subscribe to ETSI V2X topics on an MQTT broker and publish corresponding ROS topics (`/SPATEM`, `/MAPEM`, `/CAM`, `/topicA`, `/topicB`).

* **Inputs / requirements:**

  * MQTT broker: `broker.hivemq.com` (host & port `1883`).
  * Topic names: e.g. `ika_acdc_22/SPATEM`, `ika_acdc_22/MAPEM`, `ika_acdc_22/CAM`, `ika_acdc_22/objectList_a`, `ika_acdc_22/objectList_b`. &#x20;

* **Expected output / artifact:**

  * Config file(s): `v2x_tp_params.yaml`, `v2x_object_fusion_params.yaml`.
  * Running MQTT↔ROS node that publishes `/SPATEM`, `/MAPEM`, `/CAM`, `/topicA`, `/topicB`.

* **Tools / technologies:** ROS (catkin), `mqtt_client` (or in-repo `mqtt_launchpack`), YAML config.

* **Step-by-step subtasks:**

  1. Create `catkin` workspace and clone required packages (`mqtt_launchpack`, `etsi_visualization`, `object_fusion_wrapper`, planner packages).

     ```bash
     catkin build
     source devel/setup.bash
     ```
  2. Create MQTT parameter file `v2x_tp_params.yaml` and `v2x_object_fusion_params.yaml`:

     ```yaml
     broker:
       host: broker.hivemq.com
       port: 1883

     topics:
       - mqtt_topic: ika_acdc_22/SPATEM
         ros_topic: /SPATEM
       - mqtt_topic: ika_acdc_22/MAPEM
         ros_topic: /MAPEM
       - mqtt_topic: ika_acdc_22/CAM
         ros_topic: /CAM
       - mqtt_topic: ika_acdc_22/objectList_a
         ros_topic: /topicA
       - mqtt_topic: ika_acdc_22/objectList_b
         ros_topic: /topicB
     ```
  3. Launch mqtt node:

     ```bash
     roslaunch mqtt_launchpack v2x_tp_mqtt.launch params:=/path/to/v2x_tp_params.yaml
     ```
  4. Validate ROS topics exist:

     ```bash
     rostopic list | grep -E "/SPATEM|/MAPEM|/CAM|/topicA|/topicB"
     ```

* **Notes / tips:**

  * If broker is unreliable, run a local mosquitto for tests.
  * Use `rostopic echo -n 1 /SPATEM` to inspect incoming messages and confirm message format.
  * Keep topic-to-message-type mapping documented in repo README.

* **Optional improvements:**

  * Add unit test that simulates an MQTT publish and asserts ROS topic receives message.
  * Add TLS auth / credentials if the broker will change to a secured instance.

---

### B — Task: ETSI Visualization in RViz (MAPViz, SPATViz, ETSIViz)

**Source:** v2i notes (SPAT/MAP visualizer).&#x20;

* **Objective:** Visualize MAPEM (intersection topology), SPATEM (signal state + timing) and CAM (vehicle positions) in RViz using `etsi_visualization`.
* **Inputs / requirements:**

  * ROS topics `/MAPEM`, `/SPATEM`, `/CAM`.
  * Implementation stubs: `MAPViz.cpp`, `SPATViz.cpp`, `ETSIViz.cpp`.
* **Expected outputs / artifacts:**

  * RViz config file (e.g., `ETSIViz.rviz` or `cloud_processing_vizu.launch`).
  * Implemented C++ nodes producing RViz MarkerArrays for intersections, signals, vehicles.
* **Tools / technologies:** ROS, RViz, C++ (ROS nodelets or nodes), `etsi_visualization` package.
* **Step-by-step subtasks:**

  1. Configure `start_ETSIViz.launch` with topic parameters:

     ```xml
     <param name="SPAT_Topic_Name" value="/SPATEM" />
     <param name="MAP_Topic_Name"  value="/MAPEM" />
     <param name="CAM_Topic_Name"  value="/CAM" />
     ```
  2. Implement `MAPViz.cpp`:

     * Parse `msg.intersections` and compute `n_intersections = msg.intersections.size();`.
     * Create lane / polygon markers for ingress lanes. (Hint: filter egress vs ingress.)
  3. Implement `SPATViz.cpp`:

     * Fill `SignalGroup` struct:

       ```cpp
       sg.sg_id = spat_intsctn.states[m].signalGroup;
       sg.next_change = spat_intsctn.states[m].state_time_speed[0].timing_likelyTime;
       ```
     * Map `eventState` to colors (e.g., red=2, green=5). Example:

       ```cpp
       switch(state) {
         case 2: // Red
           marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; break;
         case 5: // Green
           marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0; break;
         ...
       }
       ```
  4. Implement `ETSIViz.cpp` to combine markers and publish a MarkerArray.
  5. Implement CAM processing: extract vehicle id / lon / lat / speed -> marker:

     ```cpp
     obj.IdInternal = msg.header_stationID;
     float lon = msg.basic_container.referencePosition_longitude;
     float lat = msg.basic_container.referencePosition_latitude;
     float v_x = msg.high_freq_container.speed_speedValue * cos(msg.high_freq_container.heading_headingValue);
     float v_y = msg.high_freq_container.speed_speedValue * sin(msg.high_freq_container.heading_headingValue);
     ```
  6. Launch RViz and verify visual markers appear and change color when SPATEM messages change.
* **Notes / tips:**

  * Start with a single intersection test dataset before enabling multiple intersections.
  * Use frame id `map` and consistent transforms.
  * Add text markers showing `time_to_change` for each signal group.
* **Optional improvements:**

  * Export short GIFs demonstrating a red→green cycle.
  * Add an interactive RViz plugin or toggles to turn on/off layers (MAP, SPAT, CAM).

---

### C — Task: Modify Trajectory Planner to React to SPATEM/MAPEM

**Source:** v2i notes (planner integration).&#x20;

* **Objective:** Make the vehicle trajectory planner stop at red lights and resume at green using SPATEM/MAPEM data.
* **Inputs / requirements:**

  * ROS topics `/SPATEM`, `/MAPEM`.
  * Planner interface code: `v2x_planner_interface.cpp`.
  * Existing vehicle guidance launch `vehicle_guidance_v2x.launch`.
* **Expected output / artifact:**

  * Updated planner that marks waypoints / path segments as blocked/unblocked by traffic lights.
  * Demonstration log / rosbag showing planner behavior across a red→green event.
* **Tools / technologies:** ROS (planner package), C++.
* **Step-by-step subtasks:**

  1. Add params to `vehicle_guidance_v2x.launch`:

     ```xml
     <param name="topic_spatem" value="/SPATEM" />
     <param name="topic_mapem"  value="/MAPEM" />
     ```
  2. Parse MAPEM in `v2x_planner_interface.cpp`:

     * Count intersections:

       ```cpp
       int n_intersections = msg.intersections.size();
       ```
     * Filter ingress lanes only:

       ```cpp
       bool is_egress_lane = lane.directionalUse != definitions::v2x_MAP_Lane::LaneDirection_ingressPath;
       if (is_egress_lane) continue;
       ```
     * Map signalGroup -> lane geometry.
  3. Parse SPATEM and set traffic light states:

     ```cpp
     if(spat_intsctn.states[m].state_time_speed[0].eventState == 5 ||
        spat_intsctn.states[m].state_time_speed[0].eventState == 6)
       trafficlights[k].red = false;
     else
       trafficlights[k].red = true;
     ```

     *(eventState mapping per your ETSI enum; test with sample messages).*
  4. Planner reaction:

     * For any red signal within stopping distance on the planned route, insert stop waypoint (velocity=0) at safe stopping point.
     * On green, resume planned speed (or recompute trajectory).
  5. Produce tests:

     * Unit test: given SPATEM message with red, planner output has halt at point X.
     * Integration test: run simulator or rosbag and record behavior.
* **Notes / tips:**

  * Keep safety margin (stop distance) conservative for demo; expose as parameter.
  * Consider debouncing / hysteresis for quick blips (only react if red for > N ms).
* **Optional improvements:**

  * Add playback plot comparing planned speed vs time for red/green cycles.
  * Add a metrics script that computes "successful stop percentage" and "latency to resume".

---

### D — Task: Cloud Object Fusion (Station A + Station B → Fused ObjectList)

**Source:** collective cloud notes (object fusion & Kalman filter).&#x20;

* **Objective:** Convert MQTT object lists (from multiple infrastructure stations) to ROS topics and fuse them into a single object list (`/TopicFusion`) using a Kalman filter / probabilistic fusion.
* **Inputs / requirements:**

  * MQTT topics: `ika_acdc_22/objectList_a`, `ika_acdc_22/objectList_b`.
  * ROS bridge producing `/topicA`, `/topicB`.
  * Fusion node package: `object_fusion_wrapper` with `config_inout.yaml`.
* **Expected output / artifact:**

  * Fused object list published on `/TopicFusion` in frame `map`.
  * RViz visualization showing fused objects (e.g., green markers).
* **Tools / technologies:** ROS nodes/nodelets, Kalman filter math (C++/Python), RViz.
* **Step-by-step subtasks:**

  1. Create `v2x_object_fusion_params.yaml` for MQTT → ROS topic mapping (see A).
  2. Update `object_fusion_wrapper/config_inout.yaml`:

     ```yaml
     input_topics:
       object_lists: [ /topicA, /topicB ]
     ego_motion: null  # remove if no ego vehicle
     output_topic: /TopicFusion
     frame_id: map
     ```
  3. Implement or configure fusion algorithm:

     * Use a Kalman filter per object track: prediction & update steps; tune `Q`, `R`, `P0`.
     * Resolve associaton: nearest-neighbor or gating + Hungarian for assignment.
  4. Launch fusion node:

     ```bash
     roslaunch object_fusion_wrapper fusion.launch config:=/path/to/config_inout.yaml
     ```
  5. Visualize fused list with RViz launch `cloud_processing_vizu.launch`.
* **Notes / tips:**

  * Start with deterministic association (closest by Euclidean distance) to validate pipeline; then upgrade to more robust association.
  * Use `rostopic hz` to monitor message rates (object lists frequently update).
  * Keep track IDs consistent across fusion steps.
* **Optional improvements:**

  * Add confidence/uncertainty visualization (e.g., covariance ellipses).
  * Export comparison metrics: e.g., MSE of fused positions vs ground truth (if available), track continuity score.

---

### E — Task: Deliverables, Documentation & Demos (cross-task)

**Objective:** Make module recruiter-friendly and reproducible.

* **Step-by-step subtasks:**

  1. **README (module root)** — TL;DR, how to run (commands), expected outputs, troubleshooting.

     * Add a short demo script:

       ```bash
       # build and start
       catkin build
       source devel/setup.bash
       roslaunch mqtt_launchpack v2x_tp_mqtt.launch params:=./v2x_tp_params.yaml
       roslaunch etsi_visualization start_ETSIViz.launch
       roslaunch vehicle_guidance vehicle_guidance_v2x.launch
       ```
  2. **Notebooks** — A single Jupyter notebook `module05_demo.ipynb` summarizing:

     * Pipeline diagram (MQTT→ROS→RViz→Planner→Fusion).
     * Screenshots / GIFs of visualization.
     * Plots of planner speed/time for a red→green cycle.
  3. **Demo artifacts**:

     * RViz config file(s): `ETSIViz.rviz`, `cloud_processing_vizu.launch`.
     * Example `rosbag` recordings (or links/placeholders) demonstrating functionality.
     * GIFs: `red_to_green_demo.gif`, `fusion_demo.gif`.
  4. **Code docs**:

     * Inline code comments for `MAPViz.cpp`, `SPATViz.cpp`, `v2x_planner_interface.cpp`, and fusion nodes.
     * API / message type description file `MSG_TYPES.md`.
* **Optional improvements:**

  * Add a short screencast: `module05_demo.mp4`.
  * Automated CI test: run a small launch with synthetic MQTT messages and assert outputs.

---

## 3) Artifacts / Deliverables (what to produce)

* **Configs & Launches**

  * `v2x_tp_params.yaml` (MQTT↔ROS mapping).&#x20;
  * `v2x_object_fusion_params.yaml` (cloud fusion mapping).&#x20;
  * `vehicle_guidance_v2x.launch`, `start_ETSIViz.launch`, `cloud_processing_vizu.launch`.
* **Code**

  * `MAPViz.cpp`, `SPATViz.cpp`, `ETSIViz.cpp` (visualization).&#x20;
  * `v2x_planner_interface.cpp` (planner integration).&#x20;
  * `object_fusion_wrapper` code (Kalman filter & association) and `config_inout.yaml`.&#x20;
* **Notebooks & Docs**

  * `module05_demo.ipynb` (runbook with screenshots & metrics).
  * `README.md` (module overview + run instructions).
  * `MSG_TYPES.md` (ETSI types used).
* **Visuals & Media**

  * RViz configs (`ETSIViz.rviz`), demo GIFs (`red_to_green_demo.gif`, `fusion_demo.gif`), optional screencast `module05_demo.mp4`.
* **Test artifacts**

  * `sample_spatem.msg` / `sample_mapem.msg` (example messages).
  * `module05_demo.bag` (rosbag) — placeholder link: `[Download demo bag](PLACEHOLDER)`.

*(Add file links in the README as you commit files, placeholders below are included where appropriate.)*

---

## 4) Progress Tracking / Status Notes (GitHub-friendly table & templates)

* **Suggested columns for each GitHub Issue / Project card:**

  * **Title** — short task name
  * **Assignee** — your GitHub handle
  * **Status** — `To Do` / `In Progress` / `Blocked` / `Review` / `Done`
  * **Priority** — `P0` / `P1` / `P2`
  * **Related PR** — `#PR_NUM` (placeholder link)
  * **Notes** — short implementation notes / blockers.

**Markdown table template (paste into project README / tracking doc):**

| Task                |                                       Subtask |    Status   | Issue     | PR     |
| ------------------- | --------------------------------------------: | :---------: | --------- | ------ |
| MQTT bridge         | Create `v2x_tp_params.yaml`, launch mqtt node |    To Do    | #ISSUE-01 | -      |
| ETSI Viz            |         Implement `SPATViz.cpp` color mapping | In Progress | #ISSUE-02 | #PR-12 |
| Planner integration |                 Add SPATEM param + stop logic |    To Do    | #ISSUE-03 | -      |
| Cloud fusion        |    Configure `config_inout.yaml`, fusion node |    To Do    | #ISSUE-04 | -      |
| Docs & Demo         |                      Notebook + README + GIFs |    To Do    | #ISSUE-05 | -      |

**GitHub Issue template (example)**

```
Title: [Module05] Implement SPATViz color & timing markers

Description:
- Implement SPATViz.cpp:
  - Parse signal groups and timings
  - Map eventState to marker colors
  - Publish MarkerArray to /etsi_visualization/spat_markers

Acceptance criteria:
- RViz shows colored markers for SPATEM messages.
- Next-change time displayed as text marker.

Files to change:
- etsi_visualization/src/SPATViz.cpp
- etsi_visualization/launch/start_ETSIViz.launch

Labels: module05, etsi, viz
Assignee: @yourhandle
```

**Pull Request template (example)**

```
PR title: [Module05] SPATViz: Add signal color mapping + timing text

Summary:
- Implement mapping of eventState -> RGB in SPATViz.
- Publish text markers with next change time.

Testing:
- Launch mqtt bridge + ETSI viz, publish sample SPATEM message (included sample_spatem.msg)
- Verified markers in RViz.

Related issue: #ISSUE-02
```

---

## 5) Summary / TL;DR (one paragraph)

Module 05 demonstrates end-to-end V2X integration: an MQTT ↔ ROS bridge ingests SPATEM/MAPEM/CAM and infrastructure object lists, an ETSI visualization package renders intersections/traffic-lights/vehicles in RViz, a planner consumes SPATEM/MAPEM to stop/resume at signals, and a cloud fusion pipeline fuses multi-station object lists (Kalman filter + association) into a single `/TopicFusion`. Deliverables include MQTT params, launch files, C++ ROS nodes (`MAPViz`, `SPATViz`, `ETSIViz`, `v2x_planner_interface`), fusion configs, a demo notebook with GIFs, and rosbag samples — all ready to be converted into GitHub issues and project cards for step-by-step implementation and tracking. &#x20;

---

## Quick checklist (copy into README / Project board)

* [ ] Create `v2x_tp_params.yaml` and start mqtt bridge.&#x20;
* [ ] Implement `MAPViz.cpp` to draw intersections.&#x20;
* [ ] Implement `SPATViz.cpp` color + timing markers.&#x20;
* [ ] Implement CAM visualization in `ETSIViz.cpp`.&#x20;
* [ ] Update `v2x_planner_interface.cpp` to parse MAPEM & SPATEM and enforce stop/resume behavior.&#x20;
* [ ] Configure cloud fusion (`config_inout.yaml`) and run `fusion.launch`.&#x20;
* [ ] Produce `module05_demo.ipynb` with GIFs and metrics.


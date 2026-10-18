# Wildlife Monitoring GUI

ROS 2 Humble / PyQt5 GUI for monitoring the Husky simulation.

## Current features

- Three image panes: SLAM map | thermal + YOLO | live RGB camera.
- Middle pane: the thermal detector's annotated image (hot regions, track
  IDs, temperature) live; while the Husky verifies a candidate it switches
  to YOLO's annotated RGB image (projected thermal region, YOLO boxes and
  votes) and keeps the last one for 10 s.
- Map markers coloured by mission result when `/husky1/mission/status` is
  available (red exotic, green native, dark grey not an animal, amber
  unresolved, blue being checked, grey waiting); otherwise by the registry
  class (boar red, deer and kangaroo green).
- SLAM occupancy map.
- Husky position and heading on the map using TF.
- Odometry position and heading.
- Latest RGB animal detections, including class, confidence and bounding box.
- Receiving/stale indicators for camera, odometry and detections.
- Mission status bar above the map: mission state (Exploring, Investigating
  candidate, MISSION COMPLETE), what the robot is doing now (candidate,
  attempt, Husky phase or explore_lite status) and result counts. It turns
  orange when `mission_node` stops publishing for more than 3 s.
- "Mission results" tab: every candidate known to `mission_node` with its
  mission state (Waiting, Investigating, EXOTIC, Native, Not an animal,
  Unresolved, Duplicate), class, confidence, map position, attempts and last
  result; a live summary that becomes the final mission summary on
  COMPLETE (exotic and native animals with positions, unresolved candidates
  to check manually).
- When `spatial_evaluator` runs, the summary also shows its comparison with
  simulation ground truth (matched heat sources, mean error, within 1.0 m,
  boars correctly classified).
- "Candidate registry" tab: the raw `candidate_registry` snapshot.
- Coverage line under the counts (and in the summary): share of the survey
  area mapped by LiDAR and checked by the thermal camera, from
  `coverage_tracker.py`.

Detection rows represent the latest processed image, not unique animals
across the mission. Detection box coordinates are pixels, not map coordinates.

## Requirements

Use Ubuntu with ROS 2 Humble and a working Robotic_Studio1 simulation setup.

Install GUI dependencies:

```bash
sudo apt update
sudo apt install \
  python3-pyqt5 \
  python3-numpy \
  ros-humble-cv-bridge \
  ros-humble-tf2-ros \
  ros-humble-vision-msgs
```

The commands below assume the workspace is `~/41068_ws`.
Replace this path if your workspace is elsewhere.

## Build

From the workspace root:

```bash
cd ~/41068_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select wildlife_gui
source ~/41068_ws/install/setup.bash
```

The simulation package and its dependencies must already be built.

## Run

### Terminal 1: Start the simulation

Skip this step if the simulation is already running.

```bash
source /opt/ros/humble/setup.bash
source ~/41068_ws/install/setup.bash

ros2 launch 41068_ignition_bringup 41068_ignition.launch.py \
  husky:=true \
  parrot:=false \
  slam:=true \
  nav2:=true \
  rviz:=true \
  world:=large_demo
```

### Terminal 2: Start the GUI

```bash
source /opt/ros/humble/setup.bash
source ~/41068_ws/install/setup.bash

ros2 run wildlife_gui gui
```

The GUI remaps `/tf` and `/tf_static` to `/husky1/tf` and
`/husky1/tf_static` itself (needed for the robot marker on the map).
Remaps given on the command line take precedence.

Closing the GUI does not stop the simulation.

### Terminal 3: Start RGB detection (optional)

The camera and map work without YOLO. The detection table requires the
animal detector to be running.

This requires Ultralytics and the trained model:
`41068_ignition_bringup/weights/my_model.pt`.

```bash
source /opt/ros/humble/setup.bash
source ~/41068_ws/install/setup.bash

ros2 launch 41068_ignition_bringup 41068_yolo.launch.py \
  namespace:=husky1 \
  use_sim_time:=true \
  image_topic:=camera/image \
  model_path:=$HOME/41068_ws/src/Robotic_Studio1/41068_ignition_bringup/weights/my_model.pt \
  continuous:=true \
  max_inference_hz:=2.0 \
  device:=cpu
```

The example uses CPU inference with a maximum requested rate of 2 Hz.
Actual detection updates may be slower.

## ROS interfaces

| Interface | Type | Purpose |
|---|---|---|
| `/husky1/camera/image` | `sensor_msgs/msg/Image` | Live RGB camera |
| `/husky1/odom` | `nav_msgs/msg/Odometry` | Odometry position and heading |
| `/husky1/map` | `nav_msgs/msg/OccupancyGrid` | SLAM map |
| `/husky1/tf` | `tf2_msgs/msg/TFMessage` | Dynamic transforms |
| `/husky1/tf_static` | `tf2_msgs/msg/TFMessage` | Static transforms |
| `/husky1/perception/rgb/detections` | `vision_msgs/msg/Detection2DArray` | Latest RGB detections |
| `/husky1/candidates/registry_json` | `std_msgs/msg/String` (JSON) | Candidate registry table and map markers |
| `/husky1/mission/status` | `std_msgs/msg/String` (JSON) | Mission state, candidate results and summary |
| `/husky1/evaluation/summary` | `std_msgs/msg/String` (JSON) | Optional ground-truth check from `spatial_evaluator` |
| `/husky1/mission/coverage` | `std_msgs/msg/String` (JSON) | Map and thermal coverage from `coverage_tracker` |
| `/husky1/perception/thermal/annotated_image` | `sensor_msgs/msg/Image` | Middle pane, live |
| `/husky1/perception/rgb/annotated_image` | `sensor_msgs/msg/Image` | Middle pane while YOLO verifies |

Map frame: `husky1_map`.

Robot frame: `husky1_base_link`.

The position above the camera/map uses the odometry frame.
The position inside the map panel uses the map frame.
These values can differ.

## Troubleshooting

### Package not found

Build from the workspace root and source its installation:

```bash
source ~/41068_ws/install/setup.bash
```

### Camera is stale

Check that the simulation is running and the camera topic has a publisher:

```bash
ros2 topic info /husky1/camera/image -v
```

The GUI camera subscription uses RELIABLE QoS, matching the working RViz
configuration tested in this project.

### Map appears but the robot marker is missing

Run the GUI with both TF remappings shown above.

Check the transform:

```bash
ros2 run tf2_ros tf2_echo husky1_map husky1_base_link \
  --ros-args \
  -r /tf:=/husky1/tf \
  -r /tf_static:=/husky1/tf_static
```

### Detection table is waiting or empty

Start the RGB detector and point the camera towards an animal.

- Waiting: no detection message has been received yet.
- Empty table: the latest message contained no detections.
- Stale: the table is showing previous results.

Inspect the output:

```bash
ros2 topic echo /husky1/perception/rgb/detections \
  --qos-reliability reliable \
  --once
```

### Missing ultralytics

The optional detector needs Ultralytics installed in the Python environment
used by the ROS executable. This is separate from the GUI dependencies.

### Qt xcb plugin error after installing OpenCV

The GUI includes a PyQt5 plugin-path workaround before QApplication is
created. Keep this code in `gui_node.py` when modifying the application.
# Mission layer: thermal candidate → Husky investigation → YOLO → resume

This connects the existing perception, localisation, exploration and Nav2
parts into one autonomous loop (C1 proposal, sections 2.5 and 3.3).

```
thermal_candidate_detector ──► candidate_localizer ──► candidate_registry
                                                             │ /husky1/candidates/registry_json
                                                             ▼
explore_lite ◄── pause / resume ── mission_node ──► /husky1/mission/status (UI)
                                     │      ▲
            /husky1/mission/investigate      │ /husky1/perception/thermal_yolo/result
                                     ▼      │
                                husky_node ─┼──► Nav2 NavigateToPose + Spin
                                     │      │
                   /husky1/perception/verify_now ──► animal_yolo_detector
```

## One investigation, step by step

1. The thermal detector publishes a heat region. The localiser turns it into
   a map point. After 3 observations the registry creates `candidate_N`.
2. `mission_node` sees the new record on `registry_json`, picks the nearest
   uninvestigated candidate, and **pauses** explore_lite
   (`/husky1/explore/resume` = `false`).
3. After `pause_settle_sec` (explore_lite cancels *every* Nav2 goal when it
   pauses), the mission sends `investigate` (attempt 1) to `husky_node`.
4. `husky_node` picks an observation pose 3.0 m from the candidate on the
   robot's side, in free SLAM-map space, and sends it to Nav2. If Nav2 fails,
   it tries another pose (at most 3 per attempt, R4).
5. On arrival it turns to face the candidate (Nav2 `Spin`), waits 1.5 s, and
   calls `/husky1/perception/verify_now`.
6. YOLO starts its 5-frame thermal-gated vote and replies with a
   `request_id`. The consensus JSON carries the same `request_id`.
   `41068_mission.launch.py` starts YOLO with `auto_verify_on_heat:=false`, so
   YOLO does **not** run while exploring or driving; it only runs after this
   call at the observation pose. (Standalone `41068_thermal_yolo.launch.py`
   keeps the old behaviour: it votes whenever heat appears.)
7. The mission picks the thermal region that points at the candidate (same
   thermal track ID, or bearing within 15°) and decides:

| YOLO result for this candidate | Mission decision |
|---|---|
| exotic class, mean confidence ≥ 0.70 | `CONFIRMED_EXOTIC` |
| native class, mean confidence ≥ 0.70 | `CONFIRMED_NATIVE` |
| heat clearly seen but no animal box, 2 times | `REJECTED` (non-animal heat, e.g. campfire) |
| low confidence / too far / partial / not in view / no heat / unreachable / timeout | next attempt (2.2 m, view rotated +50°; then 2.6 m, −50°) |
| 3 attempts without a decision | `UNRESOLVED` |

8. If another candidate is waiting, the Husky goes there directly. Otherwise
   the mission **resumes** explore_lite.

## Files

| File | Change |
|---|---|
| `scripts/mission_node.py` | Rewritten: registry input, nearest-first, pause → settle → dispatch, result matching, 0.70 threshold, 3 attempts, Rejected/Unresolved, resume only when the Husky is idle, JSON status for the UI. |
| `scripts/husky_node.py` | Rewritten: observation-pose selection against the SLAM map, Nav2 goal retries (max 3), face the candidate, `verify_now`, status reports, cancel. |
| `scripts/animal_yolo_detector.py` | Added the `perception/verify_now` service, `request_id` / `trigger` fields in the consensus JSON, and the `auto_verify_on_heat` parameter (default `true` = old behaviour). |
| `launch/41068_thermal_yolo.launch.py` | New launch argument `auto_verify_on_heat` (default `true`). |
| `launch/41068_mission.launch.py` | New: explore_lite + perception + localiser + registry + mission + husky in one launch. |
| `config/mission_params.yaml` | New: all mission / husky parameters. |
| `package.xml` | `exec_depend explore_lite_msgs`. |
| `CMakeLists.txt` | Installs `perception_overlay.py` (imported by the YOLO node; only worked before with `--symlink-install`). |

## Build

```bash
cd ~/41068_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select 41068_ignition_bringup
source install/setup.bash
```

## Run the end-to-end test

Every terminal: `source /opt/ros/humble/setup.bash && source ~/41068_ws/install/setup.bash`
(plus `unset ROS_LOCALHOST_ONLY` if that is how your machine is set up).

**Terminal 1 – simulation, SLAM, Nav2, RViz**
```bash
ros2 launch 41068_ignition_bringup 41068_ignition_husky.launch.py \
  slam:=true nav2:=true rviz:=true world:=large_demo
```

**Terminal 2 – explore + perception + localisation + mission** (wait until Nav2 is active)
```bash
ros2 launch 41068_ignition_bringup 41068_mission.launch.py
```
Switch parts off if you already run them elsewhere, e.g. `explore:=false`
or `perception:=false`.

**Terminal 3 – watch the mission**
```bash
ros2 topic echo /husky1/mission/status --field data
ros2 topic echo /husky1/mission/husky_status --field data   # other terminal
```

**Terminal 4 – GUI (optional)**
```bash
ros2 run wildlife_gui gui
```

### Expected log for one boar

```
mission_node: New candidate candidate_1 at (x, y)
mission_node: Selected candidate_1 ... / Pause sent to exploration.
husky_node:   Investigate candidate_1 (attempt 1) ... Nav2 goal 1/3 ...
husky_node:   Observation pose reached.
animal_yolo_detector: Starting YOLO verification (mission request 1) ...
animal_yolo_detector: Consensus classification after 5 RGB frames (mission request 1): heat_4=boar 0.86 (4/5 votes)
mission_node: candidate_1: YOLO result -> exotic ...
mission_node: candidate_1 -> CONFIRMED_EXOTIC (boar 0.86) after 1 attempt(s).
mission_node: Resume sent to exploration (investigation finished).
```

### Test the Husky ↔ Nav2 ↔ YOLO part without the thermal chain

Start Terminal 2 with `spatial:=false` (perception stays on, because
`verify_now` is needed), then inject a candidate by hand at the position of
an animal from `large_demo.sdf`:

```bash
ros2 topic pub --once /husky1/mission/test_candidate geometry_msgs/msg/Point "{x: 6.0, y: 2.0, z: 0.0}"
```

### Useful checks

```bash
ros2 service call /husky1/perception/verify_now std_srvs/srv/Trigger   # YOLO answers?
ros2 topic pub --once /husky1/explore/resume std_msgs/msg/Bool "{data: true}"  # manual resume
ros2 topic echo /husky1/perception/thermal_yolo/result --field data
```

## Tuning

| Parameter (config/mission_params.yaml) | Default | When to change |
|---|---|---|
| `confirm_confidence` | 0.70 | R6 threshold. |
| `standoff_distances` | [3.0, 2.2, 2.6] | Poses too close to trees → larger. YOLO too_far → smaller. |
| `bearing_tolerance_deg` | 15 | Results rejected as `not_in_view` although the animal is centred → larger. |
| `duplicate_radius` | 1.5 | Two real animals closer than this → the second becomes DUPLICATE. 0 disables. |
| `pause_settle_sec` | 1.5 | Husky goal cancelled right after the pause → larger. |
| `look_from_current_within` | 4.0 | Attempt 1 only turns and looks when already this close. 0 = always drive. |

## Notes and limitations

- The flow drives to the candidate first. C1 3.3 lists "YOLO from the current
  pose" as attempt 1; that only happens here when the robot is already within
  `look_from_current_within` of the candidate.
- `candidate_registry` still publishes `state: uninvestigated`. Mission states
  are on `/husky1/mission/status`, which uses the same `frame_id` /
  `candidates[]` keys as `registry_json`, so a GUI panel can read either topic.
- `/Yolo_result` (CSV) is still published but no longer used by the mission.
- Nav2's goal checker ignores the final heading (`yaw_goal_tolerance: 3.14`);
  `husky_node` therefore turns with the `Spin` behaviour after arriving.

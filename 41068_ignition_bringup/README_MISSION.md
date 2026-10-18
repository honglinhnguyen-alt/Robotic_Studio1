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
| heat clearly seen but no animal box, 2 times from directions ≥ 30° apart | `UNRESOLVED` (reason `no_trained_animal_seen`, e.g. campfire) |
| low confidence / too far / partial / not in view / no heat / unreachable / timeout | next attempt (3.0 m, view rotated +50°; then 2.5 m, −50°) |
| 3 attempts without a decision | `UNRESOLVED` with a reason |

There is no "rejected / not an animal" outcome: YOLO knows three species, so
a heat source without a box may still be an untrained or hidden animal.
Every candidate ends confirmed or UNRESOLVED with one reason, chosen from
its attempts in this order (the first that occurred wins, so a possible
animal is never hidden behind a weaker reason):

| Reason | Meaning for the ranger |
|---|---|
| `low_confidence` (+ class and score) | Possible animal, YOLO below 0.70: check first |
| `no_trained_animal_seen` | Clear views, none of the trained species (fire, other species, hidden animal) |
| `too_far` / `partial_view` | Never seen closely or fully |
| `not_visible` | Heat source not visible from the observation poses |
| `unreachable` | No observation pose could be reached |
| `system_error` | YOLO unavailable or timed out |

If a later detection falls within `duplicate_radius` of an UNRESOLVED
candidate (any reason except `no_trained_animal_seen`), that candidate is
investigated once more (`max_reopen`) instead of being skipped.

8. If another candidate is waiting, the Husky goes there directly. Otherwise
   the mission **resumes** explore_lite.

## Files

| File | Change |
|---|---|
| `scripts/mission_node.py` | Rewritten: registry input, nearest-first, pause → settle → dispatch, result matching, 0.70 threshold, 3 attempts, Unresolved with reason/Unresolved, resume only when the Husky is idle, JSON status for the UI. |
| `scripts/husky_node.py` | Rewritten: observation-pose selection against the SLAM map, Nav2 goal retries (max 3), face the candidate, `verify_now`, status reports, cancel. |
| `scripts/animal_yolo_detector.py` | Added the `perception/verify_now` service, `request_id` / `trigger` fields in the consensus JSON, and the `auto_verify_on_heat` parameter (default `true` = old behaviour). |
| `launch/41068_thermal_yolo.launch.py` | New launch argument `auto_verify_on_heat` (default `true`). |
| `launch/41068_mission.launch.py` | New: explore_lite + perception + localiser + registry + mission + husky in one launch. |
| `config/mission_params.yaml` | New: all mission / husky parameters; `coverage_tracker` section. |
| `scripts/coverage_tracker.py` | New: R1 coverage — LiDAR-mapped and thermally checked share of the survey area. |
| `scripts/heat_source_logger.py` | Event log; also writes the run record (mission summary JSON + one row in `runs.csv`). |
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

The same launch also starts, unless switched off:

| Argument | Node | Output |
|---|---|---|
| `evaluate:=true` | `spatial_evaluator` | `/husky1/evaluation/summary`; ground truth from `worlds/<world>.sdf` (`world:=` must match Terminal 1) or `ground_truth_file:=` |
| `coverage:=true` | `coverage_tracker.py` | `/husky1/mission/coverage` (JSON), `/husky1/mission/thermal_coverage_grid` (OccupancyGrid for RViz) |
| `record:=true` | `heat_source_logger.py` | `~/wildtrack_runs/heat_sources_<stamp>.log`, `mission_summary_<stamp>.json`, one row in `~/wildtrack_runs/runs.csv` |

Name each scenario so the runs can be compared later, e.g.
```bash
ros2 launch 41068_ignition_bringup 41068_mission.launch.py run_label:=nominal_1
```
Do not also start `heat_source_logger.py` by hand while `record:=true`, or
the run is recorded twice.

### Run record (`~/wildtrack_runs/runs.csv`)

The logger writes the record when the mission reaches COMPLETE, rewrites it
if the mission leaves COMPLETE for a new candidate and reaches it again, and
writes it once more when stopped (Ctrl-C). A run stopped before COMPLETE is
kept with `completed=False`. One row per run:

- `duration_s`: ROS (simulation) time from the first status after
  INITIALISING to COMPLETE; `wall_duration_s`: logger wall-clock lifetime.
- `exotic` … `pending`: mission_node candidate counts at the end.
- `lidar_coverage_pct`, `thermal_coverage_pct`: last coverage_tracker values.
- `truth_objects` … `boars_classified`: last spatial_evaluator summary
  (R7 `mean_planar_error_m` / `within_tolerance`, R10 `boars_classified`).

`mission_summary_<stamp>.json` holds the same row plus the final mission
status (every candidate with its history fields), coverage and evaluation.

### Coverage (R1)

`coverage_tracker.py` keeps a 0.2 m grid over the survey boundary (a circle
of `boundary_radius` = 30 m around the start; large_demo is ringed by cliffs
35–38 m out). Every 0.5 s it ray-casts the thermal camera's 60° field of
view, up to `thermal_range_m` = 10 m, on the SLAM map (occupied cells stop a
ray) and marks the cells it reaches.

- accessible ≈ boundary cells − cells the SLAM map marks occupied
- `lidar_coverage` = known-free cells / accessible
- `thermal_coverage` = known-free cells the thermal camera has looked at / accessible
- `thermal_of_mapped` = the same over known-free cells only

Unknown cells that are in fact inside rocks or unreachable count as
accessible, so `lidar_coverage` stays below 100 % even after a complete
exploration. The value is reported, not used: the mission still completes
when explore_lite has no frontier left. In RViz, add a Map display on
`/husky1/mission/thermal_coverage_grid`: 0 (light) = thermally checked,
100 (dark) = mapped but never in the thermal view.

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
| `standoff_distances` | [3.5, 3.0, 2.5] | Observation distance per attempt: each retry a little closer. YOLO too_far → smaller. |
| `safety_radius` | 2.0 | No observation pose within safety_radius + goal_tolerance (2.5 m) of any known heat source, including other candidates; "look from here" only from >= 2.0 m. |
| `bearing_tolerance_deg` | 15 | Results rejected as `not_in_view` although the animal is centred → larger. |
| `duplicate_radius` | 2.5 | A new record this close to an investigated one is the same animal seen from another side (DUPLICATE, not visited). large_demo keeps heat sources >= 6 m apart; lower it, or set `duplicate_requires_shared_track: true`, for worlds with animals closer together. 0 disables. |
| `pause_settle_sec` | 1.5 | Husky goal cancelled right after the pause → larger. |
| `look_from_current_within` | 4.0 | Attempt 1 only turns and looks when already this close. 0 = always drive. |
| `boundary_radius` (coverage_tracker) | 30.0 | Survey area for R1 coverage; change with the world. |
| `thermal_range_m` (coverage_tracker) | 10.0 | Distance at which the thermal camera counts an area as checked. |

## Notes and limitations

- Return home: with `return_to_init: true` (explore_lite params) the Husky
  drives back to its start pose when no frontier is left. A heat source seen
  on the way is still investigated; the return restarts afterwards.

- The flow drives to the candidate first. C1 3.3 lists "YOLO from the current
  pose" as attempt 1; that only happens here when the robot is already within
  `look_from_current_within` of the candidate.
- `candidate_registry` still publishes `state: uninvestigated`. Mission states
  are on `/husky1/mission/status`, which uses the same `frame_id` /
  `candidates[]` keys as `registry_json`, so a GUI panel can read either topic.
- `/Yolo_result` (CSV) is still published but no longer used by the mission.
- Nav2's goal checker ignores the final heading (`yaw_goal_tolerance: 3.14`);
  `husky_node` therefore turns with the `Spin` behaviour after arriving.

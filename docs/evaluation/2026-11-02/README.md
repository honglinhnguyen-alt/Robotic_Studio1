# Evaluation batch, 2 Nov 2026 (species temperatures)

18 missions on `large_demo`, run with
`41068_demo.launch.py rviz:=false gui:=false` (thermal sweep on, return home),
6 runs per condition, interleaved:

| Condition | Layout |
|---|---|
| nominal_world | world file layout (`seed:=world`) |
| nominal_seed | `seed:=11..16`, animals in the open (>= 3 m from obstacles) |
| cover_seed | `seed:=21..26 profile:=cover`, every animal 1-2 m from a tree |

Every pair of heat sources >= 6 m apart; 5 boars, 4 deer, 3 kangaroos,
2 campfires per run. Heat source temperatures: boar 328.15 K, deer
325.15 K, kangaroo 322.15 K, campfire 330.15 K (the 19 Oct batch had every
heat source at 330.15 K). Seeds 11-14 and 21-24 give the same layouts as the
19 Oct batch.

16 runs used commit 53d6741. `nominal_world_4` and `cover_seed26` were run
again after the explore_lite fix described below (branch
`explore-pause-return`); their first runs are kept in the local batch folder
under `superseded/` and listed below.

Files: `runs_table.md` (per run and per condition), `confusion_matrix.md`
(ground truth x final mission verdict, all runs), `fig_*.png`, the raw
`runs.csv` written by heat_source_logger and `batch.csv` (completion and
return home per run, as seen by the batch script). Regenerate with
`python3 41068_ignition_bringup/scripts/analyse_runs.py <runs_dir> <out_dir>`.

## Summary (18 runs)

| Metric | Result |
|---|---|
| Missions completed (batch script) | 18/18 |
| Returned home | 17/18 |
| Heat sources found | 244/252 (96.8%) |
| False positives | 0 |
| Found sources within 1 m | 243/244; mean planar error 0.30 m, max 1.05 m |
| Boars confirmed exotic (R10, target >= 80%) | 87/90 (96.7%) |
| Thermal coverage at the end | 93.0-97.6% |
| Mission time | 14.9-21.8 min (mean 1052 s) |

No animal was given the wrong species. 9 of 36 campfires were reported as
animals (8 kangaroo, 1 deer), 26 were left unresolved, 1 was missed. Missed
animals: 4 deer, 2 kangaroos, 1 boar.

## Known issues

- **Return-home conflict (fixed, runs replaced).** In the first runs of
  `nominal_world_4` and `cover_seed26`, explore_lite sent its return-home goal
  after the mission had paused exploration for the thermal sweep. It and
  husky_node then kept preempting each other's Nav2 goals until explore gave
  up after 3 attempts; the mission never reached COMPLETE (first
  cover_seed26: 9/14 found, thermal coverage 87.8%; first nominal_world_4:
  13/14). explore_lite now neither sends nor retries the return home while
  paused. With the fix both layouts completed and returned home (14/14 and
  13/14 found).
- **Stuck on the way home (not fixed).** `nominal_world_5` completed the
  mission, but the Husky made no progress on the way home for 5 min (Nav2
  "Failed to make progress"). This run is kept as it is.
- **Late candidate after return home.** In `nominal_world_3` and
  `cover_seed26` a new candidate appeared less than 1 s after the Husky was
  home. The mission left COMPLETE to investigate it, so `runs.csv` (and the
  "Completed" column of `runs_table.md`) say False, and that candidate's
  verdict is the raw `unknown_heat_source` in the confusion matrix (one boar,
  one kangaroo). `batch.csv` records the COMPLETE state and return home.

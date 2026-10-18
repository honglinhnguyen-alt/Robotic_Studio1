# Evaluation batch, 19 Oct 2026

12 full missions on `c4-demo-fixes` (commit 18fcfcf), `large_demo`, run with
`41068_demo.launch.py` (thermal sweep on, return home), 4 runs per condition:

| Condition | Layout |
|---|---|
| nominal_world | world file layout (`world:=large_demo`) |
| nominal_seed | `seed:=11..14`, animals in the open (>= 3 m from obstacles) |
| cover_seed | `seed:=21..24 profile:=cover`, every animal 1-2 m from a tree |

Every pair of heat sources >= 6 m apart; 5 boars, 4 deer, 3 kangaroos,
2 campfires per run.

Files: `runs_table.md` (per run and per condition), `confusion_matrix.md`
(ground truth x final mission verdict, all runs), `fig_*.png`, the raw
`runs.csv` written by heat_source_logger and `batch.csv` (completion and
return home per run). Regenerate with
`python3 41068_ignition_bringup/scripts/analyse_runs.py <runs_dir> <out_dir>`.

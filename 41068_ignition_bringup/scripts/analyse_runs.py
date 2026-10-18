#!/usr/bin/env python3
"""Tables and figures for a batch of Wild Track mission runs (C4/C5 evidence).

    python3 analyse_runs.py <runs_dir> <out_dir>

<runs_dir> is the logger/evaluator output directory (the one holding runs.csv,
evaluation/run_*/ and heat_sources_*.log). Writes to <out_dir>:
  runs_table.md        one row per run + per-condition means
  confusion_matrix.md  ground truth class x final mission verdict
  fig_detection.png    recall / boar classification / accuracy per condition
  fig_localization.png planar error of every matched heat source (R7)
  fig_coverage.png     thermal coverage before and after the sweep (R1)
  fig_duration.png     mission time per run
The condition of a run is its run_label without the trailing run number
(nominal_world_1 -> nominal_world, cover_seed21 -> cover_seed).
"""

import csv
import glob
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette, first three slots (all-pairs validated).
CONDITION_COLOURS = ['#2a78d6', '#eb6834', '#1baf7a']
TEXT = '#0b0b0b'
MUTED = '#52514e'
GRID = '#e4e3df'
SURFACE = '#fcfcfb'


def condition_of(label):
    return re.sub(r'(_\d+|\d+)$', '', label)


def load_runs(runs_dir):
    with open(os.path.join(runs_dir, 'runs.csv'), newline='') as f:
        rows = [r for r in csv.DictReader(f)]
    batch = {}
    path = os.path.join(runs_dir, '..', 'batch.csv')
    if os.path.exists(path):
        with open(path, newline='') as f:
            batch = {r['label']: r for r in csv.DictReader(f)}
    for row in rows:
        row['condition'] = condition_of(row['run_label'])
        row.update({f'batch_{k}': v for k, v in batch.get(row['run_label'], {}).items()})
        log = row.get('log_file', '')
        log = os.path.join(runs_dir, os.path.basename(log)) if log else ''
        row['coverage_before_sweep'] = ''
        if os.path.exists(log):
            text = open(log).read()
            m = re.search(r'THERMAL SWEEP: .*thermal coverage (\d+)%', text)
            if m:
                row['coverage_before_sweep'] = float(m.group(1))
    return rows


def evaluation_for(runs_dir, stamp):
    """Last evaluation rows of the evaluator run started closest to stamp."""
    target = datetime.strptime(stamp, '%Y%m%d_%H%M%S')
    best = None
    for d in glob.glob(os.path.join(runs_dir, 'evaluation', 'run_*')):
        m = re.search(r'run_(\d{8}_\d{6})', d)
        gap = abs((datetime.strptime(m.group(1), '%Y%m%d_%H%M%S') - target).total_seconds())
        if gap < 120 and (best is None or gap < best[0]):
            best = (gap, d)
    if best is None:
        return []
    with open(os.path.join(best[1], 'evaluation.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    last = max(int(r['evaluation_index']) for r in rows)
    return [r for r in rows if int(r['evaluation_index']) == last]


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', fontsize=11, color=TEXT)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)


def strip(ax, groups, conditions, offset_scale=0.12):
    """One dot per value, conditions side by side, mean as a short bar."""
    for i, cond in enumerate(conditions):
        values = [v for v in groups.get(cond, []) if v is not None]
        xs = [i + offset_scale * ((k % 5) - 2) / 2 for k in range(len(values))]
        ax.scatter(xs, values, s=36, color=CONDITION_COLOURS[i], edgecolor=SURFACE,
                   linewidth=1.5, zorder=3)
        if values:
            m = statistics.mean(values)
            ax.plot([i - 0.25, i + 0.25], [m, m], color=TEXT, linewidth=2, zorder=4)
            ax.annotate(f'{m:.2f}' if m < 10 else f'{m:.0f}', (i + 0.28, m),
                        va='center', fontsize=8, color=TEXT)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9, color=TEXT)
    ax.set_xlim(-0.6, len(conditions) - 0.4)


def main():
    runs_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    runs = load_runs(runs_dir)
    conditions = sorted({r['condition'] for r in runs},
                        key=lambda c: ('world' not in c, 'cover' in c, c))

    confusion = Counter()
    errors = defaultdict(list)
    for run in runs:
        rows = evaluation_for(runs_dir, run['stamp'])
        run['eval_rows'] = rows
        for r in rows:
            if r['status'] == 'matched':
                errors[run['condition']].append(num(r['planar_error_m']))
                confusion[(r['ground_truth_class'], r['estimate_class'])] += 1
            elif r['status'] == 'missed':
                confusion[(r['ground_truth_class'], 'missed')] += 1
            elif r['status'] == 'false_positive':
                confusion[('(no truth)', r['estimate_class'])] += 1

    # ---------------- tables ----------------
    columns = [
        ('run_label', 'Run'), ('completed', 'Done'), ('batch_returned_home', 'Home'),
        ('duration_s', 'Time (s)'), ('candidates', 'Cand.'), ('exotic', 'Exotic'),
        ('native', 'Native'), ('unresolved', 'Unres.'), ('matched', 'Found'),
        ('truth_objects', 'Truth'), ('false_positives', 'FP'),
        ('mean_planar_error_m', 'Mean err (m)'), ('max_planar_error_m', 'Max err (m)'),
        ('within_tolerance', '<=1 m'), ('classification_accuracy', 'Class acc.'),
        ('boars_classified', 'Boars ok'), ('boars', 'Boars'),
        ('coverage_before_sweep', 'Thermal % before sweep'),
        ('thermal_coverage_pct', 'Thermal % end'), ('lidar_coverage_pct', 'Map %'),
        ('batch_sweep_looks', 'Sweep looks'),
    ]
    lines = ['| ' + ' | '.join(h for _, h in columns) + ' |',
             '|' + '---|' * len(columns)]
    for run in runs:
        lines.append('| ' + ' | '.join(str(run.get(k, '')) for k, _ in columns) + ' |')
    lines += ['', '| Condition | Runs | Completed | Returned home | Time (s) | Recall | '
              'Precision | Mean err (m) | Within 1 m | Class acc. | Boars ok / total | '
              'Thermal % end |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for cond in conditions:
        rs = [r for r in runs if r['condition'] == cond]
        boars_ok = sum(int(num(r['boars_classified']) or 0) for r in rs)
        boars = sum(int(num(r['boars']) or 0) for r in rs)
        within = sum(int(num(r['within_tolerance']) or 0) for r in rs)
        matched = sum(int(num(r['matched']) or 0) for r in rs)

        def fmt(v, d=2):
            return '' if v is None else f'{v:.{d}f}'
        lines.append(
            f'| {cond} | {len(rs)} | {sum(r["completed"] == "True" for r in rs)} | '
            f'{sum(r.get("batch_returned_home") == "1" for r in rs)} | '
            f'{fmt(mean(num(r["duration_s"]) for r in rs), 0)} | '
            f'{fmt(mean(num(r["recall"]) for r in rs))} | '
            f'{fmt(mean(num(r["precision"]) for r in rs))} | '
            f'{fmt(mean(num(r["mean_planar_error_m"]) for r in rs))} | '
            f'{within}/{matched} | '
            f'{fmt(mean(num(r["classification_accuracy"]) for r in rs))} | '
            f'{boars_ok}/{boars} | '
            f'{fmt(mean(num(r["thermal_coverage_pct"]) for r in rs), 1)} |')
    with open(os.path.join(out_dir, 'runs_table.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    truths = ['boar', 'deer', 'kangaroo', 'campfire', '(no truth)']
    verdicts = ['boar', 'deer', 'kangaroo', 'unresolved', 'missed']
    other = sorted({v for (_, v) in confusion} - set(verdicts))
    verdicts += other
    cm = ['| Truth \\ verdict | ' + ' | '.join(verdicts) + ' |',
          '|' + '---|' * (len(verdicts) + 1)]
    for t in truths:
        cm.append(f'| {t} | ' + ' | '.join(str(confusion.get((t, v), 0)) for v in verdicts)
                  + ' |')
    with open(os.path.join(out_dir, 'confusion_matrix.md'), 'w') as f:
        f.write('\n'.join(cm) + '\n')

    # ---------------- figures ----------------
    by_cond = defaultdict(lambda: defaultdict(list))
    for r in runs:
        c = r['condition']
        by_cond['recall'][c].append(num(r['recall']))
        by_cond['accuracy'][c].append(num(r['classification_accuracy']))
        boars = num(r['boars'])
        ok = num(r['boars_classified'])
        by_cond['boar'][c].append(ok / boars if boars else None)
        by_cond['duration'][c].append((num(r['duration_s']) or 0) / 60.0)
        by_cond['thermal_end'][c].append(num(r['thermal_coverage_pct']))
        by_cond['thermal_before'][c].append(num(r['coverage_before_sweep']))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), facecolor=SURFACE)
    for ax, key, title in zip(axes, ('recall', 'boar', 'accuracy'),
                              ('Heat sources found (recall)',
                               'Boars confirmed exotic (R10)',
                               'Class correct when found')):
        strip(ax, by_cond[key], conditions)
        style(ax, title, 'fraction per run')
        ax.set_ylim(0, 1.05)
    axes[1].axhline(0.8, color=MUTED, linestyle='--', linewidth=1)
    axes[1].annotate('R10 target 0.80', (len(conditions) - 0.45, 0.81), fontsize=7,
                     color=MUTED, ha='right', va='bottom')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_detection.png'), dpi=160)

    fig, ax = plt.subplots(figsize=(6, 3.6), facecolor=SURFACE)
    strip(ax, errors, conditions, offset_scale=0.3)
    style(ax, 'Localization error of every matched heat source (R7)', 'planar error (m)')
    ax.axhline(1.0, color=MUTED, linestyle='--', linewidth=1)
    ax.annotate('R7 limit 1.0 m', (len(conditions) - 0.45, 1.02), fontsize=7, color=MUTED,
                ha='right', va='bottom')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_localization.png'), dpi=160)

    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    xs, labels = 0, []
    for i, cond in enumerate(conditions):
        rs = [r for r in runs if r['condition'] == cond]
        for r in rs:
            before = num(r['coverage_before_sweep'])
            after = num(r['thermal_coverage_pct'])
            if before is not None and after is not None:
                ax.plot([xs, xs], [before, after], color=GRID, linewidth=2, zorder=1)
                ax.scatter([xs], [before], s=28, facecolor=SURFACE,
                           edgecolor=CONDITION_COLOURS[i], linewidth=1.5, zorder=2)
            if after is not None:
                ax.scatter([xs], [after], s=36, color=CONDITION_COLOURS[i], zorder=3)
            labels.append(r['run_label'].replace('nominal_', 'n_').replace('cover_', 'c_'))
            xs += 1
        xs += 0.6
    style(ax, 'Thermal coverage: hollow = before sweep, filled = end (R1)',
          'thermal coverage (%)')
    ax.axhline(95, color=MUTED, linestyle='--', linewidth=1)
    ax.annotate('R1 target 95 %', (0.01, 95.05), xycoords=('axes fraction', 'data'),
                fontsize=7, color=MUTED, ha='left', va='bottom')
    positions = []
    x = 0
    for cond in conditions:
        for _ in [r for r in runs if r['condition'] == cond]:
            positions.append(x)
            x += 1
        x += 0.6
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7, color=TEXT)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_coverage.png'), dpi=160)

    fig, ax = plt.subplots(figsize=(6, 3.6), facecolor=SURFACE)
    strip(ax, by_cond['duration'], conditions)
    style(ax, 'Mission time per run (simulation time)', 'minutes')
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_duration.png'), dpi=160)
    print(f'{len(runs)} runs -> {out_dir}')


if __name__ == '__main__':
    main()

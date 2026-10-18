"""Unit tests for coverage_tracker and the heat_source_logger run record.

Run from the package root (ROS sourced, the nodes import rclpy):
    python3 -m pytest test/test_coverage_and_run_record.py -q
"""

import csv
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from coverage_tracker import (  # noqa: E402
    SurveyGrid,
    coverage_stats,
    sample_map,
    visible_points,
)
from heat_source_logger import (  # noqa: E402
    RUN_CSV_FIELDS,
    run_record,
    upsert_run_row,
)


def make_map(size=200, resolution=0.1, origin=-10.0):
    """Free 20 m x 20 m map with a wall at x = 3 m for |y| < 1 m."""
    grid = np.zeros((size, size), dtype=np.int16)
    xs = origin + (np.arange(size) + 0.5) * resolution
    wall_col = int((3.0 - origin) / resolution)
    rows = np.where(np.abs(xs) < 1.0)[0]
    grid[rows, wall_col] = 100

    def lookup(x, y):
        return sample_map(grid, origin, origin, resolution, x, y)
    return grid, lookup


def test_sample_map_returns_unknown_outside():
    grid, lookup = make_map()
    values = lookup(np.array([0.0, 3.05, 50.0]), np.array([0.0, 0.0, 0.0]))
    assert values.tolist() == [0, 100, -1]


def test_visible_points_respect_fov_range_and_obstacles():
    _, lookup = make_map()
    xs, ys = visible_points(0.0, 0.0, 0.0, math.radians(60.0), 5.0, 0.1, lookup)

    ranges = np.hypot(xs, ys)
    bearings = np.degrees(np.arctan2(ys, xs))
    assert ranges.max() <= 5.0 + 1e-6
    assert np.abs(bearings).max() <= 30.0 + 1e-6
    # Straight ahead the wall at x = 3 m blocks the ray.
    ahead = np.abs(ys) < 0.05
    assert xs[ahead].max() < 3.0
    # Off to the side (y > 1 m) the ray passes the wall's end.
    side = ys > 1.5
    assert xs[side].max() > 3.0


def test_coverage_stats_counts_mapped_and_thermally_checked_area():
    survey = SurveyGrid(0.0, 0.0, 5.0, 0.5)
    values = np.full(survey.inside.shape, -1, dtype=np.int16)
    right = survey.xs > 0.0
    values[right] = 0                          # right half mapped free
    values[right & (survey.ys > 3.0)] = 100    # a few occupied cells
    survey.mark_seen(survey.xs[right & (survey.ys < 0.0)],
                     survey.ys[right & (survey.ys < 0.0)])

    stats = coverage_stats(survey, values)

    inside = survey.inside
    occupied = inside & right & (survey.ys > 3.0)
    free = inside & right & ~(survey.ys > 3.0)
    accessible = inside.sum() - occupied.sum()
    assert stats['lidar_coverage'] == pytest.approx(free.sum() / accessible)
    checked = inside & right & (survey.ys < 0.0)
    assert stats['thermal_coverage'] == pytest.approx(checked.sum() / accessible)
    assert stats['thermal_of_mapped'] == pytest.approx(checked.sum() / free.sum())
    assert 0.0 < stats['thermal_coverage'] < stats['lidar_coverage'] < 1.0


STATUS = {
    'mission_state': 'COMPLETE',
    'exploration_status': 'exploration_complete',
    'counts': {'exotic': 2, 'native': 3, 'unresolved': 2,
               'unresolved_low_confidence': 1,
               'pending': 0, 'duplicate': 0},
    'candidates': [{}] * 7,
}
EVALUATION = {
    'ground_truth_count': 14, 'matched_count': 8, 'false_positive_count': 1,
    'precision': 0.888888, 'recall': 0.571428, 'mean_planar_error_m': 0.4123,
    'maximum_planar_error_m': 1.31, 'within_tolerance_count': 7,
    'classification_accuracy': 0.75,
    'per_class': {'boar': {'ground_truth': 5, 'detected': 3,
                           'correctly_classified': 2}},
}
COVERAGE = {'lidar_coverage': 0.8712, 'thermal_coverage': 0.4321,
            'accessible_m2': 2700.0}


def test_run_record_row_has_every_csv_field():
    record = run_record(
        stamp='20261018_120000', run_label='nominal_1', world='large_demo',
        completed=True, mission_start=12.0, mission_end=612.5,
        wall_seconds=700.0, status=STATUS, coverage=COVERAGE,
        evaluation=EVALUATION, log_file='x.log')

    row = record['row']
    assert set(RUN_CSV_FIELDS) == set(row)
    assert row['duration_s'] == pytest.approx(600.5)
    assert row['exotic'] == 2 and row['candidates'] == 7
    assert row['lidar_coverage_pct'] == pytest.approx(87.1)
    assert row['thermal_coverage_pct'] == pytest.approx(43.2)
    assert row['boars'] == 5 and row['boars_classified'] == 2
    assert row['mean_planar_error_m'] == pytest.approx(0.412)


def test_run_record_without_inputs_leaves_fields_empty():
    record = run_record('s', '', '', False, None, 5.0, 1.0, None, None, None, 'x')
    assert record['row']['duration_s'] == ''
    assert record['row']['recall'] == ''
    assert record['row']['completed'] is False


def test_upsert_replaces_the_row_of_the_same_run(tmp_path):
    path = tmp_path / 'runs.csv'
    first = run_record('a', 'nominal_1', 'large_demo', False, 0.0, 10.0, 10.0,
                       STATUS, None, None, 'a.log')['row']
    other = run_record('b', 'nominal_2', 'large_demo', True, 0.0, 20.0, 20.0,
                       STATUS, None, None, 'b.log')['row']
    upsert_run_row(path, first)
    upsert_run_row(path, other)
    upsert_run_row(path, dict(first, completed=True, duration_s=30.0))

    with path.open(newline='') as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [row['stamp'] for row in rows] == ['b', 'a']
    assert rows[1]['completed'] == 'True'
    assert rows[1]['duration_s'] == '30.0'

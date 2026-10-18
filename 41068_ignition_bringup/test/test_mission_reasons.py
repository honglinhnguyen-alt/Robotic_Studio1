"""Unit tests for mission_node's UNRESOLVED reasons and early stop.

Run from the package root (ROS sourced, the node imports rclpy):
    python3 -m pytest test/test_mission_reasons.py -q
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from mission_node import unresolved_reason, views_differ  # noqa: E402


def attempt(outcome, class_name=None, confidence=None):
    return {'attempt': 1, 'outcome': outcome,
            'class_name': class_name, 'confidence': confidence}


def test_possible_animal_wins_over_every_other_reason():
    history = [attempt('too_far'), attempt('low_confidence', 'deer', 0.55),
               attempt('low_confidence', 'boar', 0.66), attempt('unreachable')]
    assert unresolved_reason(history) == ('low_confidence', 'boar 0.66')


def test_reason_priority_order():
    assert unresolved_reason([attempt('unreachable'), attempt('no_animal')]) == \
        ('no_trained_animal_seen', '')
    assert unresolved_reason([attempt('not_in_view'), attempt('too_far')]) == ('too_far', '')
    assert unresolved_reason([attempt('no_heat'), attempt('unreachable')]) == \
        ('not_visible', '')
    assert unresolved_reason([attempt('verify_timeout')]) == ('system_error', '')
    assert unresolved_reason([]) == ('system_error', '')


def test_early_stop_needs_two_different_directions():
    sep = math.radians(30.0)
    assert not views_differ([0.5], sep)
    assert not views_differ([0.5, 0.6], sep)                    # same side twice
    assert views_differ([0.5, 0.5 + math.radians(50)], sep)     # attempt 2 offset
    assert views_differ([math.pi - 0.1, -math.pi + 0.2], sep) is False  # wraps


def test_unchecked_clusters_returns_large_unseen_areas_inside_them():
    import numpy as np
    from mission_node import unchecked_clusters
    grid = np.full((100, 100), -1, dtype=np.int16)
    grid[10:90, 10:90] = 0          # thermally checked
    grid[20:40, 60:80] = 100        # 20 x 20 cells unchecked = 4 m^2 at 0.1 m
    grid[70:73, 20:23] = 100        # 9 cells = 0.09 m^2, too small
    clusters = unchecked_clusters(grid, -5.0, -5.0, 0.1, 3.0)
    assert len(clusters) == 1
    x, y, area = clusters[0]
    assert abs(area - 4.0) < 1e-6
    assert 1.0 <= x <= 3.0 and -3.0 <= y <= -1.0   # cols 60-80, rows 20-40
    col = int(round((x + 5.0) / 0.1 - 0.5))
    row = int(round((y + 5.0) / 0.1 - 0.5))
    assert grid[row, col] == 100

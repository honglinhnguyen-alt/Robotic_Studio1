"""Unit tests for husky_node's observation-pose free-space check.

Run from the package root (ROS sourced, the node imports rclpy):
    python3 -m pytest test/test_husky_viewpoints.py -q
"""

import os
import sys

import pytest
from nav_msgs.msg import OccupancyGrid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from husky_node import GridChecker, pick_viewpoints  # noqa: E402


def grid_10m():
    """10 m x 10 m map from (0, 0): free, unknown band x > 8 m, one wall cell."""
    grid = OccupancyGrid()
    grid.info.resolution = 0.5
    grid.info.width = 20
    grid.info.height = 20
    data = []
    for row in range(20):
        for col in range(20):
            data.append(-1 if col >= 16 else 0)
    data[10 * 20 + 5] = 100        # wall at (2.5..3.0, 5.0..5.5)
    grid.data = data
    return grid


def test_free_unknown_and_occupied_inside_the_map():
    checker = GridChecker(grid_10m())
    assert checker.is_clear(1.0, 1.0, 0.6, allow_unknown=False)
    assert not checker.is_clear(9.0, 1.0, 0.6, allow_unknown=False)
    assert checker.is_clear(9.0, 1.0, 0.6, allow_unknown=True)
    assert not checker.is_clear(2.7, 5.2, 0.6, allow_unknown=True)


def test_pose_off_the_map_is_never_clear():
    checker = GridChecker(grid_10m())
    assert not checker.is_clear(150.0, 0.0, 0.6, allow_unknown=True)
    assert not checker.is_clear(-1.0, 5.0, 0.6, allow_unknown=True)
    poses = [(150.0, 0.0, 0.0), (9.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
    assert pick_viewpoints(poses, checker, 0.6, 3) == [(1.0, 1.0, 0.0), (9.0, 1.0, 0.0)]


def test_poses_near_other_heat_sources_are_dropped():
    from husky_node import outside_safety, viewpoint_candidates
    poses = viewpoint_candidates((0.0, 0.0), (5.0, 0.0), 3.0, 0.0, [0.0, 1.5708])
    # (3, 0) is 1 m from another animal at (3, 1); (0, 3) is clear of it.
    kept = outside_safety(poses, [(3.0, 1.0)], 3.0)
    assert len(kept) == 1
    assert kept[0][0] == pytest.approx(0.0, abs=1e-3)
    assert kept[0][1] == pytest.approx(3.0)

"""Unit tests for the stop-turn-go gate (no ROS graph needed).

Run from the package root (ROS sourced, the node imports rclpy):
    python3 -m pytest test/test_stop_turn_go.py -q
"""

import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from stop_turn_go_controller import (  # noqa: E402
    MotionGate,
    corridor_distance,
    heading_to_path,
)

STRAIGHT_X = [[0.1 * i, 0.0] for i in range(50)]


def test_heading_to_path_straight_and_left():
    error, distance = heading_to_path(STRAIGHT_X, 0.0, 0.0, 0.0, 0.8)
    assert error == pytest.approx(0.0)
    assert distance == pytest.approx(0.8, abs=0.11)

    error, _ = heading_to_path(STRAIGHT_X, 0.0, 0.0, -math.pi / 2, 0.8)
    assert error == pytest.approx(math.pi / 2)


def test_heading_to_path_uses_point_ahead_of_closest():
    # Robot already half-way along the path: target is ahead of it, not the start.
    error, distance = heading_to_path(STRAIGHT_X, 2.5, 0.0, 0.0, 0.8)
    assert error == pytest.approx(0.0)
    assert distance == pytest.approx(0.8, abs=0.11)
    assert heading_to_path([], 0.0, 0.0, 0.0, 0.8) is None


def test_corridor_sees_right_front_in_0_to_2pi_scan():
    # 360 beams over [0, 2pi): the old sector check missed angles near 2pi.
    ranges = [math.inf] * 360
    ranges[355] = 0.7          # -5 deg, right front
    ranges[90] = 0.4           # beside the chassis, cannot be hit driving straight
    ranges[60] = 1.0           # 60 deg left: y = 0.87 m, outside the corridor
    distance = corridor_distance(ranges, 0.0, 2 * math.pi / 360, 0.2, 40.0, 0.45, 0.35)
    assert distance == pytest.approx(0.7 * math.cos(math.radians(5)), abs=1e-6)


def run(gate, steps, req_v, req_w, heading, obstacle=False, start=0.0, dt=0.05):
    """Step the gate with a still robot; return the outputs."""
    outputs = []
    for i in range(steps):
        outputs.append(gate.step(start + i * dt, req_v, req_w, heading, obstacle, 0.0, 0.0))
    return outputs


def test_large_heading_error_turns_in_place_then_drives_straight():
    gate = MotionGate()
    out = run(gate, 5, 0.8, 0.2, (0.6, 2.0))
    assert out[-1][0] == 0.0 and out[-1][1] > 0.0          # turning left, no linear
    assert gate.mode == MotionGate.TURNING

    # Between exit (0.08) and enter (0.35): keep turning (hysteresis).
    out = run(gate, 3, 0.8, 0.2, (0.2, 2.0), start=1.0)
    assert out[-1][0] == 0.0 and out[-1][1] > 0.0

    # Aligned: stop first, then drive straight at DWB's speed.
    out = run(gate, 5, 0.8, 0.2, (0.03, 2.0), start=2.0)
    assert out[0] == (0.0, 0.0)
    assert out[-1] == (0.8, 0.0)
    assert gate.mode == MotionGate.MOVING


def test_small_heading_error_drives_without_stopping():
    gate = MotionGate()
    run(gate, 5, 0.6, 0.3, (0.1, 2.0))
    out = run(gate, 5, 0.6, -0.3, (-0.3, 2.0), start=1.0)     # below enter threshold
    assert all(o == (0.6, 0.0) for o in out)


def test_nav2_pure_rotation_is_passed_through():
    gate = MotionGate()
    out = run(gate, 5, 0.0, -0.8, None)
    assert out[-1] == (0.0, -1.0)


def test_slow_rotation_request_is_raised_to_a_speed_that_turns_the_husky():
    # RotationShim asked for 0.32 rad/s from standstill; the simulated Husky
    # does not turn below ~1.0 rad/s commanded, so this stalled for 10 s.
    gate = MotionGate()
    out = run(gate, 5, 0.0, 0.32, (1.0, 2.0))
    assert out[-1] == (0.0, 1.0)


def test_obstacle_blocks_driving_but_not_turning():
    gate = MotionGate()
    assert run(gate, 5, 0.6, 0.0, (0.0, 2.0), obstacle=True)[-1] == (0.0, 0.0)
    out = run(gate, 5, 0.6, 0.0, (1.0, 2.0), obstacle=True, start=1.0)
    assert out[-1][0] == 0.0 and out[-1][1] > 0.0


def test_no_turn_onto_the_last_path_point():
    gate = MotionGate()
    out = run(gate, 5, 0.3, 0.0, (1.2, 0.2))      # target only 0.2 m away
    assert out[-1] == (0.3, 0.0)


def test_waits_for_the_chassis_to_stop_before_switching():
    gate = MotionGate()
    run(gate, 5, 0.8, 0.0, (0.0, 2.0))
    assert gate.mode == MotionGate.MOVING
    # Large error while still rolling at 0.8 m/s: stop, do not turn yet.
    out = [gate.step(1.0 + 0.05 * i, 0.8, 0.0, (0.8, 2.0), False, 0.8, 0.0)
           for i in range(5)]
    assert all(o == (0.0, 0.0) for o in out)
    # Once odometry says it stopped, it turns.
    assert gate.step(2.0, 0.8, 0.0, (0.8, 2.0), False, 0.0, 0.0)[1] > 0.0


def test_never_linear_and_angular_together():
    rng = random.Random(4)
    gate = MotionGate(allow_reverse=True)
    now = 0.0
    for _ in range(5000):
        now += 0.05
        heading = None if rng.random() < 0.2 else (
            rng.uniform(-math.pi, math.pi), rng.uniform(0.0, 3.0))
        v, w = gate.step(now, rng.uniform(-0.5, 1.0), rng.uniform(-1.5, 1.5), heading,
                         rng.random() < 0.2, rng.uniform(-0.1, 0.1), rng.uniform(-0.2, 0.2))
        assert v == 0.0 or w == 0.0


def test_backup_recovery_reverses_straight_when_allowed():
    gate = MotionGate(allow_reverse=True)
    assert run(gate, 5, -0.05, 0.0, (0.5, 2.0))[-1] == (-0.05, 0.0)
    gate = MotionGate(allow_reverse=False)
    assert run(gate, 5, -0.05, 0.0, (0.5, 2.0))[-1] == (0.0, 0.0)

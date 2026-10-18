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

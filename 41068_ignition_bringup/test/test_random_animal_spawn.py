"""Unit tests for the world-driven animal layout (no Gazebo needed).

Run from the package root (ROS sourced):
    python3 -m pytest test/test_random_animal_spawn.py -q
"""

import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from random_animal_spawn import (  # noqa: E402
    choose_layout,
    ground_truth_payload,
    load_world,
    split_world,
)

WORLD = os.path.join(os.path.dirname(__file__), '..', 'worlds', 'large_demo.sdf')


@pytest.fixture(scope='module')
def world():
    name, includes = load_world(WORLD)
    return name, split_world(includes)


def test_world_is_split_into_animals_campfires_and_obstacles(world):
    name, (animals, campfires, obstacles) = world
    assert name == 'large_demo'
    assert len(animals) == 12
    assert len(campfires) == 2
    assert not any(o['model'] in ('boar', 'campfire', 'forest_plane') for o in obstacles)
    assert any('tree' in o['model'] for o in obstacles)


@pytest.mark.parametrize('seed', [1, 2, 3, 7])
def test_nominal_layout_keeps_spacing_and_clearance(world, seed):
    _, (animals, campfires, obstacles) = world
    layout = choose_layout(random.Random(seed), animals, campfires, obstacles)
    points = list(layout.values()) + [(c['x'], c['y'], 0.0) for c in campfires]
    assert len(layout) == len(animals)
    for i, (x, y, _) in enumerate(points):
        for (u, v, _) in points[i + 1:]:
            assert math.hypot(x - u, y - v) >= 6.0
    for x, y, _ in layout.values():
        assert 5.0 <= math.hypot(x, y) <= 22.0
        assert min(math.hypot(x - o['x'], y - o['y']) for o in obstacles) >= 3.0


def test_cover_layout_puts_every_animal_next_to_a_tree(world):
    _, (animals, campfires, obstacles) = world
    trees = [o for o in obstacles if 'tree' in o['model']]
    layout = choose_layout(random.Random(7), animals, campfires, obstacles,
                           profile='cover')
    for x, y, _ in layout.values():
        assert 1.0 <= min(math.hypot(x - t['x'], y - t['y']) for t in trees) <= 2.0


def test_same_seed_same_layout(world):
    _, (animals, campfires, obstacles) = world
    first = choose_layout(random.Random(5), animals, campfires, obstacles)
    assert first == choose_layout(random.Random(5), animals, campfires, obstacles)


def test_ground_truth_lists_moved_animals_and_fixed_campfires(world):
    _, (animals, campfires, obstacles) = world
    layout = choose_layout(random.Random(1), animals, campfires, obstacles)
    payload = ground_truth_payload(animals, campfires, layout)
    assert payload['frame_id'] == 'husky1_map'
    by_id = {o['id']: o for o in payload['objects']}
    assert len(by_id) == 14
    x, y, _ = layout['boar_01']
    assert by_id['boar_01']['position'] == {'x': x, 'y': y, 'z': 0.0}
    assert by_id['boar_01']['class_name'] == 'boar'
    assert by_id['male_deer_01']['class_name'] == 'deer'
    assert by_id['campfire1']['class_name'] == 'campfire'

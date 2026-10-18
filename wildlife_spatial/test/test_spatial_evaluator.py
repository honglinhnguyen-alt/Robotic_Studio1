import json
from pathlib import Path

import pytest
from vision_msgs.msg import Detection3D
from vision_msgs.msg import ObjectHypothesisWithPose

from wildlife_spatial.spatial_evaluator import evaluate_detections
from wildlife_spatial.spatial_evaluator import ground_truth_objects_from_world
from wildlife_spatial.spatial_evaluator import load_ground_truth
from wildlife_spatial.spatial_evaluator import load_world_ground_truth


def make_detection(identifier, class_name, x, y, z=0.0):
    detection = Detection3D()
    detection.id = identifier
    detection.bbox.center.position.x = x
    detection.bbox.center.position.y = y
    detection.bbox.center.position.z = z
    result = ObjectHypothesisWithPose()
    result.hypothesis.class_id = class_name
    result.hypothesis.score = 1.0
    detection.results.append(result)
    return detection


def test_evaluation_matches_nearest_candidates_and_calculates_metrics():
    estimates = [
        make_detection('candidate_1', 'kangaroo', 4.024, 6.909, 0.559),
        make_detection('candidate_2', 'deer', -3.1, 5.0, 0.4),
    ]
    truth = [
        make_detection('kangaroo1', 'kangaroo', 4.0, 7.0),
        make_detection('male_deer1', 'deer', -3.0, 5.0),
        make_detection('boar1', 'boar', 7.0, -5.0),
    ]

    metrics = evaluate_detections(estimates, truth, maximum_distance=2.0)

    assert metrics['matched_count'] == 2
    assert metrics['false_positive_count'] == 0
    assert metrics['missed_count'] == 1
    assert metrics['precision'] == pytest.approx(1.0)
    assert metrics['recall'] == pytest.approx(2 / 3)
    assert metrics['classification_accuracy'] == pytest.approx(1.0)
    assert metrics['mean_planar_error_m'] == pytest.approx(
        (0.0941116 + 0.1) / 2,
        abs=1e-6,
    )


def test_evaluation_reports_false_positive_and_wrong_class():
    estimates = [
        make_detection('candidate_1', 'boar', 4.1, 7.0),
        make_detection('candidate_2', 'deer', 20.0, 20.0),
    ]
    truth = [make_detection('kangaroo1', 'kangaroo', 4.0, 7.0)]

    metrics = evaluate_detections(estimates, truth, maximum_distance=1.0)

    assert metrics['matched_count'] == 1
    assert metrics['false_positive_count'] == 1
    assert metrics['missed_count'] == 0
    assert metrics['classification_accuracy'] == pytest.approx(0.0)


def test_load_ground_truth_json(tmp_path):
    path = tmp_path / 'truth.json'
    path.write_text(json.dumps({
        'frame_id': 'husky1_map',
        'objects': [{
            'id': 'kangaroo1',
            'class_name': 'kangaroo',
            'position': {'x': 4.0, 'y': 7.0, 'z': 0.0},
        }],
    }))

    message = load_ground_truth(path)

    assert message.header.frame_id == 'husky1_map'
    assert len(message.detections) == 1
    assert message.detections[0].id == 'kangaroo1'
    assert message.detections[0].bbox.center.position.x == pytest.approx(4.0)
    assert (
        message.detections[0].results[0].hypothesis.class_id
        == 'kangaroo'
    )


WORLD_SDF = """<?xml version="1.0"?>
<sdf version="1.8">
  <world name="test_world">
    <include>
      <uri>model://boar</uri>
      <name>boar_01</name>
      <pose>1 2 0 0 0 -2</pose>
    </include>
    <include>
      <uri>model://male_deer</uri>
      <name>male_deer_01</name>
      <pose>-3.5 4 0.1 0 0 0</pose>
    </include>
    <include>
      <uri>model://campfire</uri>
      <name>campfire1</name>
      <pose>5 -6 0 0 0 0</pose>
    </include>
    <include>
      <uri>model://oak_tree</uri>
      <name>oak1</name>
      <pose>0 0 0 0 0 0</pose>
    </include>
  </world>
</sdf>
"""

PACKAGE_DIR = Path(__file__).resolve().parents[1]
LARGE_DEMO_WORLD = (
    PACKAGE_DIR.parent / '41068_ignition_bringup' / 'worlds' / 'large_demo.sdf'
)


def test_world_ground_truth_keeps_heat_sources_only(tmp_path):
    path = tmp_path / 'world.sdf'
    path.write_text(WORLD_SDF)

    message = load_world_ground_truth(path, 'husky1_map')

    assert message.header.frame_id == 'husky1_map'
    assert [d.id for d in message.detections] == [
        'boar_01', 'male_deer_01', 'campfire1']
    assert [d.results[0].hypothesis.class_id for d in message.detections] == [
        'boar', 'deer', 'campfire']
    assert message.detections[1].bbox.center.position.x == pytest.approx(-3.5)
    assert message.detections[1].bbox.center.position.z == pytest.approx(0.1)


def test_campfire_is_correct_without_animal_label():
    estimates = [
        make_detection('candidate_1', 'unknown_heat_source', 5.2, -6.0),
        make_detection('candidate_2', 'boar', 1.1, 2.0),
    ]
    truth = [
        make_detection('campfire1', 'campfire', 5.0, -6.0),
        make_detection('boar_01', 'boar', 1.0, 2.0),
    ]
    assert evaluate_detections(
        estimates, truth, 2.0)['classification_accuracy'] == pytest.approx(1.0)

    estimates[0] = make_detection('candidate_1', 'boar', 5.2, -6.0)
    metrics = evaluate_detections(estimates, truth, 2.0)
    assert metrics['classification_accuracy'] == pytest.approx(0.5)


def test_per_class_recall_and_localization_tolerance():
    estimates = [
        make_detection('candidate_1', 'boar', 1.2, 2.0),
        make_detection('candidate_2', 'deer', 11.5, 0.0),
    ]
    truth = [
        make_detection('boar_01', 'boar', 1.0, 2.0),
        make_detection('boar_02', 'boar', 10.0, 0.0),
        make_detection('boar_03', 'boar', -8.0, -8.0),
    ]

    metrics = evaluate_detections(
        estimates, truth, 2.0, localization_tolerance=1.0)

    assert metrics['within_tolerance_count'] == 1
    boar = metrics['per_class']['boar']
    assert boar['ground_truth'] == 3
    assert boar['detected'] == 2
    assert boar['correctly_classified'] == 1
    assert boar['detection_recall'] == pytest.approx(2 / 3)
    assert boar['classification_recall'] == pytest.approx(1 / 3)


def test_static_ground_truth_matches_large_demo_world():
    if not LARGE_DEMO_WORLD.exists():
        pytest.skip('large_demo.sdf is not next to this package')
    static = json.loads(
        (PACKAGE_DIR / 'config' / 'static_large_demo_ground_truth.json')
        .read_text())

    assert static['objects'] == ground_truth_objects_from_world(
        LARGE_DEMO_WORLD)

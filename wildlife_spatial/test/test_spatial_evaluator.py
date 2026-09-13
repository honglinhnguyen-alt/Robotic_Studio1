import json

import pytest
from vision_msgs.msg import Detection3D
from vision_msgs.msg import ObjectHypothesisWithPose

from wildlife_spatial.spatial_evaluator import evaluate_detections
from wildlife_spatial.spatial_evaluator import load_ground_truth


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

import pytest
from builtin_interfaces.msg import Time
from vision_msgs.msg import Detection3D
from vision_msgs.msg import ObjectHypothesisWithPose

from wildlife_spatial.candidate_registry import CandidateRegistryStore
from wildlife_spatial.candidate_registry import registry_payload


def make_detection(x, y, z, source_id, class_name, confidence):
    detection = Detection3D()
    detection.id = source_id
    detection.bbox.center.position.x = x
    detection.bbox.center.position.y = y
    detection.bbox.center.position.z = z
    result = ObjectHypothesisWithPose()
    result.hypothesis.class_id = class_name
    result.hypothesis.score = confidence
    detection.results.append(result)
    return detection


def test_nearby_observations_merge_and_average_position():
    store = CandidateRegistryStore(merge_distance=1.0)
    first = make_detection(4.0, 6.9, 0.5, 'thermal_1', 'kangaroo', 0.90)
    second = make_detection(4.2, 7.1, 0.7, 'thermal_1', 'kangaroo', 0.92)

    record, is_new, distance = store.update(first, Time(sec=10))
    assert is_new
    assert distance is None
    assert record.candidate_id == 'candidate_1'

    record, is_new, distance = store.update(second, Time(sec=12))
    assert not is_new
    assert distance == pytest.approx(0.2828427)
    assert len(store.records) == 1
    assert record.observation_count == 2
    assert [record.x, record.y, record.z] == pytest.approx([4.1, 7.0, 0.6])
    assert record.confidence == pytest.approx(0.92)
    assert record.last_seen_ns == 12_000_000_000


def test_distant_observation_creates_another_candidate():
    store = CandidateRegistryStore(merge_distance=1.0)
    first = make_detection(0.0, 0.0, 0.0, 'thermal_1', 'kangaroo', 0.9)
    distant = make_detection(2.0, 0.0, 0.0, 'thermal_2', 'boar', 0.8)

    store.update(first, Time(sec=1))
    record, is_new, _ = store.update(distant, Time(sec=2))

    assert is_new
    assert record.candidate_id == 'candidate_2'
    assert len(store.records) == 2


def test_lower_confidence_observation_does_not_replace_classification():
    store = CandidateRegistryStore(merge_distance=1.0)
    strong = make_detection(1.0, 1.0, 0.0, 'thermal_1', 'kangaroo', 0.92)
    weak = make_detection(1.1, 1.0, 0.0, 'thermal_7', 'boar', 0.60)

    store.update(strong, Time(sec=1))
    record, _, _ = store.update(weak, Time(sec=2))

    assert record.class_name == 'kangaroo'
    assert record.confidence == pytest.approx(0.92)
    assert record.source_ids == {'thermal_1', 'thermal_7'}


def test_yolo_label_replaces_unknown_heat_source():
    store = CandidateRegistryStore(merge_distance=1.0)
    thermal = make_detection(
        1.0, 1.0, 0.0, 'thermal_1', 'unknown_heat_source', 0.0
    )
    classified = make_detection(
        1.1, 1.0, 0.0, 'thermal_1', 'kangaroo', 0.72
    )

    store.update(thermal, Time(sec=1))
    record, _, _ = store.update(classified, Time(sec=2))

    assert record.class_name == 'kangaroo'
    assert record.confidence == pytest.approx(0.72)


def test_unknown_heat_source_does_not_erase_yolo_label():
    store = CandidateRegistryStore(merge_distance=1.0)
    classified = make_detection(
        1.0, 1.0, 0.0, 'thermal_1', 'kangaroo', 0.72
    )
    thermal = make_detection(
        1.1, 1.0, 0.0, 'thermal_1', 'unknown_heat_source', 0.0
    )

    store.update(classified, Time(sec=1))
    record, _, _ = store.update(thermal, Time(sec=2))

    assert record.class_name == 'kangaroo'
    assert record.confidence == pytest.approx(0.72)


def test_registry_payload_contains_mission_fields():
    store = CandidateRegistryStore(merge_distance=1.0)
    detection = make_detection(
        4.0,
        6.9,
        0.5,
        'thermal_1',
        'kangaroo',
        0.92,
    )
    store.update(detection, Time(sec=10, nanosec=500_000_000))

    payload = registry_payload(store.records, 'husky1_map')

    assert payload['frame_id'] == 'husky1_map'
    assert payload['candidate_count'] == 1
    candidate = payload['candidates'][0]
    assert candidate['candidate_id'] == 'candidate_1'
    assert candidate['state'] == 'uninvestigated'
    assert candidate['class_name'] == 'kangaroo'
    assert candidate['position']['x'] == pytest.approx(4.0)
    assert candidate['first_seen'] == pytest.approx(10.5)

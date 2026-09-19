import numpy as np
import pytest
from builtin_interfaces.msg import Time
from std_msgs.msg import Header

from wildlife_spatial.candidate_localizer import bounding_box_indices
from wildlife_spatial.candidate_localizer import make_camera_point
from wildlife_spatial.candidate_localizer import make_candidate_marker
from wildlife_spatial.candidate_localizer import make_localized_detection
from wildlife_spatial.candidate_localizer import make_projected_detection
from wildlife_spatial.candidate_localizer import project_thermal_box
from wildlife_spatial.candidate_localizer import robust_position
from vision_msgs.msg import Detection2D
from vision_msgs.msg import ObjectHypothesisWithPose


def test_bounding_box_indices_select_inner_pixels():
    indices = bounding_box_indices(
        center_x=3.5,
        center_y=2.5,
        size_x=4.0,
        size_y=4.0,
        cloud_width=8,
        cloud_height=6,
        inner_fraction=0.5,
        pixel_stride=1,
    )

    assert indices.tolist() == [19, 20, 27, 28]


def test_bounding_box_indices_clamps_to_cloud_edges():
    indices = bounding_box_indices(
        center_x=0.0,
        center_y=0.0,
        size_x=4.0,
        size_y=4.0,
        cloud_width=4,
        cloud_height=3,
        inner_fraction=1.0,
        pixel_stride=1,
    )

    assert indices.tolist() == [0, 1, 2, 4, 5, 6, 8, 9, 10]


def test_robust_position_filters_invalid_ranges_and_uses_median():
    points = np.array([
        [1.8, 0.0, 0.0],
        [2.0, 0.1, 0.0],
        [2.2, 0.0, 0.0],
        [np.nan, 0.0, 0.0],
        [100.0, 0.0, 0.0],
    ])

    position, spread, valid_count = robust_position(
        points,
        min_range=0.2,
        max_range=20.0,
        min_valid_points=3,
    )

    assert position == pytest.approx([2.0, 0.0, 0.0])
    assert spread == pytest.approx(0.2)
    assert valid_count == 3


def test_robust_position_rejects_too_few_valid_points():
    points = np.array([
        [2.0, 0.0, 0.0],
        [np.nan, 0.0, 0.0],
    ])

    with pytest.raises(ValueError, match='only 1 valid XYZ points'):
        robust_position(
            points,
            min_range=0.2,
            max_range=20.0,
            min_valid_points=2,
        )


def test_camera_point_and_marker_preserve_localization_result():
    header = Header(stamp=Time(sec=12, nanosec=34), frame_id='husky1_map')
    point = make_camera_point(header, [2.5, -0.4, 0.3])
    marker = make_candidate_marker(point, 'thermal_1', spread=0.25)

    assert point.header.frame_id == 'husky1_map'
    assert [point.point.x, point.point.y, point.point.z] == [2.5, -0.4, 0.3]
    assert marker.header.frame_id == 'husky1_map'
    assert marker.pose.position == point.point
    assert marker.pose.orientation.w == 1.0
    assert marker.scale.x == pytest.approx(0.5)
    assert marker.scale.y == pytest.approx(0.5)
    assert marker.scale.z == pytest.approx(0.5)
    assert marker.color.a == pytest.approx(0.85)


def test_candidate_marker_id_is_stable_per_identifier():
    point = make_camera_point(Header(frame_id='husky1_map'), [1.0, 2.0, 3.0])

    first = make_candidate_marker(point, 'thermal_1', spread=0.01)
    repeated = make_candidate_marker(point, 'thermal_1', spread=0.01)
    another = make_candidate_marker(point, 'thermal_2', spread=0.01)

    assert first.id == repeated.id
    assert first.id != another.id
    assert first.scale.x == pytest.approx(0.2)


def test_localized_detection_preserves_id_and_classification():
    source = Detection2D()
    source.id = 'thermal_1'
    result = ObjectHypothesisWithPose()
    result.hypothesis.class_id = 'kangaroo'
    result.hypothesis.score = 0.92
    source.results.append(result)
    point = make_camera_point(Header(frame_id='husky1_map'), [4.0, 6.9, 0.5])

    localized = make_localized_detection(source, point, spread=0.25)

    assert localized.id == 'thermal_1'
    assert localized.header.frame_id == 'husky1_map'
    assert localized.bbox.center.position == point.point
    assert localized.bbox.size.x == pytest.approx(0.5)
    assert localized.results[0].hypothesis.class_id == 'kangaroo'
    assert localized.results[0].hypothesis.score == pytest.approx(0.92)


def test_thermal_box_projects_to_depth_image_center():
    projected = project_thermal_box(
        box=(139.5, 99.5, 179.5, 139.5),
        thermal_width=320,
        thermal_height=240,
        depth_width=720,
        depth_height=480,
        thermal_hfov_deg=60.0,
        depth_hfov_deg=120.0,
        margin=1.5,
    )

    center_x = (projected[0] + projected[2]) / 2.0
    center_y = (projected[1] + projected[3]) / 2.0
    assert center_x == pytest.approx(359.5)
    assert center_y == pytest.approx(239.5)
    assert projected[2] - projected[0] > 0.0
    assert projected[3] - projected[1] > 0.0


def test_projected_thermal_detection_defaults_to_unknown_heat_source():
    source = Detection2D()
    source.header.frame_id = 'husky1_thermal_camera_link'

    projected = make_projected_detection(
        source,
        'thermal_1',
        (300.0, 180.0, 420.0, 300.0),
    )

    assert projected.id == 'thermal_1'
    assert projected.bbox.center.position.x == pytest.approx(360.0)
    assert projected.bbox.center.position.y == pytest.approx(240.0)
    assert projected.bbox.size_x == pytest.approx(120.0)
    assert projected.bbox.size_y == pytest.approx(120.0)
    assert projected.results[0].hypothesis.class_id == 'unknown_heat_source'
    assert projected.results[0].hypothesis.score == pytest.approx(0.0)

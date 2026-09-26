import numpy as np
import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from wildlife_spatial.candidate_localizer import bounding_box_indices
from wildlife_spatial.candidate_localizer import depth_window_for_box
from wildlife_spatial.candidate_localizer import detection_hot_mask
from wildlife_spatial.candidate_localizer import dominant_range_cluster
from wildlife_spatial.candidate_localizer import hot_pixel_position
from wildlife_spatial.candidate_localizer import make_camera_point
from wildlife_spatial.candidate_localizer import make_candidate_marker
from wildlife_spatial.candidate_localizer import make_localized_detection
from wildlife_spatial.candidate_localizer import make_projected_detection
from wildlife_spatial.candidate_localizer import project_thermal_box
from wildlife_spatial.candidate_localizer import robust_position
from wildlife_spatial.candidate_localizer import thermal_image_to_kelvin
from wildlife_spatial.candidate_localizer import thermal_pixel_indices
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


IDENTITY = np.eye(3)
NO_OFFSET = np.zeros(3)


def rock_in_front_of_deer(deer_has_depth=True):
    """
    Deer 8 m away, its middle hidden by a cold rock 3 m away.

    Depth points are sampled per depth pixel (about 1/207 rad apart) and the
    hot mask per thermal pixel (1/277 rad), like the Husky cameras. The deer
    stays one hot region (legs beside the rock, back above it), so its box
    covers the rock.
    """
    def classify(slope_y, slope_z):
        rock = (np.abs(slope_y) < 0.2 / 3.0) & (slope_z < 0.3 / 3.0)
        deer = (
            ~rock
            & (np.abs(slope_y) < 0.7 / 8.0)
            & (slope_z > -0.38 / 8.0)
            & (slope_z < 0.9 / 8.0)
        )
        return rock, deer

    slopes = np.arange(-0.2, 0.2, 1.0 / 207.0)
    slope_y, slope_z = (grid.ravel() for grid in np.meshgrid(slopes, slopes))
    rock, deer = classify(slope_y, slope_z)
    if not deer_has_depth:
        deer[:] = False
    distance = np.where(rock, 3.0, 8.0)[rock | deer]
    points = np.column_stack((
        distance,
        slope_y[rock | deer] * distance,
        slope_z[rock | deer] * distance,
    ))

    tan_half = np.tan(np.radians(30.0))
    columns = np.arange(320)
    rows = np.arange(240)
    pixel_y = -((columns + 0.5) / 160.0 - 1.0) * tan_half
    pixel_z = -((rows + 0.5) / 120.0 - 1.0) * tan_half * 0.75
    grid_y, grid_z = np.meshgrid(pixel_y, pixel_z)
    _, hot_mask = classify(grid_y, grid_z)
    hot_rows, hot_columns = np.nonzero(hot_mask)
    box = (
        float(hot_columns.min()),
        float(hot_rows.min()),
        float(hot_columns.max() + 1),
        float(hot_rows.max() + 1),
    )
    return points, hot_mask, box


def test_depth_on_hot_pixels_ignores_rock_in_front_of_deer():
    points, hot_mask, box = rock_in_front_of_deer()
    column, row, valid = thermal_pixel_indices(
        points, IDENTITY, NO_OFFSET, 320, 240, 60.0
    )
    # Previous approach: median of the central part of the box.
    width = box[2] - box[0]
    height = box[3] - box[1]
    central = (
        valid
        & (np.abs(column + 0.5 - (box[0] + box[2]) / 2.0) < 0.375 * width)
        & (np.abs(row + 0.5 - (box[1] + box[3]) / 2.0) < 0.375 * height)
    )
    box_median, _, _ = robust_position(points[central], 0.2, 20.0, 20)
    assert box_median[0] == pytest.approx(3.0, abs=0.1)

    mask = detection_hot_mask(hot_mask, box)
    position, spread, used, hot_count = hot_pixel_position(
        points, column, row, valid, mask, 0.2, 20.0, 20, 1.0
    )
    assert position[0] == pytest.approx(8.0, abs=0.05)
    assert abs(position[1]) < 0.7


def test_hot_target_without_depth_is_rejected_instead_of_using_rock():
    # Beyond the depth camera range the deer has no depth points; only the
    # rock in front of it does.
    points, hot_mask, box = rock_in_front_of_deer(deer_has_depth=False)
    column, row, valid = thermal_pixel_indices(
        points, IDENTITY, NO_OFFSET, 320, 240, 60.0
    )
    mask = detection_hot_mask(hot_mask, box)

    with pytest.raises(ValueError, match='only 0 depth points on hot'):
        hot_pixel_position(
            points, column, row, valid, mask, 0.2, 20.0, 20, 1.0
        )


def test_dominant_range_cluster_keeps_largest_group():
    near = np.tile([5.0, 0.0, 0.0], (10, 1))
    far = np.tile([8.0, 0.0, 0.0], (30, 1))
    cluster = dominant_range_cluster(np.vstack((near, far)), 1.0)

    assert len(cluster) == 30
    assert np.allclose(cluster[:, 0], 8.0)


def test_thermal_pixel_indices_follow_camera_axes():
    points = np.array([
        [10.0, 0.0, 0.0],
        [10.0, 1.0, 0.0],
        [10.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
    ])
    column, row, valid = thermal_pixel_indices(
        points, IDENTITY, NO_OFFSET, 320, 240, 60.0
    )

    assert valid.tolist() == [True, True, True, False]
    assert (column[0], row[0]) == (160, 120)
    assert column[1] < 160
    assert row[2] < 120


def test_detection_hot_mask_erodes_edges_but_keeps_small_regions():
    hot = np.zeros((240, 320), dtype=bool)
    hot[100:110, 100:110] = True
    hot[50:53, 200:203] = True

    large = detection_hot_mask(hot, (100.0, 100.0, 110.0, 110.0))
    small = detection_hot_mask(hot, (200.0, 50.0, 203.0, 53.0))

    assert int(large.sum()) == 64
    assert int(small.sum()) == 9


def test_thermal_image_to_kelvin_reads_padded_mono16_rows():
    values = np.array([[29315, 33015, 0], [1, 2, 3]], dtype='<u2')
    rows = np.zeros((2, 8), dtype=np.uint8)
    rows[:, :6] = values.view(np.uint8).reshape(2, 6)
    image = Image()
    image.height = 2
    image.width = 3
    image.encoding = 'mono16'
    image.is_bigendian = 0
    image.step = 8
    image.data = rows.tobytes()

    kelvin = thermal_image_to_kelvin(image, 0.01)

    assert kelvin.shape == (2, 3)
    assert kelvin[0, 0] == pytest.approx(293.15)
    assert kelvin[0, 1] == pytest.approx(330.15)


def test_depth_window_covers_the_thermal_box():
    # Thermal camera 6 cm to the right of the depth camera.
    top, bottom, left, right = depth_window_for_box(
        (150.0, 110.0, 170.0, 130.0),
        IDENTITY,
        np.array([0.0, 0.06, 0.0]),
        320, 240, 60.0,
        720, 480, 120.0,
    )

    assert left < 360 < right
    assert top < 240 < bottom
    assert right - left < 60
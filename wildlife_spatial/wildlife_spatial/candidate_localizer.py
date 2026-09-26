#!/usr/bin/env python3

"""Localize thermal heat detections with depth point clouds for R2."""

from collections import deque
from math import atan
from math import ceil
from math import floor
from math import radians
from math import tan
import zlib

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer
from tf2_ros import TransformException
from tf2_ros import TransformListener
from visualization_msgs.msg import Marker
from vision_msgs.msg import Detection2D
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection3D
from vision_msgs.msg import Detection3DArray
from vision_msgs.msg import ObjectHypothesisWithPose


NANOSECONDS_PER_SECOND = 1_000_000_000


def detection_box(detection):
    """Return a Detection2D box as x1, y1, x2, y2 coordinates."""
    center = detection.bbox.center
    if hasattr(center, 'position'):
        center_x = center.position.x
        center_y = center.position.y
    else:
        center_x = center.x
        center_y = center.y
    half_width = float(detection.bbox.size_x) / 2.0
    half_height = float(detection.bbox.size_y) / 2.0
    return (
        float(center_x) - half_width,
        float(center_y) - half_height,
        float(center_x) + half_width,
        float(center_y) + half_height,
    )


def _vertical_fov(horizontal_fov, width, height):
    """Derive vertical field of view for square image pixels."""
    return 2.0 * atan(tan(horizontal_fov / 2.0) * height / width)


def _project_axis(
    coordinate,
    source_size,
    source_fov,
    destination_size,
    destination_fov,
):
    """Project one pixel coordinate between aligned pinhole cameras."""
    normalized = 2.0 * float(coordinate) / (source_size - 1) - 1.0
    ray_tangent = normalized * tan(source_fov / 2.0)
    destination_normalized = ray_tangent / tan(destination_fov / 2.0)
    return (
        (destination_normalized + 1.0)
        * (destination_size - 1)
        / 2.0
    )


def project_thermal_box(
    box,
    thermal_width,
    thermal_height,
    depth_width,
    depth_height,
    thermal_hfov_deg,
    depth_hfov_deg,
    margin=1.5,
    offset_x=0.0,
    offset_y=0.0,
):
    """Project a thermal pixel box into the aligned RGB-D image."""
    if min(thermal_width, thermal_height, depth_width, depth_height) <= 1:
        raise ValueError('camera image dimensions must exceed one pixel')
    if thermal_hfov_deg <= 0.0 or depth_hfov_deg <= 0.0:
        raise ValueError('camera fields of view must be positive')
    if margin <= 0.0:
        raise ValueError('thermal projection margin must be positive')

    thermal_hfov = radians(float(thermal_hfov_deg))
    depth_hfov = radians(float(depth_hfov_deg))
    thermal_vfov = _vertical_fov(
        thermal_hfov,
        thermal_width,
        thermal_height,
    )
    depth_vfov = _vertical_fov(
        depth_hfov,
        depth_width,
        depth_height,
    )
    x1, y1, x2, y2 = box
    projected = (
        _project_axis(
            x1, thermal_width, thermal_hfov, depth_width, depth_hfov
        ),
        _project_axis(
            y1, thermal_height, thermal_vfov, depth_height, depth_vfov
        ),
        _project_axis(
            x2, thermal_width, thermal_hfov, depth_width, depth_hfov
        ),
        _project_axis(
            y2, thermal_height, thermal_vfov, depth_height, depth_vfov
        ),
    )
    projected_x1, projected_y1, projected_x2, projected_y2 = projected
    center_x = (projected_x1 + projected_x2) / 2.0 + float(offset_x)
    center_y = (projected_y1 + projected_y2) / 2.0 + float(offset_y)
    half_width = abs(projected_x2 - projected_x1) * margin / 2.0
    half_height = abs(projected_y2 - projected_y1) * margin / 2.0
    return (
        max(0.0, center_x - half_width),
        max(0.0, center_y - half_height),
        min(float(depth_width - 1), center_x + half_width),
        min(float(depth_height - 1), center_y + half_height),
    )


def make_projected_detection(
    source,
    identifier,
    projected_box,
    classification_results=None,
):
    """Create a depth-image detection from one thermal candidate."""
    x1, y1, x2, y2 = projected_box
    detection = Detection2D()
    detection.header = source.header
    detection.id = identifier
    center = detection.bbox.center
    if hasattr(center, 'position'):
        center.position.x = (x1 + x2) / 2.0
        center.position.y = (y1 + y2) / 2.0
        center.theta = 0.0
    else:
        center.x = (x1 + x2) / 2.0
        center.y = (y1 + y2) / 2.0
        center.theta = 0.0
    detection.bbox.size_x = max(0.0, x2 - x1)
    detection.bbox.size_y = max(0.0, y2 - y1)

    if classification_results:
        detection.results = classification_results
    else:
        result = ObjectHypothesisWithPose()
        result.hypothesis.class_id = 'unknown_heat_source'
        result.hypothesis.score = 0.0
        detection.results.append(result)
    return detection


def stamp_nanoseconds(stamp):
    """Convert a ROS time message to an integer nanosecond timestamp."""
    return stamp.sec * NANOSECONDS_PER_SECOND + stamp.nanosec


def closest_stamped(messages, target_stamp):
    """Return the message nearest to a target stamp and the time gap."""
    if not messages:
        return None, None

    target_ns = stamp_nanoseconds(target_stamp)
    message = min(
        messages,
        key=lambda item: abs(
            stamp_nanoseconds(item.header.stamp) - target_ns
        ),
    )
    difference_ns = abs(stamp_nanoseconds(message.header.stamp) - target_ns)
    return message, difference_ns / NANOSECONDS_PER_SECOND


def closest_cloud(clouds, target_stamp):
    """Return the cloud nearest to a target stamp and the absolute time gap."""
    return closest_stamped(clouds, target_stamp)


def bounding_box_indices(
    center_x,
    center_y,
    size_x,
    size_y,
    cloud_width,
    cloud_height,
    inner_fraction=0.5,
    pixel_stride=2,
):
    """Return flat cloud indices sampled from the inner detection box."""
    if cloud_width <= 0 or cloud_height <= 1:
        raise ValueError('point cloud must be organized')
    if not 0.0 < inner_fraction <= 1.0:
        raise ValueError('inner_fraction must be in the range (0, 1]')
    if pixel_stride < 1:
        raise ValueError('pixel_stride must be at least one')

    inner_width = max(1.0, float(size_x) * inner_fraction)
    inner_height = max(1.0, float(size_y) * inner_fraction)
    left = max(0, int(ceil(float(center_x) - inner_width / 2.0)))
    right = min(
        cloud_width - 1,
        int(floor(float(center_x) + inner_width / 2.0)),
    )
    top = max(0, int(ceil(float(center_y) - inner_height / 2.0)))
    bottom = min(
        cloud_height - 1,
        int(floor(float(center_y) + inner_height / 2.0)),
    )

    if left > right or top > bottom:
        return np.empty(0, dtype=np.int64)

    return np.fromiter(
        (
            row * cloud_width + column
            for row in range(top, bottom + 1, pixel_stride)
            for column in range(left, right + 1, pixel_stride)
        ),
        dtype=np.int64,
    )


def robust_position(points, min_range, max_range, min_valid_points):
    """Filter invalid XYZ samples and return median position and spread."""
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(xyz).all(axis=1)
    ranges = np.linalg.norm(xyz, axis=1)
    valid = xyz[
        finite
        & (ranges >= float(min_range))
        & (ranges <= float(max_range))
    ]

    if len(valid) < min_valid_points:
        raise ValueError(
            f'only {len(valid)} valid XYZ points; '
            f'{min_valid_points} required'
        )

    position = np.median(valid, axis=0)
    distances_from_position = np.linalg.norm(valid - position, axis=1)
    spread = float(np.median(distances_from_position))
    return position, spread, len(valid)


def extract_position_from_cloud(
    cloud,
    detection,
    inner_fraction,
    pixel_stride,
    min_range,
    max_range,
    min_valid_points,
):
    """Estimate a camera-frame position from an organized cloud and box."""
    field_names = {field.name for field in cloud.fields}
    if not {'x', 'y', 'z'}.issubset(field_names):
        raise ValueError('point cloud does not contain x, y and z fields')

    center = detection.bbox.center
    if hasattr(center, 'position'):
        center_x = center.position.x
        center_y = center.position.y
    else:
        center_x = center.x
        center_y = center.y

    indices = bounding_box_indices(
        center_x,
        center_y,
        detection.bbox.size_x,
        detection.bbox.size_y,
        cloud.width,
        cloud.height,
        inner_fraction,
        pixel_stride,
    )
    if len(indices) == 0:
        raise ValueError('detection box does not overlap the point cloud')

    samples = point_cloud2.read_points(
        cloud,
        field_names=['x', 'y', 'z'],
        skip_nans=False,
        uvs=indices,
    )
    xyz = np.column_stack((samples['x'], samples['y'], samples['z']))
    position, spread, valid_count = robust_position(
        xyz,
        min_range,
        max_range,
        min_valid_points,
    )
    return position, spread, valid_count, len(indices)


def thermal_image_to_kelvin(message, kelvin_per_count=0.01):
    """Convert a mono16/16UC1 (counts) or 32FC1 thermal image to kelvin."""
    encoding = message.encoding.lower()
    if encoding in ('mono16', '16uc1'):
        dtype = np.dtype(np.uint16)
        scale = float(kelvin_per_count)
    elif encoding == '32fc1':
        dtype = np.dtype(np.float32)
        scale = 1.0
    else:
        raise ValueError(
            f'unsupported thermal image encoding {message.encoding!r}'
        )
    dtype = dtype.newbyteorder('>' if message.is_bigendian else '<')

    # bytes() accepts the array.array, bytes or list forms of uint8[] data.
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    row_bytes = message.width * dtype.itemsize
    if message.step < row_bytes or raw.size < message.height * message.step:
        raise ValueError('thermal image data does not match its size')
    rows = raw[:message.height * message.step].reshape(
        message.height,
        message.step,
    )
    values = np.ascontiguousarray(rows[:, :row_bytes]).view(dtype)
    return values.reshape(message.height, message.width).astype(
        np.float32
    ) * scale


def erode_mask(mask):
    """Remove the outer pixel ring of every region in a boolean mask."""
    eroded = mask.copy()
    eroded[1:, :] &= mask[:-1, :]
    eroded[:-1, :] &= mask[1:, :]
    eroded[:, 1:] &= mask[:, :-1]
    eroded[:, :-1] &= mask[:, 1:]
    return eroded


def detection_hot_mask(
    hot_mask,
    box,
    margin_px=1,
    erode=True,
    minimum_eroded_pixels=10,
):
    """
    Return the hot thermal pixels that belong to one detection box.

    Cold pixels inside the box (a tree trunk in front of the animal, the
    ground between its legs, the background) are excluded. Erosion removes
    the boundary pixels, where the thermal and depth images can disagree by
    one pixel, unless too few pixels would remain.
    """
    height, width = hot_mask.shape
    x1, y1, x2, y2 = box
    left = max(0, int(floor(x1)) - margin_px)
    top = max(0, int(floor(y1)) - margin_px)
    right = min(width, int(ceil(x2)) + margin_px)
    bottom = min(height, int(ceil(y2)) + margin_px)
    selected = np.zeros_like(hot_mask, dtype=bool)
    if left >= right or top >= bottom:
        return selected

    selected[top:bottom, left:right] = hot_mask[top:bottom, left:right]
    if erode:
        eroded = erode_mask(hot_mask)
        eroded_box = np.zeros_like(selected)
        eroded_box[top:bottom, left:right] = eroded[top:bottom, left:right]
        if int(eroded_box.sum()) >= minimum_eroded_pixels:
            return eroded_box
    return selected


def transform_to_matrix(transform):
    """Convert a geometry_msgs Transform to a rotation matrix and offset."""
    q = transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    rotation = np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ])
    t = transform.translation
    return rotation, np.array([t.x, t.y, t.z])


def depth_window_for_box(
    box,
    rotation,
    translation,
    thermal_width,
    thermal_height,
    thermal_hfov_deg,
    depth_width,
    depth_height,
    depth_hfov_deg,
    near=1.0,
    far=20.0,
    margin_px=4,
):
    """
    Return the depth-image rows and columns that can see a thermal box.

    rotation and translation move cloud points into the thermal frame. The
    corner rays of the box are placed at the near and far distances and
    projected into the depth image, so camera offset and robot motion are
    covered. Returns top, bottom, left, right (exclusive end indices).
    """
    x1, y1, x2, y2 = box
    tan_thermal_h = tan(radians(float(thermal_hfov_deg)) / 2.0)
    tan_thermal_v = tan_thermal_h * thermal_height / thermal_width
    tan_depth_h = tan(radians(float(depth_hfov_deg)) / 2.0)
    tan_depth_v = tan_depth_h * depth_height / depth_width
    rotation = np.asarray(rotation)
    translation = np.asarray(translation)

    corners = []
    for u in (x1, x2):
        for v in (y1, y2):
            ray = np.array([
                1.0,
                -(2.0 * u / thermal_width - 1.0) * tan_thermal_h,
                -(2.0 * v / thermal_height - 1.0) * tan_thermal_v,
            ])
            for distance in (near, far):
                corners.append(rotation.T @ (ray * distance - translation))
    corners = np.array(corners)
    if np.any(corners[:, 0] <= 1e-6):
        return 0, depth_height, 0, depth_width

    columns = depth_width / 2.0 * (
        1.0 - corners[:, 1] / corners[:, 0] / tan_depth_h
    )
    rows = depth_height / 2.0 * (
        1.0 - corners[:, 2] / corners[:, 0] / tan_depth_v
    )
    left = max(0, int(floor(columns.min())) - margin_px)
    right = min(depth_width, int(ceil(columns.max())) + margin_px + 1)
    top = max(0, int(floor(rows.min())) - margin_px)
    bottom = min(depth_height, int(ceil(rows.max())) + margin_px + 1)
    return top, bottom, left, right


def cloud_points(cloud, pixel_stride, min_range, max_range, window=None):
    """
    Return finite in-range XYZ points of a cloud as an N x 3 array.

    window (top, bottom, left, right) limits an organized cloud to a part
    of the depth image.
    """
    field_names = {field.name for field in cloud.fields}
    if not {'x', 'y', 'z'}.issubset(field_names):
        raise ValueError('point cloud does not contain x, y and z fields')

    samples = point_cloud2.read_points(
        cloud,
        field_names=['x', 'y', 'z'],
        skip_nans=False,
    )
    if cloud.height > 1:
        grid = samples.reshape(cloud.height, cloud.width)
        if window is not None:
            top, bottom, left, right = window
            grid = grid[top:bottom, left:right]
        samples = grid[::pixel_stride, ::pixel_stride].reshape(-1)
    xyz = np.column_stack(
        (samples['x'], samples['y'], samples['z'])
    ).astype(np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    ranges = np.linalg.norm(xyz, axis=1)
    return xyz[(ranges >= float(min_range)) & (ranges <= float(max_range))]


def thermal_pixel_indices(
    points,
    rotation,
    translation,
    width,
    height,
    hfov_deg,
):
    """
    Project points into thermal pixels (x forward, y left, z up).

    rotation and translation move the points into the thermal camera frame.
    Returns the pixel column and row of every point and a mask of the points
    that land inside the image.
    """
    camera = points @ np.asarray(rotation).T + np.asarray(translation)
    forward = camera[:, 0]
    valid = forward > 1e-6
    tan_horizontal = tan(radians(float(hfov_deg)) / 2.0)
    tan_vertical = tan_horizontal * height / width
    safe_forward = np.where(valid, forward, 1.0)
    # Gazebo pinhole model: principal point at the image centre.
    column = np.floor(
        width / 2.0 * (1.0 - camera[:, 1] / safe_forward / tan_horizontal)
    )
    row = np.floor(
        height / 2.0 * (1.0 - camera[:, 2] / safe_forward / tan_vertical)
    )
    valid &= (column >= 0) & (column < width) & (row >= 0) & (row < height)
    column = np.where(valid, column, 0).astype(np.int64)
    row = np.where(valid, row, 0).astype(np.int64)
    return column, row, valid


def dominant_range_cluster(points, window):
    """Keep the densest group of points whose ranges lie within window."""
    if len(points) == 0:
        return points
    ranges = np.linalg.norm(points, axis=1)
    order = np.argsort(ranges, kind='stable')
    sorted_ranges = ranges[order]
    ends = np.searchsorted(
        sorted_ranges,
        sorted_ranges + float(window),
        side='right',
    )
    counts = ends - np.arange(len(sorted_ranges))
    start = int(np.argmax(counts))
    return points[order[start:ends[start]]]


def hot_pixel_position(
    points,
    column,
    row,
    valid,
    mask,
    min_range,
    max_range,
    min_valid_points,
    cluster_window,
):
    """
    Estimate a position from the depth points that land on hot pixels.

    Returns the median position, spread, points used and hot points found.
    """
    on_hot_pixel = np.zeros(len(points), dtype=bool)
    on_hot_pixel[valid] = mask[row[valid], column[valid]]
    hot_points = points[on_hot_pixel]
    if len(hot_points) < min_valid_points:
        raise ValueError(
            f'only {len(hot_points)} depth points on hot pixels; '
            f'{min_valid_points} required (target hidden or beyond the '
            'depth range)'
        )
    cluster = dominant_range_cluster(hot_points, cluster_window)
    position, spread, valid_count = robust_position(
        cluster,
        min_range,
        max_range,
        min_valid_points,
    )
    return position, spread, valid_count, len(hot_points)


def make_camera_point(header, position):
    """Create a stamped point from a camera-frame XYZ estimate."""
    point = PointStamped()
    point.header = header
    point.point.x = float(position[0])
    point.point.y = float(position[1])
    point.point.z = float(position[2])
    return point


def make_candidate_marker(point, identifier, spread):
    """Create a persistent RViz marker for a localized candidate."""
    marker = Marker()
    marker.header = point.header
    marker.ns = 'wildlife_candidates'
    marker.id = zlib.crc32(identifier.encode('utf-8')) & 0x7FFFFFFF
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position = point.point
    marker.pose.orientation.w = 1.0

    diameter = max(0.2, 2.0 * float(spread))
    marker.scale.x = diameter
    marker.scale.y = diameter
    marker.scale.z = diameter
    marker.color.r = 1.0
    marker.color.g = 0.45
    marker.color.b = 0.0
    marker.color.a = 0.85
    return marker


def make_localized_detection(source_detection, point, spread):
    """Represent a localized 2D observation as a standard 3D detection."""
    detection = Detection3D()
    detection.header = point.header
    detection.id = source_detection.id
    detection.results = source_detection.results
    detection.bbox.center.position = point.point
    detection.bbox.center.orientation.w = 1.0

    diameter = max(0.2, 2.0 * float(spread))
    detection.bbox.size.x = diameter
    detection.bbox.size.y = diameter
    detection.bbox.size.z = diameter
    return detection


class CandidateLocalizer(Node):
    """Localize each thermal trigger and attach optional YOLO metadata."""

    def __init__(self):
        super().__init__('candidate_localizer')

        self.declare_parameter(
            'detection_topic',
            '/husky1/perception/thermal/detections',
        )
        self.declare_parameter(
            'classification_topic',
            '/husky1/perception/thermal_yolo/detections',
        )
        self.declare_parameter(
            'point_cloud_topic',
            '/husky1/camera/depth/points',
        )
        self.declare_parameter(
            'thermal_image_topic',
            '/husky1/thermal/image_raw',
        )
        # thermal_mask: use only depth points that land on hot thermal
        # pixels. bbox: previous behaviour (median of the inner box).
        self.declare_parameter('depth_selection', 'thermal_mask')
        self.declare_parameter('hot_threshold_k', 315.0)
        self.declare_parameter('kelvin_per_count', 0.01)
        self.declare_parameter('erode_hot_mask', True)
        self.declare_parameter('depth_cluster_window', 1.0)
        self.declare_parameter('motion_compensation_frame', 'husky1_odom')
        self.declare_parameter('thermal_buffer_seconds', 2.0)
        self.declare_parameter('max_cloud_time_difference', 0.2)
        self.declare_parameter('bbox_inner_fraction', 0.5)
        self.declare_parameter('pixel_stride', 1)
        self.declare_parameter('minimum_range', 0.2)
        self.declare_parameter('maximum_range', 20.0)
        self.declare_parameter('minimum_valid_points', 20)
        self.declare_parameter('thermal_image_width', 320)
        self.declare_parameter('thermal_image_height', 240)
        self.declare_parameter('thermal_hfov_deg', 60.0)
        self.declare_parameter('depth_hfov_deg', 120.0)
        self.declare_parameter('thermal_projection_margin', 1.5)
        self.declare_parameter('thermal_depth_offset_x_px', 0.0)
        self.declare_parameter('thermal_depth_offset_y_px', 0.0)
        self.declare_parameter('classification_max_age', 10.0)
        self.declare_parameter('map_frame', 'husky1_map')
        self.declare_parameter(
            'localized_point_topic',
            '/husky1/candidates/localized_point',
        )
        self.declare_parameter(
            'marker_topic',
            '/husky1/candidates/marker',
        )
        self.declare_parameter(
            'localized_detections_topic',
            '/husky1/candidates/localized_detections',
        )
        self.declare_parameter('transform_timeout', 0.25)

        detection_topic = str(self.get_parameter('detection_topic').value)
        classification_topic = str(
            self.get_parameter('classification_topic').value
        )
        point_cloud_topic = str(self.get_parameter('point_cloud_topic').value)
        thermal_image_topic = str(
            self.get_parameter('thermal_image_topic').value
        )
        self._depth_selection = str(
            self.get_parameter('depth_selection').value
        )
        self._hot_threshold_k = float(
            self.get_parameter('hot_threshold_k').value
        )
        self._kelvin_per_count = float(
            self.get_parameter('kelvin_per_count').value
        )
        self._erode_hot_mask = bool(
            self.get_parameter('erode_hot_mask').value
        )
        self._depth_cluster_window = float(
            self.get_parameter('depth_cluster_window').value
        )
        self._motion_frame = str(
            self.get_parameter('motion_compensation_frame').value
        )
        self._thermal_buffer_seconds = float(
            self.get_parameter('thermal_buffer_seconds').value
        )
        self._max_cloud_time_difference = float(
            self.get_parameter('max_cloud_time_difference').value
        )
        self._bbox_inner_fraction = float(
            self.get_parameter('bbox_inner_fraction').value
        )
        self._pixel_stride = int(self.get_parameter('pixel_stride').value)
        self._minimum_range = float(self.get_parameter('minimum_range').value)
        self._maximum_range = float(self.get_parameter('maximum_range').value)
        self._minimum_valid_points = int(
            self.get_parameter('minimum_valid_points').value
        )
        self._thermal_image_width = int(
            self.get_parameter('thermal_image_width').value
        )
        self._thermal_image_height = int(
            self.get_parameter('thermal_image_height').value
        )
        self._thermal_hfov_deg = float(
            self.get_parameter('thermal_hfov_deg').value
        )
        self._depth_hfov_deg = float(
            self.get_parameter('depth_hfov_deg').value
        )
        self._thermal_projection_margin = float(
            self.get_parameter('thermal_projection_margin').value
        )
        self._thermal_depth_offset_x_px = float(
            self.get_parameter('thermal_depth_offset_x_px').value
        )
        self._thermal_depth_offset_y_px = float(
            self.get_parameter('thermal_depth_offset_y_px').value
        )
        self._classification_max_age = float(
            self.get_parameter('classification_max_age').value
        )
        self._map_frame = str(self.get_parameter('map_frame').value)
        localized_point_topic = str(
            self.get_parameter('localized_point_topic').value
        )
        marker_topic = str(self.get_parameter('marker_topic').value)
        localized_detections_topic = str(
            self.get_parameter('localized_detections_topic').value
        )
        self._transform_timeout = float(
            self.get_parameter('transform_timeout').value
        )

        if self._depth_selection not in ('thermal_mask', 'bbox'):
            raise ValueError(
                "depth_selection must be 'thermal_mask' or 'bbox'"
            )
        if self._kelvin_per_count <= 0.0:
            raise ValueError('kelvin_per_count must be greater than zero')
        if self._depth_cluster_window <= 0.0:
            raise ValueError('depth_cluster_window must be greater than zero')
        if self._thermal_buffer_seconds <= 0.0:
            raise ValueError(
                'thermal_buffer_seconds must be greater than zero'
            )
        if self._max_cloud_time_difference < 0.0:
            raise ValueError(
                'max_cloud_time_difference must be zero or greater'
            )
        if not 0.0 < self._bbox_inner_fraction <= 1.0:
            raise ValueError('bbox_inner_fraction must be in the range (0, 1]')
        if self._pixel_stride < 1:
            raise ValueError('pixel_stride must be at least one')
        if not 0.0 <= self._minimum_range < self._maximum_range:
            raise ValueError(
                'minimum_range must be nonnegative and below maximum_range'
            )
        if self._minimum_valid_points < 1:
            raise ValueError('minimum_valid_points must be at least one')
        if self._thermal_image_width <= 1 or self._thermal_image_height <= 1:
            raise ValueError('thermal image dimensions must exceed one pixel')
        if self._thermal_hfov_deg <= 0.0 or self._depth_hfov_deg <= 0.0:
            raise ValueError('camera fields of view must be positive')
        if self._thermal_projection_margin <= 0.0:
            raise ValueError(
                'thermal_projection_margin must be greater than zero'
            )
        if self._classification_max_age < 0.0:
            raise ValueError('classification_max_age must be nonnegative')
        if not self._map_frame:
            raise ValueError('map_frame cannot be empty')
        if self._transform_timeout < 0.0:
            raise ValueError('transform_timeout must be zero or greater')

        self._first_cloud_received = False
        self._first_thermal_image_received = False
        # A cloud waits here until a thermal frame at least as new as the
        # cloud has arrived, so it is paired with the closest thermal frame.
        self._pending_cloud = None
        self._detection_buffer = deque()
        self._hot_masks = deque()
        self._active_thermal_ids = set()
        self._classification_cache = {}
        self._tf_buffer = Buffer(
            cache_time=Duration(
                seconds=max(10.0, self._thermal_buffer_seconds + 2.0)
            )
        )
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._point_pub = self.create_publisher(
            PointStamped,
            localized_point_topic,
            10,
        )
        marker_qos = QoSProfile(depth=10)
        marker_qos.reliability = ReliabilityPolicy.RELIABLE
        marker_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._marker_pub = self.create_publisher(
            Marker,
            marker_topic,
            marker_qos,
        )
        self._localized_detections_pub = self.create_publisher(
            Detection3DArray,
            localized_detections_topic,
            10,
        )

        self._detection_sub = self.create_subscription(
            Detection2DArray,
            detection_topic,
            self._detection_callback,
            qos_profile_sensor_data,
        )
        self._classification_sub = self.create_subscription(
            Detection2DArray,
            classification_topic,
            self._classification_callback,
            qos_profile_sensor_data,
        )
        self._cloud_sub = self.create_subscription(
            PointCloud2,
            point_cloud_topic,
            self._cloud_callback,
            qos_profile_sensor_data,
        )
        if self._depth_selection == 'thermal_mask':
            self._thermal_image_sub = self.create_subscription(
                Image,
                thermal_image_topic,
                self._thermal_image_callback,
                qos_profile_sensor_data,
            )

        self.get_logger().info('Candidate localizer started')
        self.get_logger().info(
            f'Thermal trigger detections: {detection_topic}'
        )
        self.get_logger().info(
            f'Optional YOLO classifications: {classification_topic}'
        )
        self.get_logger().info(f'Point cloud: {point_cloud_topic}')
        self.get_logger().info(
            f'Localized points: {localized_point_topic}; '
            f'RViz markers: {marker_topic}; map frame: {self._map_frame}'
        )
        self.get_logger().info(
            f'Localized detections: {localized_detections_topic}'
        )
        self.get_logger().info(
            'Each cloud is paired with the closest thermal frame; '
            'maximum difference: '
            f'{self._max_cloud_time_difference:.3f} s; motion compensation '
            f'frame: {self._motion_frame or "disabled"}'
        )
        if self._depth_selection == 'thermal_mask':
            self.get_logger().info(
                'Depth selection: thermal_mask (only depth points on pixels '
                f'>= {self._hot_threshold_k:.1f} K from {thermal_image_topic}'
                f'), pixel stride={self._pixel_stride}, '
                f'range cluster={self._depth_cluster_window:.2f} m, '
                f'range=[{self._minimum_range:.1f}, '
                f'{self._maximum_range:.1f}] m, '
                f'minimum points={self._minimum_valid_points}'
            )
        else:
            self.get_logger().info(
                'Depth selection: bbox, '
                f'inner box={self._bbox_inner_fraction:.2f}, '
                f'pixel stride={self._pixel_stride}, '
                f'range=[{self._minimum_range:.1f}, '
                f'{self._maximum_range:.1f}] m, '
                f'minimum points={self._minimum_valid_points}'
            )
        self.get_logger().info(
            'Thermal camera: '
            f'{self._thermal_image_width}x{self._thermal_image_height}, '
            f'thermal HFOV={self._thermal_hfov_deg:.1f} deg, '
            f'depth HFOV={self._depth_hfov_deg:.1f} deg, '
            f'margin={self._thermal_projection_margin:.2f}'
        )

    @staticmethod
    def _stamp_seconds(stamp):
        return stamp.sec + stamp.nanosec / 1_000_000_000

    def _prune(self, buffer, newest_ns, stamp_ns_of):
        """Drop buffered items older than thermal_buffer_seconds."""
        # Simulation time can jump backwards when Gazebo is restarted.
        if buffer and stamp_ns_of(buffer[0]) > newest_ns:
            newest = buffer[-1]
            buffer.clear()
            buffer.append(newest)
            return
        cutoff_ns = newest_ns - int(
            self._thermal_buffer_seconds * NANOSECONDS_PER_SECOND
        )
        while buffer and stamp_ns_of(buffer[0]) < cutoff_ns:
            buffer.popleft()

    def _thermal_image_callback(self, message):
        """Keep the hot-pixel mask of recent thermal frames by stamp."""
        try:
            kelvin = thermal_image_to_kelvin(message, self._kelvin_per_count)
        except ValueError as error:
            self.get_logger().error(f'Cannot read thermal image: {error}')
            return
        if not self._first_thermal_image_received:
            self._first_thermal_image_received = True
            self.get_logger().info(
                'First thermal image received: '
                f'{message.width}x{message.height} {message.encoding}, '
                f'max={float(np.nanmax(kelvin)):.2f} K'
            )
        stamp_ns = stamp_nanoseconds(message.header.stamp)
        self._hot_masks.append((stamp_ns, kelvin >= self._hot_threshold_k))
        self._prune(self._hot_masks, stamp_ns, lambda item: item[0])

    def _hot_mask_for(self, stamp):
        """Return the hot mask of the thermal frame with this exact stamp."""
        target_ns = stamp_nanoseconds(stamp)
        for stamp_ns, mask in reversed(self._hot_masks):
            if abs(stamp_ns - target_ns) <= 1_000_000:
                return mask
        return None

    def _cloud_callback(self, message):
        if not self._first_cloud_received:
            self._first_cloud_received = True
            field_names = [field.name for field in message.fields]

            self.get_logger().info(
                'First point cloud received: '
                f'frame={message.header.frame_id}, '
                f'time={self._stamp_seconds(message.header.stamp):.3f}, '
                f'size={message.width}x{message.height}, '
                f'fields={field_names}'
            )

        if self._pending_cloud is not None:
            # A newer cloud arrived first: use the best thermal frame so far.
            pending = self._pending_cloud
            self._pending_cloud = None
            self._localize_cloud(pending)

        if self._waiting_for_thermal(message):
            self._pending_cloud = message
            return
        self._localize_cloud(message)

    def _waiting_for_thermal(self, cloud):
        """Return True while no thermal frame as new as the cloud exists."""
        if not self._detection_buffer:
            return True
        newest_ns = stamp_nanoseconds(
            self._detection_buffer[-1].header.stamp
        )
        return newest_ns < stamp_nanoseconds(cloud.header.stamp)

    @staticmethod
    def _thermal_identifier(detection, index):
        """Return the identifier shared by thermal and YOLO detections."""
        return detection.id or f'thermal_{index + 1}'

    def _classification_callback(self, message):
        """Cache optional YOLO labels without triggering localization."""
        timestamp_ns = stamp_nanoseconds(message.header.stamp)
        for index, detection in enumerate(message.detections):
            identifier = self._thermal_identifier(detection, index)
            if identifier not in self._active_thermal_ids:
                continue
            if not detection.results:
                continue

            self._classification_cache[identifier] = (
                timestamp_ns,
                list(detection.results),
            )
            best = max(
                detection.results,
                key=lambda result: result.hypothesis.score,
            )
            self.get_logger().info(
                f'Cached YOLO classification for {identifier}: '
                f'{best.hypothesis.class_id} '
                f'{best.hypothesis.score:.2f}'
            )

    def _classification_results(self, identifier, thermal_stamp):
        """Return a fresh cached YOLO result for one thermal candidate."""
        cached = self._classification_cache.get(identifier)
        if cached is None:
            return None

        classification_stamp_ns, results = cached
        thermal_stamp_ns = stamp_nanoseconds(thermal_stamp)
        age = abs(thermal_stamp_ns - classification_stamp_ns)
        age_seconds = age / NANOSECONDS_PER_SECOND
        if age_seconds > self._classification_max_age:
            self._classification_cache.pop(identifier, None)
            return None
        return results

    def _detection_callback(self, message):
        self.get_logger().info(
            'Thermal detection message received: '
            f'frame={message.header.frame_id}, '
            f'time={self._stamp_seconds(message.header.stamp):.3f}, '
            f'count={len(message.detections)}'
        )

        active_ids = {
            self._thermal_identifier(detection, index)
            for index, detection in enumerate(message.detections)
        }
        self._active_thermal_ids = active_ids
        for identifier in tuple(self._classification_cache):
            if identifier not in active_ids:
                self._classification_cache.pop(identifier, None)

        for index, detection in enumerate(message.detections):
            center = detection.bbox.center
            if hasattr(center, 'position'):
                center_x = center.position.x
                center_y = center.position.y
            else:
                center_x = center.x
                center_y = center.y

            self.get_logger().info(
                f'Thermal detection {index}: '
                f'id={self._thermal_identifier(detection, index)}, '
                f'centre=({center_x:.1f}, {center_y:.1f}), '
                f'size=({detection.bbox.size_x:.1f}, '
                f'{detection.bbox.size_y:.1f})'
            )

        # Empty messages are kept too: they show that nothing was hot then.
        self._detection_buffer.append(message)
        self._prune(
            self._detection_buffer,
            stamp_nanoseconds(message.header.stamp),
            lambda item: stamp_nanoseconds(item.header.stamp),
        )

        if (
            self._pending_cloud is not None
            and not self._waiting_for_thermal(self._pending_cloud)
        ):
            pending = self._pending_cloud
            self._pending_cloud = None
            self._localize_cloud(pending)

    def _cloud_to_thermal(self, cloud, thermal_header):
        """
        Look up the transform from the cloud frame to the thermal frame.

        With a motion compensation frame (odometry), the robot motion between
        the cloud stamp and the thermal stamp is included.
        """
        thermal_frame = thermal_header.frame_id
        if self._motion_frame:
            transform = self._tf_buffer.lookup_transform_full(
                thermal_frame,
                Time.from_msg(thermal_header.stamp),
                cloud.header.frame_id,
                Time.from_msg(cloud.header.stamp),
                self._motion_frame,
                Duration(seconds=self._transform_timeout),
            )
        else:
            transform = self._tf_buffer.lookup_transform(
                thermal_frame,
                cloud.header.frame_id,
                Time.from_msg(cloud.header.stamp),
                timeout=Duration(seconds=self._transform_timeout),
            )
        return transform_to_matrix(transform.transform)

    def _localize_cloud(self, cloud):
        """Localize the thermal detections closest in time to one cloud."""
        message, difference = closest_stamped(
            self._detection_buffer,
            cloud.header.stamp,
        )
        if message is None:
            self.get_logger().warning(
                'Cannot localize cloud: no thermal detection message has '
                'been received',
                throttle_duration_sec=5.0,
            )
            return

        if difference > self._max_cloud_time_difference:
            self.get_logger().warning(
                'Cannot localize cloud: closest thermal detection differs '
                f'by {difference:.3f} s, which exceeds the '
                f'{self._max_cloud_time_difference:.3f} s limit',
                throttle_duration_sec=2.0,
            )
            return

        if not message.detections:
            return

        self.get_logger().info(
            'Matched detection to point cloud: '
            f'detection_time={self._stamp_seconds(message.header.stamp):.3f}, '
            f'cloud_time={self._stamp_seconds(cloud.header.stamp):.3f}, '
            f'difference={difference:.3f} s, '
            f'cloud_frame={cloud.header.frame_id}'
        )

        hot_mask = None
        cloud_to_thermal = None
        if self._depth_selection == 'thermal_mask':
            hot_mask = self._hot_mask_for(message.header.stamp)
            if hot_mask is None:
                self.get_logger().warning(
                    'Cannot localize detection: no thermal image with stamp '
                    f'{self._stamp_seconds(message.header.stamp):.3f} '
                    '(check thermal_image_topic)',
                    throttle_duration_sec=5.0,
                )
                return
            try:
                cloud_to_thermal = self._cloud_to_thermal(
                    cloud,
                    message.header,
                )
            except TransformException as error:
                self.get_logger().warning(
                    f'Cannot transform {cloud.header.frame_id} to '
                    f'{message.header.frame_id}: {error}'
                )
                return

        try:
            map_transform = self._tf_buffer.lookup_transform(
                self._map_frame,
                cloud.header.frame_id,
                Time.from_msg(cloud.header.stamp),
                timeout=Duration(seconds=self._transform_timeout),
            )
        except TransformException as error:
            self.get_logger().warning(
                f'Cannot transform {cloud.header.frame_id} to '
                f'{self._map_frame}: {error}'
            )
            return

        localized_message = Detection3DArray()
        localized_message.header.stamp = cloud.header.stamp
        localized_message.header.frame_id = self._map_frame

        for index, detection in enumerate(message.detections):
            identifier = self._thermal_identifier(detection, index)
            classification_results = self._classification_results(
                identifier,
                message.header.stamp,
            )
            if self._depth_selection == 'thermal_mask':
                estimate = self._estimate_thermal_mask(
                    identifier,
                    detection,
                    cloud,
                    cloud_to_thermal,
                    hot_mask,
                )
                source_detection = make_projected_detection(
                    detection,
                    identifier,
                    detection_box(detection),
                    classification_results,
                )
            else:
                estimate, source_detection = self._estimate_bbox(
                    identifier,
                    detection,
                    cloud,
                    classification_results,
                )
            if estimate is None:
                continue
            position, spread, valid_count, sampled_count = estimate

            self.get_logger().info(
                f'Camera-frame estimate for {identifier}: '
                f'position=({position[0]:.3f}, {position[1]:.3f}, '
                f'{position[2]:.3f}) m, '
                f'spread={spread:.3f} m, '
                f'valid_points={valid_count}/{sampled_count}, '
                f'frame={cloud.header.frame_id}'
            )

            camera_point = make_camera_point(cloud.header, position)
            map_point = do_transform_point(camera_point, map_transform)

            self._point_pub.publish(map_point)
            self._marker_pub.publish(
                make_candidate_marker(map_point, identifier, spread)
            )
            localized_message.detections.append(
                make_localized_detection(
                    source_detection,
                    map_point,
                    spread,
                )
            )
            self.get_logger().info(
                f'Map-frame estimate for {identifier}: '
                f'position=({map_point.point.x:.3f}, '
                f'{map_point.point.y:.3f}, '
                f'{map_point.point.z:.3f}) m, '
                f'frame={map_point.header.frame_id}'
            )

        if localized_message.detections:
            self._localized_detections_pub.publish(localized_message)

    def _estimate_thermal_mask(
        self,
        identifier,
        detection,
        cloud,
        cloud_to_thermal,
        hot_mask,
    ):
        """Estimate one detection from depth points on its hot pixels."""
        box = detection_box(detection)
        rotation, translation = cloud_to_thermal
        thermal_height, thermal_width = hot_mask.shape
        mask = detection_hot_mask(hot_mask, box, erode=self._erode_hot_mask)
        try:
            window = None
            if cloud.height > 1:
                window = depth_window_for_box(
                    box,
                    rotation,
                    translation,
                    thermal_width,
                    thermal_height,
                    self._thermal_hfov_deg,
                    cloud.width,
                    cloud.height,
                    self._depth_hfov_deg,
                    near=max(self._minimum_range, 0.5),
                    far=self._maximum_range,
                )
            points = cloud_points(
                cloud,
                self._pixel_stride,
                self._minimum_range,
                self._maximum_range,
                window,
            )
            column, row, valid = thermal_pixel_indices(
                points,
                rotation,
                translation,
                thermal_width,
                thermal_height,
                self._thermal_hfov_deg,
            )
            return hot_pixel_position(
                points,
                column,
                row,
                valid,
                mask,
                self._minimum_range,
                self._maximum_range,
                self._minimum_valid_points,
                self._depth_cluster_window,
            )
        except (AssertionError, ValueError) as error:
            self.get_logger().warning(
                f'Cannot estimate {identifier}: {error}'
            )
            return None

    def _estimate_bbox(
        self,
        identifier,
        detection,
        cloud,
        classification_results,
    ):
        """Estimate one detection with the median of the projected box."""
        try:
            projected_box = project_thermal_box(
                detection_box(detection),
                self._thermal_image_width,
                self._thermal_image_height,
                cloud.width,
                cloud.height,
                self._thermal_hfov_deg,
                self._depth_hfov_deg,
                self._thermal_projection_margin,
                self._thermal_depth_offset_x_px,
                self._thermal_depth_offset_y_px,
            )
        except ValueError as error:
            self.get_logger().warning(
                f'Cannot project {identifier}: {error}'
            )
            return None, None

        projected_detection = make_projected_detection(
            detection,
            identifier,
            projected_box,
            classification_results,
        )
        projected_center = projected_detection.bbox.center
        if hasattr(projected_center, 'position'):
            projected_center_x = projected_center.position.x
            projected_center_y = projected_center.position.y
        else:
            projected_center_x = projected_center.x
            projected_center_y = projected_center.y
        classification = projected_detection.results[0].hypothesis
        self.get_logger().info(
            f'Projected {identifier} into depth image: '
            f'centre=({projected_center_x:.1f}, '
            f'{projected_center_y:.1f}), '
            f'size=({projected_detection.bbox.size_x:.1f}, '
            f'{projected_detection.bbox.size_y:.1f}), '
            f'class={classification.class_id} '
            f'{classification.score:.2f}'
        )
        try:
            estimate = extract_position_from_cloud(
                cloud,
                projected_detection,
                self._bbox_inner_fraction,
                self._pixel_stride,
                self._minimum_range,
                self._maximum_range,
                self._minimum_valid_points,
            )
        except (AssertionError, ValueError) as error:
            self.get_logger().warning(
                f'Cannot estimate {identifier}: {error}'
            )
            return None, None
        return estimate, projected_detection


def main(args=None):
    rclpy.init(args=args)
    node = CandidateLocalizer()
    # TF subscriptions use their own reentrant callback group. A second
    # thread lets a lookup wait for the transform at the newest thermal
    # stamp; the node's own callbacks still run one at a time.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
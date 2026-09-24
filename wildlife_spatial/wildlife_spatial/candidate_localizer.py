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
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
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


def closest_cloud(clouds, target_stamp):
    """Return the cloud nearest to a target stamp and the absolute time gap."""
    if not clouds:
        return None, None

    target_ns = stamp_nanoseconds(target_stamp)
    cloud = min(
        clouds,
        key=lambda item: abs(
            stamp_nanoseconds(item.header.stamp) - target_ns
        ),
    )
    difference_ns = abs(stamp_nanoseconds(cloud.header.stamp) - target_ns)
    return cloud, difference_ns / NANOSECONDS_PER_SECOND


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
        self.declare_parameter('cloud_buffer_seconds', 5.0)
        self.declare_parameter('max_cloud_time_difference', 0.5)
        self.declare_parameter('bbox_inner_fraction', 0.5)
        self.declare_parameter('pixel_stride', 2)
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
        self._cloud_buffer_seconds = float(
            self.get_parameter('cloud_buffer_seconds').value
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

        if self._cloud_buffer_seconds <= 0.0:
            raise ValueError('cloud_buffer_seconds must be greater than zero')
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
        self._cloud_buffer = deque()
        self._active_thermal_ids = set()
        self._classification_cache = {}
        self._tf_buffer = Buffer(
            cache_time=Duration(
                seconds=max(10.0, self._cloud_buffer_seconds + 2.0)
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
            f'Cloud buffer: {self._cloud_buffer_seconds:.1f} s; '
            'maximum match difference: '
            f'{self._max_cloud_time_difference:.3f} s'
        )
        self.get_logger().info(
            'Depth sampling: '
            f'inner box={self._bbox_inner_fraction:.2f}, '
            f'pixel stride={self._pixel_stride}, '
            f'range=[{self._minimum_range:.1f}, '
            f'{self._maximum_range:.1f}] m, '
            f'minimum points={self._minimum_valid_points}'
        )
        self.get_logger().info(
            'Thermal projection: '
            f'{self._thermal_image_width}x{self._thermal_image_height}, '
            f'thermal HFOV={self._thermal_hfov_deg:.1f} deg, '
            f'depth HFOV={self._depth_hfov_deg:.1f} deg, '
            f'margin={self._thermal_projection_margin:.2f}'
        )

    @staticmethod
    def _stamp_seconds(stamp):
        return stamp.sec + stamp.nanosec / 1_000_000_000

    def _cloud_callback(self, message):
        self._cloud_buffer.append(message)
        self._prune_cloud_buffer(message.header.stamp)

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

    def _prune_cloud_buffer(self, newest_stamp):
        newest_ns = stamp_nanoseconds(newest_stamp)
        buffer_ns = int(
            self._cloud_buffer_seconds * NANOSECONDS_PER_SECOND
        )

        # Simulation time can jump backwards when Gazebo is restarted.
        if (
            self._cloud_buffer
            and stamp_nanoseconds(
                self._cloud_buffer[0].header.stamp
            ) > newest_ns
        ):
            newest_cloud = self._cloud_buffer[-1]
            self._cloud_buffer.clear()
            self._cloud_buffer.append(newest_cloud)
            return

        cutoff_ns = newest_ns - buffer_ns
        while (
            self._cloud_buffer
            and stamp_nanoseconds(
                self._cloud_buffer[0].header.stamp
            ) < cutoff_ns
        ):
            self._cloud_buffer.popleft()

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

        if not message.detections:
            return

        cloud, difference = closest_cloud(
            self._cloud_buffer,
            message.header.stamp,
        )

        if cloud is None:
            self.get_logger().warning(
                'Cannot localize detection: no point cloud has been received'
            )
            return

        if difference > self._max_cloud_time_difference:
            self.get_logger().warning(
                'Cannot localize detection: closest point cloud differs by '
                f'{difference:.3f} s, which exceeds the '
                f'{self._max_cloud_time_difference:.3f} s limit'
            )
            return

        self.get_logger().info(
            'Matched detection to point cloud: '
            f'detection_time={self._stamp_seconds(message.header.stamp):.3f}, '
            f'cloud_time={self._stamp_seconds(cloud.header.stamp):.3f}, '
            f'difference={difference:.3f} s, '
            f'cloud_frame={cloud.header.frame_id}'
        )

        localized_message = Detection3DArray()
        localized_message.header.stamp = cloud.header.stamp
        localized_message.header.frame_id = self._map_frame

        for index, detection in enumerate(message.detections):
            identifier = self._thermal_identifier(detection, index)
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
                continue

            classification_results = self._classification_results(
                identifier,
                message.header.stamp,
            )
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
                position, spread, valid_count, sampled_count = (
                    extract_position_from_cloud(
                        cloud,
                        projected_detection,
                        self._bbox_inner_fraction,
                        self._pixel_stride,
                        self._minimum_range,
                        self._maximum_range,
                        self._minimum_valid_points,
                    )
                )
            except (AssertionError, ValueError) as error:
                self.get_logger().warning(
                    f'Cannot estimate {identifier}: {error}'
                )
                continue

            self.get_logger().info(
                f'Camera-frame estimate for {identifier}: '
                f'position=({position[0]:.3f}, {position[1]:.3f}, '
                f'{position[2]:.3f}) m, '
                f'spread={spread:.3f} m, '
                f'valid_points={valid_count}/{sampled_count}, '
                f'frame={cloud.header.frame_id}'
            )

            camera_point = make_camera_point(cloud.header, position)
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._map_frame,
                    cloud.header.frame_id,
                    Time.from_msg(cloud.header.stamp),
                    timeout=Duration(seconds=self._transform_timeout),
                )
                map_point = do_transform_point(camera_point, transform)
            except TransformException as error:
                self.get_logger().warning(
                    f'Cannot transform {identifier} from '
                    f'{cloud.header.frame_id} to '
                    f'{self._map_frame}: {error}'
                )
                continue

            self._point_pub.publish(map_point)
            self._marker_pub.publish(
                make_candidate_marker(map_point, identifier, spread)
            )
            localized_message.detections.append(
                make_localized_detection(
                    projected_detection,
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


def main(args=None):
    rclpy.init(args=args)
    node = CandidateLocalizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

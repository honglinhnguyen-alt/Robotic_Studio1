#!/usr/bin/env python3

"""Localize fused animal detections with depth point clouds for R2."""

from collections import deque
from math import ceil
from math import floor
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
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection3D
from vision_msgs.msg import Detection3DArray


NANOSECONDS_PER_SECOND = 1_000_000_000


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
    """Verify that the spatial pipeline receives detections and depth data."""

    def __init__(self):
        super().__init__('candidate_localizer')

        self.declare_parameter(
            'detection_topic',
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
        if not self._map_frame:
            raise ValueError('map_frame cannot be empty')
        if self._transform_timeout < 0.0:
            raise ValueError('transform_timeout must be zero or greater')

        self._first_cloud_received = False
        self._cloud_buffer = deque()
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
        self._cloud_sub = self.create_subscription(
            PointCloud2,
            point_cloud_topic,
            self._cloud_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info('Candidate localizer started')
        self.get_logger().info(f'Detections: {detection_topic}')
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

    def _detection_callback(self, message):
        self.get_logger().info(
            'Detection message received: '
            f'frame={message.header.frame_id}, '
            f'time={self._stamp_seconds(message.header.stamp):.3f}, '
            f'count={len(message.detections)}'
        )

        for index, detection in enumerate(message.detections):
            center = detection.bbox.center
            if hasattr(center, 'position'):
                center_x = center.position.x
                center_y = center.position.y
            else:
                center_x = center.x
                center_y = center.y

            self.get_logger().info(
                f'Detection {index}: '
                f'id={detection.id}, '
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
            identifier = detection.id or f'detection_{index}'
            try:
                position, spread, valid_count, sampled_count = (
                    extract_position_from_cloud(
                        cloud,
                        detection,
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
                make_localized_detection(detection, map_point, spread)
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

#!/usr/bin/env python3

"""Maintain stable R2 candidate records from localized observations."""

from dataclasses import dataclass
from dataclasses import field
import json
from math import hypot
import zlib

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from vision_msgs.msg import Detection3D
from vision_msgs.msg import Detection3DArray
from vision_msgs.msg import ObjectHypothesisWithPose


NANOSECONDS_PER_SECOND = 1_000_000_000


def stamp_nanoseconds(stamp):
    """Convert a ROS stamp to nanoseconds."""
    return stamp.sec * NANOSECONDS_PER_SECOND + stamp.nanosec


def best_classification(detection):
    """Return the highest-scoring class and confidence in a detection."""
    if not detection.results:
        return 'unknown', 0.0

    result = max(
        detection.results,
        key=lambda item: float(item.hypothesis.score),
    )
    return str(result.hypothesis.class_id or 'unknown'), float(
        result.hypothesis.score
    )


@dataclass
class CandidateRecord:
    """One persistent physical candidate in the map."""

    candidate_id: str
    x: float
    y: float
    z: float
    first_seen_ns: int
    last_seen_ns: int
    observation_count: int = 1
    class_name: str = 'unknown'
    confidence: float = 0.0
    source_ids: set = field(default_factory=set)


class CandidateRegistryStore:
    """Merge observations into stable records using planar map distance."""

    def __init__(self, merge_distance=1.0):
        if merge_distance <= 0.0:
            raise ValueError('merge_distance must be greater than zero')
        self.merge_distance = float(merge_distance)
        self.records = []

    def update(self, detection, stamp):
        """Store an observation and return record, new flag, and distance."""
        position = detection.bbox.center.position
        x = float(position.x)
        y = float(position.y)
        z = float(position.z)
        timestamp_ns = stamp_nanoseconds(stamp)
        class_name, confidence = best_classification(detection)

        record, distance = self._closest(x, y)
        if record is None or distance > self.merge_distance:
            record = CandidateRecord(
                candidate_id=f'candidate_{len(self.records) + 1}',
                x=x,
                y=y,
                z=z,
                first_seen_ns=timestamp_ns,
                last_seen_ns=timestamp_ns,
                class_name=class_name,
                confidence=confidence,
                source_ids={detection.id} if detection.id else set(),
            )
            self.records.append(record)
            return record, True, None

        previous_count = record.observation_count
        record.observation_count += 1
        weight = 1.0 / record.observation_count
        record.x += (x - record.x) * weight
        record.y += (y - record.y) * weight
        record.z += (z - record.z) * weight
        record.last_seen_ns = timestamp_ns
        if detection.id:
            record.source_ids.add(detection.id)
        if confidence >= record.confidence:
            record.class_name = class_name
            record.confidence = confidence

        assert record.observation_count == previous_count + 1
        return record, False, distance

    def _closest(self, x, y):
        if not self.records:
            return None, None

        record = min(
            self.records,
            key=lambda item: hypot(x - item.x, y - item.y),
        )
        return record, hypot(x - record.x, y - record.y)


def record_to_detection(record, header, marker_diameter=0.5):
    """Convert a registry record to a standard Detection3D message."""
    detection = Detection3D()
    detection.header = header
    detection.id = record.candidate_id
    detection.bbox.center.position.x = record.x
    detection.bbox.center.position.y = record.y
    detection.bbox.center.position.z = record.z
    detection.bbox.center.orientation.w = 1.0
    detection.bbox.size.x = marker_diameter
    detection.bbox.size.y = marker_diameter
    detection.bbox.size.z = marker_diameter

    result = ObjectHypothesisWithPose()
    result.hypothesis.class_id = record.class_name
    result.hypothesis.score = record.confidence
    result.pose.pose = detection.bbox.center
    detection.results.append(result)
    return detection


def record_markers(record, header, marker_diameter=0.5):
    """Create a sphere and text label for one registry record."""
    stable_id = zlib.crc32(
        record.candidate_id.encode('utf-8')
    ) & 0x3FFFFFFF

    sphere = Marker()
    sphere.header = header
    sphere.ns = 'candidate_registry'
    sphere.id = stable_id
    sphere.type = Marker.SPHERE
    sphere.action = Marker.ADD
    sphere.pose.position.x = record.x
    sphere.pose.position.y = record.y
    sphere.pose.position.z = record.z
    sphere.pose.orientation.w = 1.0
    sphere.scale.x = marker_diameter
    sphere.scale.y = marker_diameter
    sphere.scale.z = marker_diameter
    sphere.color.r = 0.0
    sphere.color.g = 0.75
    sphere.color.b = 1.0
    sphere.color.a = 0.9

    label = Marker()
    label.header = header
    label.ns = 'candidate_registry_labels'
    label.id = stable_id
    label.type = Marker.TEXT_VIEW_FACING
    label.action = Marker.ADD
    label.pose.position.x = record.x
    label.pose.position.y = record.y
    label.pose.position.z = record.z + 0.6
    label.pose.orientation.w = 1.0
    label.scale.z = 0.3
    label.color.r = 1.0
    label.color.g = 1.0
    label.color.b = 1.0
    label.color.a = 1.0
    label.text = (
        f'{record.candidate_id}: {record.class_name} '
        f'({record.confidence:.2f}), n={record.observation_count}'
    )
    return sphere, label


def registry_payload(records, frame_id):
    """Create the JSON-safe registry snapshot used by mission and UI code."""
    return {
        'frame_id': frame_id,
        'candidate_count': len(records),
        'candidates': [
            {
                'candidate_id': record.candidate_id,
                'position': {
                    'x': record.x,
                    'y': record.y,
                    'z': record.z,
                },
                'state': 'uninvestigated',
                'class_name': record.class_name,
                'confidence': record.confidence,
                'observation_count': record.observation_count,
                'source_ids': sorted(record.source_ids),
                'first_seen': (
                    record.first_seen_ns / NANOSECONDS_PER_SECOND
                ),
                'last_seen': (
                    record.last_seen_ns / NANOSECONDS_PER_SECOND
                ),
            }
            for record in records
        ],
    }


class CandidateRegistry(Node):
    """ROS interface for persistent candidate storage and duplicate merging."""

    def __init__(self):
        super().__init__('candidate_registry')

        self.declare_parameter(
            'input_topic',
            '/husky1/candidates/localized_detections',
        )
        self.declare_parameter(
            'registry_topic',
            '/husky1/candidates/registry',
        )
        self.declare_parameter(
            'registry_json_topic',
            '/husky1/candidates/registry_json',
        )
        self.declare_parameter(
            'marker_topic',
            '/husky1/candidates/registry_markers',
        )
        self.declare_parameter('map_frame', 'husky1_map')
        self.declare_parameter('merge_distance', 1.0)
        self.declare_parameter('marker_diameter', 0.5)

        input_topic = str(self.get_parameter('input_topic').value)
        registry_topic = str(self.get_parameter('registry_topic').value)
        registry_json_topic = str(
            self.get_parameter('registry_json_topic').value
        )
        marker_topic = str(self.get_parameter('marker_topic').value)
        self._map_frame = str(self.get_parameter('map_frame').value)
        merge_distance = float(self.get_parameter('merge_distance').value)
        self._marker_diameter = float(
            self.get_parameter('marker_diameter').value
        )

        if not self._map_frame:
            raise ValueError('map_frame cannot be empty')
        if self._marker_diameter <= 0.0:
            raise ValueError('marker_diameter must be greater than zero')

        self._store = CandidateRegistryStore(merge_distance)

        snapshot_qos = QoSProfile(depth=1)
        snapshot_qos.reliability = ReliabilityPolicy.RELIABLE
        snapshot_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._registry_pub = self.create_publisher(
            Detection3DArray,
            registry_topic,
            snapshot_qos,
        )
        self._json_pub = self.create_publisher(
            String,
            registry_json_topic,
            snapshot_qos,
        )
        self._marker_pub = self.create_publisher(
            MarkerArray,
            marker_topic,
            snapshot_qos,
        )
        self._localized_sub = self.create_subscription(
            Detection3DArray,
            input_topic,
            self._localized_callback,
            10,
        )

        self.get_logger().info('Candidate registry started')
        self.get_logger().info(f'Localized detections: {input_topic}')
        self.get_logger().info(
            f'Registry: {registry_topic}; JSON: {registry_json_topic}; '
            f'markers: {marker_topic}'
        )
        self.get_logger().info(
            f'Map frame: {self._map_frame}; '
            f'merge distance: {merge_distance:.2f} m'
        )

    def _localized_callback(self, message):
        if message.header.frame_id != self._map_frame:
            self.get_logger().warning(
                'Ignoring localized detections in frame '
                f'{message.header.frame_id}; expected {self._map_frame}'
            )
            return

        changed = False
        for detection in message.detections:
            record, is_new, distance = self._store.update(
                detection,
                message.header.stamp,
            )
            changed = True
            if is_new:
                self.get_logger().info(
                    f'Created {record.candidate_id} at '
                    f'({record.x:.3f}, {record.y:.3f}, '
                    f'{record.z:.3f}) m; class={record.class_name} '
                    f'{record.confidence:.2f}'
                )
            else:
                self.get_logger().info(
                    f'Merged observation into {record.candidate_id}: '
                    f'distance={distance:.3f} m, '
                    f'observations={record.observation_count}, '
                    f'class={record.class_name} '
                    f'{record.confidence:.2f}'
                )

        if changed:
            self._publish_snapshot(message.header)

    def _publish_snapshot(self, header):
        registry = Detection3DArray()
        registry.header = header
        markers = MarkerArray()

        for record in self._store.records:
            registry.detections.append(
                record_to_detection(
                    record,
                    header,
                    self._marker_diameter,
                )
            )
            markers.markers.extend(
                record_markers(
                    record,
                    header,
                    self._marker_diameter,
                )
            )

        json_message = String()
        json_message.data = json.dumps(
            registry_payload(self._store.records, self._map_frame),
            separators=(',', ':'),
            sort_keys=True,
        )
        self._registry_pub.publish(registry)
        self._json_pub.publish(json_message)
        self._marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = CandidateRegistry()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

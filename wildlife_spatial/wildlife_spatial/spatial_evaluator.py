#!/usr/bin/env python3

"""Log and evaluate R2 candidate estimates against simulation truth."""

import csv
from datetime import datetime
import json
from math import hypot
from math import sqrt
import os
from pathlib import Path
from statistics import mean
from statistics import median

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import String
from vision_msgs.msg import Detection3D
from vision_msgs.msg import Detection3DArray
from vision_msgs.msg import ObjectHypothesisWithPose


CSV_FIELDS = [
    'evaluation_index',
    'ros_time',
    'status',
    'estimate_id',
    'ground_truth_id',
    'estimate_class',
    'ground_truth_class',
    'estimate_x',
    'estimate_y',
    'estimate_z',
    'ground_truth_x',
    'ground_truth_y',
    'ground_truth_z',
    'planar_error_m',
    'spatial_error_m',
    'class_correct',
]


def detection_class(detection):
    """Return the most confident class label in a 3D detection."""
    if not detection.results:
        return 'unknown'
    result = max(
        detection.results,
        key=lambda item: float(item.hypothesis.score),
    )
    return str(result.hypothesis.class_id or 'unknown')


def planar_distance(first, second):
    """Return XY distance between two Detection3D centres."""
    first_position = first.bbox.center.position
    second_position = second.bbox.center.position
    return hypot(
        first_position.x - second_position.x,
        first_position.y - second_position.y,
    )


def spatial_distance(first, second):
    """Return XYZ distance between two Detection3D centres."""
    first_position = first.bbox.center.position
    second_position = second.bbox.center.position
    return sqrt(
        (first_position.x - second_position.x) ** 2
        + (first_position.y - second_position.y) ** 2
        + (first_position.z - second_position.z) ** 2
    )


def match_detections(estimates, ground_truth, maximum_distance):
    """Greedily create unique nearest pairs inside the match distance."""
    if maximum_distance <= 0.0:
        raise ValueError('maximum_distance must be greater than zero')

    possible = sorted(
        (
            (planar_distance(estimate, truth), estimate_index, truth_index)
            for estimate_index, estimate in enumerate(estimates)
            for truth_index, truth in enumerate(ground_truth)
        ),
        key=lambda item: item[0],
    )
    used_estimates = set()
    used_truth = set()
    matches = []

    for distance, estimate_index, truth_index in possible:
        if distance > maximum_distance:
            break
        if estimate_index in used_estimates or truth_index in used_truth:
            continue
        used_estimates.add(estimate_index)
        used_truth.add(truth_index)
        matches.append((estimate_index, truth_index, distance))

    unmatched_estimates = [
        index for index in range(len(estimates))
        if index not in used_estimates
    ]
    unmatched_truth = [
        index for index in range(len(ground_truth))
        if index not in used_truth
    ]
    return matches, unmatched_estimates, unmatched_truth


def evaluate_detections(estimates, ground_truth, maximum_distance):
    """Compute localization, detection, and classification metrics."""
    matches, false_positives, missed = match_detections(
        estimates,
        ground_truth,
        maximum_distance,
    )
    planar_errors = [item[2] for item in matches]
    spatial_errors = [
        spatial_distance(estimates[estimate_index], ground_truth[truth_index])
        for estimate_index, truth_index, _ in matches
    ]
    class_results = [
        detection_class(estimates[estimate_index]).lower()
        == detection_class(ground_truth[truth_index]).lower()
        for estimate_index, truth_index, _ in matches
    ]

    matched_count = len(matches)
    estimate_count = len(estimates)
    truth_count = len(ground_truth)
    return {
        'estimated_count': estimate_count,
        'ground_truth_count': truth_count,
        'matched_count': matched_count,
        'false_positive_count': len(false_positives),
        'missed_count': len(missed),
        'precision': matched_count / estimate_count if estimate_count else 0.0,
        'recall': matched_count / truth_count if truth_count else 0.0,
        'mean_planar_error_m': mean(planar_errors) if planar_errors else None,
        'median_planar_error_m': (
            median(planar_errors) if planar_errors else None
        ),
        'rmse_planar_error_m': (
            sqrt(mean([error ** 2 for error in planar_errors]))
            if planar_errors else None
        ),
        'maximum_planar_error_m': (
            max(planar_errors) if planar_errors else None
        ),
        'mean_spatial_error_m': (
            mean(spatial_errors) if spatial_errors else None
        ),
        'classification_accuracy': (
            sum(class_results) / matched_count if matched_count else None
        ),
        'matches': matches,
        'false_positive_indices': false_positives,
        'missed_indices': missed,
    }


def load_ground_truth(path):
    """Load R5 ground truth from the documented JSON interchange format."""
    with Path(path).expanduser().open(encoding='utf-8') as input_file:
        payload = json.load(input_file)

    frame_id = str(payload.get('frame_id', ''))
    if not frame_id:
        raise ValueError('ground-truth file must contain frame_id')

    message = Detection3DArray()
    message.header.frame_id = frame_id
    for item in payload.get('objects', []):
        position = item['position']
        detection = Detection3D()
        detection.header.frame_id = frame_id
        detection.id = str(item['id'])
        detection.bbox.center.position.x = float(position['x'])
        detection.bbox.center.position.y = float(position['y'])
        detection.bbox.center.position.z = float(position.get('z', 0.0))
        detection.bbox.center.orientation.w = 1.0
        result = ObjectHypothesisWithPose()
        result.hypothesis.class_id = str(item['class_name'])
        result.hypothesis.score = 1.0
        detection.results.append(result)
        message.detections.append(detection)
    return message


def evaluation_rows(metrics, estimates, ground_truth):
    """Return detailed matched and unmatched rows for one evaluation."""
    rows = []
    for estimate_index, truth_index, planar_error in metrics['matches']:
        estimate = estimates[estimate_index]
        truth = ground_truth[truth_index]
        estimate_position = estimate.bbox.center.position
        truth_position = truth.bbox.center.position
        estimate_class = detection_class(estimate)
        truth_class = detection_class(truth)
        rows.append({
            'status': 'matched',
            'estimate_id': estimate.id,
            'ground_truth_id': truth.id,
            'estimate_class': estimate_class,
            'ground_truth_class': truth_class,
            'estimate_x': estimate_position.x,
            'estimate_y': estimate_position.y,
            'estimate_z': estimate_position.z,
            'ground_truth_x': truth_position.x,
            'ground_truth_y': truth_position.y,
            'ground_truth_z': truth_position.z,
            'planar_error_m': planar_error,
            'spatial_error_m': spatial_distance(estimate, truth),
            'class_correct': estimate_class.lower() == truth_class.lower(),
        })

    for estimate_index in metrics['false_positive_indices']:
        estimate = estimates[estimate_index]
        position = estimate.bbox.center.position
        rows.append({
            'status': 'false_positive',
            'estimate_id': estimate.id,
            'estimate_class': detection_class(estimate),
            'estimate_x': position.x,
            'estimate_y': position.y,
            'estimate_z': position.z,
        })

    for truth_index in metrics['missed_indices']:
        truth = ground_truth[truth_index]
        position = truth.bbox.center.position
        rows.append({
            'status': 'missed',
            'ground_truth_id': truth.id,
            'ground_truth_class': detection_class(truth),
            'ground_truth_x': position.x,
            'ground_truth_y': position.y,
            'ground_truth_z': position.z,
        })
    return rows


class SpatialEvaluator(Node):
    """Evaluate registry snapshots and write repeatable experiment logs."""

    def __init__(self):
        super().__init__('spatial_evaluator')

        self.declare_parameter(
            'registry_topic',
            '/husky1/candidates/registry',
        )
        self.declare_parameter(
            'ground_truth_topic',
            '/simulation/ground_truth/animals',
        )
        self.declare_parameter(
            'summary_topic',
            '/husky1/evaluation/summary',
        )
        self.declare_parameter('ground_truth_file', '')
        self.declare_parameter('map_frame', 'husky1_map')
        self.declare_parameter('maximum_match_distance', 2.0)
        self.declare_parameter(
            'output_directory',
            '~/wildlife_evaluation_logs',
        )

        registry_topic = str(self.get_parameter('registry_topic').value)
        ground_truth_topic = str(
            self.get_parameter('ground_truth_topic').value
        )
        summary_topic = str(self.get_parameter('summary_topic').value)
        ground_truth_file = str(
            self.get_parameter('ground_truth_file').value
        )
        self._map_frame = str(self.get_parameter('map_frame').value)
        self._maximum_match_distance = float(
            self.get_parameter('maximum_match_distance').value
        )
        output_directory = Path(
            str(self.get_parameter('output_directory').value)
        ).expanduser()

        if not self._map_frame:
            raise ValueError('map_frame cannot be empty')
        if self._maximum_match_distance <= 0.0:
            raise ValueError(
                'maximum_match_distance must be greater than zero'
            )

        run_name = datetime.now().strftime('run_%Y%m%d_%H%M%S')
        self._run_directory = output_directory / f'{run_name}_{os.getpid()}'
        self._run_directory.mkdir(parents=True, exist_ok=False)
        self._csv_path = self._run_directory / 'evaluation.csv'
        self._summary_path = self._run_directory / 'summary.json'
        with self._csv_path.open(
            'w', newline='', encoding='utf-8'
        ) as csv_file:
            csv.DictWriter(csv_file, fieldnames=CSV_FIELDS).writeheader()

        snapshot_qos = QoSProfile(depth=1)
        snapshot_qos.reliability = ReliabilityPolicy.RELIABLE
        snapshot_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._summary_pub = self.create_publisher(
            String,
            summary_topic,
            snapshot_qos,
        )
        self._registry_sub = self.create_subscription(
            Detection3DArray,
            registry_topic,
            self._registry_callback,
            snapshot_qos,
        )
        self._ground_truth_sub = self.create_subscription(
            Detection3DArray,
            ground_truth_topic,
            self._ground_truth_callback,
            snapshot_qos,
        )

        self._registry = None
        self._ground_truth = None
        self._evaluation_index = 0
        if ground_truth_file:
            self._ground_truth = load_ground_truth(ground_truth_file)

        self.get_logger().info('Spatial evaluator started')
        self.get_logger().info(f'Registry input: {registry_topic}')
        self.get_logger().info(
            f'Ground truth: {ground_truth_topic}'
            + (f' or {ground_truth_file}' if ground_truth_file else '')
        )
        self.get_logger().info(f'Logs: {self._run_directory}')

    def _registry_callback(self, message):
        self._registry = message
        self._evaluate_if_ready()

    def _ground_truth_callback(self, message):
        self._ground_truth = message
        self._evaluate_if_ready()

    def _evaluate_if_ready(self):
        if self._registry is None or self._ground_truth is None:
            return
        if self._registry.header.frame_id != self._map_frame:
            self.get_logger().warning(
                f'Registry frame is {self._registry.header.frame_id}; '
                f'expected {self._map_frame}'
            )
            return
        if self._ground_truth.header.frame_id != self._map_frame:
            self.get_logger().warning(
                f'Ground-truth frame is {self._ground_truth.header.frame_id}; '
                f'expected {self._map_frame}'
            )
            return

        self._evaluation_index += 1
        estimates = self._registry.detections
        ground_truth = self._ground_truth.detections
        metrics = evaluate_detections(
            estimates,
            ground_truth,
            self._maximum_match_distance,
        )
        rows = evaluation_rows(metrics, estimates, ground_truth)
        ros_time = (
            self._registry.header.stamp.sec
            + self._registry.header.stamp.nanosec / 1_000_000_000
        )
        with self._csv_path.open(
            'a', newline='', encoding='utf-8'
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            for row in rows:
                writer.writerow({
                    **row,
                    'evaluation_index': self._evaluation_index,
                    'ros_time': ros_time,
                })

        public_metrics = {
            key: value for key, value in metrics.items()
            if key not in {
                'matches',
                'false_positive_indices',
                'missed_indices',
            }
        }
        summary = {
            'evaluation_index': self._evaluation_index,
            'ros_time': ros_time,
            'frame_id': self._map_frame,
            'maximum_match_distance_m': self._maximum_match_distance,
            **public_metrics,
        }
        summary_text = json.dumps(summary, indent=2, sort_keys=True)
        temporary_path = self._summary_path.with_suffix('.json.tmp')
        temporary_path.write_text(summary_text + '\n', encoding='utf-8')
        temporary_path.replace(self._summary_path)

        summary_message = String()
        summary_message.data = json.dumps(summary, separators=(',', ':'))
        self._summary_pub.publish(summary_message)

        error = summary['mean_planar_error_m']
        error_text = 'n/a' if error is None else f'{error:.3f} m'
        accuracy = summary['classification_accuracy']
        accuracy_text = 'n/a' if accuracy is None else f'{accuracy:.1%}'
        self.get_logger().info(
            f'Evaluation {self._evaluation_index}: '
            f'matched={summary["matched_count"]}/'
            f'{summary["ground_truth_count"]}, '
            f'mean planar error={error_text}, '
            f'class accuracy={accuracy_text}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = SpatialEvaluator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

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
import xml.etree.ElementTree as ElementTree

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

# Gazebo model URI name -> evaluated class. Every heat source in the world is
# listed, because the registry stores campfires as well as animals.
WORLD_MODEL_CLASSES = {
    'boar': 'boar',
    'male_deer': 'deer',
    'female_deer': 'deer',
    'kangaroo': 'kangaroo',
    'campfire': 'campfire',
}

# Heat sources that are not animals. Such a truth object counts as correctly
# classified when its estimate carries no animal label.
DEFAULT_NON_ANIMAL_CLASSES = ('campfire',)


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


def class_matches(estimate_class, truth_class, animal_classes,
                  non_animal_classes):
    """Return whether an estimate label is correct for a truth class."""
    # An animal needs its exact label. A non-animal heat source (campfire) is
    # correct when the estimate has no animal label, e.g. unknown_heat_source.
    estimate_class = estimate_class.lower()
    truth_class = truth_class.lower()
    if truth_class in non_animal_classes:
        return estimate_class not in animal_classes
    return estimate_class == truth_class


def per_class_metrics(ground_truth, matches, class_results):
    """Return detection and classification recall for each truth class."""
    per_class = {}
    for truth in ground_truth:
        entry = per_class.setdefault(detection_class(truth).lower(), {
            'ground_truth': 0,
            'detected': 0,
            'correctly_classified': 0,
        })
        entry['ground_truth'] += 1
    for (_, truth_index, _), correct in zip(matches, class_results):
        entry = per_class[detection_class(ground_truth[truth_index]).lower()]
        entry['detected'] += 1
        entry['correctly_classified'] += int(correct)
    for entry in per_class.values():
        entry['detection_recall'] = entry['detected'] / entry['ground_truth']
        entry['classification_recall'] = (
            entry['correctly_classified'] / entry['ground_truth']
        )
    return per_class


def evaluate_detections(estimates, ground_truth, maximum_distance,
                        non_animal_classes=DEFAULT_NON_ANIMAL_CLASSES,
                        localization_tolerance=1.0):
    """Compute localization, detection, and classification metrics."""
    matches, false_positives, missed = match_detections(
        estimates,
        ground_truth,
        maximum_distance,
    )
    non_animal_classes = {name.lower() for name in non_animal_classes}
    animal_classes = {
        detection_class(truth).lower() for truth in ground_truth
    } - non_animal_classes
    planar_errors = [item[2] for item in matches]
    spatial_errors = [
        spatial_distance(estimates[estimate_index], ground_truth[truth_index])
        for estimate_index, truth_index, _ in matches
    ]
    class_results = [
        class_matches(
            detection_class(estimates[estimate_index]),
            detection_class(ground_truth[truth_index]),
            animal_classes,
            non_animal_classes,
        )
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
        # R7: matched candidates whose planar error is within tolerance.
        'localization_tolerance_m': localization_tolerance,
        'within_tolerance_count': sum(
            error <= localization_tolerance for error in planar_errors
        ),
        # R10 reads per_class['boar'].
        'per_class': per_class_metrics(ground_truth, matches, class_results),
        'matches': matches,
        'class_results': class_results,
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
    return ground_truth_message(frame_id, payload.get('objects', []))


def ground_truth_message(frame_id, objects):
    """Convert JSON-format truth objects into a Detection3DArray."""
    message = Detection3DArray()
    message.header.frame_id = frame_id
    for item in objects:
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


def ground_truth_objects_from_world(path, model_classes=None):
    """Return truth objects for every heat source included in an SDF world."""
    # Reads each <include> whose model://<name> URI is in model_classes and
    # uses its <pose> (x y z roll pitch yaw). The Husky spawns at the world
    # origin with zero yaw, so SLAM's map frame coincides with the world
    # frame. Layouts moved at runtime (random_animal_spawn.py) are not seen.
    if model_classes is None:
        model_classes = WORLD_MODEL_CLASSES
    world = ElementTree.parse(Path(path).expanduser()).getroot().find('world')
    if world is None:
        raise ValueError(f'{path} contains no <world> element')

    objects = []
    for include in world.findall('include'):
        uri = (include.findtext('uri') or '').strip()
        model = uri.split('://', 1)[-1].strip('/').split('/')[0]
        if model not in model_classes:
            continue
        pose = [float(value) for value in
                (include.findtext('pose') or '0 0 0').split()]
        x, y, z = (pose + [0.0, 0.0, 0.0])[:3]
        objects.append({
            'id': (include.findtext('name') or model).strip(),
            'class_name': model_classes[model],
            'position': {'x': x, 'y': y, 'z': z},
        })
    return objects


def load_world_ground_truth(path, frame_id, model_classes=None):
    """Load ground truth from an SDF world into a Detection3DArray."""
    return ground_truth_message(
        frame_id, ground_truth_objects_from_world(path, model_classes))


def evaluation_rows(metrics, estimates, ground_truth):
    """Return detailed matched and unmatched rows for one evaluation."""
    rows = []
    for (estimate_index, truth_index, planar_error), class_correct in zip(
        metrics['matches'], metrics['class_results']
    ):
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
            'class_correct': class_correct,
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
        # SDF world used when ground_truth_file is empty.
        self.declare_parameter('ground_truth_world', '')
        self.declare_parameter('map_frame', 'husky1_map')
        self.declare_parameter('maximum_match_distance', 2.0)
        self.declare_parameter('localization_tolerance', 1.0)       # R7
        self.declare_parameter(
            'non_animal_classes', list(DEFAULT_NON_ANIMAL_CLASSES))
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
        ground_truth_world = str(
            self.get_parameter('ground_truth_world').value
        )
        self._map_frame = str(self.get_parameter('map_frame').value)
        self._maximum_match_distance = float(
            self.get_parameter('maximum_match_distance').value
        )
        self._localization_tolerance = float(
            self.get_parameter('localization_tolerance').value
        )
        self._non_animal_classes = [
            str(name) for name in
            self.get_parameter('non_animal_classes').value
        ]
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
        ground_truth_source = ground_truth_file or ground_truth_world
        if ground_truth_file:
            self._ground_truth = load_ground_truth(ground_truth_file)
        elif ground_truth_world:
            self._ground_truth = load_world_ground_truth(
                ground_truth_world, self._map_frame)

        self.get_logger().info('Spatial evaluator started')
        self.get_logger().info(f'Registry input: {registry_topic}')
        self.get_logger().info(
            f'Ground truth: {ground_truth_topic}'
            + (f' or {ground_truth_source}' if ground_truth_source else '')
        )
        if self._ground_truth is not None:
            counts = {}
            for truth in self._ground_truth.detections:
                name = detection_class(truth)
                counts[name] = counts.get(name, 0) + 1
            self.get_logger().info(
                f'Loaded {len(self._ground_truth.detections)} truth objects: '
                + ', '.join(f'{k}={v}' for k, v in sorted(counts.items()))
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
            self._non_animal_classes,
            self._localization_tolerance,
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
                'class_results',
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
        boar = summary['per_class'].get('boar')
        boar_text = (
            'n/a' if boar is None else
            f'{boar["correctly_classified"]}/{boar["ground_truth"]}'
        )
        self.get_logger().info(
            f'Evaluation {self._evaluation_index}: '
            f'matched={summary["matched_count"]}/'
            f'{summary["ground_truth_count"]}, '
            f'mean planar error={error_text}, '
            f'within {self._localization_tolerance:.1f} m='
            f'{summary["within_tolerance_count"]}, '
            f'class accuracy={accuracy_text}, '
            f'boars confirmed={boar_text}'
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

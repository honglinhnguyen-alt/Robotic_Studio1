#!/usr/bin/env python3
"""Detect persistent hot regions in the Husky thermal camera image.

Input:
  thermal/image_raw                         sensor_msgs/msg/Image (L16)

Outputs:
  thermal/image_mono8                       sensor_msgs/msg/Image
  perception/thermal/detections             vision_msgs/msg/Detection2DArray
  perception/thermal/annotated_image        sensor_msgs/msg/Image
  perception/thermal/candidate_active       std_msgs/msg/Bool
  perception/thermal/max_temperature        sensor_msgs/msg/Temperature

All topic names are relative to the node namespace. This node detects heat
only; it does not decide where the robot should navigate.

Each hot region is tracked across frames and published with a stable
image-space track ID (``heat_7``). A region must be seen in
``persistence_frames`` consecutive frames before it is published (R5), and a
track keeps its ID through ``tracking_max_missed_frames`` missing frames. These
track IDs let the YOLO node vote for the same animal in every frame; map-level
candidate IDs are still assigned by the candidate registry.
"""

from typing import Dict, List

import cv2
import cv_bridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Temperature
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from perception_common import HotRegionTracker, short_tag
from perception_overlay import (
    CONFIRMED_HEAT_COLOUR,
    TENTATIVE_HEAT_COLOUR,
    LabelPlacer,
    draw_legend,
    draw_tag,
)


class ThermalCandidateDetector(Node):
    """Convert L16 data to Kelvin and publish stable hot-region detections."""

    def __init__(self) -> None:
        super().__init__('thermal_candidate_detector')

        self.declare_parameter('image_topic', 'thermal/image_raw')
        self.declare_parameter('normalized_topic', 'thermal/image_mono8')
        self.declare_parameter(
            'detections_topic',
            'perception/thermal/detections',
        )
        self.declare_parameter(
            'annotated_topic',
            'perception/thermal/annotated_image',
        )
        self.declare_parameter(
            'candidate_active_topic',
            'perception/thermal/candidate_active',
        )
        self.declare_parameter(
            'max_temperature_topic',
            'perception/thermal/max_temperature',
        )
        self.declare_parameter('kelvin_per_count', 0.01)
        self.declare_parameter('display_min_k', 285.0)
        self.declare_parameter('display_max_k', 335.0)
        self.declare_parameter('hot_threshold_k', 315.0)
        self.declare_parameter('min_hot_area_px', 30)
        self.declare_parameter('morphology_kernel_px', 3)
        # R5: a hot region must be seen in this many consecutive frames.
        self.declare_parameter('persistence_frames', 3)
        self.declare_parameter('max_regions', 8)
        self.declare_parameter('tracking_iou_threshold', 0.2)
        self.declare_parameter('tracking_max_missed_frames', 2)
        # Also match a region whose centre moved less than this many box
        # diagonals (small, distant blobs lose IoU quickly).
        self.declare_parameter('tracking_max_center_jump', 1.0)
        self.declare_parameter('publish_annotated_image', True)
        self.declare_parameter('log_period_sec', 2.0)

        self.bridge = cv_bridge.CvBridge()
        self.tracker = HotRegionTracker(
            iou_threshold=float(self.get_parameter('tracking_iou_threshold').value),
            max_missed_frames=int(
                self.get_parameter('tracking_max_missed_frames').value
            ),
            persistence_frames=int(self.get_parameter('persistence_frames').value),
            max_center_jump=float(
                self.get_parameter('tracking_max_center_jump').value
            ),
        )
        self.last_log_ns = self.get_clock().now().nanoseconds

        image_topic = str(self.get_parameter('image_topic').value)
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.normalized_pub = self.create_publisher(
            Image,
            str(self.get_parameter('normalized_topic').value),
            qos_profile_sensor_data,
        )
        self.detections_pub = self.create_publisher(
            Detection2DArray,
            str(self.get_parameter('detections_topic').value),
            10,
        )
        self.annotated_pub = self.create_publisher(
            Image,
            str(self.get_parameter('annotated_topic').value),
            qos_profile_sensor_data,
        )
        self.candidate_active_pub = self.create_publisher(
            Bool,
            str(self.get_parameter('candidate_active_topic').value),
            10,
        )
        self.max_temperature_pub = self.create_publisher(
            Temperature,
            str(self.get_parameter('max_temperature_topic').value),
            10,
        )

        threshold = float(self.get_parameter('hot_threshold_k').value)
        minimum_area = int(self.get_parameter('min_hot_area_px').value)
        persistence = int(self.get_parameter('persistence_frames').value)
        self.get_logger().info(
            'Thermal candidate detector ready | '
            f'input={self.resolve_topic_name(image_topic)} | '
            f'hot_threshold={threshold:.2f} K | '
            f'min_area={minimum_area} px | '
            f'persistence={persistence} frames'
        )

    def _image_callback(self, msg: Image) -> None:
        try:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except cv_bridge.CvBridgeError as exc:
            self.get_logger().error(f'Thermal image conversion failed: {exc}')
            return

        raw_array = np.asarray(raw)
        if raw_array.ndim != 2:
            self.get_logger().error(
                f'Expected a single-channel thermal image, got {raw_array.shape}.'
            )
            return

        if np.issubdtype(raw_array.dtype, np.floating):
            temperature = raw_array.astype(np.float32)
        else:
            scale = float(self.get_parameter('kelvin_per_count').value)
            temperature = raw_array.astype(np.float32) * scale

        finite_temperature = np.nan_to_num(
            temperature,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        regions = self._find_hot_regions(finite_temperature)
        tracks = self.tracker.update(
            [
                (
                    region['x'],
                    region['y'],
                    region['x'] + region['width'],
                    region['y'] + region['height'],
                )
                for region in regions
            ],
            regions,
        )
        # Only confirmed tracks seen in this frame are published, so every box
        # is current. A track that misses a frame keeps its ID in the tracker
        # and is published again with the same ID when it reappears.
        confirmed_tracks = [
            track for track in tracks
            if track['confirmed'] and track['observed']
        ]
        persistence = self.tracker.persistence_frames

        self._publish_normalized(msg, finite_temperature)
        self._publish_detections(msg, confirmed_tracks)

        active_msg = Bool()
        active_msg.data = bool(confirmed_tracks)
        self.candidate_active_pub.publish(active_msg)

        max_temperature = float(finite_temperature.max())
        temperature_msg = Temperature()
        temperature_msg.header = msg.header
        temperature_msg.temperature = max_temperature
        temperature_msg.variance = 0.0
        self.max_temperature_pub.publish(temperature_msg)

        if bool(self.get_parameter('publish_annotated_image').value):
            self._publish_annotated(
                msg,
                finite_temperature,
                tracks,
                persistence=persistence,
            )

        now_ns = self.get_clock().now().nanoseconds
        log_period_ns = int(
            max(0.1, float(self.get_parameter('log_period_sec').value)) * 1e9
        )
        if now_ns - self.last_log_ns >= log_period_ns:
            self.get_logger().info(
                'Thermal frame | '
                f'min={float(finite_temperature.min()):.2f} K | '
                f'mean={float(finite_temperature.mean()):.2f} K | '
                f'max={max_temperature:.2f} K | '
                f'hot_regions={len(confirmed_tracks)} '
                f'[{", ".join(str(track["id"]) for track in confirmed_tracks)}]'
            )
            self.last_log_ns = now_ns

    def _find_hot_regions(self, temperature: np.ndarray) -> List[Dict[str, float]]:
        threshold = float(self.get_parameter('hot_threshold_k').value)
        mask = (temperature >= threshold).astype(np.uint8) * 255

        kernel_size = max(
            1,
            int(self.get_parameter('morphology_kernel_px').value),
        )
        if kernel_size > 1:
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        component_count, labels, stats, _centroids = (
            cv2.connectedComponentsWithStats(mask, connectivity=8)
        )
        minimum_area = max(
            1,
            int(self.get_parameter('min_hot_area_px').value),
        )
        regions: List[Dict[str, float]] = []
        for label in range(1, component_count):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue

            component_values = temperature[labels == label]
            regions.append({
                'x': float(x),
                'y': float(y),
                'width': float(width),
                'height': float(height),
                'area': float(area),
                'max_temperature': float(component_values.max()),
                'mean_temperature': float(component_values.mean()),
            })

        regions.sort(key=lambda region: region['area'], reverse=True)
        maximum_regions = max(1, int(self.get_parameter('max_regions').value))
        return regions[:maximum_regions]

    def _publish_normalized(self, source_msg: Image, temperature: np.ndarray) -> None:
        display_min = float(self.get_parameter('display_min_k').value)
        display_max = float(self.get_parameter('display_max_k').value)
        if display_max <= display_min:
            self.get_logger().error('display_max_k must be greater than display_min_k.')
            return
        normalized = np.clip(
            (temperature - display_min) / (display_max - display_min),
            0.0,
            1.0,
        )
        mono8 = (normalized * 255.0).astype(np.uint8)
        output_msg = self.bridge.cv2_to_imgmsg(mono8, encoding='mono8')
        output_msg.header = source_msg.header
        self.normalized_pub.publish(output_msg)

    def _publish_detections(
        self,
        source_msg: Image,
        tracks: List[Dict[str, object]],
    ) -> None:
        output = Detection2DArray()
        output.header = source_msg.header
        threshold = float(self.get_parameter('hot_threshold_k').value)
        display_max = float(self.get_parameter('display_max_k').value)
        score_denominator = max(1e-6, display_max - threshold)

        for track in tracks:
            region = track['data']
            x1, y1, x2, y2 = track['box']
            detection = Detection2D()
            detection.header = source_msg.header
            detection.id = str(track['id'])
            self._set_bbox(
                detection,
                center_x=(x1 + x2) / 2.0,
                center_y=(y1 + y2) / 2.0,
                width=x2 - x1,
                height=y2 - y1,
            )
            hypothesis = ObjectHypothesisWithPose()
            confidence = np.clip(
                (float(region['max_temperature']) - threshold) / score_denominator,
                0.0,
                1.0,
            )
            self._set_hypothesis(hypothesis, 'heat_source', float(confidence))
            detection.results.append(hypothesis)
            output.detections.append(detection)

        # Empty arrays are intentionally published so the YOLO gate can re-arm.
        self.detections_pub.publish(output)

    def _publish_annotated(
        self,
        source_msg: Image,
        temperature: np.ndarray,
        tracks: List[Dict[str, object]],
        persistence: int,
    ) -> None:
        display_min = float(self.get_parameter('display_min_k').value)
        display_max = float(self.get_parameter('display_max_k').value)
        denominator = max(1e-6, display_max - display_min)
        mono8 = (
            np.clip((temperature - display_min) / denominator, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        annotated = cv2.applyColorMap(mono8, cv2.COLORMAP_INFERNO)
        height, width = annotated.shape[:2]

        legend_lines = []
        for track in tracks:
            tag = short_tag(str(track['id']))
            temperature_k = float(track['data'].get('max_temperature', 0.0))
            if track['confirmed']:
                state = 'confirmed' if track['observed'] else 'coasting'
                legend_lines.append(
                    (f'{tag} {temperature_k:.0f} K {state}', CONFIRMED_HEAT_COLOUR)
                )
            else:
                legend_lines.append((
                    f'{tag} {temperature_k:.0f} K {track["streak"]}/{persistence}',
                    TENTATIVE_HEAT_COLOUR,
                ))
        legend_rect = draw_legend(
            annotated,
            legend_lines[:8],
            origin=(3, 3),
            scale=0.33,
            pad=3,
        )

        placer = LabelPlacer(width, height, reserved=[legend_rect])
        # Boxes only for regions seen in this frame (a coasting track is only
        # listed in the legend, its last box would be out of date).
        visible = [track for track in tracks if track['observed']]
        for track in visible:
            colour = (
                CONFIRMED_HEAT_COLOUR if track['confirmed']
                else TENTATIVE_HEAT_COLOUR
            )
            x1, y1, x2, y2 = (int(round(value)) for value in track['box'])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 1)
        for track in visible:
            colour = (
                CONFIRMED_HEAT_COLOUR if track['confirmed']
                else TENTATIVE_HEAT_COLOUR
            )
            draw_tag(
                annotated,
                placer,
                short_tag(str(track['id'])),
                track['box'],
                colour,
                scale=0.33,
                pad=1,
            )

        output_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        output_msg.header = source_msg.header
        self.annotated_pub.publish(output_msg)

    @staticmethod
    def _set_bbox(
        detection: Detection2D,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
    ) -> None:
        center = detection.bbox.center
        if hasattr(center, 'position'):
            center.position.x = center_x
            center.position.y = center_y
            center.theta = 0.0
        else:
            center.x = center_x
            center.y = center_y
            center.theta = 0.0
        detection.bbox.size_x = width
        detection.bbox.size_y = height

    @staticmethod
    def _set_hypothesis(
        hypothesis: ObjectHypothesisWithPose,
        class_name: str,
        score: float,
    ) -> None:
        if hasattr(hypothesis, 'hypothesis'):
            hypothesis.hypothesis.class_id = class_name
            hypothesis.hypothesis.score = score
        else:
            hypothesis.id = class_name
            hypothesis.score = score


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ThermalCandidateDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
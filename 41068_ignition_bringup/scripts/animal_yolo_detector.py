#!/usr/bin/env python3
"""YOLO RGB perception node with optional thermal-triggered inference.

Inputs:
  camera/image                              sensor_msgs/msg/Image
  perception/thermal/detections             vision_msgs/msg/Detection2DArray

Outputs:
  perception/rgb/detections                 vision_msgs/msg/Detection2DArray
  perception/rgb/annotated_image            sensor_msgs/msg/Image
  perception/rgb/confirmation_image         sensor_msgs/msg/Image
  perception/thermal_yolo/detections        vision_msgs/msg/Detection2DArray
  perception/thermal_yolo/result            std_msgs/msg/String (JSON)
  Yolo_result                               std_msgs/msg/String (legacy CSV)

Service:
  perception/rgb/classify_now               std_srvs/srv/Trigger

When ``thermal_gate_enabled`` is true, YOLO remains idle until the thermal
detector publishes at least one hot region. The hot region is projected into
the RGB image using both cameras' fields of view; only a YOLO box associated
with that region is accepted as the heat-source classification.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import cv_bridge
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from perception_common import (
    associate_detections,
    category_for_class,
    center_size_to_xyxy,
    project_box_between_cameras,
)


PACKAGE_NAME = '41068_ignition_bringup'


class AnimalYoloDetector(Node):
    """Run the trained animal model continuously, on demand, or after heat."""

    def __init__(self) -> None:
        super().__init__('animal_yolo_detector')

        default_model = str(
            Path(get_package_share_directory(PACKAGE_NAME))
            / 'weights'
            / 'my_model.pt'
        )

        self.declare_parameter('model_path', default_model)
        self.declare_parameter('image_topic', 'camera/image')
        self.declare_parameter('confidence', 0.50)
        self.declare_parameter('iou', 0.70)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('continuous', True)
        self.declare_parameter('max_inference_hz', 3.0)
        self.declare_parameter('device', '')
        self.declare_parameter('publish_annotated_image', True)

        # Thermal-gated mode takes precedence over continuous mode so the same
        # RGB frame is not inferred twice.
        self.declare_parameter('thermal_gate_enabled', False)
        self.declare_parameter(
            'thermal_detections_topic',
            'perception/thermal/detections',
        )
        self.declare_parameter('thermal_image_width', 320)
        self.declare_parameter('thermal_image_height', 240)
        self.declare_parameter('thermal_hfov_deg', 60.0)
        self.declare_parameter('rgb_hfov_deg', 120.0)
        self.declare_parameter('thermal_roi_margin', 1.5)
        self.declare_parameter('thermal_rgb_offset_x_px', 0.0)
        self.declare_parameter('thermal_rgb_offset_y_px', 0.0)
        self.declare_parameter('minimum_thermal_overlap', 0.05)
        self.declare_parameter('max_rgb_thermal_age_sec', 1.0)
        self.declare_parameter('thermal_retrigger_sec', 0.0)
        self.declare_parameter('verification_frames', 5)
        self.declare_parameter('verification_min_votes', 3)

        # The supplied model contains boar, deer, and kangaroo. These arrays are
        # parameters so the category policy can change without a code edit.
        self.declare_parameter('native_classes', ['kangaroo'])
        self.declare_parameter('exotic_classes', ['boar', 'deer'])
        self.declare_parameter(
            'fused_detections_topic',
            'perception/thermal_yolo/detections',
        )
        self.declare_parameter(
            'classification_topic',
            'perception/thermal_yolo/result',
        )
        self.declare_parameter('publish_legacy_result', True)
        self.declare_parameter('legacy_result_topic', 'Yolo_result')

        model_path = Path(str(self.get_parameter('model_path').value)).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f'YOLO weights not found: {model_path}')

        self.bridge = cv_bridge.CvBridge()
        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(str(model_path))
        self.get_logger().info(f'YOLO classes: {self.model.names}')

        self.detections_pub = self.create_publisher(
            Detection2DArray,
            'perception/rgb/detections',
            10,
        )
        self.annotated_pub = self.create_publisher(
            Image,
            'perception/rgb/annotated_image',
            10,
        )
        self.confirmation_pub = self.create_publisher(
            Image,
            'perception/rgb/confirmation_image',
            10,
        )
        self.fused_detections_pub = self.create_publisher(
            Detection2DArray,
            str(self.get_parameter('fused_detections_topic').value),
            10,
        )
        self.classification_pub = self.create_publisher(
            String,
            str(self.get_parameter('classification_topic').value),
            10,
        )
        self.legacy_result_pub = self.create_publisher(
            String,
            str(self.get_parameter('legacy_result_topic').value),
            10,
        )

        image_topic = str(self.get_parameter('image_topic').value)
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.classify_service = self.create_service(
            Trigger,
            'perception/rgb/classify_now',
            self._classify_now_callback,
        )

        self.thermal_sub = None
        thermal_gate_enabled = bool(
            self.get_parameter('thermal_gate_enabled').value
        )
        if thermal_gate_enabled:
            thermal_topic = str(
                self.get_parameter('thermal_detections_topic').value
            )
            self.thermal_sub = self.create_subscription(
                Detection2DArray,
                thermal_topic,
                self._thermal_callback,
                10,
            )
        else:
            thermal_topic = '(disabled)'

        self.latest_image_msg: Optional[Image] = None
        self.latest_frame_number = 0
        self.last_processed_frame_number = -1
        self.latest_thermal_msg: Optional[Detection2DArray] = None
        self.thermal_active = False
        self.thermal_candidate_ids: List[str] = []
        self.last_thermal_trigger_ns = -10**30
        self.inference_running = False
        self.verification_active = False
        self.verification_candidate_ids: List[str] = []
        self.verification_votes: Dict[str, List[Dict[str, object]]] = {}
        self.verification_frames_seen = 0
        self.verification_last_frame_number = -1
        self.verification_last_inference_ns = -10**30

        max_hz = max(0.1, float(self.get_parameter('max_inference_hz').value))
        self.inference_timer = self.create_timer(
            1.0 / max_hz,
            self._continuous_tick,
        )

        if thermal_gate_enabled and bool(self.get_parameter('continuous').value):
            self.get_logger().warning(
                'thermal_gate_enabled is true, so thermal gating takes '
                'precedence over continuous inference.'
            )

        mode = 'thermal-gated' if thermal_gate_enabled else (
            'continuous' if bool(self.get_parameter('continuous').value)
            else 'service-only'
        )
        result_topic = str(self.get_parameter('classification_topic').value)
        thermal_resolved = (
            self.resolve_topic_name(thermal_topic)
            if thermal_gate_enabled
            else thermal_topic
        )
        self.get_logger().info(
            'Animal YOLO detector ready | '
            f'mode={mode} | '
            f'rgb={self.resolve_topic_name(image_topic)} | '
            f'thermal={thermal_resolved} | '
            f'result={self.resolve_topic_name(result_topic)} | '
            f'verification={self._verification_frame_target()} frames/'
            f'{self._verification_min_votes(self._verification_frame_target())} votes'
        )

    def _image_callback(self, msg: Image) -> None:
        """Keep the newest RGB frame and feed an active verification window."""
        self.latest_image_msg = msg
        self.latest_frame_number += 1
        if self.verification_active:
            self._maybe_run_verification()

    def _thermal_callback(self, msg: Detection2DArray) -> None:
        """Start or update a multi-frame verification for every heat region."""
        if not msg.detections:
            self.thermal_active = False
            self.thermal_candidate_ids = []
            self.latest_thermal_msg = None
            self._reset_verification()
            return

        candidate_ids = self._candidate_ids(msg)
        previous_ids = set(self.thermal_candidate_ids)
        candidates_changed = (
            self.thermal_active
            and previous_ids
            and set(candidate_ids) != previous_ids
        )
        self.thermal_candidate_ids = candidate_ids
        self.latest_thermal_msg = msg
        now_ns = self.get_clock().now().nanoseconds
        retrigger_sec = max(
            0.0,
            float(self.get_parameter('thermal_retrigger_sec').value),
        )
        retrigger_due = (
            retrigger_sec > 0.0
            and now_ns - self.last_thermal_trigger_ns >= int(retrigger_sec * 1e9)
        )
        if not self.thermal_active:
            self.thermal_active = True
            self._start_verification(msg, reason='new heat event')
        elif candidates_changed:
            self._start_verification(
                msg,
                reason='thermal candidate set changed',
            )
        elif not self.verification_active and retrigger_due:
            self._start_verification(msg, reason='periodic retrigger')

        # A current thermal message keeps the RGB/thermal timestamp comparison
        # valid while the five-frame RGB verification is in progress.
        if self.verification_active:
            self._maybe_run_verification()

    def _start_verification(
        self,
        thermal_msg: Detection2DArray,
        reason: str,
    ) -> None:
        self.latest_thermal_msg = thermal_msg
        self.verification_candidate_ids = self._candidate_ids(thermal_msg)
        self.verification_votes = {
            candidate_id: [] for candidate_id in self.verification_candidate_ids
        }
        self.verification_frames_seen = 0
        # Require the next RGB callback, rather than reusing an older cached
        # frame that may have been captured before the heat was confirmed.
        self.verification_last_frame_number = self.latest_frame_number
        self.verification_last_inference_ns = -10**30
        self.verification_active = True
        frame_count = self._verification_frame_target()
        required_votes = self._verification_min_votes(frame_count)
        self.get_logger().info(
            f'Starting YOLO verification ({reason}) for '
            f'{len(self.verification_candidate_ids)} thermal region(s): '
            f'{frame_count} fresh RGB frames, {required_votes} matching votes required.'
        )

    def _reset_verification(self) -> None:
        self.verification_active = False
        self.verification_candidate_ids = []
        self.verification_votes = {}
        self.verification_frames_seen = 0

    def _maybe_run_verification(self) -> None:
        if (
            not self.verification_active
            or self.latest_thermal_msg is None
            or self.latest_image_msg is None
            or self.inference_running
        ):
            return
        if self.latest_frame_number == self.verification_last_frame_number:
            return

        now_ns = self.get_clock().now().nanoseconds
        maximum_hz = max(
            0.1,
            float(self.get_parameter('max_inference_hz').value),
        )
        minimum_interval_ns = int(1e9 / maximum_hz)
        if now_ns - self.verification_last_inference_ns < minimum_interval_ns:
            return

        thermal_msg = self.latest_thermal_msg
        rgb_msg = self.latest_image_msg
        max_age = max(
            0.0,
            float(self.get_parameter('max_rgb_thermal_age_sec').value),
        )
        thermal_stamp = self._stamp_seconds(thermal_msg.header.stamp)
        rgb_stamp = self._stamp_seconds(rgb_msg.header.stamp)
        if thermal_stamp > 0.0 and rgb_stamp > 0.0:
            difference = abs(rgb_stamp - thermal_stamp)
            if difference > max_age:
                # Wait for the lagging sensor rather than voting with an
                # unsynchronised pair of images.
                return

        self.verification_last_frame_number = self.latest_frame_number
        self.verification_last_inference_ns = now_ns
        self._run_inference(
            rgb_msg,
            publish_confirmation=False,
            thermal_detections=thermal_msg,
            collect_verification=True,
        )

    def _verification_frame_target(self) -> int:
        return max(1, int(self.get_parameter('verification_frames').value))

    def _verification_min_votes(self, frame_target: int) -> int:
        requested = max(
            1,
            int(self.get_parameter('verification_min_votes').value),
        )
        return min(frame_target, requested)

    def _continuous_tick(self) -> None:
        if bool(self.get_parameter('thermal_gate_enabled').value):
            return
        if not bool(self.get_parameter('continuous').value):
            return
        if self.latest_image_msg is None:
            return
        if self.latest_frame_number == self.last_processed_frame_number:
            return

        msg = self.latest_image_msg
        frame_number = self.latest_frame_number
        self._run_inference(msg, publish_confirmation=False)
        self.last_processed_frame_number = frame_number

    def _classify_now_callback(self, _request, response):
        """Classify the newest RGB frame after navigation reaches a viewpoint."""
        if self.latest_image_msg is None:
            response.success = False
            response.message = 'No RGB camera frame has been received yet.'
            return response
        if self.inference_running:
            response.success = False
            response.message = 'YOLO inference is already running.'
            return response

        labels = self._run_inference(
            self.latest_image_msg,
            publish_confirmation=True,
        )
        response.success = True
        response.message = (
            'Detected: ' + ', '.join(labels)
            if labels
            else 'No trained animal detected in the latest RGB frame.'
        )
        return response

    def _run_inference(
        self,
        source_msg: Image,
        publish_confirmation: bool,
        thermal_detections: Optional[Detection2DArray] = None,
        collect_verification: bool = False,
    ) -> List[str]:
        if self.inference_running:
            return []
        self.inference_running = True
        try:
            try:
                image_bgr = self.bridge.imgmsg_to_cv2(
                    source_msg,
                    desired_encoding='bgr8',
                )
            except cv_bridge.CvBridgeError as exc:
                self.get_logger().error(f'RGB image conversion failed: {exc}')
                if thermal_detections is not None and not collect_verification:
                    self._publish_error_result(
                        source_msg,
                        thermal_detections,
                        'rgb_conversion_failed',
                    )
                return []

            predict_kwargs = {
                'source': image_bgr,
                'conf': float(self.get_parameter('confidence').value),
                'iou': float(self.get_parameter('iou').value),
                'imgsz': int(self.get_parameter('imgsz').value),
                'verbose': False,
            }
            device = str(self.get_parameter('device').value).strip()
            if device:
                predict_kwargs['device'] = device

            try:
                result = self.model.predict(**predict_kwargs)[0]
            except Exception as exc:  # Ultralytics raises several backend types.
                self.get_logger().error(f'YOLO inference failed: {exc}')
                if thermal_detections is not None and not collect_verification:
                    self._publish_error_result(
                        source_msg,
                        thermal_detections,
                        'yolo_inference_failed',
                    )
                return []

            detections_msg = Detection2DArray()
            detections_msg.header = source_msg.header
            records: List[Dict[str, object]] = []

            if result.boxes is not None:
                xyxy = result.boxes.xyxy.detach().cpu().numpy()
                class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
                scores = result.boxes.conf.detach().cpu().numpy()
                for box, class_id, score_value in zip(xyxy, class_ids, scores):
                    x1, y1, x2, y2 = [float(value) for value in box]
                    class_name = str(result.names[int(class_id)])
                    score = float(score_value)
                    detection = self._make_detection(
                        source_msg,
                        class_name,
                        score,
                        (x1, y1, x2, y2),
                    )
                    detections_msg.detections.append(detection)
                    records.append({
                        'box': (x1, y1, x2, y2),
                        'class_name': class_name,
                        'score': score,
                        'detection': detection,
                    })

            # An empty array means "inference ran and found nothing".
            self.detections_pub.publish(detections_msg)

            if thermal_detections is not None:
                (
                    labels,
                    annotated,
                    result_items,
                    matched_records,
                ) = self._publish_thermal_fusion(
                    source_msg,
                    thermal_detections,
                    image_bgr,
                    records,
                    publish_result=not collect_verification,
                )
                if collect_verification:
                    verification_frame_number = self.verification_frames_seen + 1
                    consensus_image = self._record_verification_frame(
                        source_msg,
                        thermal_detections,
                        image_bgr,
                        result_items,
                        matched_records,
                    )
                    if consensus_image is not None:
                        annotated = consensus_image
            else:
                labels = [
                    f'{record["class_name"]} {float(record["score"]):.2f}'
                    for record in records
                ]
                annotated = result.plot(show=False)

            publish_image = bool(
                self.get_parameter('publish_annotated_image').value
            )
            if publish_image or publish_confirmation:
                annotated_msg = self.bridge.cv2_to_imgmsg(
                    annotated,
                    encoding='bgr8',
                )
                annotated_msg.header = source_msg.header
                if publish_image:
                    self.annotated_pub.publish(annotated_msg)
                if publish_confirmation:
                    self.confirmation_pub.publish(annotated_msg)

            if collect_verification:
                provisional = ', '.join(labels) if labels else 'no matches'
                self.get_logger().info(
                    f'YOLO verification frame '
                    f'{verification_frame_number}/'
                    f'{self._verification_frame_target()}: {provisional}'
                )
            elif labels:
                self.get_logger().info('Accepted classification: ' + ', '.join(labels))
            elif thermal_detections is not None:
                self.get_logger().warning(
                    'Thermal candidate was not matched to a trained YOLO class.'
                )
            return labels
        finally:
            self.inference_running = False

    def _publish_thermal_fusion(
        self,
        rgb_msg: Image,
        thermal_msg: Detection2DArray,
        image_bgr,
        records: List[Dict[str, object]],
        publish_result: bool = True,
    ) -> Tuple[
        List[str],
        object,
        List[Dict[str, object]],
        List[Optional[Dict[str, object]]],
    ]:
        image_height, image_width = image_bgr.shape[:2]
        thermal_boxes = [
            self._detection_box(detection)
            for detection in thermal_msg.detections
        ]
        candidate_ids = self._candidate_ids(thermal_msg)
        projected_rois = [
            project_box_between_cameras(
                box,
                source_size=(
                    int(self.get_parameter('thermal_image_width').value),
                    int(self.get_parameter('thermal_image_height').value),
                ),
                destination_size=(image_width, image_height),
                source_hfov_deg=float(
                    self.get_parameter('thermal_hfov_deg').value
                ),
                destination_hfov_deg=float(
                    self.get_parameter('rgb_hfov_deg').value
                ),
                margin=float(self.get_parameter('thermal_roi_margin').value),
                offset_pixels=(
                    float(self.get_parameter('thermal_rgb_offset_x_px').value),
                    float(self.get_parameter('thermal_rgb_offset_y_px').value),
                ),
            )
            for box in thermal_boxes
        ]
        rgb_boxes = [record['box'] for record in records]
        rgb_scores = [float(record['score']) for record in records]
        assignments = associate_detections(
            projected_rois,
            rgb_boxes,
            rgb_scores,
            minimum_overlap=float(
                self.get_parameter('minimum_thermal_overlap').value
            ),
        )

        native_classes = self.get_parameter('native_classes').value
        exotic_classes = self.get_parameter('exotic_classes').value
        fused_msg = Detection2DArray()
        fused_msg.header = rgb_msg.header
        result_items = []
        labels = []
        matched_records = []

        for region_index, (
            candidate_id,
            thermal_box,
            projected_roi,
            assignment,
        ) in enumerate(
            zip(candidate_ids, thermal_boxes, projected_rois, assignments)
        ):
            if assignment is None:
                result_items.append({
                    'thermal_region': region_index,
                    'candidate_id': candidate_id,
                    'status': 'unresolved',
                    'class_name': 'unknown_heat_source',
                    'category': 'unknown',
                    'confidence': 0.0,
                    'thermal_bbox': self._box_payload(thermal_box),
                    'projected_rgb_roi': self._box_payload(projected_roi),
                    'rgb_bbox': None,
                })
                matched_records.append(None)
                continue

            record = records[assignment]
            class_name = str(record['class_name'])
            score = float(record['score'])
            category = category_for_class(
                class_name,
                native_classes,
                exotic_classes,
            )
            record['detection'].id = candidate_id
            fused_msg.detections.append(record['detection'])
            result_items.append({
                'thermal_region': region_index,
                'candidate_id': candidate_id,
                'status': 'classified',
                'class_name': class_name,
                'category': category,
                'confidence': round(score, 6),
                'thermal_bbox': self._box_payload(thermal_box),
                'projected_rgb_roi': self._box_payload(projected_roi),
                'rgb_bbox': self._box_payload(record['box']),
            })
            labels.append(f'{class_name} {score:.2f} [{category}]')
            matched_records.append(record)

        if publish_result:
            self.fused_detections_pub.publish(fused_msg)
        overall_status = (
            'classified'
            if any(item['status'] == 'classified' for item in result_items)
            else 'unresolved'
        )
        payload = {
            'status': overall_status,
            'thermal_stamp': self._stamp_payload(thermal_msg.header.stamp),
            'rgb_stamp': self._stamp_payload(rgb_msg.header.stamp),
            'items': result_items,
        }
        if publish_result:
            result_msg = String()
            result_msg.data = json.dumps(
                payload,
                separators=(',', ':'),
                sort_keys=True,
            )
            self.classification_pub.publish(result_msg)

        if publish_result and bool(
            self.get_parameter('publish_legacy_result').value
        ):
            classified = [
                item for item in result_items
                if item['status'] == 'classified'
                and item['category'] in ('native', 'exotic')
            ]
            if classified:
                best = max(classified, key=lambda item: item['confidence'])
                legacy_msg = String()
                legacy_msg.data = (
                    f'{best["class_name"]},{best["confidence"]:.6f},'
                    f'{str(best["category"] == "exotic").lower()}'
                )
                self.legacy_result_pub.publish(legacy_msg)

        annotated = self._annotate_thermal_result(
            image_bgr,
            projected_rois,
            matched_records,
            candidate_ids,
        )
        return labels, annotated, result_items, matched_records

    def _record_verification_frame(
        self,
        rgb_msg: Image,
        thermal_msg: Detection2DArray,
        image_bgr,
        result_items: List[Dict[str, object]],
        matched_records: List[Optional[Dict[str, object]]],
    ):
        """Accumulate one vote per thermal candidate from one fresh RGB frame."""
        for item, record in zip(result_items, matched_records):
            candidate_id = str(item['candidate_id'])
            self.verification_votes.setdefault(candidate_id, [])
            if item['status'] != 'classified' or record is None:
                continue
            self.verification_votes[candidate_id].append({
                'class_name': str(item['class_name']),
                'category': str(item['category']),
                'confidence': float(item['confidence']),
                'record': record,
            })

        self.verification_frames_seen += 1
        if self.verification_frames_seen < self._verification_frame_target():
            return None

        return self._publish_verification_consensus(
            rgb_msg,
            thermal_msg,
            image_bgr,
            result_items,
        )

    def _publish_verification_consensus(
        self,
        rgb_msg: Image,
        thermal_msg: Detection2DArray,
        image_bgr,
        latest_items: List[Dict[str, object]],
    ):
        """Publish one majority-vote result containing every thermal region."""
        frame_count = self.verification_frames_seen
        required_votes = self._verification_min_votes(frame_count)
        latest_by_id = {
            str(item['candidate_id']): item for item in latest_items
        }
        result_items: List[Dict[str, object]] = []
        representative_records: List[Optional[Dict[str, object]]] = []
        fused_msg = Detection2DArray()
        fused_msg.header = rgb_msg.header
        log_items = []

        for region_index, candidate_id in enumerate(
            self.verification_candidate_ids
        ):
            latest = latest_by_id.get(candidate_id, {})
            votes = self.verification_votes.get(candidate_id, [])
            counts = Counter(str(vote['class_name']) for vote in votes)
            vote_summary = dict(sorted(counts.items()))

            winning_class = None
            winning_votes: List[Dict[str, object]] = []
            if counts:
                winning_class = max(
                    counts,
                    key=lambda class_name: (
                        counts[class_name],
                        sum(
                            float(vote['confidence'])
                            for vote in votes
                            if vote['class_name'] == class_name
                        ) / counts[class_name],
                    ),
                )
                winning_votes = [
                    vote for vote in votes
                    if vote['class_name'] == winning_class
                ]

            common = {
                'thermal_region': region_index,
                'candidate_id': candidate_id,
                'thermal_bbox': latest.get('thermal_bbox'),
                'projected_rgb_roi': latest.get('projected_rgb_roi'),
                'frames_evaluated': frame_count,
                'required_votes': required_votes,
                'vote_summary': vote_summary,
            }
            if winning_class is None or len(winning_votes) < required_votes:
                result_items.append({
                    **common,
                    'status': 'unresolved',
                    'class_name': 'unknown_heat_source',
                    'category': 'unknown',
                    'confidence': 0.0,
                    'agreement': round(
                        len(winning_votes) / max(1, frame_count),
                        6,
                    ),
                    'rgb_bbox': None,
                })
                representative_records.append(None)
                provisional = winning_class or 'no_match'
                log_items.append(
                    f'{candidate_id}=unresolved '
                    f'({provisional} {len(winning_votes)}/{frame_count} votes)'
                )
                continue

            representative_vote = max(
                winning_votes,
                key=lambda vote: float(vote['confidence']),
            )
            representative = representative_vote['record']
            mean_confidence = sum(
                float(vote['confidence']) for vote in winning_votes
            ) / len(winning_votes)
            consensus_detection = self._make_detection(
                rgb_msg,
                winning_class,
                mean_confidence,
                representative['box'],
            )
            consensus_detection.id = candidate_id
            display_record = dict(representative)
            display_record['score'] = mean_confidence
            display_record['detection'] = consensus_detection
            item = {
                **common,
                'status': 'classified',
                'class_name': winning_class,
                'category': str(representative_vote['category']),
                'confidence': round(mean_confidence, 6),
                'agreement': round(len(winning_votes) / frame_count, 6),
                'rgb_bbox': self._box_payload(representative['box']),
            }
            result_items.append(item)
            representative_records.append(display_record)
            fused_msg.detections.append(consensus_detection)
            log_items.append(
                f'{candidate_id}={winning_class} {mean_confidence:.2f} '
                f'({len(winning_votes)}/{frame_count} votes)'
            )

        self.fused_detections_pub.publish(fused_msg)
        classified_count = sum(
            item['status'] == 'classified' for item in result_items
        )
        payload = {
            'status': 'classified' if classified_count else 'unresolved',
            'classified_regions': classified_count,
            'total_regions': len(result_items),
            'verification_frames': frame_count,
            'required_votes': required_votes,
            'thermal_stamp': self._stamp_payload(thermal_msg.header.stamp),
            'rgb_stamp': self._stamp_payload(rgb_msg.header.stamp),
            'items': result_items,
        }
        result_msg = String()
        result_msg.data = json.dumps(
            payload,
            separators=(',', ':'),
            sort_keys=True,
        )
        self.classification_pub.publish(result_msg)

        # The legacy CSV contract represents only one mission candidate. Keep
        # publishing its best result for compatibility; the JSON topic above
        # is the authoritative multi-candidate interface.
        if bool(self.get_parameter('publish_legacy_result').value):
            classified = [
                item for item in result_items
                if item['status'] == 'classified'
                and item['category'] in ('native', 'exotic')
            ]
            if classified:
                best = max(
                    classified,
                    key=lambda item: float(item['confidence']),
                )
                legacy_msg = String()
                legacy_msg.data = (
                    f'{best["class_name"]},{float(best["confidence"]):.6f},'
                    f'{str(best["category"] == "exotic").lower()}'
                )
                self.legacy_result_pub.publish(legacy_msg)

        self.get_logger().info(
            f'Consensus classification after {frame_count} RGB frames: '
            + '; '.join(log_items)
        )
        projected_rois = [
            self._payload_box(item.get('projected_rgb_roi'))
            for item in result_items
        ]
        annotated = self._annotate_thermal_result(
            image_bgr,
            projected_rois,
            representative_records,
            self.verification_candidate_ids,
        )
        self.last_thermal_trigger_ns = self.get_clock().now().nanoseconds
        self._reset_verification()
        return annotated

    def _annotate_thermal_result(
        self,
        image_bgr,
        projected_rois: Sequence[Sequence[float]],
        matched_records: Sequence[Optional[Dict[str, object]]],
        candidate_ids: Optional[Sequence[str]] = None,
    ):
        annotated = image_bgr.copy()
        native_classes = self.get_parameter('native_classes').value
        exotic_classes = self.get_parameter('exotic_classes').value

        for region_index, (roi, record) in enumerate(
            zip(projected_rois, matched_records)
        ):
            rx1, ry1, rx2, ry2 = [int(round(value)) for value in roi]
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 165, 255), 2)
            cv2.putText(
                annotated,
                'THERMAL '
                + (
                    str(candidate_ids[region_index])
                    if candidate_ids is not None
                    and region_index < len(candidate_ids)
                    else f'ROI {region_index + 1}'
                ),
                (rx1, max(18, ry1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

            if record is None:
                cv2.putText(
                    annotated,
                    'UNKNOWN HEAT SOURCE',
                    (rx1, min(annotated.shape[0] - 8, ry2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                continue

            x1, y1, x2, y2 = [
                int(round(value)) for value in record['box']
            ]
            class_name = str(record['class_name'])
            score = float(record['score'])
            category = category_for_class(
                class_name,
                native_classes,
                exotic_classes,
            )
            colour = (0, 200, 0) if category == 'native' else (255, 100, 0)
            if category == 'unknown':
                colour = (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 3)
            cv2.putText(
                annotated,
                f'{class_name} {score:.2f} [{category.upper()}]',
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                colour,
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _publish_error_result(
        self,
        rgb_msg: Image,
        thermal_msg: Detection2DArray,
        reason: str,
    ) -> None:
        fused_msg = Detection2DArray()
        fused_msg.header = rgb_msg.header
        self.fused_detections_pub.publish(fused_msg)
        result_msg = String()
        result_msg.data = json.dumps({
            'status': 'error',
            'reason': reason,
            'thermal_stamp': self._stamp_payload(thermal_msg.header.stamp),
            'rgb_stamp': self._stamp_payload(rgb_msg.header.stamp),
            'items': [],
        }, separators=(',', ':'), sort_keys=True)
        self.classification_pub.publish(result_msg)

    @staticmethod
    def _make_detection(
        source_msg: Image,
        class_name: str,
        score: float,
        box: Sequence[float],
    ) -> Detection2D:
        x1, y1, x2, y2 = (float(value) for value in box)
        detection = Detection2D()
        detection.header = source_msg.header
        detection.id = ''
        AnimalYoloDetector._set_bbox(
            detection,
            center_x=(x1 + x2) / 2.0,
            center_y=(y1 + y2) / 2.0,
            width=max(0.0, x2 - x1),
            height=max(0.0, y2 - y1),
        )
        hypothesis = ObjectHypothesisWithPose()
        AnimalYoloDetector._set_hypothesis(
            hypothesis,
            class_name=class_name,
            score=score,
        )
        detection.results.append(hypothesis)
        return detection

    @staticmethod
    def _detection_box(detection: Detection2D) -> Tuple[float, float, float, float]:
        center = detection.bbox.center
        if hasattr(center, 'position'):
            center_x = float(center.position.x)
            center_y = float(center.position.y)
        else:
            center_x = float(center.x)
            center_y = float(center.y)
        return center_size_to_xyxy(
            center_x,
            center_y,
            float(detection.bbox.size_x),
            float(detection.bbox.size_y),
        )

    @staticmethod
    def _candidate_ids(msg: Detection2DArray) -> List[str]:
        """Return unique stable IDs, with a fallback for older thermal nodes."""
        identifiers = []
        used = set()
        for index, detection in enumerate(msg.detections):
            base = str(detection.id).strip() or f'thermal_{index + 1}'
            candidate_id = base
            duplicate_index = 2
            while candidate_id in used:
                candidate_id = f'{base}_{duplicate_index}'
                duplicate_index += 1
            identifiers.append(candidate_id)
            used.add(candidate_id)
        return identifiers

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

    @staticmethod
    def _stamp_seconds(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def _stamp_payload(stamp) -> Dict[str, int]:
        return {'sec': int(stamp.sec), 'nanosec': int(stamp.nanosec)}

    @staticmethod
    def _box_payload(box: Sequence[float]) -> Dict[str, float]:
        x1, y1, x2, y2 = (float(value) for value in box)
        return {
            'x1': round(x1, 3),
            'y1': round(y1, 3),
            'x2': round(x2, 3),
            'y2': round(y2, 3),
        }

    @staticmethod
    def _payload_box(payload) -> Tuple[float, float, float, float]:
        if not isinstance(payload, dict):
            return (0.0, 0.0, 0.0, 0.0)
        return (
            float(payload.get('x1', 0.0)),
            float(payload.get('y1', 0.0)),
            float(payload.get('x2', 0.0)),
            float(payload.get('y2', 0.0)),
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AnimalYoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

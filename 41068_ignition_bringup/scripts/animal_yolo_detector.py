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

Each RGB frame is paired with the thermal frame closest in time (the RGB
camera runs at 3 Hz and the thermal camera at 10 Hz), and frames captured
while the robot turns quickly are skipped, because the two images would no
longer show the same scene. A hot region that is too small in the RGB image
for YOLO is reported as ``too_far``, and one cut by the thermal image edge as
``partial``, instead of ``unknown_heat_source``. Neither gets a class vote.
"""

import json
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import cv_bridge
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from nav_msgs.msg import Odometry
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
    short_tag,
)
from perception_overlay import (
    THERMAL_ROI_COLOUR,
    LabelPlacer,
    colour_for_status,
    draw_legend,
    draw_tag,
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
        # Pair each RGB frame with the thermal frame closest in time; reject
        # pairs further apart than this.
        self.declare_parameter('max_rgb_thermal_age_sec', 0.15)
        self.declare_parameter('thermal_buffer_sec', 2.0)
        # Cancel the verification window after this many empty thermal frames.
        self.declare_parameter('thermal_empty_reset_frames', 3)
        # Skip verification frames while the robot turns faster than this
        # [rad/s]; 0 disables the check.
        self.declare_parameter('max_verification_yaw_rate', 0.35)
        self.declare_parameter('odometry_topic', 'odometry')
        # A projected hot region shorter than this in the RGB image is too
        # small for YOLO; it is reported as too_far, not unknown_heat_source.
        self.declare_parameter('min_classify_height_px', 18.0)
        # A hot region touching the left/right edge of the thermal image is
        # only partly visible, so its centre is biased and it could be matched
        # to a neighbouring animal that only the wider RGB camera sees. Such
        # regions are reported as partial and get no vote. 0 disables.
        self.declare_parameter('thermal_edge_margin_px', 1.0)
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

        self.odometry_sub = None
        if (
            thermal_gate_enabled
            and float(self.get_parameter('max_verification_yaw_rate').value) > 0.0
        ):
            self.odometry_sub = self.create_subscription(
                Odometry,
                str(self.get_parameter('odometry_topic').value),
                self._odometry_callback,
                qos_profile_sensor_data,
            )

        self.latest_image_msg: Optional[Image] = None
        self.latest_frame_number = 0
        self.last_processed_frame_number = -1
        self.latest_thermal_msg: Optional[Detection2DArray] = None
        self.thermal_buffer = deque()
        self.window_candidate_ids = set()
        self.latest_yaw_rate: Optional[float] = None
        self.empty_thermal_frames = 0
        self.last_turn_log_ns = -10**30
        self.last_pairing_log_ns = -10**30
        self.thermal_active = False
        self.thermal_candidate_ids: List[str] = []
        self.last_thermal_trigger_ns = -10**30
        self.inference_running = False
        self.verification_active = False
        self.verification_candidate_ids: List[str] = []
        self.verification_votes: Dict[str, List[Dict[str, object]]] = {}
        self.verification_status_counts: Dict[str, Counter] = {}
        self.verification_last_match: Dict[str, Tuple[int, Dict[str, object]]] = {}
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

    def _odometry_callback(self, msg: Odometry) -> None:
        self.latest_yaw_rate = abs(float(msg.twist.twist.angular.z))

    def _thermal_callback(self, msg: Detection2DArray) -> None:
        """Start or update a multi-frame verification for every heat region."""
        self._buffer_thermal(msg)
        if not msg.detections:
            # One empty frame (a source hidden for a moment) does not cancel
            # the verification window; several in a row do.
            self.empty_thermal_frames += 1
            if self.empty_thermal_frames >= max(
                1,
                int(self.get_parameter('thermal_empty_reset_frames').value),
            ):
                self.thermal_active = False
                self.thermal_candidate_ids = []
                self.window_candidate_ids = set()
                self.latest_thermal_msg = None
                self._reset_verification()
            return
        self.empty_thermal_frames = 0

        candidate_ids = self._candidate_ids(msg)
        # Stable track IDs come from the thermal node, so only a genuinely new
        # heat source restarts the window. A source that briefly disappears
        # keeps its votes.
        new_ids = [
            candidate_id for candidate_id in candidate_ids
            if candidate_id not in self.window_candidate_ids
        ]
        candidates_changed = self.thermal_active and bool(new_ids)
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
                reason='new thermal candidate ' + ', '.join(new_ids),
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
        self.window_candidate_ids = set(self.verification_candidate_ids)
        self.verification_votes = {
            candidate_id: [] for candidate_id in self.verification_candidate_ids
        }
        self.verification_status_counts = {}
        self.verification_last_match = {}
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
        self.verification_status_counts = {}
        self.verification_last_match = {}
        self.verification_frames_seen = 0

    def _buffer_thermal(self, msg: Detection2DArray) -> None:
        """Keep recent thermal frames so each RGB frame gets its closest one."""
        stamp = self._stamp_seconds(msg.header.stamp)
        self.thermal_buffer.append((stamp, msg))
        keep = max(0.2, float(self.get_parameter('thermal_buffer_sec').value))
        while self.thermal_buffer and stamp - self.thermal_buffer[0][0] > keep:
            self.thermal_buffer.popleft()

    def _closest_thermal(
        self,
        rgb_stamp: float,
    ) -> Tuple[Optional[Detection2DArray], float, float]:
        """Return ``(message, |dt|, newest thermal stamp)`` for an RGB stamp."""
        if not self.thermal_buffer:
            return None, float('inf'), float('-inf')
        stamp, msg = min(
            self.thermal_buffer,
            key=lambda item: abs(item[0] - rgb_stamp),
        )
        return msg, abs(stamp - rgb_stamp), self.thermal_buffer[-1][0]

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

        rgb_msg = self.latest_image_msg
        max_yaw_rate = float(self.get_parameter('max_verification_yaw_rate').value)
        if (
            max_yaw_rate > 0.0
            and self.latest_yaw_rate is not None
            and self.latest_yaw_rate > max_yaw_rate
        ):
            # While turning, the 3 Hz RGB frame and the 10 Hz thermal frame
            # show different scenes and the projected box lands beside the
            # animal. Skip this frame; the window continues afterwards.
            self.verification_last_frame_number = self.latest_frame_number
            if now_ns - self.last_turn_log_ns > int(2e9):
                self.last_turn_log_ns = now_ns
                self.get_logger().info(
                    f'Skipping YOLO verification frames while turning '
                    f'(|w|={self.latest_yaw_rate:.2f} rad/s > {max_yaw_rate:.2f}).'
                )
            return

        max_age = max(
            0.0,
            float(self.get_parameter('max_rgb_thermal_age_sec').value),
        )
        rgb_stamp = self._stamp_seconds(rgb_msg.header.stamp)
        thermal_msg = self.latest_thermal_msg
        if rgb_stamp > 0.0:
            closest, difference, newest_thermal = self._closest_thermal(rgb_stamp)
            if closest is None or difference > max_age:
                if newest_thermal < rgb_stamp:
                    # The thermal stream is behind this RGB frame: wait for
                    # the next thermal message and try the same frame again.
                    return
                # No thermal frame close to this RGB frame: skip it.
                self.verification_last_frame_number = self.latest_frame_number
                if now_ns - self.last_pairing_log_ns > int(5e9):
                    self.last_pairing_log_ns = now_ns
                    self.get_logger().warning(
                        f'No thermal frame within {max_age:.2f} s of the RGB frame '
                        f'(closest {difference:.2f} s); skipping it. If this '
                        'repeats, check use_sim_time and the camera timestamps, '
                        'or raise max_rgb_thermal_age_sec.'
                    )
                return
            thermal_msg = closest
        if not thermal_msg.detections:
            # At the RGB capture time the heat source was not visible.
            self.verification_last_frame_number = self.latest_frame_number
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
        # The same projection without the search margin gives the real size of
        # the hot region in the RGB image, used for the too_far decision.
        tight_rois = [
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
                margin=1.0,
                offset_pixels=(
                    float(self.get_parameter('thermal_rgb_offset_x_px').value),
                    float(self.get_parameter('thermal_rgb_offset_y_px').value),
                ),
            )
            for box in thermal_boxes
        ]
        minimum_height = float(self.get_parameter('min_classify_height_px').value)
        edge_margin = float(self.get_parameter('thermal_edge_margin_px').value)
        thermal_width = int(self.get_parameter('thermal_image_width').value)
        at_edge = [
            edge_margin > 0.0
            and (box[0] <= edge_margin - 1.0 or box[2] >= thermal_width - edge_margin)
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
        # Regions cut by the thermal image edge are not classified (see
        # thermal_edge_margin_px); their YOLO box stays free for others.
        assignments = [
            None if edge else assignment
            for assignment, edge in zip(assignments, at_edge)
        ]

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
            tight_roi,
            assignment,
        ) in enumerate(
            zip(
                candidate_ids,
                thermal_boxes,
                projected_rois,
                tight_rois,
                assignments,
            )
        ):
            roi_height = float(tight_roi[3] - tight_roi[1])
            if assignment is None:
                if at_edge[region_index]:
                    status, class_name = 'partial', 'partial_view'
                elif roi_height < minimum_height:
                    status, class_name = 'too_far', 'too_far_to_classify'
                else:
                    status, class_name = 'unresolved', 'unknown_heat_source'
                result_items.append({
                    'thermal_region': region_index,
                    'candidate_id': candidate_id,
                    'status': status,
                    'class_name': class_name,
                    'category': 'unknown',
                    'confidence': 0.0,
                    'rgb_roi_height_px': round(roi_height, 1),
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
                'rgb_roi_height_px': round(roi_height, 1),
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
            result_items,
            title=(
                f'Verifying frame {self.verification_frames_seen + 1}/'
                f'{self._verification_frame_target()}'
                if self.verification_active else 'Thermal + YOLO'
            ),
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
        frame_index = self.verification_frames_seen + 1
        for item, record in zip(result_items, matched_records):
            candidate_id = str(item['candidate_id'])
            self.verification_votes.setdefault(candidate_id, [])
            self.verification_status_counts.setdefault(candidate_id, Counter())[
                str(item['status'])
            ] += 1
            if item['status'] != 'classified' or record is None:
                continue
            self.verification_last_match[candidate_id] = (frame_index, record)
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
                # Most frequent reason this source got no class. Ties go to
                # "too far" / "partial": both mean "look again from closer".
                statuses = self.verification_status_counts.get(candidate_id, Counter())
                reason = max(
                    ('too_far', 'partial', 'unresolved'),
                    key=lambda name: (statuses.get(name, 0), name != 'unresolved'),
                )
                if statuses.get(reason, 0) == 0:
                    reason = 'unresolved'
                class_names = {
                    'too_far': 'too_far_to_classify',
                    'partial': 'partial_view',
                    'unresolved': 'unknown_heat_source',
                }
                result_items.append({
                    **common,
                    'status': reason,
                    'class_name': class_names[reason],
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
                    f'{candidate_id}={reason} '
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
        # Draw YOLO boxes from the final frame only: a box kept from an older
        # frame would sit beside the animal if the robot has moved since.
        display_records: List[Optional[Dict[str, object]]] = []
        for item in result_items:
            last = self.verification_last_match.get(str(item['candidate_id']))
            if (
                item['status'] == 'classified'
                and last is not None
                and last[0] == frame_count
            ):
                display_records.append(last[1])
            else:
                display_records.append(None)
        annotated = self._annotate_thermal_result(
            image_bgr,
            projected_rois,
            display_records,
            self.verification_candidate_ids,
            result_items,
            title=f'Result after {frame_count} RGB frames',
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
        items: Optional[Sequence[Dict[str, object]]] = None,
        title: Optional[str] = None,
    ):
        """Draw short tags on the image and the full result in a legend.

        Orange box: hot region projected from the thermal camera.
        Coloured box: the YOLO detection matched to it (green native,
        blue exotic). The legend lists every hot region once:
        ``H7 deer 0.87 EXOTIC 4/5``, ``H8 too far``, ``H9 no animal``.
        """
        annotated = image_bgr.copy()
        height, width = annotated.shape[:2]
        items = list(items) if items is not None else []
        tags = []
        for region_index in range(len(projected_rois)):
            if candidate_ids is not None and region_index < len(candidate_ids):
                tags.append(short_tag(str(candidate_ids[region_index])))
            else:
                tags.append(f'R{region_index + 1}')

        legend_lines = []
        if title:
            legend_lines.append((title, None))
        colours = []
        for region_index, tag in enumerate(tags):
            item = items[region_index] if region_index < len(items) else {}
            status = str(item.get('status', 'unresolved'))
            category = str(item.get('category', 'unknown'))
            colour = colour_for_status(status, category)
            colours.append(colour)
            if status == 'classified':
                text = (
                    f'{tag} {item.get("class_name", "?")} '
                    f'{float(item.get("confidence", 0.0)):.2f} {category.upper()}'
                )
                if 'frames_evaluated' in item:
                    votes = sum(
                        int(count)
                        for name, count in dict(
                            item.get('vote_summary', {})
                        ).items()
                        if name == item.get('class_name')
                    )
                    text += f' {votes}/{int(item["frames_evaluated"])}'
            elif status == 'too_far':
                text = f'{tag} too far to classify'
            elif status == 'partial':
                text = f'{tag} at thermal image edge'
            else:
                text = f'{tag} no animal match'
            legend_lines.append((text, colour))
        legend_rect = draw_legend(annotated, legend_lines, origin=(6, 6))

        placer = LabelPlacer(width, height, reserved=[legend_rect])
        visible = []
        for roi, record, colour, tag in zip(
            projected_rois,
            matched_records,
            colours,
            tags,
        ):
            rx1, ry1, rx2, ry2 = [int(round(value)) for value in roi]
            if rx2 <= rx1 and ry2 <= ry1:
                # Not in this frame (listed in the legend only).
                continue
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), THERMAL_ROI_COLOUR, 1)
            if record is not None:
                x1, y1, x2, y2 = [int(round(value)) for value in record['box']]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            visible.append((record['box'] if record is not None else roi, colour, tag))
        for anchor, colour, tag in visible:
            draw_tag(annotated, placer, tag, anchor, colour)
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
#!/usr/bin/env python3
"""YOLO RGB perception node for the Husky (ROS 2).

Scope: RGB YOLO only. No thermal-camera code is included here.

Forward-facing interfaces are deliberately modality-specific so a thermal
pipeline can be added later without changing this node:

  input:
    camera/image                         sensor_msgs/msg/Image

  output:
    perception/rgb/detections           vision_msgs/msg/Detection2DArray
    perception/rgb/annotated_image      sensor_msgs/msg/Image
    perception/rgb/confirmation_image   sensor_msgs/msg/Image

  service:
    perception/rgb/classify_now         std_srvs/srv/Trigger

All names are relative. If the node is launched in namespace ``husky1``, the
interfaces become /husky1/camera/image, /husky1/perception/rgb/detections, etc.

The Detection2DArray header is copied from the source RGB image. This is
important for future fusion: a later fusion/localisation node can synchronize
RGB detections with thermal/depth data by timestamp while keeping the YOLO node
unchanged.
"""

from pathlib import Path
from typing import Optional

import cv_bridge
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


PACKAGE_NAME = '41068_ignition_bringup'


class AnimalYoloDetector(Node):
    """Run the trained animal YOLO model on the Husky RGB camera."""

    def __init__(self) -> None:
        super().__init__('animal_yolo_detector')

        default_model = str(
            Path(get_package_share_directory(PACKAGE_NAME))
            / 'weights'
            / 'my_model.pt'
        )

        # Parameters are kept generic so this node can be reused or remapped later.
        self.declare_parameter('model_path', default_model)
        self.declare_parameter('image_topic', 'camera/image')
        self.declare_parameter('confidence', 0.50)
        self.declare_parameter('iou', 0.70)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('continuous', True)
        self.declare_parameter('max_inference_hz', 3.0)
        self.declare_parameter('device', '')
        self.declare_parameter('publish_annotated_image', True)

        model_path = Path(str(self.get_parameter('model_path').value)).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f'YOLO weights not found: {model_path}')

        self.bridge = cv_bridge.CvBridge()
        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(str(model_path))
        self.get_logger().info(f'YOLO classes: {self.model.names}')

        # Standard structured detection output. This is the topic that future
        # localisation/fusion code should consume.
        self.detections_pub = self.create_publisher(
            Detection2DArray,
            'perception/rgb/detections',
            10,
        )

        # Human/debug visualization. Downstream logic should NOT parse this image.
        self.annotated_pub = self.create_publisher(
            Image,
            'perception/rgb/annotated_image',
            5,
        )

        # Used by the later investigation workflow after R3 reaches an
        # observation pose and requests a fresh RGB classification.
        self.confirmation_pub = self.create_publisher(
            Image,
            'perception/rgb/confirmation_image',
            5,
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

        # Store only the newest frame. The timer performs inference, which avoids
        # building a queue of old camera frames when inference is slower than the camera.
        self.latest_image_msg: Optional[Image] = None
        self.latest_frame_number = 0
        self.last_processed_frame_number = -1

        max_hz = max(0.1, float(self.get_parameter('max_inference_hz').value))
        self.inference_timer = self.create_timer(1.0 / max_hz, self._continuous_tick)

        self.get_logger().info(
            'Animal YOLO detector ready | '
            f'input={self.resolve_topic_name(image_topic)} | '
            f'detections={self.resolve_topic_name("perception/rgb/detections")} | '
            f'continuous={bool(self.get_parameter("continuous").value)} | '
            f'max_hz={max_hz:.2f}'
        )

    def _image_callback(self, msg: Image) -> None:
        """Keep the newest RGB frame; inference is performed by the timer/service."""
        self.latest_image_msg = msg
        self.latest_frame_number += 1

    def _continuous_tick(self) -> None:
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
        """Classify the newest RGB frame on demand and publish a confirmation image."""
        if self.latest_image_msg is None:
            response.success = False
            response.message = 'No RGB camera frame has been received yet.'
            return response

        labels = self._run_inference(
            self.latest_image_msg,
            publish_confirmation=True,
        )

        response.success = True
        if labels:
            response.message = 'Detected: ' + ', '.join(labels)
        else:
            response.message = 'No trained animal detected in the latest RGB frame.'
        return response

    def _run_inference(self, source_msg: Image, publish_confirmation: bool):
        try:
            image_bgr = self.bridge.imgmsg_to_cv2(
                source_msg,
                desired_encoding='bgr8',
            )
        except cv_bridge.CvBridgeError as exc:
            self.get_logger().error(f'RGB image conversion failed: {exc}')
            return []

        conf = float(self.get_parameter('confidence').value)
        iou = float(self.get_parameter('iou').value)
        imgsz = int(self.get_parameter('imgsz').value)
        device = str(self.get_parameter('device').value).strip()

        predict_kwargs = {
            'source': image_bgr,
            'conf': conf,
            'iou': iou,
            'imgsz': imgsz,
            'verbose': False,
        }
        if device:
            predict_kwargs['device'] = device

        result = self.model.predict(**predict_kwargs)[0]

        # IMPORTANT: copy the source image header. A later fusion/localisation node
        # can then pair this detection result with the sensor frame that produced it.
        detections_msg = Detection2DArray()
        detections_msg.header = source_msg.header

        labels = []

        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
            scores = result.boxes.conf.detach().cpu().numpy()

            for box, class_id, score in zip(xyxy, class_ids, scores):
                x1, y1, x2, y2 = [float(v) for v in box]
                class_name = str(result.names[int(class_id)])
                score = float(score)

                detection = Detection2D()
                detection.header = source_msg.header

                # R2 owns candidate IDs / duplicate handling, so R1 deliberately
                # leaves Detection2D.id empty rather than inventing tracking IDs.
                detection.id = ''

                self._set_bbox(
                    detection,
                    center_x=(x1 + x2) / 2.0,
                    center_y=(y1 + y2) / 2.0,
                    width=max(0.0, x2 - x1),
                    height=max(0.0, y2 - y1),
                )

                hypothesis = ObjectHypothesisWithPose()
                self._set_hypothesis(
                    hypothesis,
                    class_name=class_name,
                    score=score,
                )
                detection.results.append(hypothesis)
                detections_msg.detections.append(detection)

                labels.append(f'{class_name} {score:.2f}')

        # Publish even when the array is empty. An empty result at a timestamp is
        # useful information for later fusion and evaluation.
        self.detections_pub.publish(detections_msg)

        publish_image = bool(self.get_parameter('publish_annotated_image').value)
        if publish_image or publish_confirmation:
            annotated = result.plot(show=False)
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            annotated_msg.header = source_msg.header

            if publish_image:
                self.annotated_pub.publish(annotated_msg)
            if publish_confirmation:
                self.confirmation_pub.publish(annotated_msg)

        if labels:
            self.get_logger().info('Detected: ' + ', '.join(labels))

        return labels

    @staticmethod
    def _set_bbox(detection: Detection2D, center_x: float, center_y: float,
                  width: float, height: float) -> None:
        """Fill BoundingBox2D while tolerating vision_msgs API variants."""
        center = detection.bbox.center

        # Current vision_msgs uses vision_msgs/Pose2D -> Point2D position.
        if hasattr(center, 'position'):
            center.position.x = center_x
            center.position.y = center_y
            center.theta = 0.0
        else:
            # Compatibility with distributions where BoundingBox2D used
            # geometry_msgs/Pose2D directly.
            center.x = center_x
            center.y = center_y
            center.theta = 0.0

        detection.bbox.size_x = width
        detection.bbox.size_y = height

    @staticmethod
    def _set_hypothesis(hypothesis: ObjectHypothesisWithPose,
                        class_name: str, score: float) -> None:
        """Fill ObjectHypothesisWithPose across common vision_msgs versions."""
        if hasattr(hypothesis, 'hypothesis'):
            hypothesis.hypothesis.class_id = class_name
            hypothesis.hypothesis.score = score
        else:
            # Older message layout compatibility.
            hypothesis.id = class_name
            hypothesis.score = score


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
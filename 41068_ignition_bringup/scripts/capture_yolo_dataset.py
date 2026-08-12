#!/usr/bin/env python3
"""Capture JPEG images from a ROS 2 camera topic for a small YOLO dataset.

Default behaviour for the Husky demo:
- subscribe to relative topic ``camera/image``
- when launched in namespace ``husky1`` this becomes ``/husky1/camera/image``
- save one image every 4.5 seconds
- stop automatically after 200 images (~15 minutes)
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class YoloDatasetCapture(Node):
    def __init__(self):
        super().__init__('yolo_dataset_capture')

        self.declare_parameter('image_topic', 'camera/image')
        self.declare_parameter('interval_seconds', 4.5)
        self.declare_parameter('max_images', 200)
        self.declare_parameter('output_directory', '~/41068_ws/yolo_dataset')
        self.declare_parameter('jpeg_quality', 95)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.interval_seconds = float(self.get_parameter('interval_seconds').value)
        self.max_images = int(self.get_parameter('max_images').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)

        base_dir = Path(str(self.get_parameter('output_directory').value)).expanduser()
        session_name = datetime.now().strftime('session_%Y%m%d_%H%M%S')
        self.output_dir = base_dir / session_name / 'images'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.saved_count = 0
        self.start_wall_time: Optional[float] = None
        self.next_save_wall_time: Optional[float] = None
        self.finished = False
        self.reported_format = False

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info('YOLO dataset capture node ready.')
        self.get_logger().info(f'Camera topic: {self.resolve_topic_name(self.image_topic)}')
        self.get_logger().info(f'Save interval: {self.interval_seconds:.2f} s')
        self.get_logger().info(f'Max images: {self.max_images}')
        self.get_logger().info(f'Output: {self.output_dir}')
        self.get_logger().info('Timer starts when the first camera image arrives.')

    def image_callback(self, msg: Image) -> None:
        if self.finished:
            return

        now = time.monotonic()

        if self.start_wall_time is None:
            self.start_wall_time = now
            # Wait one full interval before the first saved frame. With the
            # defaults, image #200 is therefore saved at about 900 seconds.
            self.next_save_wall_time = now + self.interval_seconds
            self.get_logger().info('First camera image received. Capture timer started.')

        if not self.reported_format:
            self.reported_format = True
            self.get_logger().info(
                f'Image format: {msg.width}x{msg.height}, encoding={msg.encoding}, step={msg.step}'
            )

        if now < self.next_save_wall_time:
            return

        frame = self.ros_image_to_cv2(msg)
        if frame is None:
            return

        self.saved_count += 1
        filename = self.output_dir / f'frame_{self.saved_count:04d}.jpg'

        ok = cv2.imwrite(
            str(filename),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )

        if not ok:
            self.get_logger().error(f'Failed to save {filename}')
            self.saved_count -= 1
            return

        elapsed = now - self.start_wall_time
        self.get_logger().info(
            f'Saved {self.saved_count:03d}/{self.max_images}: '
            f'{filename.name}  (elapsed {elapsed:.1f} s)'
        )

        # Advance from the previous target time instead of using "now" so a
        # slightly late callback does not make the schedule drift over 15 min.
        self.next_save_wall_time += self.interval_seconds

        # If the simulator briefly stalls and we missed one or more target
        # times, skip those old targets rather than saving many near-identical
        # images immediately after recovery.
        while self.next_save_wall_time <= now:
            self.next_save_wall_time += self.interval_seconds

        if self.saved_count >= self.max_images:
            self.finished = True
            self.get_logger().info('Capture complete.')
            self.get_logger().info(f'Saved {self.saved_count} images to: {self.output_dir}')
            rclpy.shutdown()

    def ros_image_to_cv2(self, msg: Image) -> Optional[np.ndarray]:
        """Convert common sensor_msgs/Image encodings to an OpenCV image.

        This intentionally does not require cv_bridge, matching the lightweight
        style already used by basic_autonomy_demo.py in this package.
        """
        encoding = msg.encoding.lower()

        if encoding in ('rgb8', 'bgr8'):
            channels = 3
        elif encoding in ('rgba8', 'bgra8'):
            channels = 4
        elif encoding in ('mono8', '8uc1'):
            channels = 1
        else:
            self.get_logger().error(
                f'Unsupported camera encoding: {msg.encoding}',
                throttle_duration_sec=5.0,
            )
            return None

        try:
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            rows = raw.reshape((msg.height, msg.step))
        except ValueError:
            self.get_logger().error(
                'Could not reshape ROS image buffer.',
                throttle_duration_sec=5.0,
            )
            return None

        useful_bytes = msg.width * channels
        if msg.step < useful_bytes:
            self.get_logger().error(
                f'Invalid image step {msg.step}; expected at least {useful_bytes}.',
                throttle_duration_sec=5.0,
            )
            return None

        image_data = rows[:, :useful_bytes]

        if channels == 1:
            return image_data.reshape((msg.height, msg.width)).copy()

        image = image_data.reshape((msg.height, msg.width, channels))

        if encoding == 'rgb8':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == 'bgr8':
            return image.copy()
        if encoding == 'rgba8':
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding == 'bgra8':
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        return None


def main(args=None):
    rclpy.init(args=args)
    node = YoloDatasetCapture()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Capture stopped by user.')
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()
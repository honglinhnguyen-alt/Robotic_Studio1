import math
import threading
import time

import numpy as np

from nav_msgs.msg import OccupancyGrid
from rclpy.time import Time
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from tf2_ros import Buffer, TransformListener, TransformException

from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt5.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPen,
    QPolygonF,
)
from PyQt5.QtWidgets import QWidget


def quaternion_yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class MapPanel(QWidget):
    def __init__(self, node):
        super().__init__()

        self.node = node
        self.robot_frame = "husky1_base_link"

        self.lock = threading.Lock()
        self.pending_map = None

        self.map_image = None
        self.map_info = None
        self.map_frame = ""
        self.robot_pose = None

        self.last_tf_stamp = None
        self.last_tf_change = None

        self.status = "Waiting for /husky1/map..."

        self.setMinimumSize(300, 300)

        self.tf_buffer = Buffer(node=node)
        self.tf_listener = TransformListener(
            self.tf_buffer,
            node,
            spin_thread=False,
        )

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_sub = node.create_subscription(
            OccupancyGrid,
            "/husky1/map",
            self.receive_map,
            map_qos,
        )

        # This timer runs on the GUI thread.
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)

    def receive_map(self, msg):
        # ROS callback: only store the latest message.
        with self.lock:
            self.pending_map = msg

    def refresh(self):
        with self.lock:
            msg = self.pending_map
            self.pending_map = None

        if msg is not None:
            try:
                self.convert_map(msg)
            except Exception as exc:
                self.map_image = None
                self.map_info = None
                self.robot_pose = None
                self.status = f"Map error: {exc}"
                self.update()
                return

        if self.map_image is not None:
            self.update_robot_pose()

        self.update()

    def convert_map(self, msg):
        width = msg.info.width
        height = msg.info.height

        if (
            width <= 0
            or height <= 0
            or msg.info.resolution <= 0
            or len(msg.data) != width * height
            or not msg.header.frame_id
        ):
            raise ValueError("Invalid map dimensions, resolution or frame")

        cells = np.asarray(msg.data, dtype=np.int16).reshape(
            height, width
        )

        # SLAM occupancy values:
        # -1 = unknown, 0 = free, 100 = occupied.
        gray = np.full((height, width), 150, dtype=np.uint8)
        known = cells >= 0

        gray[known] = (
            255.0 * (
                1.0 - np.clip(cells[known], 0, 100) / 100.0
            )
        ).astype(np.uint8)

        # Map Y increases upwards; screen Y increases downwards.
        gray = np.ascontiguousarray(np.flipud(gray))

        self.map_image = QImage(
            gray.data,
            width,
            height,
            gray.strides[0],
            QImage.Format_Grayscale8,
        ).copy()

        self.map_info = msg.info
        self.map_frame = msg.header.frame_id

    def update_robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                Time(),
            )

            stamp = (
                transform.header.stamp.sec,
                transform.header.stamp.nanosec,
            )
            now = time.monotonic()

            if stamp != self.last_tf_stamp:
                self.last_tf_stamp = stamp
                self.last_tf_change = now

            # Avoid presenting an old TF pose as a live position.
            if (
                self.last_tf_change is not None
                and now - self.last_tf_change > 3.0
            ):
                self.robot_pose = None
                self.status = "Map available | Robot TF stale"
                return

            position = transform.transform.translation
            yaw = quaternion_yaw(transform.transform.rotation)

            self.robot_pose = (position.x, position.y, yaw)

            self.status = (
                f"{self.map_frame} | "
                f"X: {position.x:.2f} m | "
                f"Y: {position.y:.2f} m | "
                f"Yaw: {math.degrees(yaw):.1f}°"
            )

        except TransformException:
            self.robot_pose = None
            self.status = (
                f"Waiting for TF: "
                f"{self.map_frame} <- {self.robot_frame}"
            )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#202830"))

        painter.setPen(QColor("white"))
        painter.drawText(
            QRectF(10, 5, self.width() - 20, 55),
            Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap,
            self.status,
        )

        if self.map_image is None:
            painter.end()
            return

        # Fit the full map inside the panel.
        area = QRectF(
            10,
            65,
            self.width() - 20,
            self.height() - 75,
        )

        width = self.map_info.width
        height = self.map_info.height

        scale = min(
            area.width() / width,
            area.height() / height,
        )

        left = area.x() + (area.width() - width * scale) / 2
        top = area.y() + (area.height() - height * scale) / 2

        target = QRectF(
            left,
            top,
            width * scale,
            height * scale,
        )

        painter.drawImage(target, self.map_image)

        if self.robot_pose is not None:
            self.draw_robot(painter, left, top, scale)

        painter.end()

    def draw_robot(self, painter, left, top, scale):
        x, y, yaw = self.robot_pose
        origin = self.map_info.origin
        origin_yaw = quaternion_yaw(origin.orientation)

        # Convert map-frame coordinates into grid-local coordinates.
        dx = x - origin.position.x
        dy = y - origin.position.y

        grid_x = (
            math.cos(origin_yaw) * dx
            + math.sin(origin_yaw) * dy
        ) / self.map_info.resolution

        grid_y = (
            -math.sin(origin_yaw) * dx
            + math.cos(origin_yaw) * dy
        ) / self.map_info.resolution

        if not (
            0 <= grid_x < self.map_info.width
            and 0 <= grid_y < self.map_info.height
        ):
            painter.setPen(QColor("yellow"))
            painter.drawText(
                15,
                self.height() - 15,
                "Robot is outside current map bounds",
            )
            return

        screen_x = left + grid_x * scale
        screen_y = (
            top + (self.map_info.height - grid_y) * scale
        )

        # Fixed-size arrow for visibility; not the robot footprint.
        painter.save()
        painter.setClipRect(
            QRectF(
                left,
                top,
                self.map_info.width * scale,
                self.map_info.height * scale,
            )
        )
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(screen_x, screen_y)
        painter.rotate(-math.degrees(yaw - origin_yaw))

        painter.setPen(QPen(QColor("white"), 2))
        painter.setBrush(QColor("#ff5533"))
        painter.drawPolygon(
            QPolygonF([
                QPointF(14, 0),
                QPointF(-9, -8),
                QPointF(-5, 0),
                QPointF(-9, 8),
            ])
        )
        painter.restore()
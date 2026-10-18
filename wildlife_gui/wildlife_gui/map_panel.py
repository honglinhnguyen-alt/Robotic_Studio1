import math
import threading
import time
import json

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
from std_msgs.msg import String
from html import escape
from PyQt5.QtWidgets import QToolTip

from wildlife_gui.mission_panel import CANDIDATE_STATES
from wildlife_gui.mission_panel import MISSION_STATUS_TOPIC


# Without mission status: colour by the registry class (deer is native).
CLASS_COLOURS = {
    "kangaroo": "#15803d",
    "deer": "#15803d",
    "boar": "#dc2626",
}
UNKNOWN_COLOUR = "#9ca3af"
# Mission status older than this falls back to the registry markers.
MISSION_STALE_SEC = 5.0
LEGEND = [
    ("#dc2626", "Exotic"),
    ("#15803d", "Native"),
    ("#4b5563", "Not animal"),
    ("#b45309", "Unresolved"),
    ("#2563eb", "Checking"),
    ("#6b7280", "Waiting"),
]
LEGEND_ITEM_WIDTH = 110


def marker_colour(animal):
    """Marker colour: mission state when known, else registry class."""
    state = animal.get("state")
    if state:
        return CANDIDATE_STATES.get(state, (state, UNKNOWN_COLOUR))[1]
    return CLASS_COLOURS.get(animal["class_name"], UNKNOWN_COLOUR)


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
        self.animal_hit_targets = []
        self.setMouseTracking(True)        

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
        self.pending_registry = None
        self.animals = []
        self.registry_frame = ""
        self.registry_status = "Animals: waiting for registry..."

        self.registry_sub = node.create_subscription(
            String,
            "/husky1/candidates/registry_json",
            self.receive_registry,
            map_qos,
        )

        # mission_node's status uses the registry keys plus each candidate's
        # mission state, so markers turn red/green once the type is known.
        self.pending_mission = None
        self.mission_animals = None
        self.mission_frame = ""
        self.last_mission_time = None
        self.mission_sub = node.create_subscription(
            String,
            MISSION_STATUS_TOPIC,
            self.receive_mission,
            10,
        )
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)

    def receive_map(self, msg):
        # ROS callback: only store the latest message.
        with self.lock:
            self.pending_map = msg

    def receive_registry(self, msg):
        with self.lock:
            self.pending_registry = msg.data

    def receive_mission(self, msg):
        with self.lock:
            self.pending_mission = msg.data
            self.last_mission_time = time.monotonic()

    def update_mission(self, text):
        try:
            payload = json.loads(text)
            frame = payload["frame_id"]
            animals = []
            for candidate in payload["candidates"]:
                position = candidate["position"]
                x = float(position["x"])
                y = float(position["y"])
                if not math.isfinite(x) or not math.isfinite(y):
                    continue
                animals.append({
                    "id": str(candidate["candidate_id"]),
                    "class_name": str(candidate.get("class_name") or "unknown")
                    .strip().lower(),
                    "state": str(candidate.get("state") or ""),
                    "x": x,
                    "y": y,
                })
            self.mission_animals = animals
            self.mission_frame = str(frame)
        except (ValueError, TypeError, KeyError, AttributeError):
            # Keep the previous mission snapshot.
            pass

    def update_registry(self, text):
        try:
            payload = json.loads(text)

            if not isinstance(payload, dict):
                raise ValueError("Registry must be a JSON object")

            frame = payload.get("frame_id")
            candidates = payload.get("candidates")

            if not isinstance(frame, str) or not frame.strip():
                raise ValueError("Missing registry frame")

            if not isinstance(candidates, list):
                raise ValueError("Candidates must be a list")

            animals = []
            seen_ids = set()

            for candidate in candidates:
                identifier = candidate["candidate_id"]

                if (
                    not isinstance(identifier, str)
                    or not identifier.strip()
                    or identifier in seen_ids
                ):
                    raise ValueError("Invalid or duplicate candidate ID")

                position = candidate["position"]
                x = float(position["x"])
                y = float(position["y"])

                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("Animal coordinates must be finite")

                class_name = candidate.get("class_name", "unknown")
                if not isinstance(class_name, str):
                    raise ValueError("Invalid class name")

                animals.append({
                    "id": identifier,
                    "class_name": class_name.strip().lower(),
                    "x": x,
                    "y": y,
                })
                seen_ids.add(identifier)

            # Replace the snapshot only after validating all records.
            self.animals = animals
            self.registry_frame = frame
            self.registry_status = f"Animal records: {len(animals)}"

        except (ValueError, TypeError, KeyError) as exc:
            # Keep the previous valid snapshot.
            self.registry_status = (
                f"Registry error; keeping previous records: {exc}"
            )

    def refresh(self):
        with self.lock:
            msg = self.pending_map
            registry_text = self.pending_registry
            mission_text = self.pending_mission

            self.pending_map = None
            self.pending_registry = None
            self.pending_mission = None

        if registry_text is not None:
            self.update_registry(registry_text)
        if mission_text is not None:
            self.update_mission(mission_text)

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

        animal_status = self.registry_status
        self.animal_hit_targets = []

        # The legend wraps onto extra rows when the panel is narrow.
        legend_per_row = max(1, int((self.width() - 10) // LEGEND_ITEM_WIDTH))
        legend_rows = -(-len(LEGEND) // legend_per_row)
        legend_height = 18 * (legend_rows - 1)

        if self.map_image is not None:
            area = QRectF(
                10,
                65,
                self.width() - 20,
                self.height() - 130 - legend_height,
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

            animal_status = self.draw_animals(
                painter, left, top, scale
            )

            if self.robot_pose is not None:
                self.draw_robot(painter, left, top, scale)

        painter.setPen(QColor("white"))
        painter.drawText(
            QRectF(
                10,
                self.height() - 60 - legend_height,
                self.width() - 20,
                32,
            ),
            Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap,
            animal_status,
        )

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        for index, (color, label) in enumerate(LEGEND):
            legend_x = 10 + (index % legend_per_row) * LEGEND_ITEM_WIDTH
            legend_y = (
                self.height() - 15 - legend_height
                + 18 * (index // legend_per_row)
            )
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(color))
            painter.drawEllipse(
                QPointF(legend_x + 5, legend_y),
                5.0,
                5.0,
            )

            painter.setPen(QColor("white"))
            painter.drawText(
                QRectF(legend_x + 15, legend_y - 10, 95, 20),
                Qt.AlignLeft | Qt.AlignVCenter,
                label,
            )

        painter.restore()
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
                self.height() - 70,
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

    def draw_animals(self, painter, left, top, scale):
        mission_fresh = (
            self.mission_animals is not None
            and self.last_mission_time is not None
            and time.monotonic() - self.last_mission_time < MISSION_STALE_SEC
        )
        if mission_fresh:
            animals = self.mission_animals
            frame = self.mission_frame
            source_status = f"Candidates: {len(animals)} (mission results)"
        else:
            animals = self.animals
            frame = self.registry_frame
            source_status = self.registry_status

        if not frame:
            return source_status

        if frame != self.map_frame:
            return (
                f"Animals hidden: candidate frame "
                f"{frame} != map frame {self.map_frame}"
            )

        origin = self.map_info.origin
        origin_yaw = quaternion_yaw(origin.orientation)
        cosine = math.cos(origin_yaw)
        sine = math.sin(origin_yaw)
        resolution = self.map_info.resolution

        target = QRectF(
            left,
            top,
            self.map_info.width * scale,
            self.map_info.height * scale,
        )

        outside_count = 0

        painter.save()
        painter.setClipRect(target)
        painter.setRenderHint(QPainter.Antialiasing)

        for animal in animals:
            dx = animal["x"] - origin.position.x
            dy = animal["y"] - origin.position.y

            grid_x = (cosine * dx + sine * dy) / resolution
            grid_y = (-sine * dx + cosine * dy) / resolution

            if not (
                0 <= grid_x < self.map_info.width
                and 0 <= grid_y < self.map_info.height
            ):
                outside_count += 1
                continue

            screen_x = left + grid_x * scale
            screen_y = (
                top + (self.map_info.height - grid_y) * scale
            )

            color = QColor(marker_colour(animal))

            # Larger marker with a dark outline.
            painter.setPen(QPen(QColor("#111827"), 2))
            painter.setBrush(color)
            painter.drawEllipse(
                QPointF(screen_x, screen_y),
                8.0,
                8.0,
            )

            self.animal_hit_targets.append(
                (screen_x, screen_y, animal)
            )

    

        painter.restore()

        status = source_status
        if outside_count:
            status += f" | Outside map: {outside_count}"

        return status  
    
    def mouseMoveEvent(self, event):
        nearest_animal = None
        nearest_distance = 14.0

        for screen_x, screen_y, animal in self.animal_hit_targets:
            distance = math.hypot(
                event.pos().x() - screen_x,
                event.pos().y() - screen_y,
            )

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_animal = animal

        if nearest_animal is None:
            QToolTip.hideText()
        else:
            animal = nearest_animal
            identifier = escape(animal["id"])
            class_name = escape(animal["class_name"])
            state = animal.get("state")
            state_line = (
                f"Mission: {escape(CANDIDATE_STATES.get(state, (state,))[0])}<br>"
                if state else ""
            )

            QToolTip.showText(
                event.globalPos(),
                (
                    f"<b>{identifier}</b><br>"
                    f"{state_line}"
                    f"Class: {class_name}<br>"
                    f"X: {animal['x']:.2f} m<br>"
                    f"Y: {animal['y']:.2f} m"
                ),
                self,
            )

        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)      
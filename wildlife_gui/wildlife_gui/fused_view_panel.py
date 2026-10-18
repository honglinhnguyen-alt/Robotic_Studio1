import threading
import time

from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


THERMAL_TOPIC = "/husky1/perception/thermal/annotated_image"
YOLO_TOPIC = "/husky1/perception/rgb/annotated_image"

# Keep the last YOLO verification frame on screen this long before going
# back to the live thermal view.
YOLO_HOLD_SEC = 10.0
STALE_AFTER_SEC = 3.0


class FusedViewPanel(QWidget):
    """Thermal detections live, YOLO + thermal fusion while verifying."""

    # The thermal detector publishes its annotated image continuously (hot
    # regions, track IDs, temperature). The YOLO node only runs when the
    # Husky verifies a candidate; its annotated RGB image then shows the
    # projected thermal region together with the YOLO boxes and votes.

    def __init__(self, node):
        super().__init__()
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.pending = {"thermal": None, "yolo": None}
        self.last_time = {"thermal": None, "yolo": None}
        self.shown_image = None
        self.error = ""

        self.caption = QLabel("Thermal + YOLO: waiting for images...")
        self.caption.setStyleSheet("font-weight: bold;")
        self.caption.setWordWrap(True)

        self.image_label = QLabel("Waiting for thermal camera...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 180)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setStyleSheet("background-color: #202020; color: white;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.caption)
        layout.addWidget(self.image_label, stretch=1)

        self.thermal_sub = node.create_subscription(
            Image, THERMAL_TOPIC,
            lambda msg: self.receive("thermal", msg), qos_profile_sensor_data)
        self.yolo_sub = node.create_subscription(
            Image, YOLO_TOPIC,
            lambda msg: self.receive("yolo", msg), qos_profile_sensor_data)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)

    def receive(self, source, msg):
        # ROS thread: store only the newest frame.
        with self.lock:
            self.pending[source] = msg
            self.last_time[source] = time.monotonic()

    def refresh(self):
        # Qt thread: choose the source and update widgets.
        now = time.monotonic()
        with self.lock:
            yolo_time = self.last_time["yolo"]
            thermal_time = self.last_time["thermal"]
            showing_yolo = yolo_time is not None and now - yolo_time < YOLO_HOLD_SEC
            source = "yolo" if showing_yolo else "thermal"
            msg = self.pending[source]
            self.pending[source] = None
            source_time = self.last_time[source]

        if msg is not None:
            self.show_image(msg)

        if source_time is None:
            self.caption.setText("Thermal + YOLO: waiting for the thermal camera...")
            return
        age = now - source_time
        if source == "yolo":
            text = f"YOLO + thermal verification ({age:.0f} s ago)"
        elif age > STALE_AFTER_SEC:
            text = f"Thermal detections: stale ({age:.0f} s)"
        else:
            text = "Thermal detections (live)"
        if thermal_time is None:
            text += " | thermal camera: waiting"
        if self.error:
            text += f" | image error: {self.error}"
        self.caption.setText(text)

    def show_image(self, msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8").copy()
            height, width, _ = rgb.shape
            self.shown_image = QImage(
                rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888
            ).copy()
            self.error = ""
            self.draw()
        except Exception as exc:
            self.error = str(exc)

    def draw(self):
        if self.shown_image is None:
            return
        self.image_label.setPixmap(
            QPixmap.fromImage(self.shown_image).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.draw()

    def stop(self):
        self.timer.stop()

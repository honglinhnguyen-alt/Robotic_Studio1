import math
import sys
import threading
import time
import os
from PyQt5.QtCore import QLibraryInfo

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
    qos_profile_sensor_data,
)

from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

from PyQt5.QtWidgets import QSplitter
from wildlife_gui.map_panel import MapPanel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from wildlife_gui.detections_panel import DetectionsPanel
from wildlife_gui.animal_records_panel import AnimalRecordsPanel
from wildlife_gui.mission_panel import MissionPanel
from wildlife_gui.fused_view_panel import FusedViewPanel
from PyQt5.QtWidgets import QTabWidget

class WildlifeNode(Node):
    """Receive ROS data without accessing GUI widgets."""

    def __init__(self):
        super().__init__("wildlife_gui")

        self.data_lock = threading.Lock()

        self.pending_image = None
        self.pending_odom = None

        self.last_image_time = None
        self.last_odom_time = None
        self.image_count = 0

        # Match the camera QoS used by RViz.
        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.image_sub = self.create_subscription(
            Image,
            "/husky1/camera/image",
            self.receive_image,
            camera_qos,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/husky1/odom",
            self.receive_odom,
            qos_profile_sensor_data,
        )

    def receive_image(self, msg):
        with self.data_lock:
            # Replace the pending image instead of accumulating frames.
            self.pending_image = msg
            self.last_image_time = time.monotonic()
            self.image_count += 1

    def receive_odom(self, msg):
        with self.data_lock:
            self.pending_odom = msg
            self.last_odom_time = time.monotonic()

    def take_latest(self):
        with self.data_lock:
            image_msg = self.pending_image
            odom_msg = self.pending_odom

            self.pending_image = None
            self.pending_odom = None

        return image_msg, odom_msg

    def get_status(self):
        with self.data_lock:
            return (
                self.last_image_time,
                self.last_odom_time,
                self.image_count,
            )


class WildlifeWindow(QWidget):
    """Display the latest ROS data on the Qt GUI thread."""

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.bridge = CvBridge()

        self.latest_image = None
        self.image_error = ""

        self.rate_start_time = time.monotonic()
        self.rate_start_count = 0
        self.receive_rate = 0.0

        self.setWindowTitle("Wildlife Monitoring — Husky")
        self.resize(900, 700)

        title = QLabel("Husky Wildlife Monitoring")
        title.setStyleSheet(
            "font-size: 24px; font-weight: bold;"
        )

        self.position_label = QLabel(
            "Position: waiting for odometry..."
        )
        self.position_label.setStyleSheet("font-size: 17px;")

        self.connection_label = QLabel("Waiting for ROS data...")
        self.connection_label.setWordWrap(True)

        self.camera_label = QLabel("Waiting for RGB camera...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(320, 180)
        self.camera_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Ignored,
        )
        self.camera_label.setStyleSheet(
            "background-color: #202020; color: white;"
        )

        self.map_panel = MapPanel(self.node)
        # Middle: thermal detections, or YOLO + thermal while verifying.
        self.fused_panel = FusedViewPanel(self.node)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.map_panel)
        splitter.addWidget(self.fused_panel)
        splitter.addWidget(self.camera_label)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([450, 450, 450])

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.position_label)
        layout.addWidget(self.connection_label)

        # Mission state stays visible above the map and camera.
        self.mission_panel = MissionPanel(self.node)
        layout.addWidget(self.mission_panel.status_bar)

        self.detections_panel = DetectionsPanel(self.node)
        self.animal_records_panel = AnimalRecordsPanel(self.node)

        self.records_tabs = QTabWidget()
        self.records_tabs.addTab(
            self.mission_panel,
            "Mission results",
        )
        self.records_tabs.addTab(
            self.animal_records_panel,
            "Candidate registry",
        )
        self.records_tabs.addTab(
            self.detections_panel,
            "Latest RGB detections",
        )

        # Top: map and camera. Bottom: animal tables.
        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.addWidget(splitter)
        self.main_splitter.addWidget(self.records_tabs)
        self.main_splitter.setChildrenCollapsible(False)

        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setSizes([450, 300])

        layout.addWidget(self.main_splitter, stretch=1)

        self.resize(1400, 850)

        # Display new data at up to 20 updates per second.
        self.display_timer = QTimer(self)
        self.display_timer.timeout.connect(self.update_display)
        self.display_timer.start(50)

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(500)

    def update_display(self):
        image_msg, odom_msg = self.node.take_latest()

        if odom_msg is not None:
            self.show_odometry(odom_msg)

        if image_msg is not None:
            self.show_image(image_msg)

    def show_image(self, msg):
        try:
            # Copy to obtain a contiguous RGB array.
            rgb = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="rgb8",
            ).copy()

            height, width, _ = rgb.shape

            # QImage owns a copy after the array goes out of scope.
            self.latest_image = QImage(
                rgb.data,
                width,
                height,
                rgb.strides[0],
                QImage.Format_RGB888,
            ).copy()

            self.draw_image()
            self.image_error = ""

        except Exception as exc:
            self.image_error = str(exc)
            self.latest_image = None
            self.camera_label.setText(
                f"Image conversion/display error:\n{exc}"
            )

    def draw_image(self):
        if self.latest_image is None:
            return

        pixmap = QPixmap.fromImage(self.latest_image)

        self.camera_label.setPixmap(
            pixmap.scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            )
        )

    def show_odometry(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        yaw = math.atan2(
            2.0 * (
                orientation.w * orientation.z
                + orientation.x * orientation.y
            ),
            1.0 - 2.0 * (
                orientation.y * orientation.y
                + orientation.z * orientation.z
            ),
        )

        self.position_label.setText(
            f"Frame: {msg.header.frame_id}  |  "
            f"X: {position.x:.2f} m  |  "
            f"Y: {position.y:.2f} m  |  "
            f"Yaw: {math.degrees(yaw):.1f}°"
        )

    def update_status(self):
        now = time.monotonic()
        image_time, odom_time, image_count = self.node.get_status()

        elapsed = now - self.rate_start_time

        # Approximate callback reception rate over the last interval.
        if elapsed >= 2.0:
            self.receive_rate = (
                image_count - self.rate_start_count
            ) / elapsed

            self.rate_start_time = now
            self.rate_start_count = image_count

        def describe(timestamp):
            if timestamp is None:
                return "waiting"

            age = now - timestamp

            if age < 3.0:
                return "receiving"

            return f"stale ({age:.0f}s)"

        camera_status = describe(image_time)

        if self.image_error:
            camera_status += " | image error"
        elif image_time is not None and now - image_time >= 3.0:
            camera_status += " — showing last image"

        self.connection_label.setText(
            f"Camera: {camera_status}  |  "
            f"Images received: {image_count}  |  "
            f"Receive rate: {self.receive_rate:.1f} Hz  |  "
            f"Odometry: {describe(odom_time)}"
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if hasattr(self, "camera_label"):
            self.draw_image()

    def stop_timers(self):
        self.display_timer.stop()
        self.status_timer.stop()
        self.map_panel.timer.stop()
        self.detections_panel.timer.stop()
        self.animal_records_panel.stop()
        self.mission_panel.stop()
        self.fused_panel.stop()

    def closeEvent(self, event):
        self.stop_timers()
        super().closeEvent(event)


def main(args=None):
    # Use the Qt platform plugins belonging to PyQt5.
    # This must run after importing cv_bridge/OpenCV,
    # but before creating QApplication.
    qt_plugins = QLibraryInfo.location(QLibraryInfo.PluginsPath)

    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
        qt_plugins, "platforms"
    )

    # Remove OpenCV's font-directory override, if present.
    if "cv2" in os.environ.get("QT_QPA_FONTDIR", ""):
        os.environ.pop("QT_QPA_FONTDIR", None)

    # The Husky's TF is published on /husky1/tf; without these remaps the
    # map shows "Waiting for TF" and no robot marker. Explicit remaps on the
    # command line still take precedence.
    ros_args = list(sys.argv if args is None else args)
    if not any(arg.startswith("/tf:=") for arg in ros_args):
        ros_args += [
            "--ros-args",
            "-r", "/tf:=/husky1/tf",
            "-r", "/tf_static:=/husky1/tf_static",
        ]
    rclpy.init(args=ros_args)
    app = QApplication(sys.argv)

    node = WildlifeNode()
    window = WildlifeWindow(node)

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    # ROS callbacks run independently of Qt's event loop.
    ros_thread = threading.Thread(
        target=executor.spin,
        daemon=True,
    )
    ros_thread.start()

    window.show()

    try:
        app.exec_()
    finally:
        window.stop_timers()

        # Finish ROS callbacks before destroying the node.
        executor.shutdown()
        ros_thread.join()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
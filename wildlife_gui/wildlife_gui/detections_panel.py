import threading
import time

from vision_msgs.msg import Detection2DArray
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)


class DetectionsPanel(QWidget):
    def __init__(self, node):
        super().__init__()

        self.lock = threading.Lock()
        self.pending_msg = None
        self.last_received = None

        self.frame_id = ""
        self.image_stamp = ""
        self.detection_count = 0

        title = QLabel("Animal detections — latest processed image")
        title.setStyleSheet(
            "font-size: 18px; font-weight: bold;"
        )

        self.status_label = QLabel("Waiting for detections...")
        self.status_label.setWordWrap(True)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "Row",
            "Class",
            "Confidence",
            "Box centre (px)",
            "Box size (px)",
        ])

        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self.table.setMinimumHeight(150)

        note = QLabel(
            "Rows are detections in one image, not unique animal IDs."
        )
        note.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.status_label)
        layout.addWidget(self.table)
        layout.addWidget(note)

        detection_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.subscription = node.create_subscription(
            Detection2DArray,
            "/husky1/perception/rgb/detections",
            self.receive_detections,
            detection_qos,
        )

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)

    def receive_detections(self, msg):
        # Runs on the ROS thread. Do not change GUI widgets here.
        with self.lock:
            self.pending_msg = msg
            self.last_received = time.monotonic()

    def refresh(self):
        # Runs on the GUI thread.
        with self.lock:
            msg = self.pending_msg
            received_at = self.last_received
            self.pending_msg = None

        if msg is not None:
            self.populate_table(msg)

        if received_at is None:
            self.status_label.setText(
                "Waiting for /husky1/perception/rgb/detections..."
            )
            return

        age = time.monotonic() - received_at

        if age < 3.0:
            state = "Receiving"
            self.status_label.setStyleSheet("color: #267326;")
        else:
            state = f"Stale ({age:.0f}s) — showing previous results"
            self.status_label.setStyleSheet("color: #b45f06;")

        self.status_label.setText(
            f"{state} | "
            f"Detections: {self.detection_count} | "
            f"Image stamp: {self.image_stamp} | "
            f"Frame: {self.frame_id}"
        )

    def populate_table(self, msg):
        self.frame_id = msg.header.frame_id
        stamp = msg.header.stamp
        self.image_stamp = f"{stamp.sec}.{stamp.nanosec:09d}"
        self.detection_count = len(msg.detections)

        # Replace previous results; an empty array clears the table.
        self.table.setRowCount(self.detection_count)

        for row, detection in enumerate(msg.detections):
            if detection.results:
                best = max(
                    detection.results,
                    key=lambda result: result.hypothesis.score,
                )
                class_name = best.hypothesis.class_id
                confidence = f"{best.hypothesis.score:.1%}"
            else:
                class_name = "Unknown"
                confidence = "—"

            box = detection.bbox
            centre = box.center.position

            values = [
                str(row + 1),
                class_name,
                confidence,
                f"({centre.x:.1f}, {centre.y:.1f})",
                f"{box.size_x:.1f} × {box.size_y:.1f}",
            ]

            for column, value in enumerate(values):
                self.table.setItem(
                    row,
                    column,
                    QTableWidgetItem(value),
                )

    def stop(self):
        self.timer.stop()
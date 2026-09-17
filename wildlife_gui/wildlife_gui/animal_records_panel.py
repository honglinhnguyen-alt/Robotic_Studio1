import json
import math
import threading

from std_msgs.msg import String
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


class AnimalRecordsPanel(QWidget):
    """Display the latest complete candidate registry snapshot."""

    def __init__(self, node):
        super().__init__()

        self.lock = threading.Lock()
        self.pending_json = None

        self.status_label = QLabel(
            "Waiting for /husky1/candidates/registry_json..."
        )
        self.status_label.setWordWrap(True)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Candidate ID",
            "Class",
            "Confidence",
            "X (m)",
            "Y (m)",
            "Observations",
            "State",
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
            "Positions are in the registry frame. "
            "Records update when observations arrive."
        )
        note.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.table)
        layout.addWidget(note)

        registry_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.subscription = node.create_subscription(
            String,
            "/husky1/candidates/registry_json",
            self.receive_registry,
            registry_qos,
        )

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(200)

    def receive_registry(self, msg):
        # ROS thread: store data without changing Qt widgets.
        with self.lock:
            self.pending_json = msg.data

    @staticmethod
    def parse_snapshot(text):
        data = json.loads(text)

        if not isinstance(data, dict):
            raise ValueError("Registry must be a JSON object")

        candidates = data.get("candidates")
        frame = data.get("frame_id")

        if not isinstance(candidates, list):
            raise ValueError("Missing or invalid candidates list")

        if not isinstance(frame, str) or not frame.strip():
            raise ValueError("Missing registry frame_id")

        rows = []
        seen_ids = set()

        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("Invalid candidate record")

            candidate_id = candidate["candidate_id"]

            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError("Invalid candidate ID")

            if candidate_id in seen_ids:
                raise ValueError("Duplicate candidate ID")

            seen_ids.add(candidate_id)

            position = candidate["position"]
            x = float(position["x"])
            y = float(position["y"])
            confidence = float(candidate["confidence"])

            if not all(math.isfinite(v) for v in (x, y, confidence)):
                raise ValueError("Non-finite position or confidence")

            if not 0.0 <= confidence <= 1.0:
                raise ValueError("Confidence outside [0, 1]")

            rows.append([
                candidate_id,
                str(candidate.get("class_name", "Unknown")),
                f"{confidence:.1%}",
                f"{x:.2f}",
                f"{y:.2f}",
                str(candidate.get("observation_count", "—")),
                str(candidate.get("state", "—")),
            ])

        return frame, rows

    def refresh(self):
        # Qt thread: update widgets here.
        with self.lock:
            text = self.pending_json
            self.pending_json = None

        if text is None:
            return

        try:
            frame, rows = self.parse_snapshot(text)
        except (ValueError, TypeError, KeyError) as exc:
            self.status_label.setText(
                f"Registry error: {exc}. "
                "Table unchanged; any visible rows are from earlier data."
            )
            self.status_label.setStyleSheet("color: #b45f06;")
            return

        # Each message is a complete snapshot, so replace the table.
        self.table.setRowCount(len(rows))

        for row_index, values in enumerate(rows):
            for column, value in enumerate(values):
                self.table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(value),
                )

        self.status_label.setStyleSheet("")

        if rows:
            self.status_label.setText(
                f"Latest registry snapshot | "
                f"Candidates: {len(rows)} | Frame: {frame}"
            )
        else:
            self.status_label.setText(
                f"No candidates recorded | Frame: {frame}"
            )

    def stop(self):
        self.timer.stop()
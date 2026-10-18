import json
import math
import threading
import time

from std_msgs.msg import String
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


MISSION_STATUS_TOPIC = "/husky1/mission/status"
EVALUATION_TOPIC = "/husky1/evaluation/summary"
COVERAGE_TOPIC = "/husky1/mission/coverage"

# A status older than this means mission_node is not publishing.
STALE_AFTER_SEC = 3.0

# mission_node MissionState -> (operator text, colour)
MISSION_STATES = {
    "INITIALISING": ("Starting up", "#6b7280"),
    "COVERAGE": ("Exploring", "#2563eb"),
    "PAUSING": ("Pausing exploration", "#d97706"),
    "INVESTIGATING": ("Investigating candidate", "#d97706"),
    "SWEEPING": ("Thermal sweep (checking unseen areas)", "#7c3aed"),
    "COMPLETE": ("MISSION COMPLETE", "#15803d"),
}

# mission_node CandidateState (lower case) -> (operator text, colour)
CANDIDATE_STATES = {
    "uninvestigated": ("Waiting", "#6b7280"),
    "assigned": ("Assigned", "#2563eb"),
    "investigating": ("Investigating", "#2563eb"),
    "confirmed_exotic": ("EXOTIC", "#dc2626"),
    "confirmed_native": ("Native", "#15803d"),
    "unresolved": ("Unresolved", "#b45309"),
    "duplicate": ("Duplicate", "#9ca3af"),
}

COUNT_LABELS = [
    ("exotic", "Exotic"),
    ("native", "Native"),
    ("unresolved", "Unresolved"),
    ("unresolved_low_confidence", "of which possible animal"),
    ("pending", "Pending"),
    ("duplicate", "Duplicate"),
]

# Why a candidate ended UNRESOLVED (mission_node.REASON_PRIORITY order:
# a possible animal first).
REASONS = [
    ("low_confidence", "possible animal, low confidence"),
    ("no_trained_animal_seen", "no trained animal seen"),
    ("too_far", "too far to classify"),
    ("partial_view", "only partly in view"),
    ("not_visible", "heat source not visible"),
    ("unreachable", "no observation pose reachable"),
    ("system_error", "perception or navigation error"),
]
REASON_TEXT = dict(REASONS)
REASON_ORDER = {key: index for index, (key, _) in enumerate(REASONS)}


def result_text(row):
    """Reason for UNRESOLVED candidates, else the last attempt's outcome."""
    if row["state"] == "unresolved" and row["reason"]:
        text = REASON_TEXT.get(row["reason"], row["reason"].replace("_", " "))
        return f"{text}: {row['reason_detail']}" if row["reason_detail"] else text
    return row["last_outcome"].replace("_", " ")


def parse_mission_status(text):
    """Validate one /husky1/mission/status JSON message."""
    data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError("Mission status must be a JSON object")

    mission_state = data.get("mission_state")
    if not isinstance(mission_state, str) or not mission_state:
        raise ValueError("Missing mission_state")

    counts = data.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Missing counts")

    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Missing or invalid candidates list")

    rows = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Invalid candidate record")

        candidate_id = candidate["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("Invalid candidate ID")

        position = candidate["position"]
        x = float(position["x"])
        y = float(position["y"])
        confidence = float(candidate.get("confidence") or 0.0)

        if not all(math.isfinite(v) for v in (x, y, confidence)):
            raise ValueError("Non-finite position or confidence")

        rows.append({
            "candidate_id": candidate_id,
            "state": str(candidate.get("state", "")),
            "class_name": str(candidate.get("class_name") or "unknown"),
            "confidence": confidence,
            "x": x,
            "y": y,
            "attempts": int(candidate.get("attempts") or 0),
            "last_outcome": str(candidate.get("last_outcome") or ""),
            "reason": str(candidate.get("reason") or ""),
            "reason_detail": str(candidate.get("reason_detail") or ""),
        })

    return {
        "mission_state": mission_state,
        "exploration_status": str(data.get("exploration_status") or ""),
        "exploration_paused": bool(data.get("exploration_paused")),
        "current_candidate": data.get("current_candidate"),
        "attempt": int(data.get("attempt") or 0),
        "phase": str(data.get("phase") or ""),
        "husky_phase": str(data.get("husky_phase") or ""),
        "counts": {key: int(counts.get(key, 0)) for key, _ in COUNT_LABELS},
        "candidates": rows,
    }


def describe_activity(status):
    """Return one line saying what the robot is doing now."""
    state = status["mission_state"]
    current = status["current_candidate"]

    if state == "COMPLETE":
        return "Exploration finished and every candidate has a result."

    if state == "SWEEPING":
        phase = status["phase"] or "planning"
        return f"Looking at areas the thermal camera has not seen ({phase})"

    if current and state in ("PAUSING", "INVESTIGATING"):
        text = f"Candidate {current}"
        if status["attempt"]:
            text += f", attempt {status['attempt']}"
        phase = status["husky_phase"] or status["phase"]
        if phase:
            text += f" ({phase.replace('_', ' ')})"
        return text

    exploration = status["exploration_status"].replace("_", " ")
    if status["exploration_paused"]:
        return f"Exploration paused | explore_lite: {exploration or 'n/a'}"
    return f"Searching for heat sources | explore_lite: {exploration or 'n/a'}"


def summary_lines(status):
    """Return plain-text lines for the mission summary box."""
    counts = status["counts"]
    lines = [
        "  ".join(f"{label}: {counts[key]}" for key, label in COUNT_LABELS)
    ]

    exotic = [
        row for row in status["candidates"]
        if row["state"] == "confirmed_exotic"
    ]
    native = [
        row for row in status["candidates"]
        if row["state"] == "confirmed_native"
    ]
    unresolved = [
        row for row in status["candidates"]
        if row["state"] == "unresolved"
    ]

    for title, rows in (
        ("Exotic animals", exotic),
        ("Native animals", native),
    ):
        if rows:
            lines.append(f"{title}:")
            lines.extend(
                f"  {row['candidate_id']}: {row['class_name']} "
                f"{row['confidence']:.0%} at "
                f"({row['x']:.1f}, {row['y']:.1f}) m"
                for row in rows
            )

    if unresolved:
        lines.append("Needs a manual check (possible animals first):")
        unresolved.sort(key=lambda row: REASON_ORDER.get(row["reason"], len(REASONS)))
        lines.extend(
            f"  {row['candidate_id']} at ({row['x']:.1f}, {row['y']:.1f}) m"
            f" - {result_text(row) or 'n/a'}"
            for row in unresolved
        )

    return lines


def coverage_line(text):
    """Return one line from a coverage_tracker JSON message."""
    data = json.loads(text)
    return (
        f"Coverage: map {100.0 * float(data['lidar_coverage']):.0f}%  |  "
        f"thermally checked {100.0 * float(data['thermal_coverage']):.0f}% "
        f"of ~{float(data['accessible_m2']):.0f} m\u00b2 survey area"
    )


def evaluation_line(text):
    """Return one line from a spatial_evaluator summary JSON message."""
    data = json.loads(text)
    error = data.get("mean_planar_error_m")
    error_text = "n/a" if error is None else f"{error:.2f} m"
    line = (
        f"Matched {data.get('matched_count', 0)}/"
        f"{data.get('ground_truth_count', 0)} heat sources | "
        f"mean error {error_text} | "
        f"within {data.get('localization_tolerance_m', 1.0):.1f} m: "
        f"{data.get('within_tolerance_count', 0)}"
    )
    boar = (data.get("per_class") or {}).get("boar")
    if boar:
        line += (
            f" | boars correctly classified "
            f"{boar['correctly_classified']}/{boar['ground_truth']}"
        )
    return line


class MissionPanel(QWidget):
    """Mission state, candidate investigation results and summary."""

    def __init__(self, node):
        super().__init__()

        self.lock = threading.Lock()
        self.pending_status = None
        self.pending_evaluation = None
        self.pending_coverage = None
        self.coverage_text = ""
        self.last_status_time = None

        # Status bar placed above the map by the main window.
        self.status_bar = QWidget()
        self.state_label = QLabel("Mission: waiting for mission_node...")
        self.state_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #6b7280;"
        )
        self.activity_label = QLabel(
            f"Waiting for {MISSION_STATUS_TOPIC}..."
        )
        self.activity_label.setWordWrap(True)
        self.counts_label = QLabel("")
        self.counts_label.setStyleSheet("font-size: 15px;")
        self.coverage_label = QLabel(f"Coverage: waiting for {COVERAGE_TOPIC}...")
        self.coverage_label.setStyleSheet("font-size: 15px;")

        bar_layout = QVBoxLayout(self.status_bar)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.addWidget(self.state_label)
        bar_layout.addWidget(self.activity_label)
        bar_layout.addWidget(self.counts_label)
        bar_layout.addWidget(self.coverage_label)

        # Tab content: candidate table and summary.
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Candidate",
            "Mission state",
            "Class",
            "Conf.",
            "Position (m)",
            "Attempts",
            "Result / reason",
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(120)

        self.summary_box = QGroupBox("Mission summary (live)")
        self.summary_text = QPlainTextEdit("No candidates yet.")
        self.summary_text.setReadOnly(True)
        self.summary_text.setStyleSheet("font-family: monospace;")
        self.evaluation_label = QLabel("")
        self.evaluation_label.setWordWrap(True)
        self.evaluation_label.setVisible(False)

        summary_layout = QVBoxLayout(self.summary_box)
        summary_layout.addWidget(self.summary_text)
        summary_layout.addWidget(self.evaluation_label)

        # Side by side so the summary gets the full tab height.
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(self.summary_box)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([760, 440])

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        # mission_node publishes every 0.5 s with volatile durability.
        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        # spatial_evaluator latches its latest summary.
        evaluation_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.status_sub = node.create_subscription(
            String,
            MISSION_STATUS_TOPIC,
            self.receive_status,
            status_qos,
        )
        self.evaluation_sub = node.create_subscription(
            String,
            EVALUATION_TOPIC,
            self.receive_evaluation,
            evaluation_qos,
        )

        self.coverage_sub = node.create_subscription(
            String,
            COVERAGE_TOPIC,
            self.receive_coverage,
            status_qos,
        )

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(200)

    def receive_status(self, msg):
        # ROS thread: store data without changing Qt widgets.
        with self.lock:
            self.pending_status = msg.data
            self.last_status_time = time.monotonic()

    def receive_evaluation(self, msg):
        with self.lock:
            self.pending_evaluation = msg.data

    def receive_coverage(self, msg):
        with self.lock:
            self.pending_coverage = msg.data

    def refresh(self):
        # Qt thread: update widgets here.
        with self.lock:
            status_text = self.pending_status
            evaluation_text = self.pending_evaluation
            coverage_text = self.pending_coverage
            last_status_time = self.last_status_time
            self.pending_status = None
            self.pending_evaluation = None
            self.pending_coverage = None

        if coverage_text is not None:
            try:
                self.coverage_text = coverage_line(coverage_text)
                self.coverage_label.setText(self.coverage_text)
            except (ValueError, TypeError, KeyError) as exc:
                self.coverage_label.setText(f"Coverage error: {exc}")

        if evaluation_text is not None:
            try:
                self.evaluation_label.setText(
                    "Simulation check (spatial_evaluator vs ground truth): "
                    + evaluation_line(evaluation_text)
                )
                self.evaluation_label.setVisible(True)
            except (ValueError, TypeError, KeyError) as exc:
                self.evaluation_label.setText(f"Evaluation error: {exc}")

        if status_text is not None:
            try:
                self.show_status(parse_mission_status(status_text))
            except (ValueError, TypeError, KeyError) as exc:
                self.activity_label.setText(
                    f"Mission status error: {exc}. Showing earlier data."
                )

        if last_status_time is not None:
            age = time.monotonic() - last_status_time
            if age > STALE_AFTER_SEC:
                self.state_label.setText(
                    f"Mission: no status for {age:.0f} s "
                    "(is mission_node running?)"
                )
                self.state_label.setStyleSheet(
                    "font-size: 18px; font-weight: bold; color: #b45309;"
                )

    def show_status(self, status):
        state = status["mission_state"]
        text, color = MISSION_STATES.get(state, (state, "#6b7280"))
        self.state_label.setText(f"Mission: {text}")
        self.state_label.setStyleSheet(
            f"font-size: 18px; font-weight: bold; color: {color};"
        )
        self.activity_label.setText(describe_activity(status))
        self.counts_label.setText(
            "  |  ".join(
                f"{label}: {status['counts'][key]}"
                for key, label in COUNT_LABELS
            )
        )

        rows = status["candidates"]
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            state_text, state_color = CANDIDATE_STATES.get(
                row["state"], (row["state"], "#6b7280")
            )
            values = [
                row["candidate_id"],
                state_text,
                row["class_name"].replace("_", " "),
                f"{row['confidence']:.0%}",
                f"({row['x']:.1f}, {row['y']:.1f})",
                str(row["attempts"]),
                result_text(row),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 1:
                    item.setForeground(QColor("white"))
                    item.setBackground(QColor(state_color))
                self.table.setItem(row_index, column, item)

        self.summary_box.setTitle(
            "Final mission summary" if state == "COMPLETE"
            else "Mission summary (live)"
        )
        lines = summary_lines(status)
        if self.coverage_text:
            lines.insert(1, self.coverage_text)
        text = "\n".join(lines)
        if text != self.summary_text.toPlainText():
            # Rewriting the same text would reset the operator's scrolling.
            self.summary_text.setPlainText(text)

    def stop(self):
        self.timer.stop()

#!/usr/bin/env python3
"""Mission node (R4): decide which heat candidate to investigate, and when.

End-to-end flow (all topics are absolute, so the node can run in any namespace)

  candidate_registry  --/husky1/candidates/registry_json-->  mission_node
  mission_node        --/husky1/explore/resume (False)---->  explore_lite (pause)
  mission_node        --/husky1/mission/investigate------->  husky_node
  husky_node          --NavigateToPose / Spin------------->  Nav2
  husky_node          --/husky1/perception/verify_now----->  animal_yolo_detector
  husky_node          --/husky1/mission/husky_status------>  mission_node
  animal_yolo_detector--/husky1/perception/thermal_yolo/result--> mission_node
  mission_node        --/husky1/explore/resume (True)----->  explore_lite (resume)
  mission_node        --/husky1/mission/status------------>  user interface
  mission_node        --/husky1/mission/log--------------->  heat_source_logger
      (one readable line per event: new heat source, trial k: where the
       Husky goes and what YOLO saw, final result)

Candidate states follow the C1 proposal (section 3.3):
  UNINVESTIGATED -> ASSIGNED -> INVESTIGATING -> CONFIRMED_EXOTIC /
  CONFIRMED_NATIVE / REJECTED / UNRESOLVED
DUPLICATE marks a registry record that lies next to a candidate that was
already investigated AND was built from the same thermal track, so the same
animal is not visited twice. A different animal standing close by keeps its
own record and waits in the queue.

Only the mission node changes candidate states. One candidate is investigated
at a time, nearest first. Each candidate gets at most ``max_attempts``
observation attempts (R4). A YOLO consensus is accepted only when it carries
the request_id returned by the verification this mission asked for, and only
the thermal region that points at the candidate is used.
"""

import json
import math
from enum import Enum

import rclpy
import tf2_ros
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool
from std_msgs.msg import String

try:  # Built by m-explore-ros2; optional so the mission still runs without it.
    from explore_lite_msgs.msg import ExploreStatus
except ImportError:  # pragma: no cover - depends on the workspace
    ExploreStatus = None


class MissionState(Enum):
    INITIALISING = 0
    COVERAGE = 1
    PAUSING = 2
    INVESTIGATING = 3
    COMPLETE = 4


class CandidateState(Enum):
    UNINVESTIGATED = 0
    ASSIGNED = 1
    INVESTIGATING = 2
    CONFIRMED_EXOTIC = 3
    CONFIRMED_NATIVE = 4
    REJECTED = 5
    UNRESOLVED = 6
    DUPLICATE = 7


FINISHED_STATES = {
    CandidateState.CONFIRMED_EXOTIC,
    CandidateState.CONFIRMED_NATIVE,
    CandidateState.REJECTED,
    CandidateState.UNRESOLVED,
    CandidateState.DUPLICATE,
}
# Explore status strings that mean "no frontier left".
EXPLORATION_DONE = {
    'exploration_complete',
    'returning_to_origin',
    'returned_to_origin',
}
# Husky statuses that end an attempt without a YOLO result.
HUSKY_FAILURES = {
    'unreachable',
    'no_heat_in_view',
    'verify_unavailable',
    'verify_failed',
}


# --------------------------------------------------------------------------
# Pure helpers (no ROS) - imported by the offline tests
# --------------------------------------------------------------------------

def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def thermal_bearing(box, image_width, hfov_rad):
    """Bearing of a thermal box centre in the robot frame [rad, left > 0]."""
    centre_x = 0.5 * (float(box['x1']) + float(box['x2']))
    focal = (0.5 * image_width) / math.tan(0.5 * hfov_rad)
    return math.atan2(0.5 * image_width - centre_x, focal)


def match_result_item(
    items,
    candidate_xy,
    source_ids,
    robot_pose,
    image_width,
    hfov_rad,
    tolerance_rad,
    foreign_ids=frozenset(),
):
    """
    Return ``(item, bearing_error)`` for the thermal region of this candidate.

    1. A region whose thermal track ID was localized into this registry
       record (``source_ids``) is the candidate.
    2. Otherwise the region whose bearing is closest to the direction of the
       candidate is used, if the error is within ``tolerance_rad``.
    A region whose track ID belongs to ANOTHER candidate (``foreign_ids``) is
    never used: that is a neighbouring animal, which keeps its own queue slot.
    """
    if not items:
        return None, None
    for item in items:
        if item.get('candidate_id') in source_ids:
            return item, 0.0
    items = [item for item in items if item.get('candidate_id') not in foreign_ids]
    if not items:
        return None, None
    if robot_pose is None:
        return (items[0], None) if len(items) == 1 else (None, None)

    robot_x, robot_y, robot_yaw = robot_pose
    expected = wrap_angle(
        math.atan2(candidate_xy[1] - robot_y, candidate_xy[0] - robot_x)
        - robot_yaw
    )
    best_item = None
    best_error = math.inf
    for item in items:
        box = item.get('thermal_bbox')
        if not isinstance(box, dict):
            continue
        error = abs(wrap_angle(
            thermal_bearing(box, image_width, hfov_rad) - expected
        ))
        if error < best_error:
            best_item, best_error = item, error
    if best_item is not None and best_error <= tolerance_rad:
        return best_item, best_error
    return None, (best_error if math.isfinite(best_error) else None)


def classify_outcome(item, confirm_confidence):
    """Map one YOLO consensus item to a mission outcome string."""
    if item is None:
        return 'not_in_view'
    status = str(item.get('status', ''))
    if status == 'classified':
        confidence = float(item.get('confidence', 0.0))
        category = str(item.get('category', 'unknown'))
        if confidence >= confirm_confidence and category in ('exotic', 'native'):
            return category
        return 'low_confidence'
    if status == 'unresolved':
        # Heat region clearly visible and big enough, but no animal box.
        return 'no_animal'
    return status or 'unknown'   # too_far / partial


# Text for a trial that ended without a usable YOLO answer.
TRIAL_TEXT = {
    'unreachable': 'could not reach an observation pose',
    'no_heat_in_view': 'heat source not visible from the observation pose',
    'verify_unavailable': 'YOLO service not available',
    'verify_failed': 'YOLO could not be started',
    'verify_timeout': 'no YOLO result (timed out)',
    'husky_timeout': 'Husky did not reach an observation pose in time',
}


def describe_yolo(item, outcome, confirm_confidence):
    """What YOLO saw for this candidate in one trial (for the mission log)."""
    if outcome == 'no_heat':
        return 'YOLO -> heat source lost during the vote'
    if item is None:
        return 'YOLO -> heat source not found in the image'
    votes = item.get('vote_summary') or {}
    frames = item.get('frames_evaluated', '?')
    best = max(votes.values()) if votes else 0
    status = item.get('status')
    if status == 'classified':
        text = (f'YOLO -> {item.get("class_name")} {float(item.get("confidence", 0.0)):.2f} '
                f'({item.get("category")}, {best}/{frames} frames)')
        if outcome == 'low_confidence':
            text += f' - below {confirm_confidence:.2f}, not accepted'
        return text
    if status == 'too_far':
        return f'YOLO -> too far to classify ({best}/{frames} frames)'
    if status == 'partial':
        return 'YOLO -> heat source only partly in view'
    return f'YOLO -> no trained animal (unknown heat source, {best}/{frames} frames)'


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

class MissionNode(Node):

    def __init__(self):
        super().__init__('mission_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('map_frame', 'husky1_map')
        self.declare_parameter('base_frame', 'husky1_base_link')
        self.declare_parameter('registry_topic', '/husky1/candidates/registry_json')
        self.declare_parameter('test_candidate_topic', '/husky1/mission/test_candidate')
        self.declare_parameter('yolo_result_topic', '/husky1/perception/thermal_yolo/result')
        self.declare_parameter('husky_command_topic', '/husky1/mission/investigate')
        self.declare_parameter('husky_status_topic', '/husky1/mission/husky_status')
        self.declare_parameter('explore_resume_topic', '/husky1/explore/resume')
        self.declare_parameter('explore_status_topic', '/husky1/explore/status')
        self.declare_parameter('mission_status_topic', '/husky1/mission/status')
        self.declare_parameter('mission_log_topic', '/husky1/mission/log')

        self.declare_parameter('confirm_confidence', 0.70)          # R6
        self.declare_parameter('max_attempts', 3)                   # R4
        self.declare_parameter('reject_after_no_animal_views', 2)
        self.declare_parameter('duplicate_radius', 1.5)             # 0 disables
        # Only call it a duplicate if both records share a thermal track ID.
        self.declare_parameter('duplicate_requires_shared_track', True)
        self.declare_parameter('pause_settle_sec', 1.5)
        self.declare_parameter('husky_timeout_sec', 330.0)
        self.declare_parameter('verify_timeout_sec', 25.0)
        self.declare_parameter('bearing_tolerance_deg', 15.0)
        self.declare_parameter('thermal_image_width', 320)
        self.declare_parameter('thermal_hfov_deg', 60.0)
        self.declare_parameter('resume_exploration_on_start', True)
        self.declare_parameter('status_period_sec', 0.5)

        p = self.get_parameter
        self.map_frame = str(p('map_frame').value)
        self.base_frame = str(p('base_frame').value)
        self.confirm_confidence = float(p('confirm_confidence').value)
        self.max_attempts = max(1, int(p('max_attempts').value))
        self.reject_after = max(1, int(p('reject_after_no_animal_views').value))
        self.duplicate_radius = float(p('duplicate_radius').value)
        self.duplicate_requires_shared_track = bool(
            p('duplicate_requires_shared_track').value)
        self.pause_settle_sec = float(p('pause_settle_sec').value)
        self.husky_timeout_sec = float(p('husky_timeout_sec').value)
        self.verify_timeout_sec = float(p('verify_timeout_sec').value)
        self.bearing_tolerance = math.radians(float(p('bearing_tolerance_deg').value))
        self.thermal_width = int(p('thermal_image_width').value)
        self.thermal_hfov = math.radians(float(p('thermal_hfov_deg').value))

        # ---------------- Mission ----------------
        self.state = MissionState.INITIALISING
        self.exploration_status = 'unknown'
        self.exploration_complete = False
        self.exploration_paused = False
        self.pause_started = None
        self.last_explore_command = -math.inf
        self.explore_resend_sec = 1.0

        # ---------------- Candidates ----------------
        self.candidates = {}
        self.next_test_id = 1
        self.current_candidate = None
        self.current_attempt = 0
        self.current_request_id = None
        self.phase = 'idle'          # waiting_husky | verifying
        self.phase_started = None
        self.husky_phase = ''

        # ---------------- TF (robot pose in map) ----------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------------- Subscribers ----------------
        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.registry_sub = self.create_subscription(
            String, str(p('registry_topic').value), self._registry_callback, latched)
        self.test_candidate_sub = self.create_subscription(
            Point, str(p('test_candidate_topic').value), self._test_candidate_callback, 10)
        self.result_sub = self.create_subscription(
            String, str(p('yolo_result_topic').value), self._result_callback, 10)
        self.husky_status_sub = self.create_subscription(
            String, str(p('husky_status_topic').value), self._husky_status_callback, 10)
        self.explore_status_sub = None
        if ExploreStatus is not None:
            explore_qos = QoSProfile(depth=10)
            explore_qos.reliability = ReliabilityPolicy.RELIABLE
            explore_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
            self.explore_status_sub = self.create_subscription(
                ExploreStatus, str(p('explore_status_topic').value),
                self._explore_status_callback, explore_qos)
        else:
            self.get_logger().warning(
                'explore_lite_msgs not found: mission cannot detect '
                'exploration completion (it still pauses/resumes).')

        # ---------------- Publishers ----------------
        self.husky_command_pub = self.create_publisher(
            String, str(p('husky_command_topic').value), 10)
        self.exploration_resume_pub = self.create_publisher(
            Bool, str(p('explore_resume_topic').value), 10)
        self.ui_status_pub = self.create_publisher(
            String, str(p('mission_status_topic').value), 10)
        self.event_pub = self.create_publisher(
            String, str(p('mission_log_topic').value), 50)
        self.going_logged = False

        self.timer = self.create_timer(0.2, self._mission_tick)
        self.status_timer = self.create_timer(
            max(0.1, float(p('status_period_sec').value)), self._publish_ui_status)

        self.get_logger().info(
            'Mission node started | registry=%s | result=%s | husky=%s'
            % (p('registry_topic').value, p('yolo_result_topic').value,
               p('husky_command_topic').value))

    # ======================================================================
    # Main state machine
    # ======================================================================

    def _mission_tick(self):
        now = self._now()
        self._enforce_exploration_state(now)

        if self.state == MissionState.INITIALISING:
            if bool(self.get_parameter('resume_exploration_on_start').value):
                self._resume_exploration('mission start')
            self.get_logger().info('Mission initialised -> COVERAGE')
            self.state = MissionState.COVERAGE
            return

        if self.state in (MissionState.COVERAGE, MissionState.COMPLETE):
            candidate_id = self._select_next_candidate()
            if candidate_id is not None:
                self._begin_investigation(candidate_id)
                return
            if self.state == MissionState.COVERAGE and self.exploration_complete:
                self.state = MissionState.COMPLETE
                self.get_logger().info('Exploration complete and no candidate left -> COMPLETE. '
                                       + self._summary_text())
            return

        if self.state == MissionState.PAUSING:
            # explore_lite cancels ALL Nav2 goals when it pauses. Wait until
            # that cancel is done before the Husky node sends its own goal.
            if now - self.pause_started >= self.pause_settle_sec:
                self._dispatch_attempt()
            return

        if self.state == MissionState.INVESTIGATING:
            elapsed = now - self.phase_started
            if self.phase == 'waiting_husky' and elapsed > self.husky_timeout_sec:
                self.get_logger().warning(
                    f'{self.current_candidate}: Husky did not reach a view in '
                    f'{self.husky_timeout_sec:.0f} s.')
                self._send_husky_cancel()
                self._attempt_failed('husky_timeout')
            elif self.phase == 'verifying' and elapsed > self.verify_timeout_sec:
                self._attempt_failed('verify_timeout')

    # ======================================================================
    # Investigation steps
    # ======================================================================

    def _begin_investigation(self, candidate_id):
        candidate = self.candidates[candidate_id]
        candidate['state'] = CandidateState.ASSIGNED
        self.current_candidate = candidate_id
        self.current_attempt = 0
        self.current_request_id = None
        self.get_logger().info(
            f'Selected {candidate_id} at ({candidate["x"]:.2f}, {candidate["y"]:.2f}) '
            f'[registry class={candidate["registry_class"]}]')
        if self.exploration_paused:
            self._dispatch_attempt()
            return
        self._pause_exploration()
        self.pause_started = self._now()
        self.state = MissionState.PAUSING
        self._publish_ui_status()

    def _dispatch_attempt(self):
        candidate = self.candidates[self.current_candidate]
        self.current_attempt += 1
        self.current_request_id = None
        candidate['state'] = CandidateState.INVESTIGATING
        candidate['attempts'] = self.current_attempt
        self.phase = 'waiting_husky'
        self.phase_started = self._now()
        self.husky_phase = 'requested'
        self.going_logged = False
        self.state = MissionState.INVESTIGATING
        self._publish_json(self.husky_command_pub, {
            'command': 'investigate',
            'candidate_id': self.current_candidate,
            'attempt': self.current_attempt,
            'max_attempts': self.max_attempts,
            'x': candidate['x'],
            'y': candidate['y'],
            'frame_id': self.map_frame,
        })
        self.get_logger().info(
            f'{self.current_candidate}: attempt {self.current_attempt}/'
            f'{self.max_attempts} sent to Husky node.')

    def _attempt_failed(self, reason, detail=None, text=None):
        candidate = self.candidates[self.current_candidate]
        self._event('trial', self.current_candidate,
                    f'Trial {self.current_attempt}: '
                    f'{text or TRIAL_TEXT.get(reason, reason.replace("_", " "))}')
        candidate['last_outcome'] = reason
        entry = {'attempt': self.current_attempt, 'outcome': reason}
        entry.update(detail or {})
        candidate['history'].append(entry)
        self.get_logger().info(
            f'{self.current_candidate}: attempt {self.current_attempt} -> {reason}')
        if self.current_attempt >= self.max_attempts:
            self._finish_candidate(CandidateState.UNRESOLVED)
        else:
            self._dispatch_attempt()

    def _finish_candidate(self, state, class_name=None, confidence=None):
        candidate_id = self.current_candidate
        candidate = self.candidates[candidate_id]
        candidate['state'] = state
        if class_name is not None:
            candidate['class_name'] = class_name
            candidate['confidence'] = float(confidence or 0.0)
        self.get_logger().info(
            f'{candidate_id} -> {state.name}'
            + (f' ({class_name} {float(confidence or 0.0):.2f})' if class_name else '')
            + f' after {self.current_attempt} attempt(s).')
        trials = self.current_attempt
        if state == CandidateState.CONFIRMED_EXOTIC:
            result = (f'{str(class_name).upper()} (EXOTIC), confidence '
                      f'{float(confidence or 0.0):.2f}, after {trials} trial(s)')
        elif state == CandidateState.CONFIRMED_NATIVE:
            result = (f'{str(class_name).upper()} (NATIVE), confidence '
                      f'{float(confidence or 0.0):.2f}, after {trials} trial(s)')
        elif state == CandidateState.REJECTED:
            result = f'NOT A TRAINED ANIMAL (heat source rejected) after {trials} trial(s)'
        else:
            result = f'UNRESOLVED after {trials} trial(s)'
        self._event('result', candidate_id,
                    f'RESULT {candidate_id} at ({candidate["x"]:.2f}, {candidate["y"]:.2f}): '
                    f'{result}', state=state.name.lower(), trials=trials, summary=result)

        self.current_candidate = None
        self.current_attempt = 0
        self.current_request_id = None
        self.phase = 'idle'
        self._mark_duplicates()
        self._publish_ui_status()   # the UI sees the decision before the next goal

        next_candidate = self._select_next_candidate()
        if next_candidate is not None:
            # Exploration is still paused: go straight to the next candidate.
            self._begin_investigation(next_candidate)
            return
        self._resume_exploration('investigation finished')
        self.state = MissionState.COVERAGE
        self._publish_ui_status()

    # ======================================================================
    # Callbacks
    # ======================================================================

    def _registry_callback(self, msg):
        try:
            data = json.loads(msg.data)
            records = data.get('candidates', [])
        except (ValueError, AttributeError) as error:
            self.get_logger().warning(f'Bad registry JSON: {error}')
            return
        for record in records:
            try:
                candidate_id = str(record['candidate_id'])
                position = record['position']
                x, y = float(position['x']), float(position['y'])
                z = float(position.get('z', 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            source_ids = {str(s) for s in record.get('source_ids', []) or []}
            candidate = self.candidates.get(candidate_id)
            if candidate is None:
                candidate = self._new_candidate(candidate_id, x, y, z)
                candidate['source_ids'].update(source_ids)
                duplicate_of = self._finished_neighbour(candidate_id, x, y)
                self._event('new', candidate_id,
                            f'NEW HEAT SOURCE {candidate_id} at ({x:.2f}, {y:.2f})')
                if duplicate_of is not None:
                    candidate['state'] = CandidateState.DUPLICATE
                    candidate['last_outcome'] = f'duplicate_of:{duplicate_of}'
                    self.get_logger().info(
                        f'{candidate_id} is within {self.duplicate_radius:.1f} m of '
                        f'{duplicate_of} (same thermal track); marked DUPLICATE.')
                    self._event('result', candidate_id,
                                f'RESULT {candidate_id} at ({x:.2f}, {y:.2f}): same object '
                                f'as {duplicate_of} (skipped)', state='duplicate', trials=0,
                                summary=f'same object as {duplicate_of} (skipped)')
                else:
                    self.get_logger().info(
                        f'New candidate {candidate_id} at ({x:.2f}, {y:.2f})')
            else:
                candidate['x'], candidate['y'], candidate['z'] = x, y, z
            candidate['registry_class'] = str(record.get('class_name', 'unknown'))
            candidate['registry_confidence'] = float(record.get('confidence', 0.0) or 0.0)
            candidate['observation_count'] = int(record.get('observation_count', 0) or 0)
            candidate['source_ids'].update(source_ids)

    def _test_candidate_callback(self, msg):
        """Inject a candidate by hand (tests without thermal/registry)."""
        candidate_id = f'test_{self.next_test_id}'
        self.next_test_id += 1
        self._new_candidate(candidate_id, float(msg.x), float(msg.y), float(msg.z))
        self.get_logger().info(
            f'Test candidate {candidate_id} at ({msg.x:.2f}, {msg.y:.2f})')
        self._event('new', candidate_id,
                    f'NEW HEAT SOURCE {candidate_id} at ({msg.x:.2f}, {msg.y:.2f}) (test)')

    def _husky_status_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        if (
            self.state != MissionState.INVESTIGATING
            or data.get('candidate_id') != self.current_candidate
            or int(data.get('attempt', -1)) != self.current_attempt
        ):
            return  # stale message from an older attempt
        status = str(data.get('status', ''))
        self.husky_phase = status
        candidate = self.candidates[self.current_candidate]
        view = data.get('viewpoint') or {}
        if view:
            distance = math.hypot(view['x'] - candidate['x'], view['y'] - candidate['y'])
        if status == 'navigating' and view:
            try_index = int(data.get('try_index', 1) or 1)
            prefix = 'Husky ->' if try_index == 1 else \
                f'pose blocked, Husky -> (try {try_index})'
            self._event('going', self.current_candidate,
                        f'Trial {self.current_attempt}: {prefix} observation pose '
                        f'({view["x"]:.2f}, {view["y"]:.2f}), {distance:.1f} m from it')
            self.going_logged = True
        elif status in ('facing', 'settling') and not self.going_logged:
            self._event('going', self.current_candidate,
                        f'Trial {self.current_attempt}: already '
                        + (f'{distance:.1f} m away, ' if view else '')
                        + 'observing from the current position')
            self.going_logged = True
        if status == 'verifying':
            self.current_request_id = data.get('request_id')
            self.phase = 'verifying'
            self.phase_started = self._now()
            self.get_logger().info(
                f'{self.current_candidate}: YOLO verification started '
                f'(request {self.current_request_id}).')
        elif status in HUSKY_FAILURES:
            self._attempt_failed(status)

    def _result_callback(self, msg):
        if self.state != MissionState.INVESTIGATING or self.phase != 'verifying':
            return
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        if data.get('request_id') is None or data.get('request_id') != self.current_request_id:
            return  # automatic result or an older request

        candidate = self.candidates[self.current_candidate]
        if data.get('status') == 'no_heat':
            outcome, item, error = 'no_heat', None, None
        else:
            foreign_ids = set()
            for other_id, other in self.candidates.items():
                if other_id != self.current_candidate:
                    foreign_ids |= other['source_ids']
            item, error = match_result_item(
                data.get('items', []) or [],
                (candidate['x'], candidate['y']),
                candidate['source_ids'],
                self._robot_pose(),
                self.thermal_width,
                self.thermal_hfov,
                self.bearing_tolerance,
                foreign_ids - candidate['source_ids'],
            )
            outcome = classify_outcome(item, self.confirm_confidence)

        if item is not None and item.get('candidate_id'):
            candidate['source_ids'].add(str(item['candidate_id']))
        detail = {
            'class_name': None if item is None else item.get('class_name'),
            'confidence': None if item is None else item.get('confidence'),
            'bearing_error_deg': None if error is None else round(math.degrees(error), 1),
        }
        self.get_logger().info(
            f'{self.current_candidate}: YOLO result -> {outcome} {detail}')

        trial_text = describe_yolo(item, outcome, self.confirm_confidence)
        if outcome in ('exotic', 'native'):
            self._event('trial', self.current_candidate,
                        f'Trial {self.current_attempt}: {trial_text}')
            candidate['last_outcome'] = outcome
            candidate['history'].append(
                {'attempt': self.current_attempt, 'outcome': outcome, **detail})
            state = (CandidateState.CONFIRMED_EXOTIC if outcome == 'exotic'
                     else CandidateState.CONFIRMED_NATIVE)
            self._finish_candidate(state, item.get('class_name'), item.get('confidence'))
            return
        if outcome == 'no_animal':
            candidate['no_animal_views'] += 1
            if candidate['no_animal_views'] >= self.reject_after:
                self._event('trial', self.current_candidate,
                            f'Trial {self.current_attempt}: {trial_text}')
                candidate['last_outcome'] = outcome
                candidate['history'].append(
                    {'attempt': self.current_attempt, 'outcome': outcome, **detail})
                self._finish_candidate(
                    CandidateState.REJECTED, 'non_animal_heat_source', 0.0)
                return
        self._attempt_failed(outcome, detail, text=trial_text)

    def _explore_status_callback(self, msg):
        self.exploration_status = str(msg.status)
        self.exploration_complete = self.exploration_status in EXPLORATION_DONE
        self.get_logger().info(f'Exploration status: {self.exploration_status}')

    # ======================================================================
    # Helpers
    # ======================================================================

    def _new_candidate(self, candidate_id, x, y, z):
        candidate = {
            'x': x, 'y': y, 'z': z,
            'state': CandidateState.UNINVESTIGATED,
            'class_name': None, 'confidence': 0.0,
            'registry_class': 'unknown', 'registry_confidence': 0.0,
            'observation_count': 0,
            'source_ids': set(),
            'attempts': 0, 'no_animal_views': 0,
            'last_outcome': '', 'history': [],
        }
        self.candidates[candidate_id] = candidate
        return candidate

    def _finished_neighbour(self, candidate_id, x, y):
        if self.duplicate_radius <= 0.0:
            return None
        for other_id, other in self.candidates.items():
            if other_id == candidate_id:
                continue
            if other['state'] in (CandidateState.UNINVESTIGATED, CandidateState.DUPLICATE):
                continue
            if math.hypot(x - other['x'], y - other['y']) > self.duplicate_radius:
                continue
            if self.duplicate_requires_shared_track and not (
                    self.candidates[candidate_id]['source_ids'] & other['source_ids']):
                continue  # a different animal standing close by
            return other_id
        return None

    def _mark_duplicates(self):
        for candidate_id, candidate in self.candidates.items():
            if candidate['state'] != CandidateState.UNINVESTIGATED:
                continue
            duplicate_of = self._finished_neighbour(candidate_id, candidate['x'], candidate['y'])
            if duplicate_of is not None:
                candidate['state'] = CandidateState.DUPLICATE
                candidate['last_outcome'] = f'duplicate_of:{duplicate_of}'
                self.get_logger().info(f'{candidate_id} marked DUPLICATE of {duplicate_of}.')
                self._event('result', candidate_id,
                            f'RESULT {candidate_id} at ({candidate["x"]:.2f}, '
                            f'{candidate["y"]:.2f}): same object as {duplicate_of} (skipped)',
                            state='duplicate', trials=0,
                            summary=f'same object as {duplicate_of} (skipped)')

    def _select_next_candidate(self):
        pending = [cid for cid, c in self.candidates.items()
                   if c['state'] == CandidateState.UNINVESTIGATED]
        if not pending:
            return None
        pose = self._robot_pose()
        if pose is None:
            return pending[0]
        return min(pending, key=lambda cid: math.hypot(
            self.candidates[cid]['x'] - pose[0], self.candidates[cid]['y'] - pose[1]))

    def _robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, Time())
        except tf2_ros.TransformException:
            return None
        t = transform.transform
        return (t.translation.x, t.translation.y, yaw_from_quaternion(t.rotation))

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _event(self, kind, candidate_id, text, **extra):
        """One readable line for heat_source_logger (published in order)."""
        candidate = self.candidates.get(candidate_id, {})
        payload = {'kind': kind, 'candidate_id': candidate_id, 'text': text,
                   'x': candidate.get('x'), 'y': candidate.get('y')}
        payload.update(extra)
        self._publish_json(self.event_pub, payload)

    def _send_husky_cancel(self):
        self._publish_json(self.husky_command_pub, {'command': 'cancel'})

    def _resume_exploration(self, reason):
        msg = Bool()
        msg.data = True
        self.exploration_resume_pub.publish(msg)
        self.exploration_paused = False
        self.last_explore_command = self._now()
        self.get_logger().info(f'Resume sent to exploration ({reason}).')

    def _pause_exploration(self):
        msg = Bool()
        msg.data = False
        self.exploration_resume_pub.publish(msg)
        self.exploration_paused = True
        self.last_explore_command = self._now()
        self.get_logger().info('Pause sent to exploration.')

    def _enforce_exploration_state(self, now):
        """Re-send pause/resume until explore_lite reports the wanted state.

        The resume topic is volatile: a pause sent before explore_lite has
        subscribed (a candidate in the first seconds) was lost, and explore
        kept sending Nav2 goals that pre-empted every observation pose.
        """
        status = self.exploration_status
        running = status in ('exploration_started', 'exploration_in_progress')
        if self.exploration_paused and running:
            wanted = False
        elif not self.exploration_paused and status == 'exploration_paused':
            wanted = True
        else:
            return
        if now - self.last_explore_command < self.explore_resend_sec:
            return
        self.last_explore_command = now
        msg = Bool()
        msg.data = wanted
        self.exploration_resume_pub.publish(msg)
        self.get_logger().warning(
            f'explore_lite reports {status}: re-sending '
            f'{"resume" if wanted else "pause"}.')

    @staticmethod
    def _publish_json(publisher, payload):
        msg = String()
        msg.data = json.dumps(payload, separators=(',', ':'))
        publisher.publish(msg)

    def _counts(self):
        counts = {'exotic': 0, 'native': 0, 'rejected': 0, 'unresolved': 0,
                  'pending': 0, 'duplicate': 0}
        for candidate in self.candidates.values():
            state = candidate['state']
            if state == CandidateState.CONFIRMED_EXOTIC:
                counts['exotic'] += 1
            elif state == CandidateState.CONFIRMED_NATIVE:
                counts['native'] += 1
            elif state == CandidateState.REJECTED:
                counts['rejected'] += 1
            elif state == CandidateState.UNRESOLVED:
                counts['unresolved'] += 1
            elif state == CandidateState.DUPLICATE:
                counts['duplicate'] += 1
            else:
                counts['pending'] += 1
        return counts

    def _summary_text(self):
        c = self._counts()
        return (f'exotic={c["exotic"]} native={c["native"]} rejected={c["rejected"]} '
                f'unresolved={c["unresolved"]} duplicate={c["duplicate"]}')

    # =============  GUI information  =================================

    def _publish_ui_status(self):
        """Mission status + animal information for the UI (JSON).

        ``frame_id`` and ``candidates`` use the same keys as registry_json,
        so the GUI panels can read this topic to show mission states.
        """
        pose = self._robot_pose()
        candidates = []
        for candidate_id, c in self.candidates.items():
            class_name = c['class_name'] or c['registry_class']
            confidence = c['confidence'] if c['class_name'] else c['registry_confidence']
            candidates.append({
                'candidate_id': candidate_id,
                'position': {'x': c['x'], 'y': c['y'], 'z': c['z']},
                'state': c['state'].name.lower(),
                'class_name': class_name,
                'confidence': max(0.0, min(1.0, float(confidence or 0.0))),
                'observation_count': c['observation_count'],
                'attempts': c['attempts'],
                'last_outcome': c['last_outcome'],
            })
        self._publish_json(self.ui_status_pub, {
            'frame_id': self.map_frame,
            'mission_state': self.state.name,
            'exploration_status': self.exploration_status,
            'exploration_paused': self.exploration_paused,
            'robot': None if pose is None else {
                'x': round(pose[0], 3), 'y': round(pose[1], 3), 'yaw': round(pose[2], 3)},
            'current_candidate': self.current_candidate,
            'attempt': self.current_attempt,
            'phase': self.phase,
            'husky_phase': self.husky_phase,
            'counts': self._counts(),
            'candidate_count': len(candidates),
            'candidates': candidates,
        })


def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
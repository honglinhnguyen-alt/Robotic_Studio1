#!/usr/bin/env python3
"""Husky node (R3): go and look at one candidate for the mission node.

For every ``investigate`` command from the mission node it:

  1. picks an observation pose ``standoff`` metres from the candidate, on the
     robot's side first, facing the candidate, and not in an occupied cell of
     the SLAM map;
  2. sends it to Nav2 (NavigateToPose). A failed goal is retried from another
     pose, at most ``max_goal_tries`` poses per attempt (R4);
  3. turns in place to face the candidate (Nav2 Spin), because the Nav2 goal
     checker ignores the final heading (yaw_goal_tolerance 3.14);
  4. waits ``settle_sec`` so the thermal track is stable, then calls
     ``/husky1/perception/verify_now`` so YOLO votes on fresh frames;
  5. reports every step on ``/husky1/mission/husky_status`` (JSON).

The mission node decides what the YOLO result means and whether another
attempt is needed. Attempt k uses ``standoff_distances[k-1]`` and rotates the
viewing direction by ``attempt_angle_offsets_deg[k-1]`` around the candidate,
so a retry sees the candidate from a different place.

Command (std_msgs/String JSON on /husky1/mission/investigate):
  {"command": "investigate", "candidate_id": "candidate_1", "attempt": 1,
   "x": 4.2, "y": -3.1}
  {"command": "cancel"}

Status (std_msgs/String JSON on /husky1/mission/husky_status):
  navigating | facing | settling | verifying (+request_id) |
  unreachable | no_heat_in_view | verify_unavailable | verify_failed | cancelled
"""

import json
import math

import rclpy
import tf2_ros
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from nav2_msgs.action import Spin
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import Trigger


# --------------------------------------------------------------------------
# Pure helpers (no ROS) - imported by the offline tests
# --------------------------------------------------------------------------

def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def viewpoint_candidates(candidate_xy, robot_xy, standoff, attempt_offset, search_offsets):
    """
    Return observation poses ``(x, y, yaw)`` around the candidate, best first.

    The base direction points from the candidate towards the robot, so the
    first pose is on the robot's side (shortest drive, animal already seen
    from there). ``attempt_offset`` rotates the whole search for later
    attempts; ``search_offsets`` are tried in order around that direction.
    Every pose faces the candidate.
    """
    cx, cy = candidate_xy
    base = 0.0 if robot_xy is None else math.atan2(robot_xy[1] - cy, robot_xy[0] - cx)
    poses = []
    for offset in search_offsets:
        angle = base + attempt_offset + offset
        x = cx + standoff * math.cos(angle)
        y = cy + standoff * math.sin(angle)
        poses.append((x, y, math.atan2(cy - y, cx - x)))
    return poses


class GridChecker:
    """Free-space test on a nav_msgs/OccupancyGrid (origin yaw assumed 0)."""

    def __init__(self, grid, occupied_threshold=50):
        info = grid.info
        self.resolution = float(info.resolution)
        self.width = int(info.width)
        self.height = int(info.height)
        self.origin_x = float(info.origin.position.x)
        self.origin_y = float(info.origin.position.y)
        self.data = grid.data
        self.occupied_threshold = occupied_threshold

    def value(self, col, row):
        if col < 0 or row < 0 or col >= self.width or row >= self.height:
            return -1
        return int(self.data[row * self.width + col])

    def is_clear(self, x, y, radius, allow_unknown):
        col = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        # Off the SLAM map Nav2 cannot plan at all; each such goal only failed
        # after ~20 s of recoveries (9 goals per candidate).
        if not (0 <= col < self.width and 0 <= row < self.height):
            return False
        centre = self.value(col, row)
        if centre >= self.occupied_threshold:
            return False
        if centre < 0 and not allow_unknown:
            return False
        cells = int(math.ceil(radius / self.resolution))
        for d_row in range(-cells, cells + 1):
            for d_col in range(-cells, cells + 1):
                if (d_col * d_col + d_row * d_row) * self.resolution ** 2 > radius ** 2:
                    continue
                if self.value(col + d_col, row + d_row) >= self.occupied_threshold:
                    return False
        return True


def pick_viewpoints(poses, checker, radius, max_count):
    """Keep poses whose surroundings are free: known-free first, then unknown."""
    if checker is None:
        return poses[:max_count]
    known = [p for p in poses if checker.is_clear(p[0], p[1], radius, allow_unknown=False)]
    unknown = [p for p in poses
               if p not in known and checker.is_clear(p[0], p[1], radius, allow_unknown=True)]
    return (known + unknown)[:max_count]


def outside_safety(poses, points, radius):
    """Drop poses closer than radius to any of the given (x, y) heat sources."""
    return [p for p in poses
            if all(math.hypot(p[0] - x, p[1] - y) >= radius for x, y in points)]


def list_value(values, attempt):
    """Value for attempt k (1-based); the last one repeats."""
    values = list(values)
    return values[min(max(attempt, 1), len(values)) - 1]


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

class HuskyNode(Node):

    def __init__(self):
        super().__init__('husky_node')

        self.declare_parameter('map_frame', 'husky1_map')
        self.declare_parameter('base_frame', 'husky1_base_link')
        self.declare_parameter('command_topic', '/husky1/mission/investigate')
        self.declare_parameter('status_topic', '/husky1/mission/husky_status')
        self.declare_parameter('map_topic', '/husky1/map')
        self.declare_parameter('navigate_action', '/husky1/navigate_to_pose')
        self.declare_parameter('spin_action', '/husky1/spin')
        self.declare_parameter('verify_service', '/husky1/perception/verify_now')

        # Attempt k uses element k-1 (the last element repeats).
        self.declare_parameter('standoff_distances', [3.5, 3.0, 2.5])
        self.declare_parameter('attempt_angle_offsets_deg', [0.0, 50.0, -50.0])
        self.declare_parameter(
            'search_offsets_deg',
            [0.0, 25.0, -25.0, 50.0, -50.0, 80.0, -80.0, 120.0, -120.0, 180.0])
        self.declare_parameter('clearance_radius', 0.6)
        self.declare_parameter('max_goal_tries', 3)                 # R4
        self.declare_parameter('nav_timeout_sec', 90.0)
        # Attempt 1: if the robot is already this close, only turn and look.
        self.declare_parameter('look_from_current_within', 4.0)     # 0 disables
        self.declare_parameter('min_view_distance', 1.5)
        # The Husky never stops closer than this to any heat source (robot
        # centre to source estimate). Observation poses are placed at least
        # safety_radius + goal_tolerance away so Nav2's goal tolerance cannot
        # bring it inside.
        self.declare_parameter('safety_radius', 2.0)
        self.declare_parameter('goal_tolerance', 0.5)       # Nav2 xy_goal_tolerance
        self.declare_parameter('face_tolerance_deg', 10.0)
        self.declare_parameter('spin_time_allowance_sec', 15.0)
        self.declare_parameter('settle_sec', 1.5)
        self.declare_parameter('verify_retries', 4)
        self.declare_parameter('verify_retry_sec', 1.0)

        p = self.get_parameter
        self.map_frame = str(p('map_frame').value)
        self.base_frame = str(p('base_frame').value)

        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.map_sub = self.create_subscription(
            OccupancyGrid, str(p('map_topic').value), self._map_callback, latched)
        self.command_sub = self.create_subscription(
            String, str(p('command_topic').value), self._command_callback, 10)
        self.status_pub = self.create_publisher(String, str(p('status_topic').value), 10)

        self.nav_client = ActionClient(self, NavigateToPose, str(p('navigate_action').value))
        self.spin_client = ActionClient(self, Spin, str(p('spin_action').value))
        self.verify_client = self.create_client(Trigger, str(p('verify_service').value))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.grid = None
        self.job = None            # current investigation (dict) or None
        self.job_token = 0         # ignores callbacks from older jobs/goals
        self.goal_handle = None
        self.one_shot = None
        self.watchdog = self.create_timer(1.0, self._watchdog)

        self.get_logger().info(
            f'Husky node started | nav={p("navigate_action").value} | '
            f'verify={p("verify_service").value}')

    # ======================================================================
    # Mission commands
    # ======================================================================

    def _command_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            self.get_logger().warning(f'Bad command: {msg.data}')
            return
        command = data.get('command')
        if command == 'cancel':
            job = self.job
            self._stop_job('mission cancel')
            if job is not None:
                self._report('cancelled', job=job)
            return
        if command != 'investigate':
            return
        try:
            candidate_id = str(data['candidate_id'])
            attempt = int(data['attempt'])
            cx, cy = float(data['x']), float(data['y'])
            # Other known heat sources: no observation pose next to them.
            avoid = [(float(x), float(y)) for x, y in data.get('avoid', []) or []]
        except (KeyError, TypeError, ValueError):
            self.get_logger().warning(f'Incomplete investigate command: {msg.data}')
            return
        if (self.job is not None and self.job['candidate_id'] == candidate_id
                and self.job['attempt'] == attempt):
            return  # duplicate command
        if self.job is not None:
            self._stop_job('replaced by a new command')
        self._start_job(candidate_id, attempt, cx, cy, avoid)

    def _start_job(self, candidate_id, attempt, cx, cy, avoid=()):
        self.job_token += 1
        self.job = {
            'candidate_id': candidate_id, 'attempt': attempt,
            'x': cx, 'y': cy, 'token': self.job_token,
            'phase': 'planning', 'viewpoints': [], 'next_index': 0,
            'nav_started': None, 'verify_tries': 0, 'viewpoint': None,
            'goal_seq': 0,
        }
        self.get_logger().info(
            f'Investigate {candidate_id} (attempt {attempt}) at ({cx:.2f}, {cy:.2f})')

        robot = self._robot_pose()
        safety = float(self.get_parameter('safety_radius').value)
        keep_away = safety + float(self.get_parameter('goal_tolerance').value)
        within = float(self.get_parameter('look_from_current_within').value)
        if attempt == 1 and robot is not None and within > 0.0:
            distance = math.hypot(cx - robot[0], cy - robot[1])
            nearest_other = min(
                (math.hypot(x - robot[0], y - robot[1]) for x, y in avoid),
                default=math.inf)
            closest = max(float(self.get_parameter('min_view_distance').value), safety)
            if closest <= distance <= within and nearest_other >= safety:
                self.get_logger().info(
                    f'Already {distance:.1f} m from the candidate: look from here.')
                self.job['viewpoint'] = {'x': robot[0], 'y': robot[1], 'yaw': robot[2]}
                self._face_candidate()
                return

        standoff = max(
            float(list_value(self.get_parameter('standoff_distances').value, attempt)),
            keep_away)
        offset = math.radians(float(list_value(
            self.get_parameter('attempt_angle_offsets_deg').value, attempt)))
        search = [math.radians(float(v))
                  for v in self.get_parameter('search_offsets_deg').value]
        poses = outside_safety(
            viewpoint_candidates(
                (cx, cy), None if robot is None else robot[:2], standoff, offset, search),
            avoid, keep_away)
        checker = GridChecker(self.grid) if self.grid is not None else None
        self.job['viewpoints'] = pick_viewpoints(
            poses, checker,
            float(self.get_parameter('clearance_radius').value),
            max(1, int(self.get_parameter('max_goal_tries').value)))
        if not self.job['viewpoints']:
            self._fail('unreachable', 'no free observation pose around the candidate')
            return
        self._try_next_viewpoint()

    # ======================================================================
    # Navigation
    # ======================================================================

    def _try_next_viewpoint(self):
        job = self.job
        if job['next_index'] >= len(job['viewpoints']):
            self._fail('unreachable', f'{len(job["viewpoints"])} observation poses failed')
            return
        x, y, yaw = job['viewpoints'][job['next_index']]
        job['next_index'] += 1
        job['viewpoint'] = {'x': round(x, 3), 'y': round(y, 3), 'yaw': round(yaw, 3)}

        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self._fail('unreachable', 'Nav2 NavigateToPose server not available')
            return

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        job['phase'] = 'navigating'
        job['nav_started'] = self._now()
        job['goal_seq'] += 1
        self._report('navigating', try_index=job['next_index'])
        self.get_logger().info(
            f'Nav2 goal {job["next_index"]}/{len(job["viewpoints"])}: '
            f'({x:.2f}, {y:.2f}) facing the candidate')
        token = (job['token'], job['goal_seq'])
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(lambda f: self._goal_response(f, token))

    def _goal_response(self, future, token):
        if not self._current(*token):
            return
        try:
            handle = future.result()
        except Exception as error:  # noqa: BLE001 - rclpy raises several types
            self.get_logger().warning(f'Nav2 goal send failed: {error}')
            self._try_next_viewpoint()
            return
        if not handle.accepted:
            self.get_logger().warning('Nav2 goal rejected.')
            self._try_next_viewpoint()
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(
            lambda f: self._navigation_result(f, token))

    def _navigation_result(self, future, token):
        if not self._current(*token) or self.job['phase'] != 'navigating':
            return
        self.goal_handle = None
        try:
            status = future.result().status
        except Exception as error:  # noqa: BLE001
            self.get_logger().warning(f'Nav2 result failed: {error}')
            status = GoalStatus.STATUS_ABORTED
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Observation pose reached.')
            self._face_candidate()
            return
        self.get_logger().warning(f'Nav2 goal ended with status {status}; trying another pose.')
        self._try_next_viewpoint()

    # ======================================================================
    # Face the candidate, settle, verify
    # ======================================================================

    def _face_candidate(self):
        job = self.job
        job['phase'] = 'facing'
        robot = self._robot_pose()
        if robot is None:
            self._settle()
            return
        error = wrap_angle(math.atan2(job['y'] - robot[1], job['x'] - robot[0]) - robot[2])
        tolerance = math.radians(float(self.get_parameter('face_tolerance_deg').value))
        if abs(error) <= tolerance or not self.spin_client.wait_for_server(timeout_sec=1.0):
            self._settle()
            return
        self._report('facing', turn_deg=round(math.degrees(error), 1))
        goal = Spin.Goal()
        goal.target_yaw = float(error)
        goal.time_allowance = Duration(
            seconds=float(self.get_parameter('spin_time_allowance_sec').value)).to_msg()
        job['goal_seq'] += 1
        token = (job['token'], job['goal_seq'])
        self.spin_client.send_goal_async(goal).add_done_callback(
            lambda f: self._spin_response(f, token))

    def _spin_response(self, future, token):
        if not self._current(*token):
            return
        try:
            handle = future.result()
        except Exception:  # noqa: BLE001
            handle = None
        if handle is None or not handle.accepted:
            self._settle()
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self._spin_done(token))

    def _spin_done(self, token):
        if self._current(*token) and self.job['phase'] == 'facing':
            self.goal_handle = None
            self._settle()

    def _settle(self):
        self.job['phase'] = 'settling'
        self._report('settling')
        self._after(float(self.get_parameter('settle_sec').value), self._call_verify)

    def _call_verify(self):
        job = self.job
        if job is None:
            return
        job['phase'] = 'verify_call'
        job['verify_tries'] += 1
        if not self.verify_client.service_is_ready():
            self._retry_verify('verify_unavailable', 'verify_now service not available')
            return
        token = job['token']
        self.verify_client.call_async(Trigger.Request()).add_done_callback(
            lambda f: self._verify_response(f, token))

    def _verify_response(self, future, token):
        if not self._current(token):
            return
        try:
            response = future.result()
        except Exception as error:  # noqa: BLE001
            self._retry_verify('verify_failed', str(error))
            return
        try:
            info = json.loads(response.message)
        except ValueError:
            info = {'status': response.message}
        if response.success:
            self.get_logger().info(
                f'YOLO verification started (request {info.get("request_id")}).')
            job = self.job
            self._end_job()          # Husky is done; the mission decides next
            self._report('verifying', job=job, request_id=info.get('request_id'),
                         regions=info.get('regions'))
            return
        status = str(info.get('status', 'verify_failed'))
        if status not in ('no_heat_in_view', 'verify_unavailable'):
            status = 'verify_failed'
        self._retry_verify(status, response.message)

    def _retry_verify(self, status, detail):
        # The thermal detector needs a few frames after the robot stops
        # turning before it publishes the heat region again.
        if self.job['verify_tries'] < int(self.get_parameter('verify_retries').value):
            self.job['phase'] = 'settling'
            self._after(float(self.get_parameter('verify_retry_sec').value), self._call_verify)
            return
        self._fail(status, detail)

    # ======================================================================
    # Job bookkeeping
    # ======================================================================

    def _watchdog(self):
        job = self.job
        if job is None or job['phase'] != 'navigating' or job['nav_started'] is None:
            return
        if self._now() - job['nav_started'] > float(self.get_parameter('nav_timeout_sec').value):
            self.get_logger().warning('Nav2 goal timed out; trying another pose.')
            self._cancel_goal()          # its late result is ignored (goal_seq)
            job['nav_started'] = None
            self._try_next_viewpoint()

    def _fail(self, status, detail):
        self.get_logger().warning(f'{self.job["candidate_id"]}: {status} ({detail})')
        job = self.job
        self._end_job()              # end first: the mission may reply at once
        self._report(status, job=job, detail=detail)

    def _stop_job(self, reason):
        if self.job is None:
            return
        self.get_logger().info(f'Stopping {self.job["candidate_id"]}: {reason}')
        self._cancel_goal()
        self._end_job()

    def _end_job(self):
        self.job = None
        self.job_token += 1          # stale callbacks are ignored from now on
        self.goal_handle = None
        if self.one_shot is not None:
            self.one_shot.cancel()
            self.one_shot = None

    def _cancel_goal(self):
        if self.job is not None:
            self.job['goal_seq'] += 1    # ignore the cancelled goal's result
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None

    def _current(self, token, goal_seq=None):
        if self.job is None or self.job['token'] != token:
            return False
        return goal_seq is None or self.job['goal_seq'] == goal_seq

    def _after(self, seconds, callback):
        """Run ``callback`` once after ``seconds`` (only for the current job)."""
        if self.one_shot is not None:
            self.one_shot.cancel()
        token = self.job['token']

        def fire():
            if self.one_shot is not None:
                self.one_shot.cancel()
                self.one_shot = None
            if self._current(token):
                callback()
        self.one_shot = self.create_timer(max(0.01, seconds), fire)

    def _report(self, status, job=None, **extra):
        job = job or self.job or {}
        payload = {
            'candidate_id': job.get('candidate_id'),
            'attempt': job.get('attempt'),
            'status': status,
            'viewpoint': job.get('viewpoint'),
        }
        payload.update(extra)
        msg = String()
        msg.data = json.dumps(payload, separators=(',', ':'))
        self.status_pub.publish(msg)

    def _map_callback(self, msg):
        self.grid = msg

    def _robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except tf2_ros.TransformException:
            return None
        t = transform.transform
        return (t.translation.x, t.translation.y, yaw_from_quaternion(t.rotation))

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = HuskyNode()
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
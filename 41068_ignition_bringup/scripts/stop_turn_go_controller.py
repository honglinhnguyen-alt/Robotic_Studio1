#!/usr/bin/env python3
"""Strict stop-turn-go velocity gate for the Husky.

This node sits between Nav2 and the Gazebo velocity bridge:

    Nav2 /husky1/cmd_vel -> this node -> /husky1/cmd_vel_stg -> Gazebo

Nav2 keeps doing the planning, costmaps, obstacle avoidance and replanning.
This node only changes HOW the motion is executed: the Husky either turns in
place or drives straight, never both at once (linear.x and angular.z are
never non-zero together), with a short stop between the two.

When to turn is decided from Nav2's global path, not from DWB's angular
request. DWB asks for small curvatures all the time; reacting to those made
the robot stop every few centimetres. Instead:

  heading error = direction to a point `lookahead_distance` ahead on the
                  path (from the path point closest to the robot) - yaw

  DRIVE  while |error| <= turn_enter_threshold, straight, at the speed DWB
         asks for (DWB still slows down near obstacles and the goal);
  TURN   in place when |error| > turn_enter_threshold, until
         |error| <= turn_exit_threshold (hysteresis), then DRIVE again.

Pure rotations requested by Nav2 itself (RotationShim aligning to a new path,
the Spin behaviour used by husky_node to face a candidate, recoveries) are
passed through. Without a fresh path the gate falls back to DWB's angular
request with the same hysteresis.

A front safety corridor (robot width + margin, straight ahead) stops forward
motion when the LiDAR sees something inside it; turning is still allowed so
Nav2 can align with a replanned path.
"""

import math
import time

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan


# --------------------------------------------------------------------------
# Pure helpers (no ROS) - imported by the offline tests
# --------------------------------------------------------------------------

def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def heading_to_path(path_xy, robot_x, robot_y, robot_yaw, lookahead):
    """Return (heading error [rad], distance to the lookahead point [m]).

    The lookahead point lies `lookahead` metres along the path after the path
    point closest to the robot (the last point if the path is shorter).
    Returns None for an empty path.
    """
    if path_xy is None or len(path_xy) == 0:
        return None
    path_xy = np.asarray(path_xy, dtype=float)
    closest = int(np.argmin(np.hypot(path_xy[:, 0] - robot_x, path_xy[:, 1] - robot_y)))
    target = path_xy[-1]
    travelled = 0.0
    for index in range(closest + 1, len(path_xy)):
        travelled += float(np.hypot(*(path_xy[index] - path_xy[index - 1])))
        if travelled >= lookahead:
            target = path_xy[index]
            break
    dx = float(target[0]) - robot_x
    dy = float(target[1]) - robot_y
    return wrap_angle(math.atan2(dy, dx) - robot_yaw), math.hypot(dx, dy)


def corridor_distance(ranges, angle_min, angle_increment, range_min, range_max,
                      half_width, start=0.0):
    """Closest forward distance of a LiDAR point inside the straight corridor.

    The corridor is |y| <= half_width, x > start in the scan frame; points
    beside the chassis (x <= start) cannot be hit by driving straight. Works
    for scans over [-pi, pi] as well as [0, 2 pi].
    """
    ranges = np.asarray(ranges, dtype=float)
    angles = angle_min + angle_increment * np.arange(len(ranges))
    valid = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)
    ranges = np.where(valid, ranges, 0.0)
    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)
    inside = valid & (xs > start) & (np.abs(ys) <= half_width)
    return float(xs[inside].min()) if inside.any() else math.inf


class MotionGate:
    """STOPPED / TURNING / MOVING state machine; at most one motion at a time."""

    STOPPED = 'STOPPED'
    TURNING = 'TURNING'
    MOVING = 'MOVING'

    def __init__(self, turn_enter=0.35, turn_exit=0.08, rotate_gain=1.5,
                 min_turn_speed=1.0, max_turn_speed=1.0, max_linear_speed=1.0,
                 linear_request=0.02, angular_request=0.05,
                 fallback_turn_enter=0.30, fallback_turn_exit=0.10,
                 min_turn_distance=0.3, minimum_stop_time=0.10,
                 linear_settled=0.05, angular_settled=0.10, settle_timeout=1.0,
                 allow_reverse=False):
        self.turn_enter = turn_enter
        self.turn_exit = turn_exit
        self.rotate_gain = rotate_gain
        self.min_turn_speed = min_turn_speed
        self.max_turn_speed = max_turn_speed
        self.max_linear_speed = max_linear_speed
        self.linear_request = linear_request
        self.angular_request = angular_request
        self.fallback_turn_enter = fallback_turn_enter
        self.fallback_turn_exit = fallback_turn_exit
        self.min_turn_distance = min_turn_distance
        self.minimum_stop_time = minimum_stop_time
        self.linear_settled = linear_settled
        self.angular_settled = angular_settled
        self.settle_timeout = settle_timeout
        self.allow_reverse = allow_reverse

        self.mode = self.STOPPED
        self.stop_started = 0.0
        self.reason = 'start'

    def _turn_speed(self, magnitude):
        return min(self.max_turn_speed, max(self.min_turn_speed, magnitude))

    def desired(self, req_v, req_w, heading):
        """Return (mode, v, w, reason) wanted now, ignoring the stop between phases.

        heading: (error, target distance) from heading_to_path, or None.
        """
        turning = self.mode == self.TURNING
        wants_forward = req_v > self.linear_request
        wants_reverse = req_v < -self.linear_request

        # Pure rotation from Nav2 (RotationShim, Spin, recoveries): pass through.
        if not wants_forward and not wants_reverse and abs(req_w) >= self.angular_request:
            return self.TURNING, 0.0, math.copysign(self._turn_speed(abs(req_w)), req_w), \
                'nav2 rotation'

        if wants_reverse and self.allow_reverse:
            return self.MOVING, max(req_v, -self.max_linear_speed), 0.0, 'reverse'

        if not wants_forward:
            return self.STOPPED, 0.0, 0.0, 'no forward request'

        if heading is not None and heading[1] >= self.min_turn_distance:
            error = heading[0]
            limit = self.turn_exit if turning else self.turn_enter
            if abs(error) > limit:
                return self.TURNING, 0.0, math.copysign(
                    self._turn_speed(self.rotate_gain * abs(error)), error), \
                    f'path heading error {math.degrees(error):+.0f} deg'
        elif heading is None:
            # No fresh path: fall back to DWB's angular request.
            limit = self.fallback_turn_exit if turning else self.fallback_turn_enter
            if abs(req_w) > limit:
                return self.TURNING, 0.0, math.copysign(self._turn_speed(abs(req_w)), req_w), \
                    'dwb angular request (no path)'

        return self.MOVING, min(req_v, self.max_linear_speed), 0.0, 'drive straight'

    def step(self, now, req_v, req_w, heading, obstacle_ahead, meas_v, meas_w):
        """Return the (v, w) to publish. v and w are never both non-zero."""
        mode, v, w, reason = self.desired(req_v, req_w, heading)
        if mode == self.MOVING and v > 0.0 and obstacle_ahead:
            mode, v, w, reason = self.STOPPED, 0.0, 0.0, 'obstacle in front corridor'
        self.reason = reason

        if mode == self.mode and mode != self.STOPPED:
            return v, w

        # Changing motion (or stopping): go through STOPPED first.
        if self.mode != self.STOPPED:
            self.mode = self.STOPPED
            self.stop_started = now
        if mode == self.STOPPED:
            return 0.0, 0.0

        waited = now - self.stop_started
        if waited < self.minimum_stop_time:
            return 0.0, 0.0
        # Before turning the chassis must have stopped rolling, before driving
        # it must have stopped turning (skid steer). Give up waiting after
        # settle_timeout so odometry noise cannot block the robot.
        settled = (abs(meas_v) <= self.linear_settled if mode == self.TURNING
                   else abs(meas_w) <= self.angular_settled)
        if not settled and waited < self.settle_timeout:
            return 0.0, 0.0
        self.mode = mode
        return v, w


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

class StopTurnGoController(Node):
    """Turn ordinary Nav2 cmd_vel into strict stop-turn-go motion along the path."""

    def __init__(self):
        super().__init__('stop_turn_go_controller')

        # Relative topics: inside namespace /husky1 they become
        #   /husky1/cmd_vel (Nav2) -> /husky1/cmd_vel_stg (Gazebo bridge).
        self.declare_parameter('input_cmd_topic', 'cmd_vel')
        self.declare_parameter('output_cmd_topic', 'cmd_vel_stg')
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('odom_topic', 'odom')
        self.declare_parameter('path_topic', 'plan')
        self.declare_parameter('map_frame', 'husky1_map')
        self.declare_parameter('base_frame', 'husky1_base_link')

        # Turn / drive decision (path heading error, rad).
        self.declare_parameter('lookahead_distance', 0.8)       # m along the path
        self.declare_parameter('turn_enter_threshold', 0.35)    # ~20 deg -> stop and turn
        self.declare_parameter('turn_exit_threshold', 0.08)     # ~5 deg -> stop and drive
        self.declare_parameter('rotate_gain', 1.5)              # w = gain * error
        # Measured in large_demo (Gazebo Fortress): an in-place turn command of
        # 0.5-0.6 rad/s does not move the Husky at all, 0.7-0.8 only sometimes,
        # and from 1.0 rad/s it turns at its limit of ~0.5 rad/s. RotationShim
        # limits its own command by the MEASURED rate, so below the threshold
        # it never got the robot turning (10 s stalls, failed Spin recoveries).
        self.declare_parameter('minimum_turn_speed', 1.0)       # rad/s commanded
        self.declare_parameter('maximum_turn_speed', 1.0)       # rad/s commanded
        self.declare_parameter('maximum_linear_speed', 1.0)     # m/s
        self.declare_parameter('min_turn_distance', 0.3)        # m: no turning onto the last point
        self.declare_parameter('path_timeout', 10.0)            # s: older path -> fallback
        # Fallback without a path: DWB angular request [rad/s].
        self.declare_parameter('fallback_turn_enter', 0.30)
        self.declare_parameter('fallback_turn_exit', 0.10)

        self.declare_parameter('minimum_stop_time', 0.10)       # s between phases
        self.declare_parameter('linear_settled_threshold', 0.05)   # m/s measured
        self.declare_parameter('angular_settled_threshold', 0.10)  # rad/s measured
        self.declare_parameter('settle_timeout', 1.0)           # s
        self.declare_parameter('command_timeout', 0.50)         # s
        self.declare_parameter('control_frequency', 20.0)       # Hz
        self.declare_parameter('allow_reverse', False)

        # Front safety corridor in the scan frame (LiDAR is above base_link).
        self.declare_parameter('stopping_distance', 0.9)        # m ahead of the LiDAR
        self.declare_parameter('corridor_half_width', 0.45)     # m, Husky 0.35 + margin
        self.declare_parameter('corridor_start', 0.35)          # m, ignore points beside the body
        self.declare_parameter('obstacle_clear_hysteresis', 0.15)  # m
        self.declare_parameter('scan_timeout', 0.75)            # s
        self.declare_parameter('debug_log_period', 1.0)         # s, 0 disables

        p = self.get_parameter
        self.gate = MotionGate(
            turn_enter=float(p('turn_enter_threshold').value),
            turn_exit=float(p('turn_exit_threshold').value),
            rotate_gain=float(p('rotate_gain').value),
            min_turn_speed=float(p('minimum_turn_speed').value),
            max_turn_speed=float(p('maximum_turn_speed').value),
            max_linear_speed=float(p('maximum_linear_speed').value),
            fallback_turn_enter=float(p('fallback_turn_enter').value),
            fallback_turn_exit=float(p('fallback_turn_exit').value),
            min_turn_distance=float(p('min_turn_distance').value),
            minimum_stop_time=float(p('minimum_stop_time').value),
            linear_settled=float(p('linear_settled_threshold').value),
            angular_settled=float(p('angular_settled_threshold').value),
            settle_timeout=float(p('settle_timeout').value),
            allow_reverse=bool(p('allow_reverse').value),
        )
        self.map_frame = str(p('map_frame').value)
        self.base_frame = str(p('base_frame').value)
        self.lookahead = float(p('lookahead_distance').value)
        self.path_timeout = float(p('path_timeout').value)
        self.command_timeout = float(p('command_timeout').value)
        self.stopping_distance = float(p('stopping_distance').value)
        self.corridor_half_width = float(p('corridor_half_width').value)
        self.corridor_start = float(p('corridor_start').value)
        self.obstacle_clear_hysteresis = float(p('obstacle_clear_hysteresis').value)
        self.scan_timeout = float(p('scan_timeout').value)
        self.debug_log_period = float(p('debug_log_period').value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.publisher = self.create_publisher(Twist, str(p('output_cmd_topic').value), 10)
        self.create_subscription(Twist, str(p('input_cmd_topic').value), self._cmd_callback, 10)
        self.create_subscription(LaserScan, str(p('scan_topic').value), self._scan_callback, 10)
        self.create_subscription(Odometry, str(p('odom_topic').value), self._odom_callback, 10)
        self.create_subscription(Path, str(p('path_topic').value), self._path_callback, 10)

        self.requested_cmd = Twist()
        self.measured_linear = 0.0
        self.measured_angular = 0.0
        self.front_distance = math.inf
        self.obstacle_blocked = False
        self.path_xy = None
        self.path_frame = self.map_frame
        now = time.monotonic()
        self.last_cmd_time = now
        self.last_scan_time = 0.0
        self.last_path_time = 0.0
        self.last_mode_log = None
        self.last_debug_log_time = 0.0

        frequency = max(5.0, float(p('control_frequency').value))
        self.timer = self.create_timer(1.0 / frequency, self._control_loop)
        self.get_logger().info(
            'Stop-turn-go gate active: Nav2 cmd_vel -> cmd_vel_stg. Turn in place when the '
            f'path heading error exceeds {math.degrees(self.gate.turn_enter):.0f} deg, '
            f'drive straight below {math.degrees(self.gate.turn_exit):.0f} deg; '
            'linear.x and angular.z are never non-zero together.')

    def _cmd_callback(self, msg):
        self.requested_cmd = msg
        self.last_cmd_time = time.monotonic()

    def _odom_callback(self, msg):
        self.measured_linear = float(msg.twist.twist.linear.x)
        self.measured_angular = float(msg.twist.twist.angular.z)

    def _path_callback(self, msg):
        if not msg.poses:
            self.path_xy = None
            return
        self.path_xy = np.array(
            [[pose.pose.position.x, pose.pose.position.y] for pose in msg.poses])
        self.path_frame = msg.header.frame_id or self.map_frame
        self.last_path_time = time.monotonic()

    def _scan_callback(self, msg):
        self.front_distance = corridor_distance(
            msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max,
            self.corridor_half_width, self.corridor_start)
        self.last_scan_time = time.monotonic()
        # Hysteresis: enter at stopping_distance, leave with extra clearance.
        if self.obstacle_blocked:
            if self.front_distance > self.stopping_distance + self.obstacle_clear_hysteresis:
                self.obstacle_blocked = False
        elif self.front_distance < self.stopping_distance:
            self.obstacle_blocked = True

    def _heading(self, now):
        if self.path_xy is None or now - self.last_path_time > self.path_timeout:
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.path_frame, self.base_frame, Time())
        except tf2_ros.TransformException:
            return None
        t = transform.transform
        return heading_to_path(self.path_xy, t.translation.x, t.translation.y,
                               yaw_from_quaternion(t.rotation), self.lookahead)

    def _control_loop(self):
        now = time.monotonic()
        out = Twist()

        # Fail-safe: a stale Nav2 command means STOP.
        if now - self.last_cmd_time > self.command_timeout:
            self.gate.step(now, 0.0, 0.0, None, False, self.measured_linear,
                           self.measured_angular)
            self.publisher.publish(out)
            self._log_mode()
            return

        # A missing scan does not invent an obstacle; Nav2's costmaps still apply.
        scan_fresh = (self.last_scan_time > 0.0
                      and now - self.last_scan_time <= self.scan_timeout)
        heading = self._heading(now)
        v, w = self.gate.step(
            now, float(self.requested_cmd.linear.x), float(self.requested_cmd.angular.z),
            heading, self.obstacle_blocked and scan_fresh,
            self.measured_linear, self.measured_angular)
        out.linear.x = v
        out.angular.z = w
        self.publisher.publish(out)
        self._log_mode()

        if self.debug_log_period > 0.0 and now - self.last_debug_log_time >= self.debug_log_period:
            self.last_debug_log_time = now
            error = 'n/a' if heading is None else f'{math.degrees(heading[0]):+.0f}deg'
            self.get_logger().debug(
                f'STG mode={self.gate.mode} ({self.gate.reason}) | '
                f'req[v={self.requested_cmd.linear.x:+.2f}, w={self.requested_cmd.angular.z:+.2f}]'
                f' out[v={v:+.2f}, w={w:+.2f}] | path error={error} | '
                f'front={self.front_distance:.2f}m blocked={self.obstacle_blocked}')

    def _log_mode(self):
        if self.gate.mode != self.last_mode_log:
            self.last_mode_log = self.gate.mode
            self.get_logger().info(f'STG -> {self.gate.mode} ({self.gate.reason})')


def main(args=None):
    rclpy.init(args=args)
    node = StopTurnGoController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publisher.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

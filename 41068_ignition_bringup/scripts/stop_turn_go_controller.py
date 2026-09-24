#!/usr/bin/env python3
"""Strict stop-turn-go velocity gate for the Husky.

This node sits between Nav2 and the Gazebo velocity bridge.

Nav2 continues to do the planning, costmap updates, obstacle avoidance and
replanning.  This node only changes *how* the requested velocity is executed:

    STOP -> TURN IN PLACE -> STOP -> DRIVE STRAIGHT

It never publishes non-zero linear.x and angular.z at the same time.
A small front LaserScan safety zone also forces an immediate stop while Nav2's
costmap / behaviour tree updates and generates a new path to the same goal.
"""

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class StopTurnGoController(Node):
    """Convert ordinary Nav2 cmd_vel commands into strict stop-turn-go motion."""

    STOPPED = 'STOPPED'
    TURNING = 'TURNING'
    MOVING = 'MOVING'

    def __init__(self):
        super().__init__('stop_turn_go_controller')

        # Topics are relative so, inside namespace /husky1, they become:
        #   input  /husky1/cmd_vel       (Nav2 / teleop request)
        #   output /husky1/cmd_vel_stg   (sent to Gazebo bridge)
        self.declare_parameter('input_cmd_topic', 'cmd_vel')
        self.declare_parameter('output_cmd_topic', 'cmd_vel_stg')
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('odom_topic', 'odom')

        # Motion-state thresholds.
        # enter > exit gives hysteresis and avoids rapid TURN/MOVE switching.
        self.declare_parameter('turn_enter_threshold', 0.10)   # rad/s requested by Nav2
        self.declare_parameter('turn_exit_threshold', 0.035)  # rad/s requested by Nav2
        self.declare_parameter('minimum_turn_speed', 0.70)
        self.declare_parameter('linear_request_threshold', 0.015)  # m/s
        self.declare_parameter('linear_settled_threshold', 0.015)  # m/s measured
        self.declare_parameter('angular_settled_threshold', 0.045)  # rad/s measured
        self.declare_parameter('minimum_stop_time', 0.20)      # s between phases
        self.declare_parameter('command_timeout', 0.50)        # s
        self.declare_parameter('control_frequency', 20.0)      # Hz
        self.declare_parameter('allow_reverse', False)

        # Independent front safety stop.  LiDAR still feeds the Nav2 costmaps;
        # this layer only prevents forward motion while the obstacle is close.
        self.declare_parameter('stopping_distance', 1.10)      # m from LiDAR
        self.declare_parameter('front_sector_half_angle', 0.55)  # rad ~= 31.5 deg
        self.declare_parameter('obstacle_clear_hysteresis', 0.15)  # m
        self.declare_parameter('scan_timeout', 0.75)            # s

        self.input_cmd_topic = str(self.get_parameter('input_cmd_topic').value)
        self.output_cmd_topic = str(self.get_parameter('output_cmd_topic').value)
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)

        self.turn_enter = float(self.get_parameter('turn_enter_threshold').value)
        self.turn_exit = float(self.get_parameter('turn_exit_threshold').value)
        self.minimum_turn_speed = float(
            self.get_parameter('minimum_turn_speed').value
        )
        self.linear_request_threshold = float(
            self.get_parameter('linear_request_threshold').value)
        self.linear_settled = float(self.get_parameter('linear_settled_threshold').value)
        self.angular_settled = float(self.get_parameter('angular_settled_threshold').value)
        self.minimum_stop_time = float(self.get_parameter('minimum_stop_time').value)
        self.command_timeout = float(self.get_parameter('command_timeout').value)
        self.allow_reverse = bool(self.get_parameter('allow_reverse').value)

        self.stopping_distance = float(self.get_parameter('stopping_distance').value)
        self.front_sector_half_angle = float(
            self.get_parameter('front_sector_half_angle').value)
        self.obstacle_clear_hysteresis = float(
            self.get_parameter('obstacle_clear_hysteresis').value)
        self.scan_timeout = float(self.get_parameter('scan_timeout').value)

        frequency = max(5.0, float(self.get_parameter('control_frequency').value))

        self.publisher = self.create_publisher(Twist, self.output_cmd_topic, 10)
        self.create_subscription(Twist, self.input_cmd_topic, self._cmd_callback, 10)
        self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 10)

        self.requested_cmd = Twist()
        self.measured_linear = 0.0
        self.measured_angular = 0.0
        self.front_distance = math.inf

        now = time.monotonic()
        self.last_cmd_time = now
        self.last_scan_time = 0.0
        self.stop_started = now
        self.mode = self.STOPPED
        self.obstacle_blocked = False
        self.last_mode_log = None
        self.last_obstacle_log_time = 0.0
        self.last_debug_log_time = 0.0

        self.timer = self.create_timer(1.0 / frequency, self._control_loop)

        self.get_logger().info(
            'Stop-turn-go controller active: Nav2 cmd_vel -> cmd_vel_stg. '
            'Strict rule: linear.x and angular.z are never non-zero together.'
        )

    @staticmethod
    def _zero_cmd():
        return Twist()

    def _cmd_callback(self, msg: Twist):
        self.requested_cmd = msg
        self.last_cmd_time = time.monotonic()

    def _odom_callback(self, msg: Odometry):
        self.measured_linear = float(msg.twist.twist.linear.x)
        self.measured_angular = float(msg.twist.twist.angular.z)

    def _scan_callback(self, msg: LaserScan):
        minimum = math.inf
        angle = msg.angle_min

        for distance in msg.ranges:
            if -self.front_sector_half_angle <= angle <= self.front_sector_half_angle:
                if math.isfinite(distance) and msg.range_min <= distance <= msg.range_max:
                    minimum = min(minimum, float(distance))
            angle += msg.angle_increment

        self.front_distance = minimum
        self.last_scan_time = time.monotonic()

        # Hysteresis: enter the stop zone at stopping_distance, leave only after
        # gaining an additional obstacle_clear_hysteresis clearance.
        if self.obstacle_blocked:
            if minimum > self.stopping_distance + self.obstacle_clear_hysteresis:
                self.obstacle_blocked = False
        elif minimum < self.stopping_distance:
            self.obstacle_blocked = True

    def _robot_is_settled(self):
        return (
            abs(self.measured_linear) <= self.linear_settled
            and abs(self.measured_angular) <= self.angular_settled
        )

    def _enter_stopped(self):
        if self.mode != self.STOPPED:
            self.stop_started = time.monotonic()
        self.mode = self.STOPPED

    def _log_mode_if_changed(self):
        if self.mode == self.last_mode_log:
            return
        self.last_mode_log = self.mode
        self.get_logger().info(f'STG state -> {self.mode}')

    def _control_loop(self):
        now = time.monotonic()

        # Fail-safe: stale Nav2 command means STOP.
        if now - self.last_cmd_time > self.command_timeout:
            self._enter_stopped()
            self.publisher.publish(self._zero_cmd())
            self._log_mode_if_changed()
            return

        # If scan disappears, do not invent an obstacle, but report it.  Nav2's
        # own costmap source timeout / controller safety still applies.
        scan_fresh = self.last_scan_time > 0.0 and (now - self.last_scan_time) <= self.scan_timeout
        obstacle_ahead = self.obstacle_blocked if scan_fresh else False

        requested_linear = float(self.requested_cmd.linear.x)
        requested_angular = float(self.requested_cmd.angular.z)

        wants_turn = abs(requested_angular) >= self.turn_enter
        turn_finished = abs(requested_angular) <= self.turn_exit
        wants_forward = requested_linear > self.linear_request_threshold
        wants_reverse = requested_linear < -self.linear_request_threshold

        # Periodic diagnostic telemetry. This is intentionally rate-limited so
        # the terminal stays readable while showing exactly why the state
        # machine is (or is not) allowed to leave STOPPED.
        if now - self.last_debug_log_time >= 0.50:
            self.last_debug_log_time = now
            scan_age = (now - self.last_scan_time) if self.last_scan_time > 0.0 else math.inf
            self.get_logger().info(
                'STG DEBUG | '
                f'mode={self.mode} | '
                f'req[v={requested_linear:+.3f}, w={requested_angular:+.3f}] | '
                f'odom[v={self.measured_linear:+.3f}, w={self.measured_angular:+.3f}] | '
                f'front={self.front_distance:.2f}m blocked={obstacle_ahead} | '
                f'settled={self._robot_is_settled()} | '
                f'wants_turn={wants_turn} turn_finished={turn_finished} '
                f'wants_forward={wants_forward} wants_reverse={wants_reverse} | '
                f'cmd_age={now - self.last_cmd_time:.2f}s scan_age={scan_age:.2f}s'
            )

        if wants_reverse and not self.allow_reverse:
            # The requested strategy is forward-only stop-turn-go.  Block Nav2
            # recovery backup commands rather than reversing unexpectedly.
            wants_forward = False

        # --------------------------------------------------------------
        # MOVING: angular motion is forbidden.  Any meaningful steering
        # request or close obstacle first forces a complete stop.
        # --------------------------------------------------------------
        if self.mode == self.MOVING:
            if obstacle_ahead or wants_turn or wants_reverse:
                self._enter_stopped()
                self.publisher.publish(self._zero_cmd())
            elif wants_forward:
                out = Twist()
                out.linear.x = max(0.0, requested_linear) if not self.allow_reverse else requested_linear
                out.angular.z = 0.0
                self.publisher.publish(out)
            else:
                self._enter_stopped()
                self.publisher.publish(self._zero_cmd())

        # --------------------------------------------------------------
        # TURNING: linear motion is forbidden.  When Nav2 no longer asks for
        # significant rotation, STOP first; MOVING can only start afterwards.
        # --------------------------------------------------------------
        elif self.mode == self.TURNING:
            if turn_finished:
                self._enter_stopped()
                self.publisher.publish(self._zero_cmd())
            else:
                out = Twist()
                out.linear.x = 0.0

                # The simulated Husky does not reliably rotate in place
                # at very small angular velocities. Preserve Nav2's
                # requested direction, but enforce a minimum turn speed.
                turn_speed = max(
                    abs(requested_angular),
                    self.minimum_turn_speed
                )

                out.angular.z = math.copysign(
                    turn_speed,
                    requested_angular
                )

                self.publisher.publish(out)

        # --------------------------------------------------------------
        # STOPPED: publish zero until both a minimum dwell time has elapsed
        # and odometry confirms the chassis has settled.  Then choose exactly
        # one next phase: TURN or MOVE.
        # --------------------------------------------------------------
        else:
            self.publisher.publish(self._zero_cmd())

            stopped_long_enough = (now - self.stop_started) >= self.minimum_stop_time
            if not stopped_long_enough or not self._robot_is_settled():
                self._log_mode_if_changed()
                return

            # A close obstacle blocks forward motion, but rotation is allowed
            # so Nav2 can align the Husky with its newly replanned safe segment.
            if wants_turn:
                self.mode = self.TURNING
            elif obstacle_ahead:
                if now - self.last_obstacle_log_time > 1.0:
                    self.last_obstacle_log_time = now
                    self.get_logger().info(
                        f'Obstacle in front safety zone ({self.front_distance:.2f} m). '
                        'Holding STOP while Nav2 costmap / global path replans.'
                    )
            elif wants_forward:
                self.mode = self.MOVING

        self._log_mode_if_changed()


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
        rclpy.shutdown()


if __name__ == '__main__':
    main()

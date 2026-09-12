#!/usr/bin/env python3

import math
import time

import rclpy

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool


class CoverageNode(Node):

    def __init__(self):

        super().__init__('coverage_node')

        # ============================================================
        # PARAMETERS
        # ============================================================

        # Relative action name.
        # If this node is launched in /husky1 namespace,
        # "navigate_to_pose" becomes:
        #
        # /husky1/navigate_to_pose

        self.declare_parameter(
            'navigate_action',
            'navigate_to_pose'
        )

        # Robot name used for frame names
        self.declare_parameter(
            'robot_name',
            'husky1'
        )


        # ============================================================
        # ROBOT / FRAME SETUP
        # ============================================================

        self.robot_name = (
            self.get_parameter('robot_name')
            .value
            .strip()
            .strip('/')
        )

        self.map_frame = f'{self.robot_name}_map'

        self.get_logger().info(
            f'Robot name: {self.robot_name}'
        )

        self.get_logger().info(
            f'Using map frame: {self.map_frame}'
        )


        # ============================================================
        # COVERAGE STATE
        # ============================================================

        # Start paused.
        # Mission node must send True before coverage begins.
        self.paused = True

        self.coverage_complete = False

        self.current_waypoint_index = 0

        self.goal_active = False

        self.goal_handle = None

        self.waiting_for_goal_response = False

        self.last_waiting_log_time = 0.0


        # ============================================================
        # WAYPOINTS
        # ============================================================
        #
        # Temporary test waypoints.
        #
        # Later replace these coordinates with the actual
        # large_demo environment coverage pattern.
        #
        #
        # Example lawnmower pattern:
        #
        # P1 --> P2 --> P3 --> P4
        #                      |
        # P8 <-- P7 <-- P6 <-- P5
        # |
        # P9 --> ...
        #

        self.waypoints = [

            (0.0, 0.0),
            (2.0, 0.0),
            (4.0, 0.0),
            (6.0, 0.0),

            (6.0, 2.0),
            (4.0, 2.0),
            (2.0, 2.0),
            (0.0, 2.0),

            (0.0, 4.0),
            (2.0, 4.0),
            (4.0, 4.0),
            (6.0, 4.0),

            (6.0, 6.0),
            (4.0, 6.0),
            (2.0, 6.0),
            (0.0, 6.0),
        ]


        # ============================================================
        # NAV2 ACTION CLIENT
        # ============================================================

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            self.get_parameter('navigate_action').value
        )


        # ============================================================
        # SUBSCRIBER
        # ============================================================

        # Mission Node sends:
        #
        # True  = resume coverage
        # False = pause coverage

        self.resume_sub = self.create_subscription(
            Bool,
            '/husky1/explore/resume',
            self.resume_callback,
            10
        )


        # ============================================================
        # PUBLISHER
        # ============================================================

        self.coverage_complete_pub = self.create_publisher(
            Bool,
            '/husky1/coverage_complete',
            10
        )


        # ============================================================
        # TIMER
        # ============================================================

        self.timer = self.create_timer(
            0.5,
            self.coverage_tick
        )


        self.get_logger().info(
            f'Coverage Node started with '
            f'{len(self.waypoints)} waypoints.'
        )


    # ================================================================
    # PAUSE / RESUME CALLBACK
    # ================================================================

    def resume_callback(self, msg: Bool):

        # ------------------------------------------------------------
        # RESUME
        # ------------------------------------------------------------

        if msg.data:

            if self.coverage_complete:

                self.get_logger().info(
                    'Coverage already complete.'
                )

                return


            if not self.paused:

                self.get_logger().info(
                    'Coverage is already running.'
                )

                return


            self.paused = False

            self.get_logger().info(
                'Coverage resumed.'
            )


        # ------------------------------------------------------------
        # PAUSE
        # ------------------------------------------------------------

        else:

            if self.paused:

                self.get_logger().info(
                    'Coverage is already paused.'
                )

                return


            self.paused = True

            self.get_logger().info(
                'Coverage paused.'
            )


            # Cancel current Nav2 goal if one is active
            if (
                self.goal_handle is not None
                and self.goal_active
            ):

                self.get_logger().info(
                    'Cancelling current coverage goal...'
                )

                cancel_future = (
                    self.goal_handle.cancel_goal_async()
                )

                cancel_future.add_done_callback(
                    self.cancel_done_callback
                )


    # ================================================================
    # MAIN COVERAGE LOOP
    # ================================================================

    def coverage_tick(self):

        # ------------------------------------------------------------
        # Coverage complete
        # ------------------------------------------------------------

        if self.coverage_complete:
            return


        # ------------------------------------------------------------
        # Paused
        # ------------------------------------------------------------

        if self.paused:
            return


        # ------------------------------------------------------------
        # Check Nav2 action server
        # ------------------------------------------------------------

        if not self.nav_client.server_is_ready():

            if not self.nav_client.wait_for_server(
                timeout_sec=0.1
            ):

                self._log_waiting(
                    'Waiting for Nav2 NavigateToPose '
                    'action server...'
                )

                return


        # ------------------------------------------------------------
        # Check whether all waypoints are complete
        # ------------------------------------------------------------

        if (
            self.current_waypoint_index
            >= len(self.waypoints)
        ):

            self.coverage_complete = True

            self.get_logger().info(
                'All coverage waypoints completed.'
            )

            msg = Bool()
            msg.data = True

            self.coverage_complete_pub.publish(msg)

            return


        # ------------------------------------------------------------
        # Do not send another goal while one is active
        # ------------------------------------------------------------

        if self.goal_active:
            return


        # ------------------------------------------------------------
        # Do not send duplicates while waiting for Nav2 response
        # ------------------------------------------------------------

        if self.waiting_for_goal_response:
            return


        # ------------------------------------------------------------
        # Send current waypoint
        # ------------------------------------------------------------

        self.send_current_waypoint()


    # ================================================================
    # SEND CURRENT WAYPOINT
    # ================================================================

    def send_current_waypoint(self):

        if self.paused:
            return


        x, y = self.waypoints[
            self.current_waypoint_index
        ]


        self.get_logger().info(
            f'Sending coverage waypoint '
            f'{self.current_waypoint_index + 1}/'
            f'{len(self.waypoints)}: '
            f'({x:.2f}, {y:.2f}) '
            f'in frame {self.map_frame}'
        )


        # ------------------------------------------------------------
        # Build Nav2 goal
        # ------------------------------------------------------------

        goal_msg = NavigateToPose.Goal()

        goal_msg.pose = PoseStamped()

        goal_msg.pose.header.frame_id = (
            self.map_frame
        )

        goal_msg.pose.header.stamp = (
            self.get_clock().now().to_msg()
        )


        # Position

        goal_msg.pose.pose.position.x = float(x)

        goal_msg.pose.pose.position.y = float(y)

        goal_msg.pose.pose.position.z = 0.0


        # Orientation
        #
        # For now every waypoint uses yaw = 0.
        #
        # Quaternion:
        # x = 0
        # y = 0
        # z = 0
        # w = 1

        goal_msg.pose.pose.orientation.x = 0.0

        goal_msg.pose.pose.orientation.y = 0.0

        goal_msg.pose.pose.orientation.z = 0.0

        goal_msg.pose.pose.orientation.w = 1.0


        # ------------------------------------------------------------
        # Send goal
        # ------------------------------------------------------------

        self.waiting_for_goal_response = True


        send_future = (
            self.nav_client.send_goal_async(
                goal_msg,
                feedback_callback=self.feedback_callback
            )
        )


        send_future.add_done_callback(
            self.goal_response_callback
        )


    # ================================================================
    # GOAL RESPONSE CALLBACK
    # ================================================================

    def goal_response_callback(
        self,
        future
    ):

        self.waiting_for_goal_response = False


        try:

            self.goal_handle = future.result()


        except Exception as e:

            self.get_logger().error(
                f'Failed to send navigation goal: {e}'
            )

            self.goal_active = False

            self.goal_handle = None

            return


        # ------------------------------------------------------------
        # Goal rejected
        # ------------------------------------------------------------

        if not self.goal_handle.accepted:

            self.get_logger().warn(
                'Nav2 goal rejected.'
            )

            self.goal_active = False

            self.goal_handle = None

            return


        # ------------------------------------------------------------
        # Goal accepted
        # ------------------------------------------------------------

        self.goal_active = True


        self.get_logger().info(
            'Nav2 goal accepted. '
            'Waiting for result...'
        )


        result_future = (
            self.goal_handle.get_result_async()
        )


        result_future.add_done_callback(
            self.navigation_result_callback
        )


    # ================================================================
    # NAV2 FEEDBACK CALLBACK
    # ================================================================

    def feedback_callback(
        self,
        feedback_msg
    ):

        feedback = feedback_msg.feedback

        distance = getattr(
            feedback,
            'distance_remaining',
            None
        )


        if distance is not None:

            self.get_logger().info(
                f'Distance remaining: '
                f'{distance:.2f} m',
                throttle_duration_sec=5.0
            )


    # ================================================================
    # NAVIGATION RESULT CALLBACK
    # ================================================================

    def navigation_result_callback(
        self,
        future
    ):

        self.goal_active = False


        try:

            wrapped_result = future.result()

        except Exception as e:

            self.get_logger().error(
                f'Navigation result error: {e}'
            )

            self.goal_handle = None

            return


        status = wrapped_result.status


        # ------------------------------------------------------------
        # SUCCEEDED
        # ------------------------------------------------------------

        if status == GoalStatus.STATUS_SUCCEEDED:

            self.get_logger().info(
                f'Waypoint '
                f'{self.current_waypoint_index + 1} '
                f'reached successfully.'
            )


            # Move to next waypoint
            self.current_waypoint_index += 1


        # ------------------------------------------------------------
        # CANCELED
        # ------------------------------------------------------------

        elif status == GoalStatus.STATUS_CANCELED:

            self.get_logger().info(
                'Coverage goal cancelled.'
            )


            # IMPORTANT:
            #
            # Do NOT increase waypoint index.
            #
            # When coverage resumes,
            # the same waypoint will be attempted again.


        # ------------------------------------------------------------
        # ABORTED
        # ------------------------------------------------------------

        elif status == GoalStatus.STATUS_ABORTED:

            self.get_logger().warn(
                f'Waypoint '
                f'{self.current_waypoint_index + 1} '
                f'aborted.'
            )


            # For now:
            #
            # Do not increase the waypoint index.
            #
            # The timer will retry the same waypoint.
            #
            # Later we can add:
            #
            # retry 1
            # retry 2
            # retry 3
            # mark inaccessible
            # move to next waypoint


        # ------------------------------------------------------------
        # Other status
        # ------------------------------------------------------------

        else:

            self.get_logger().warn(
                f'Navigation finished '
                f'with status {status}.'
            )


        self.goal_handle = None


    # ================================================================
    # CANCEL CALLBACK
    # ================================================================

    def cancel_done_callback(
        self,
        future
    ):

        try:

            cancel_response = future.result()


            if len(
                cancel_response.goals_canceling
            ) > 0:

                self.get_logger().info(
                    'Coverage navigation goal '
                    'successfully cancelled.'
                )

            else:

                self.get_logger().warn(
                    'Nav2 did not cancel '
                    'the coverage goal.'
                )


        except Exception as e:

            self.get_logger().error(
                f'Error cancelling goal: {e}'
            )


    # ================================================================
    # WAITING LOG
    # ================================================================

    def _log_waiting(
        self,
        message
    ):

        now = time.monotonic()

        if (
            now - self.last_waiting_log_time
            > 5.0
        ):

            self.last_waiting_log_time = now

            self.get_logger().info(
                message
            )


# ====================================================================
# MAIN
# ====================================================================

def main(args=None):

    rclpy.init(args=args)

    node = CoverageNode()


    try:

        rclpy.spin(node)


    except KeyboardInterrupt:

        pass


    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
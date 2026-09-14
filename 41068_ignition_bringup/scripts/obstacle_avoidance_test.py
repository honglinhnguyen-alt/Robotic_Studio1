#!/usr/bin/env python3

import math
import time

import rclpy

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

from rclpy.action import ActionClient
from rclpy.node import Node


class ObstacleAvoidanceTest(Node):
    """
    Send one repeatable Nav2 goal for obstacle-avoidance testing.
    """

    def __init__(self):

        super().__init__('obstacle_avoidance_test')

        # --------------------------------------------------------------
        # Test parameters
        # --------------------------------------------------------------

        self.declare_parameter('goal_x', 6.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_yaw', 0.0)

        self.declare_parameter(
            'map_frame',
            'husky1_map'
        )


        # --------------------------------------------------------------
        # Nav2 action client
        #
        # This node will be launched inside namespace /husky1.
        #
        # Therefore:
        #
        #   navigate_to_pose
        #
        # automatically becomes:
        #
        #   /husky1/navigate_to_pose
        # --------------------------------------------------------------

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose',
        )


        self.goal_sent = False

        self.last_wait_log_time = 0.0
        self.last_feedback_log_time = 0.0


        # Check every 0.5 seconds until Nav2 becomes available.
        self.timer = self.create_timer(
            0.5,
            self._try_send_goal
        )


        self.get_logger().info(
            'Obstacle-avoidance test ready. '
            'Waiting for the Husky Nav2 action server.'
        )


    def _try_send_goal(self):

        # Only send one goal per test.
        if self.goal_sent:
            return


        # Wait until Nav2 NavigateToPose is running.
        if not self.nav_client.server_is_ready():

            now = time.monotonic()

            if now - self.last_wait_log_time > 2.0:

                self.last_wait_log_time = now

                self.get_logger().info(
                    'Waiting for navigate_to_pose...'
                )

            return


        # --------------------------------------------------------------
        # Read test goal parameters
        # --------------------------------------------------------------

        goal_x = float(
            self.get_parameter('goal_x').value
        )

        goal_y = float(
            self.get_parameter('goal_y').value
        )

        goal_yaw = float(
            self.get_parameter('goal_yaw').value
        )

        map_frame = str(
            self.get_parameter('map_frame').value
        )


        # --------------------------------------------------------------
        # Construct Nav2 goal
        # --------------------------------------------------------------

        goal = NavigateToPose.Goal()

        goal.pose = PoseStamped()

        goal.pose.header.frame_id = map_frame

        goal.pose.header.stamp = (
            self.get_clock().now().to_msg()
        )


        goal.pose.pose.position.x = goal_x
        goal.pose.pose.position.y = goal_y
        goal.pose.pose.position.z = 0.0


        # Convert yaw angle into quaternion.
        goal.pose.pose.orientation.z = math.sin(
            goal_yaw / 2.0
        )

        goal.pose.pose.orientation.w = math.cos(
            goal_yaw / 2.0
        )


        self.goal_sent = True


        self.get_logger().info(
            f'Sending test goal '
            f'({goal_x:.2f}, {goal_y:.2f}) '
            f'in {map_frame}.'
        )


        # --------------------------------------------------------------
        # Send the goal
        # --------------------------------------------------------------

        future = self.nav_client.send_goal_async(
            goal,
            feedback_callback=self._feedback_callback,
        )

        future.add_done_callback(
            self._goal_response_callback
        )


    def _goal_response_callback(self, future):

        goal_handle = future.result()


        if not goal_handle.accepted:

            self.get_logger().error(
                'Nav2 rejected the test goal.'
            )

            return


        self.get_logger().info(
            'Nav2 accepted the test goal.'
        )


        result_future = (
            goal_handle.get_result_async()
        )

        result_future.add_done_callback(
            self._result_callback
        )


    def _feedback_callback(self, feedback_msg):

        # Do not spam the terminal with feedback.
        now = time.monotonic()

        if now - self.last_feedback_log_time < 2.0:
            return


        self.last_feedback_log_time = now


        distance = (
            feedback_msg.feedback.distance_remaining
        )


        self.get_logger().info(
            f'Distance remaining: '
            f'{distance:.2f} m.'
        )


    def _result_callback(self, future):

        result = future.result()

        status = result.status


        if status == GoalStatus.STATUS_SUCCEEDED:

            self.get_logger().info(
                'PASS: Husky reached the goal. '
                'Check Gazebo and RViz to confirm '
                'that it avoided the box without collision.'
            )


        elif status == GoalStatus.STATUS_ABORTED:

            self.get_logger().error(
                'FAIL/REVIEW: Nav2 aborted the goal.'
            )


        elif status == GoalStatus.STATUS_CANCELED:

            self.get_logger().warn(
                'Test goal was cancelled.'
            )


        else:

            self.get_logger().warn(
                f'Navigation finished with status {status}.'
            )


def main(args=None):

    rclpy.init(args=args)

    node = ObstacleAvoidanceTest()


    try:

        rclpy.spin(node)


    except KeyboardInterrupt:

        pass


    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
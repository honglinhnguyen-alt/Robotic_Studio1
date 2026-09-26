#!/usr/bin/env python3

import math
import random

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Point, PoseStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from std_srvs.srv import Trigger

class HuskyNode(Node):

    def __init__(self):
        super().__init__('husky_node')

        # ==================================================
        # NAVIGATION SETTINGS
        # ==================================================

        self.map_frame = 'husky1_map'

        # Every observation point is 1 m away
        # from the original candidate location
        self.observation_distance = 1.0

        # ==================================================
        # CURRENT INVESTIGATION
        # ==================================================

        # Original candidate location from Mission Node
        self.candidate_x = None
        self.candidate_y = None

        self.goal_active = False
        self.goal_handle = None

        # ==================================================
        # RECEIVE INVESTIGATION TARGET FROM MISSION
        # ==================================================

        self.target_sub = self.create_subscription(
            Point,
            'navigation_target',
            self.target_callback,
            10
        )

        # ==================================================
        # NAV2 ACTION CLIENT
        # ==================================================

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            '/husky1/navigate_to_pose'
        )

        # ==================================================
        # YOLO CLASSIFICATION SERVICE CLIENT
        # ==================================================

        self.yolo_client = self.create_client(
            Trigger,
            'perception/rgb/classify_now'
        )

        self.get_logger().info(
            'Husky Node started'
        )

    # ======================================================
    # RECEIVE CANDIDATE LOCATION
    # ======================================================

    def target_callback(self, msg: Point):

        # Ignore new candidate while an investigation
        # is already active
        if self.goal_active or self.candidate_x is not None:

            self.get_logger().warn(
                'Investigation already active. '
                'New target ignored.'
            )

            return

        # Store ORIGINAL candidate location
        self.candidate_x = msg.x
        self.candidate_y = msg.y

        self.get_logger().info(
            f'Investigation target received: '
            f'({self.candidate_x:.2f}, '
            f'{self.candidate_y:.2f})'
        )

        # Generate first observation point
        observation_x, observation_y, yaw = (
            self.calculate_observation_point()
        )

        # Send observation point to Nav2
        self.send_navigation_goal(
            observation_x,
            observation_y,
            yaw
        )

    # ======================================================
    # CALCULATE OBSERVATION POINT
    # ======================================================

    def calculate_observation_point(self):

        # Random direction around the SAME candidate
        angle = random.uniform(
            0.0,
            2.0 * math.pi
        )

        # Observation point is always exactly
        # observation_distance away from candidate
        observation_x = (
            self.candidate_x
            + self.observation_distance * math.cos(angle)
        )

        observation_y = (
            self.candidate_y
            + self.observation_distance * math.sin(angle)
        )

        # Make Husky face toward the candidate
        yaw = math.atan2(
            self.candidate_y - observation_y,
            self.candidate_x - observation_x
        )

        self.get_logger().info(
            f'Observation point calculated: '
            f'({observation_x:.2f}, '
            f'{observation_y:.2f}) '
            f'around candidate '
            f'({self.candidate_x:.2f}, '
            f'{self.candidate_y:.2f})'
        )

        return observation_x, observation_y, yaw

    # ======================================================
    # SEND NAVIGATION GOAL TO NAV2
    # ======================================================

    def send_navigation_goal(self, x, y, yaw):

        if not self.nav_client.wait_for_server(
            timeout_sec=1.0
        ):

            self.get_logger().warn(
                'Nav2 NavigateToPose action server '
                'not available.'
            )

            self.goal_active = False

            return

        # Create Nav2 goal
        goal_msg = NavigateToPose.Goal()

        goal_msg.pose = PoseStamped()

        # Goal frame
        goal_msg.pose.header.frame_id = self.map_frame

        goal_msg.pose.header.stamp = (
            self.get_clock().now().to_msg()
        )

        # Goal position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # Goal orientation
        # Husky faces the candidate
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0

        goal_msg.pose.pose.orientation.z = (
            math.sin(yaw / 2.0)
        )

        goal_msg.pose.pose.orientation.w = (
            math.cos(yaw / 2.0)
        )

        self.goal_active = True

        self.get_logger().info(
            f'Sending Nav2 goal: '
            f'({x:.2f}, {y:.2f})'
        )

        send_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        send_future.add_done_callback(
            self.goal_response_callback
        )

    # ======================================================
    # NAV2 GOAL RESPONSE
    # ======================================================

    def goal_response_callback(self, future):

        try:

            self.goal_handle = future.result()

        except Exception as error:

            self.get_logger().error(
                f'Failed to send Nav2 goal: {error}'
            )

            self.goal_active = False
            self.goal_handle = None

            self.retry_navigation()

            return

        # --------------------------------------------------
        # GOAL REJECTED
        # --------------------------------------------------

        if not self.goal_handle.accepted:

            self.get_logger().warn(
                'Nav2 goal rejected. '
                'Generating another observation point.'
            )

            self.goal_active = False
            self.goal_handle = None

            self.retry_navigation()

            return

        # --------------------------------------------------
        # GOAL ACCEPTED
        # --------------------------------------------------

        self.get_logger().info(
            'Nav2 goal accepted. '
            'Navigating to observation point...'
        )

        result_future = (
            self.goal_handle.get_result_async()
        )

        result_future.add_done_callback(
            self.navigation_result_callback
        )

    # ======================================================
    # NAVIGATION FEEDBACK
    # ======================================================

    def feedback_callback(self, feedback_msg):

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
                throttle_duration_sec=2.0
            )

    # ======================================================
    # NAVIGATION RESULT
    # ======================================================

    def navigation_result_callback(self, future):

        try:

            result = future.result()

        except Exception as error:

            self.get_logger().error(
                f'Failed to get navigation result: {error}'
            )

            self.goal_active = False
            self.goal_handle = None

            self.retry_navigation()

            return

        status = result.status

        self.goal_active = False
        self.goal_handle = None

        # --------------------------------------------------
        # NAVIGATION SUCCESSFUL
        # --------------------------------------------------

        if status == GoalStatus.STATUS_SUCCEEDED:

            self.get_logger().info(
                'Observation point reached successfully.'
            )

            # Husky has reached the observation point.
            # Request YOLO classification.
            self.request_yolo_classification()

            return

        # --------------------------------------------------
        # NAVIGATION UNSUCCESSFUL
        # --------------------------------------------------

        self.get_logger().warn(
            f'Navigation unsuccessful '
            f'(Nav2 status = {status}). '
            f'Generating another observation point.'
        )

        # Keep the SAME candidate location.
        # Generate another observation point around it.
        self.retry_navigation()

    # ======================================================
    # RETRY NAVIGATION
    # ======================================================

    def retry_navigation(self):

        if (
            self.candidate_x is None
            or self.candidate_y is None
        ):

            self.get_logger().warn(
                'No active candidate. '
                'Cannot retry navigation.'
            )

            return

        self.get_logger().info(
            f'Retrying navigation around candidate '
            f'({self.candidate_x:.2f}, '
            f'{self.candidate_y:.2f})...'
        )

        # Generate a NEW random observation point
        # around the SAME candidate
        observation_x, observation_y, yaw = (
            self.calculate_observation_point()
        )

        self.send_navigation_goal(
            observation_x,
            observation_y,
            yaw
        )

    # ======================================================
    # REQUEST YOLO CLASSIFICATION
    # ======================================================

    def request_yolo_classification(self):

        # Check if YOLO service is available
        if not self.yolo_client.wait_for_service(
            timeout_sec=2.0
        ):

            self.get_logger().error(
                'YOLO classification service '
                'is not available.'
            )

            # Keep the candidate active.
            # Do not reset investigation here.
            return

        self.get_logger().info(
            'Requesting RGB YOLO classification...'
        )

        # Trigger service has an empty request
        request = Trigger.Request()

        future = self.yolo_client.call_async(
            request
        )

        future.add_done_callback(
            self.yolo_response_callback
        )

    # ======================================================
    # YOLO SERVICE RESPONSE
    # ======================================================

    def yolo_response_callback(self, future):

        try:

            response = future.result()

        except Exception as error:

            self.get_logger().error(
                f'YOLO service call failed: {error}'
            )

            return

        # --------------------------------------------------
        # YOLO REQUEST SUCCESSFUL
        # --------------------------------------------------

        if response.success:

            self.get_logger().info(
                f'YOLO classification completed: '
                f'{response.message}'
            )

            # Navigation + YOLO request completed.
            # Ready for next candidate.
            self.reset_investigation()

        # --------------------------------------------------
        # YOLO REQUEST FAILED
        # --------------------------------------------------

        else:

            self.get_logger().warn(
                f'YOLO classification failed: '
                f'{response.message}'
            )

    # ======================================================
    # RESET INVESTIGATION
    # ======================================================

    def reset_investigation(self):

        self.goal_active = False
        self.goal_handle = None

        self.candidate_x = None
        self.candidate_y = None

        self.get_logger().info(
            'Investigation complete. '
            'Waiting for next candidate.'
        )

# ==========================================================
# MAIN
# ==========================================================

def main(args=None):

    rclpy.init(args=args)

    node = HuskyNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()

if __name__ == '__main__':

    main()

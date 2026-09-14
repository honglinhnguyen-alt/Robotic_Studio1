#!/usr/bin/env python3

import math
import time
from typing import Optional

import rclpy
import tf2_ros

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan

from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class ObstacleAvoidanceTest(Node):
    """Send one Nav2 goal and print diagnostics useful for obstacle-avoidance debugging."""

    def __init__(self):
        super().__init__('obstacle_avoidance_test')

        # Test goal
        self.declare_parameter('goal_x', 4.5)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_yaw', 0.0)
        self.declare_parameter('map_frame', 'husky1_map')
        self.declare_parameter('base_frame', 'husky1_base_link')

        self.goal_x = float(self.get_parameter('goal_x').value)
        self.goal_y = float(self.get_parameter('goal_y').value)
        self.goal_yaw = float(self.get_parameter('goal_yaw').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)

        # Nav2 action client. The node is launched in /husky1, so this resolves to
        # /husky1/navigate_to_pose.
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Subscribe BEFORE sending the goal so we do not miss the short-lived
        # planner_server /plan publication.
        self.create_subscription(Path, 'plan', self._global_plan_callback, 10)
        self.create_subscription(Path, 'local_plan', self._local_plan_callback, 10)
        self.create_subscription(Twist, 'cmd_vel_nav', self._cmd_vel_callback, 10)
        self.create_subscription(OccupancyGrid, 'map', self._map_callback, 10)
        self.create_subscription(
            OccupancyGrid,
            'global_costmap/costmap',
            self._global_costmap_callback,
            10,
        )
        self.create_subscription(
            LaserScan,
            'scan',
            self._scan_callback,
            qos_profile_sensor_data,
        )

        # TF lets us calculate a distance independently of Nav2 feedback.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.goal_sent = False
        self.goal_accepted = False

        self.last_wait_log_time = 0.0

        self.global_plan: Optional[Path] = None
        self.local_plan: Optional[Path] = None
        self.map_msg: Optional[OccupancyGrid] = None
        self.global_costmap: Optional[OccupancyGrid] = None
        self.last_cmd: Optional[Twist] = None
        self.front_min_range: Optional[float] = None

        self.global_plan_count = 0
        self.local_plan_count = 0
        self.last_nav2_distance: Optional[float] = None
        self.last_recoveries = 0
        self.last_navigation_time = 0.0
        self.last_estimated_time = 0.0
        self.last_feedback_pose = None

        self.goal_timer = self.create_timer(0.5, self._try_send_goal)
        self.debug_timer = self.create_timer(1.0, self._print_debug_status)

        self.get_logger().info(
            'Obstacle-avoidance DEBUG test ready. '
            'Waiting for Husky Nav2. Diagnostics are already subscribed.'
        )

    def _try_send_goal(self):
        if self.goal_sent:
            return

        if not self.nav_client.server_is_ready():
            now = time.monotonic()
            if now - self.last_wait_log_time > 2.0:
                self.last_wait_log_time = now
                self.get_logger().info('Waiting for navigate_to_pose...')
            return

        # Do not send until we have a map. This prevents a hard-coded goal being
        # sent before SLAM has published any map at all.
        if self.map_msg is None:
            now = time.monotonic()
            if now - self.last_wait_log_time > 2.0:
                self.last_wait_log_time = now
                self.get_logger().info('Nav2 is ready; waiting for /husky1/map...')
            return

        self._print_goal_map_check(self.map_msg, 'SLAM map')

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = self.goal_x
        goal.pose.pose.position.y = self.goal_y
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation.z = math.sin(self.goal_yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(self.goal_yaw / 2.0)

        self.goal_sent = True
        self.get_logger().info(
            f'Sending test goal ({self.goal_x:.2f}, {self.goal_y:.2f}) '
            f'in {self.map_frame}.'
        )

        future = self.nav_client.send_goal_async(
            goal,
            feedback_callback=self._feedback_callback,
        )
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Nav2 rejected the test goal.')
            return

        self.goal_accepted = True
        self.get_logger().info('Nav2 accepted the test goal.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.last_nav2_distance = float(feedback.distance_remaining)
        self.last_recoveries = int(feedback.number_of_recoveries)
        self.last_feedback_pose = feedback.current_pose
        self.last_navigation_time = self._duration_to_seconds(feedback.navigation_time)
        self.last_estimated_time = self._duration_to_seconds(feedback.estimated_time_remaining)

    def _global_plan_callback(self, msg: Path):
        self.global_plan = msg
        self.global_plan_count += 1
        length = self._path_length(msg)

        if msg.poses:
            start = msg.poses[0].pose.position
            end = msg.poses[-1].pose.position
            self.get_logger().info(
                f'GLOBAL PLAN RECEIVED #{self.global_plan_count}: '
                f'poses={len(msg.poses)}, length={length:.2f} m, '
                f'start=({start.x:.2f}, {start.y:.2f}), '
                f'end=({end.x:.2f}, {end.y:.2f}).'
            )
        else:
            self.get_logger().error(
                f'GLOBAL PLAN RECEIVED #{self.global_plan_count}, but poses=[]'
            )

    def _local_plan_callback(self, msg: Path):
        self.local_plan = msg
        self.local_plan_count += 1

    def _cmd_vel_callback(self, msg: Twist):
        self.last_cmd = msg

    def _map_callback(self, msg: OccupancyGrid):
        first = self.map_msg is None
        self.map_msg = msg
        if first:
            self._print_goal_map_check(msg, 'SLAM map')

    def _global_costmap_callback(self, msg: OccupancyGrid):
        first = self.global_costmap is None
        self.global_costmap = msg
        if first:
            self._print_goal_map_check(msg, 'global costmap')

    def _scan_callback(self, msg: LaserScan):
        # Minimum valid LiDAR range in a +/-30 degree cone in front of the Husky.
        best = math.inf
        angle = msg.angle_min
        half_angle = math.radians(30.0)

        for r in msg.ranges:
            if abs(angle) <= half_angle and math.isfinite(r):
                if msg.range_min <= r <= msg.range_max:
                    best = min(best, r)
            angle += msg.angle_increment

        self.front_min_range = None if math.isinf(best) else best

    def _print_debug_status(self):
        pose = self._lookup_robot_pose()

        if pose is not None:
            x, y, yaw = pose
            direct = math.hypot(self.goal_x - x, self.goal_y - y)
            pose_text = (
                f'pose=({x:.2f},{y:.2f},{math.degrees(yaw):.0f}deg) '
                f'direct_goal={direct:.2f}m'
            )
        else:
            pose_text = 'pose=TF_UNAVAILABLE'

        nav2_text = (
            'nav2_path=N/A'
            if self.last_nav2_distance is None
            else f'nav2_path={self.last_nav2_distance:.2f}m'
        )

        if self.global_plan is None:
            global_text = 'global_plan=NONE'
        else:
            global_text = (
                f'global_plan={len(self.global_plan.poses)}poses/'
                f'{self._path_length(self.global_plan):.2f}m'
            )

        if self.local_plan is None:
            local_text = 'local_plan=NONE'
        else:
            local_text = f'local_plan={len(self.local_plan.poses)}poses'

        if self.last_cmd is None:
            cmd_text = 'cmd=N/A'
        else:
            cmd_text = (
                f'cmd(v={self.last_cmd.linear.x:.2f}, '
                f'w={self.last_cmd.angular.z:.2f})'
            )

        scan_text = (
            'front=N/A'
            if self.front_min_range is None
            else f'front={self.front_min_range:.2f}m'
        )

        map_text = self._grid_goal_summary(self.map_msg, 'map')
        costmap_text = self._grid_goal_summary(self.global_costmap, 'gcost')

        self.get_logger().info(
            'DEBUG | '
            f'{pose_text} | {nav2_text} | {global_text} | {local_text} | '
            f'{cmd_text} | {scan_text} | recoveries={self.last_recoveries} | '
            f'nav_time={self.last_navigation_time:.1f}s | '
            f'{map_text} | {costmap_text}'
        )

    def _lookup_robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1),
            )
        except Exception:
            return None

        t = transform.transform.translation
        q = transform.transform.rotation
        yaw = self._yaw_from_quaternion(q.x, q.y, q.z, q.w)
        return (t.x, t.y, yaw)

    def _grid_goal_summary(self, grid: Optional[OccupancyGrid], label: str) -> str:
        if grid is None:
            return f'{label}=N/A'

        inside, value = self._grid_value_at(grid, self.goal_x, self.goal_y)
        if not inside:
            return f'{label}_goal=OUTSIDE'
        return f'{label}_goal_cost={value}'

    def _print_goal_map_check(self, grid: OccupancyGrid, label: str):
        info = grid.info
        xmin = info.origin.position.x
        ymin = info.origin.position.y
        xmax = xmin + info.width * info.resolution
        ymax = ymin + info.height * info.resolution
        inside, value = self._grid_value_at(grid, self.goal_x, self.goal_y)

        state = f'INSIDE, cell={value}' if inside else 'OUTSIDE'
        self.get_logger().info(
            f'{label}: x=[{xmin:.2f},{xmax:.2f}], '
            f'y=[{ymin:.2f},{ymax:.2f}], res={info.resolution:.2f} m. '
            f'Goal ({self.goal_x:.2f},{self.goal_y:.2f}) is {state}.'
        )

    @staticmethod
    def _grid_value_at(grid: OccupancyGrid, x: float, y: float):
        info = grid.info
        # This test assumes the occupancy grid origin has negligible yaw, which
        # is true for the supplied SLAM/Nav2 setup.
        ix = int(math.floor((x - info.origin.position.x) / info.resolution))
        iy = int(math.floor((y - info.origin.position.y) / info.resolution))

        if ix < 0 or iy < 0 or ix >= info.width or iy >= info.height:
            return False, None

        index = iy * info.width + ix
        return True, int(grid.data[index])

    @staticmethod
    def _path_length(path: Path) -> float:
        if len(path.poses) < 2:
            return 0.0

        total = 0.0
        for a, b in zip(path.poses[:-1], path.poses[1:]):
            pa = a.pose.position
            pb = b.pose.position
            total += math.hypot(pb.x - pa.x, pb.y - pa.y)
        return total

    @staticmethod
    def _duration_to_seconds(duration_msg) -> float:
        return float(duration_msg.sec) + float(duration_msg.nanosec) * 1e-9

    @staticmethod
    def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _result_callback(self, future):
        wrapped = future.result()
        status = wrapped.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(
                'PASS: Husky reached the goal. '
                'Check Gazebo/RViz for collision-free avoidance.'
            )
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error(
                'FAIL/REVIEW: Nav2 aborted the goal. '
                f'global_plan_messages={self.global_plan_count}, '
                f'local_plan_messages={self.local_plan_count}, '
                f'recoveries={self.last_recoveries}.'
            )
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn('Test goal was cancelled.')
        else:
            self.get_logger().warn(f'Test ended with status={status}.')


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

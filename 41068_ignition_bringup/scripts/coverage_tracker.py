#!/usr/bin/env python3
"""Survey coverage (R1): how much of the survey area is mapped and thermally checked.

explore_lite stops when the LiDAR map has no reachable frontier, but the
thermal camera only sees 60 deg ahead, so a complete map does not mean every
area was checked for heat sources. This node measures both, on a fixed grid
over the survey boundary (a circle around the Husky start, map frame):

  accessible = boundary cells - cells the SLAM map marks occupied
  lidar      = known-free cells / accessible
  thermal    = known-free cells that were inside the thermal camera's
               field of view (ray-cast on the SLAM map, stopped by occupied
               cells, up to thermal_range_m) / accessible

Publishes
  /husky1/mission/coverage                 std_msgs/String (JSON)
  /husky1/mission/thermal_coverage_grid    nav_msgs/OccupancyGrid for RViz:
                                           0 = thermally checked, 100 = mapped
                                           free but not checked, -1 = other
"""

import json
import math

import numpy as np
import rclpy
import tf2_ros
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String

OCCUPIED_THRESHOLD = 50


# --------------------------------------------------------------------------
# Pure helpers (no ROS) - imported by the offline tests
# --------------------------------------------------------------------------

class SurveyGrid:
    """Fixed square grid of cell centres covering a circular survey boundary."""

    def __init__(self, center_x, center_y, radius, cell_size):
        if radius <= 0.0 or cell_size <= 0.0:
            raise ValueError('radius and cell_size must be positive')
        self.cell_size = float(cell_size)
        self.size = int(math.ceil(2.0 * radius / cell_size))
        self.origin_x = center_x - 0.5 * self.size * cell_size
        self.origin_y = center_y - 0.5 * self.size * cell_size
        centres = (np.arange(self.size) + 0.5) * cell_size
        # Row index = y, column index = x, like nav_msgs/OccupancyGrid.
        self.xs, self.ys = np.meshgrid(self.origin_x + centres, self.origin_y + centres)
        self.inside = np.hypot(self.xs - center_x, self.ys - center_y) <= radius
        self.seen = np.zeros_like(self.inside)

    def cell_index(self, x, y):
        """Return (rows, cols, valid) of the grid cells containing points x, y."""
        cols = np.floor((np.asarray(x) - self.origin_x) / self.cell_size).astype(int)
        rows = np.floor((np.asarray(y) - self.origin_y) / self.cell_size).astype(int)
        valid = (cols >= 0) & (cols < self.size) & (rows >= 0) & (rows < self.size)
        return rows, cols, valid

    def mark_seen(self, x, y):
        rows, cols, valid = self.cell_index(x, y)
        self.seen[rows[valid], cols[valid]] = True


def sample_map(grid, origin_x, origin_y, resolution, x, y):
    """Occupancy values (-1 unknown, 0..100) of a 2D map array at points x, y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cols = np.floor((x - origin_x) / resolution).astype(int)
    rows = np.floor((y - origin_y) / resolution).astype(int)
    height, width = grid.shape
    valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    values = np.full(x.shape, -1, dtype=np.int16)
    values[valid] = grid[rows[valid], cols[valid]]
    return values


def visible_points(cam_x, cam_y, cam_yaw, hfov, max_range, step, map_lookup):
    """Return x, y of ray samples inside the camera FOV before the first obstacle.

    map_lookup(x, y) -> occupancy values. Unknown cells do not block a ray,
    occupied cells (>= OCCUPIED_THRESHOLD) do and are not returned.
    """
    ray_count = max(2, int(math.ceil(hfov * max_range / step)) + 1)
    angles = cam_yaw + np.linspace(-0.5 * hfov, 0.5 * hfov, ray_count)
    ranges = np.arange(step, max_range + 1e-9, step)
    xs = cam_x + np.cos(angles)[:, None] * ranges[None, :]
    ys = cam_y + np.sin(angles)[:, None] * ranges[None, :]
    blocked = map_lookup(xs, ys) >= OCCUPIED_THRESHOLD
    # Keep samples before the first blocked sample of each ray.
    clear = np.cumsum(blocked, axis=1) == 0
    return xs[clear], ys[clear]


def coverage_stats(survey, cell_values):
    """Coverage numbers from the SLAM values sampled at every survey cell."""
    inside = survey.inside
    free = inside & (cell_values >= 0) & (cell_values < OCCUPIED_THRESHOLD)
    occupied = inside & (cell_values >= OCCUPIED_THRESHOLD)
    checked = free & survey.seen
    cell_area = survey.cell_size ** 2
    accessible = int(inside.sum() - occupied.sum())
    return {
        'boundary_m2': round(float(inside.sum() * cell_area), 1),
        'accessible_m2': round(float(accessible * cell_area), 1),
        'mapped_free_m2': round(float(free.sum() * cell_area), 1),
        'thermal_checked_m2': round(float(checked.sum() * cell_area), 1),
        'lidar_coverage': float(free.sum() / accessible) if accessible else 0.0,
        'thermal_coverage': float(checked.sum() / accessible) if accessible else 0.0,
        'thermal_of_mapped': float(checked.sum() / free.sum()) if free.sum() else 0.0,
    }


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

class CoverageTracker(Node):

    def __init__(self):
        super().__init__('coverage_tracker')
        self.declare_parameter('map_topic', '/husky1/map')
        self.declare_parameter('map_frame', 'husky1_map')
        self.declare_parameter('camera_frame', 'husky1_thermal_camera_link')
        self.declare_parameter('coverage_topic', '/husky1/mission/coverage')
        self.declare_parameter('grid_topic', '/husky1/mission/thermal_coverage_grid')
        self.declare_parameter('publish_grid', True)
        # Thermal camera: urdf_husky/husky.gazebo.xacro. Depth is clipped at
        # 10 m, so a heat source further away cannot be localized anyway.
        self.declare_parameter('thermal_hfov_deg', 60.0)
        self.declare_parameter('thermal_range_m', 10.0)
        # large_demo is ringed by rock cliffs ~35-38 m from the origin.
        self.declare_parameter('boundary_center_x', 0.0)
        self.declare_parameter('boundary_center_y', 0.0)
        self.declare_parameter('boundary_radius', 30.0)
        self.declare_parameter('cell_size', 0.2)
        self.declare_parameter('update_period_sec', 0.5)
        self.declare_parameter('publish_period_sec', 1.0)

        p = self.get_parameter
        self.map_frame = str(p('map_frame').value)
        self.camera_frame = str(p('camera_frame').value)
        self.hfov = math.radians(float(p('thermal_hfov_deg').value))
        self.max_range = float(p('thermal_range_m').value)
        self.survey = SurveyGrid(
            float(p('boundary_center_x').value),
            float(p('boundary_center_y').value),
            float(p('boundary_radius').value),
            float(p('cell_size').value),
        )
        self.publish_grid = bool(p('publish_grid').value)
        self.map_msg = None
        self.map_array = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            OccupancyGrid, str(p('map_topic').value), self._map_callback, latched)
        self.coverage_pub = self.create_publisher(String, str(p('coverage_topic').value), 10)
        self.grid_pub = self.create_publisher(
            OccupancyGrid, str(p('grid_topic').value), latched)

        self.create_timer(float(p('update_period_sec').value), self._update_seen)
        self.create_timer(float(p('publish_period_sec').value), self._publish)
        self.get_logger().info(
            f'Coverage tracker: boundary radius {float(p("boundary_radius").value):.1f} m, '
            f'thermal FOV {math.degrees(self.hfov):.0f} deg x {self.max_range:.1f} m, '
            f'camera frame {self.camera_frame}')

    def _map_callback(self, msg):
        self.map_msg = msg
        self.map_array = np.asarray(msg.data, dtype=np.int16).reshape(
            msg.info.height, msg.info.width)

    def _lookup(self, x, y):
        info = self.map_msg.info
        return sample_map(self.map_array, info.origin.position.x, info.origin.position.y,
                          info.resolution, x, y)

    def _update_seen(self):
        if self.map_msg is None:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.camera_frame, Time())
        except tf2_ros.TransformException:
            return
        t = transform.transform
        xs, ys = visible_points(
            t.translation.x, t.translation.y, yaw_from_quaternion(t.rotation),
            self.hfov, self.max_range, self.map_msg.info.resolution, self._lookup)
        self.survey.mark_seen(xs, ys)

    def _publish(self):
        if self.map_msg is None:
            return
        values = self._lookup(self.survey.xs, self.survey.ys)
        stats = coverage_stats(self.survey, values)
        msg = String()
        msg.data = json.dumps(stats, separators=(',', ':'))
        self.coverage_pub.publish(msg)

        if self.publish_grid:
            free = self.survey.inside & (values >= 0) & (values < OCCUPIED_THRESHOLD)
            grid = np.full(values.shape, -1, dtype=np.int8)
            grid[free] = 100
            grid[free & self.survey.seen] = 0
            out = OccupancyGrid()
            out.header.frame_id = self.map_frame
            out.header.stamp = self.get_clock().now().to_msg()
            out.info.resolution = self.survey.cell_size
            out.info.width = self.survey.size
            out.info.height = self.survey.size
            out.info.origin.position.x = self.survey.origin_x
            out.info.origin.position.y = self.survey.origin_y
            out.info.origin.orientation.w = 1.0
            out.data = grid.ravel().tolist()
            self.grid_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CoverageTracker()
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

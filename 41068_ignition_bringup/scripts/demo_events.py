#!/usr/bin/env python3
"""Trigger failure cases and surprises live during a demo.

  ros2 run 41068_ignition_bringup demo_events.py heat
      Spawn a campfire (330 K, not an animal) 5-7 m ahead of the Husky and
      2-3 m to the side: inside the 60 deg thermal view but off its path,
      on free SLAM-map space (no tree, rock or animal within 1.5 m).
      Expected: thermal candidate -> investigation -> REJECTED (R5/R6:
      a heat source is only an animal when YOLO says so).
  ros2 run 41068_ignition_bringup demo_events.py obstacle
      Drop a 0.6 x 0.6 x 1.2 m box 3.5 m ahead of the Husky while it is
      exploring (refused while it investigates a candidate).
      Expected: the stop-turn-go safety corridor stops the robot, Nav2
      marks the box in the costmap and replans around it (R3).
  ros2 run 41068_ignition_bringup demo_events.py unreachable
      Inject a candidate at (150, 0), outside the map. Every observation
      pose is rejected by Nav2 at once (goal off the costmap).
      Expected: 3 attempts fail within seconds -> UNRESOLVED, and
      exploration continues (R4: bounded retries, then move on).
  ros2 run 41068_ignition_bringup demo_events.py clear
      Remove every model spawned by this script.

Spawned models are not in the evaluator's ground truth, so a detected demo
campfire is counted as a false positive there.
"""

import argparse
import math
import shutil
import subprocess
import sys
import time

import json

import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import String

PREFIX = 'demo_event_'

BOX_SDF = (
    "<sdf version='1.8'><model name='{name}'><static>true</static><link name='link'>"
    "<collision name='c'><geometry><box><size>0.6 0.6 1.2</size></box></geometry>"
    "</collision><visual name='v'><geometry><box><size>0.6 0.6 1.2</size></box>"
    "</geometry><material><ambient>0.8 0.4 0.1 1</ambient>"
    "<diffuse>0.8 0.4 0.1 1</diffuse></material></visual></link></model></sdf>"
)


def gazebo_cli():
    if shutil.which('ign'):
        return 'ign', 'ignition.msgs'
    if shutil.which('gz'):
        return 'gz', 'gz.msgs'
    raise RuntimeError('Neither `ign` nor `gz` was found in PATH.')


def service(cli, msg_ns, name, reqtype, request):
    cmd = [cli, 'service', '-s', name, '--reqtype', f'{msg_ns}.{reqtype}',
           '--reptype', f'{msg_ns}.Boolean', '--timeout', '3000', '--req', request]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and 'false' not in result.stdout


def existing_demo_models(cli):
    result = subprocess.run([cli, 'model', '--list'], capture_output=True, text=True)
    return sorted({word.strip() for word in result.stdout.split()
                   if word.strip().startswith(PREFIX)})


def snapshot(timeout=5.0):
    """Husky pose (odom = world at the start), SLAM map and mission state."""
    rclpy.init()
    node = rclpy.create_node('demo_events')
    data = {}

    def on_odom(msg):
        q = msg.pose.pose.orientation
        data['pose'] = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                        math.atan2(2 * (q.w * q.z + q.x * q.y),
                                   1 - 2 * (q.y * q.y + q.z * q.z)))

    def on_status(msg):
        try:
            data['mission_state'] = json.loads(msg.data).get('mission_state')
        except ValueError:
            pass

    latched = QoSProfile(depth=1)
    latched.reliability = ReliabilityPolicy.RELIABLE
    latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
    node.create_subscription(Odometry, '/husky1/odom', on_odom, 10)
    node.create_subscription(OccupancyGrid, '/husky1/map',
                             lambda msg: data.__setitem__('map', msg), latched)
    node.create_subscription(String, '/husky1/mission/status', on_status, 10)
    deadline = time.time() + timeout
    while time.time() < deadline and not (
            'pose' in data and 'map' in data and 'mission_state' in data):
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    if 'pose' not in data:
        raise RuntimeError('No /husky1/odom received: is the simulation running?')
    return data


def map_is_free(grid, x, y, radius):
    """True when (x, y) and every cell within radius are known free."""
    if grid is None:
        return True
    info = grid.info
    res = info.resolution
    col = int((x - info.origin.position.x) / res)
    row = int((y - info.origin.position.y) / res)
    cells = int(math.ceil(radius / res))
    for d_row in range(-cells, cells + 1):
        for d_col in range(-cells, cells + 1):
            if (d_row * d_row + d_col * d_col) * res * res > radius * radius:
                continue
            c, r = col + d_col, row + d_row
            if not (0 <= c < info.width and 0 <= r < info.height):
                return False
            value = grid.data[r * info.width + c]
            if value < 0 or value >= 50:
                return False
    return True


def side_spot(pose, grid, clearance=1.5):
    """A free spot 5-7 m ahead and 2-3 m to either side of the Husky."""
    x, y, yaw = pose
    for ahead in (6.0, 5.0, 7.0):
        for side in (2.5, -2.5, 3.0, -3.0, 2.0, -2.0):
            px = x + ahead * math.cos(yaw) - side * math.sin(yaw)
            py = y + ahead * math.sin(yaw) + side * math.cos(yaw)
            if map_is_free(grid, px, py, clearance):
                return px, py
    return None


def spawn(world, kind, distance):
    cli, msg_ns = gazebo_cli()
    data = snapshot()
    x, y, yaw = data['pose']
    grid = data.get('map')
    if kind == 'heat':
        spot = side_spot(data['pose'], grid)
        if spot is None:
            raise RuntimeError('No free spot 5-7 m ahead and 2-3 m to the side; '
                               'try again when the Husky is in open ground.')
        px, py = spot
    else:
        state = data.get('mission_state')
        if state not in (None, 'COVERAGE'):
            raise RuntimeError(f'Mission is {state}: drop the obstacle while the '
                               'Husky is exploring, not while it investigates.')
        px, py = x + distance * math.cos(yaw), y + distance * math.sin(yaw)
        if not map_is_free(grid, px, py, 0.5):
            raise RuntimeError('The spot ahead is not free space (tree, rock or '
                               'unmapped); try again in a moment.')
    index = len(existing_demo_models(cli)) + 1
    name = f'{PREFIX}{kind}_{index}'
    if kind == 'heat':
        source = 'sdf_filename: "model://campfire"'
    else:
        source = 'sdf: "' + BOX_SDF.format(name=name).replace('"', '\\"') + '"'
    request = (f'{source}, name: "{name}", allow_renaming: true, '
               f'pose: {{position: {{x: {px:.2f}, y: {py:.2f}, z: 0.0}}}}')
    if not service(cli, msg_ns, f'/world/{world}/create', 'EntityFactory', request):
        raise RuntimeError(f'Gazebo refused to create {name}.')
    print(f'Spawned {name} at ({px:.2f}, {py:.2f}); Husky at ({x:.2f}, {y:.2f}).')


def inject_candidate(x, y):
    rclpy.init()
    node = rclpy.create_node('demo_events')
    pub = node.create_publisher(Point, '/husky1/mission/test_candidate', 10)
    deadline = time.time() + 3.0
    while pub.get_subscription_count() == 0 and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError('mission_node is not running.')
    msg = Point()
    msg.x, msg.y = float(x), float(y)
    pub.publish(msg)
    rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.shutdown()
    print(f'Injected a test candidate at ({x:.1f}, {y:.1f}).')


def clear(world):
    cli, msg_ns = gazebo_cli()
    names = existing_demo_models(cli)
    for name in names:
        ok = service(cli, msg_ns, f'/world/{world}/remove', 'Entity',
                     f'name: "{name}", type: MODEL')
        print(f'{"Removed" if ok else "Could not remove"} {name}')
    if not names:
        print('No demo models to remove.')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('event', choices=('heat', 'obstacle', 'unreachable', 'clear'))
    parser.add_argument('--world', default='large_demo')
    parser.add_argument('--distance', type=float, default=3.5,
                        help='obstacle: metres ahead of the Husky.')
    parser.add_argument('--x', type=float, default=150.0, help='unreachable: x [m]')
    parser.add_argument('--y', type=float, default=0.0, help='unreachable: y [m]')
    args = parser.parse_args()
    try:
        if args.event in ('heat', 'obstacle'):
            spawn(args.world, args.event, args.distance)
        elif args.event == 'unreachable':
            inject_candidate(args.x, args.y)
        else:
            clear(args.world)
    except RuntimeError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

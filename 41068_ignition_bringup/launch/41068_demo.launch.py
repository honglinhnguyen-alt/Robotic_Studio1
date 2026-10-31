"""One command for the whole Wild Track demo.

  ros2 launch 41068_ignition_bringup 41068_demo.launch.py                  # new random layout
  ros2 launch 41068_ignition_bringup 41068_demo.launch.py seed:=3          # repeat layout 3
  ros2 launch 41068_ignition_bringup 41068_demo.launch.py seed:=7 profile:=cover
  ros2 launch 41068_ignition_bringup 41068_demo.launch.py seed:=world      # world file layout

Starts, in order:
  1. Gazebo + SLAM + Nav2 (+ RViz)       41068_ignition_husky.launch.py
  2. unless seed:=world: random_animal_spawn.py moves the animals (profile:=nominal
     or cover) and writes ~/wildtrack_runs/ground_truth_<world>_<profile>_seedN.json.
     seed:=random (default) draws a new seed each launch and logs it, so a layout
     can be repeated with seed:=N.
  3. the mission stack (explore, perception, localisation, mission, evaluator,
     coverage, run record)                41068_mission.launch.py
     - after a random layout: as soon as the animals have been moved, using
       that ground truth (if moving them fails: the world file layout)
     - with seed:=world: after mission_delay seconds, using the world file layout
  4. the operator GUI (gui:=false to skip)

run_label defaults to "<profile>_seed<N>" or "<world>".
"""

import os
import random

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
from launch.actions import LogInfo
from launch.actions import OpaqueFunction
from launch.actions import RegisterEventHandler
from launch.actions import TimerAction
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _setup(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    share = FindPackageShare('41068_ignition_bringup').perform(context)
    world = arg('world')
    seed = arg('seed').strip().lower()
    if seed in ('world', 'none', 'fixed'):
        seed = ''
    elif seed in ('', 'random'):
        seed = str(random.SystemRandom().randint(1, 99999))
    profile = arg('profile')
    label = arg('run_label') or (f'{profile}_seed{seed}' if seed else world)
    ground_truth = ''
    if seed:
        ground_truth = os.path.expanduser(
            f'~/wildtrack_runs/ground_truth_{world}_{profile}_seed{seed}.json')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share, 'launch', '41068_ignition_husky.launch.py')),
        launch_arguments={
            'slam': 'true', 'nav2': 'true', 'world': world, 'rviz': arg('rviz'),
        }.items(),
    )

    def mission(ground_truth_file, run_label):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(share, 'launch', '41068_mission.launch.py')),
            launch_arguments={
                'world': world,
                'ground_truth_file': ground_truth_file,
                'run_label': run_label,
            }.items(),
        )

    actions = [sim]
    if arg('gui').lower() in ('true', '1'):
        actions.append(TimerAction(
            period=float(arg('mission_delay')),
            actions=[Node(package='wildlife_gui', executable='gui', name='wildlife_gui',
                          output='screen')]))

    if seed:
        actions.append(LogInfo(msg=f'Animal layout: {profile} seed {seed} '
                                   f'(repeat with seed:={seed}, fixed layout: seed:=world)'))
        spawn = ExecuteProcess(
            cmd=[os.path.join(get_package_prefix('41068_ignition_bringup'), 'lib',
                              '41068_ignition_bringup', 'random_animal_spawn.py'),
                 '--seed', seed, '--profile', profile,
                 '--world', os.path.join(share, 'worlds', f'{world}.sdf'),
                 '--ground-truth-out', ground_truth],
            output='screen',
        )

        def start_mission(event, _context):
            # The mission starts once the animals moved. If moving them failed,
            # they are still at the world file positions: evaluate against those.
            if event.returncode == 0:
                return [TimerAction(period=5.0, actions=[mission(ground_truth, label)])]
            return [LogInfo(msg='Animal layout failed: using the world file layout.'),
                    TimerAction(period=5.0, actions=[mission('', world)])]

        # Gazebo needs a few seconds before its set_pose service exists; the
        # script also waits for it.
        actions.append(TimerAction(period=8.0, actions=[spawn]))
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=spawn, on_exit=start_mission)))
    else:
        actions.append(TimerAction(period=float(arg('mission_delay')),
                                   actions=[mission('', label)]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='large_demo'),
        DeclareLaunchArgument('seed', default_value='random',
                              description='Animal layout: random (new layout each '
                                          'launch), N (repeat layout N) or world '
                                          '(world file layout).'),
        DeclareLaunchArgument('profile', default_value='nominal',
                              description='nominal or cover (animals next to trees).'),
        DeclareLaunchArgument('run_label', default_value='',
                              description='Scenario name in runs.csv.'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('mission_delay', default_value='20.0',
                              description='Seconds before mission/GUI start (Nav2 '
                                          'needs ~15 s).'),
        OpaqueFunction(function=_setup),
    ])

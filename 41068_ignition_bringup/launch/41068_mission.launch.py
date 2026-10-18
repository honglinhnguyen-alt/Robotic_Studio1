"""Launch the autonomy layer on top of the simulation + SLAM + Nav2.

Starts (each part can be switched off):
  explore:=true     explore_lite frontier exploration in /husky1
  perception:=true  thermal detector + thermal-gated YOLO (41068_thermal_yolo)
  spatial:=true     wildlife_spatial candidate_localizer + candidate_registry
  mission:=true     mission_node (R4) + husky_node (R3)
  evaluate:=true    wildlife_spatial spatial_evaluator (R7/R10 vs truth)
  coverage:=true    coverage_tracker (R1: LiDAR map and thermal coverage)
  record:=true      heat_source_logger: event log + run record in ~/wildtrack_runs
                    (mission_summary_<stamp>.json, one row in runs.csv)

Run the simulation first, e.g.
  ros2 launch 41068_ignition_bringup 41068_ignition_husky.launch.py \
      slam:=true nav2:=true rviz:=true world:=large_demo

world:= must name the same world as the simulation: the evaluator reads the
animal and campfire poses from worlds/<world>.sdf. After moving animals with
random_animal_spawn.py, pass ground_truth_file:=<json> instead.
run_label:= names the scenario in runs.csv, e.g. run_label:=nominal_1.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    sim_time = ParameterValue(use_sim_time, value_type=bool)
    mission_params = LaunchConfiguration('mission_params')

    bringup_share = FindPackageShare('41068_ignition_bringup')
    tf_remaps = [
        ('/tf', ['/', namespace, '/tf']),
        ('/tf_static', ['/', namespace, '/tf_static']),
    ]

    explore = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('explore_lite'), 'launch', 'explore.launch.py'])),
        launch_arguments={
            'namespace': namespace,
            'use_sim_time': use_sim_time,
        }.items(),
        condition=IfCondition(LaunchConfiguration('explore')),
    )

    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            bringup_share, 'launch', '41068_thermal_yolo.launch.py'])),
        launch_arguments={
            'namespace': namespace,
            'use_sim_time': use_sim_time,
            # YOLO only runs when husky_node calls verify_now at the
            # observation pose, never while exploring or driving.
            'auto_verify_on_heat': 'false',
        }.items(),
        condition=IfCondition(LaunchConfiguration('perception')),
    )

    localizer = Node(
        package='wildlife_spatial',
        executable='candidate_localizer',
        name='candidate_localizer',
        output='screen',
        parameters=[{'use_sim_time': sim_time}],
        remappings=tf_remaps,
        condition=IfCondition(LaunchConfiguration('spatial')),
    )
    registry = Node(
        package='wildlife_spatial',
        executable='candidate_registry',
        name='candidate_registry',
        output='screen',
        parameters=[{'use_sim_time': sim_time, 'merge_distance': 1.0}],
        condition=IfCondition(LaunchConfiguration('spatial')),
    )

    mission = Node(
        package='41068_ignition_bringup',
        executable='mission_node.py',
        name='mission_node',
        output='screen',
        parameters=[mission_params, {'use_sim_time': sim_time}],
        remappings=tf_remaps,
        condition=IfCondition(LaunchConfiguration('mission')),
    )
    husky = Node(
        package='41068_ignition_bringup',
        executable='husky_node.py',
        name='husky_node',
        output='screen',
        parameters=[mission_params, {'use_sim_time': sim_time}],
        remappings=tf_remaps,
        condition=IfCondition(LaunchConfiguration('mission')),
    )
    evaluator = Node(
        package='wildlife_spatial',
        executable='spatial_evaluator',
        name='spatial_evaluator',
        output='screen',
        parameters=[{
            'use_sim_time': sim_time,
            'ground_truth_world': PathJoinSubstitution(
                [bringup_share, 'worlds', [LaunchConfiguration('world'), '.sdf']]),
            # Takes precedence over ground_truth_world when not empty.
            'ground_truth_file': ParameterValue(
                LaunchConfiguration('ground_truth_file'), value_type=str),
            'output_directory': ParameterValue(
                LaunchConfiguration('evaluation_dir'), value_type=str),
        }],
        condition=IfCondition(LaunchConfiguration('evaluate')),
    )
    coverage = Node(
        package='41068_ignition_bringup',
        executable='coverage_tracker.py',
        name='coverage_tracker',
        output='screen',
        parameters=[mission_params, {'use_sim_time': sim_time}],
        remappings=tf_remaps,
        condition=IfCondition(LaunchConfiguration('coverage')),
    )
    recorder = Node(
        package='41068_ignition_bringup',
        executable='heat_source_logger.py',
        name='heat_source_logger',
        output='screen',
        parameters=[{
            'use_sim_time': sim_time,
            'run_label': ParameterValue(LaunchConfiguration('run_label'), value_type=str),
            'world': ParameterValue(LaunchConfiguration('world'), value_type=str),
        }],
        condition=IfCondition(LaunchConfiguration('record')),
    )

    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='husky1'),
        DeclareLaunchArgument('use_sim_time', default_value='True'),
        DeclareLaunchArgument('explore', default_value='true',
                              description='Start explore_lite.'),
        DeclareLaunchArgument('perception', default_value='true',
                              description='Start thermal detector + YOLO.'),
        DeclareLaunchArgument('spatial', default_value='true',
                              description='Start candidate localizer + registry.'),
        DeclareLaunchArgument('mission', default_value='true',
                              description='Start mission_node + husky_node.'),
        DeclareLaunchArgument(
            'mission_params',
            default_value=PathJoinSubstitution([bringup_share, 'config', 'mission_params.yaml']),
            description='Parameter file for mission_node and husky_node.'),
        DeclareLaunchArgument('evaluate', default_value='true',
                              description='Start spatial_evaluator.'),
        DeclareLaunchArgument('world', default_value='large_demo',
                              description='Simulated world; truth source for the evaluator.'),
        DeclareLaunchArgument('ground_truth_file', default_value='',
                              description='Truth JSON overriding the world file.'),
        DeclareLaunchArgument('evaluation_dir', default_value='~/wildtrack_runs/evaluation',
                              description='Parent directory of evaluator run logs.'),
        DeclareLaunchArgument('coverage', default_value='true',
                              description='Start coverage_tracker.'),
        DeclareLaunchArgument('record', default_value='true',
                              description='Start heat_source_logger (run record).'),
        DeclareLaunchArgument('run_label', default_value='',
                              description='Scenario name written to runs.csv.'),
        explore,
        perception,
        localizer,
        registry,
        mission,
        husky,
        evaluator,
        coverage,
        recorder,
    ])

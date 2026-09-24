from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='41068_ignition_bringup',
            executable='obstacle_avoidance_test.py',
            namespace='husky1',
            name='obstacle_avoidance_test',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'goal_x': 4.2,
                'goal_y': 0.0,
                'goal_yaw': 0.0,
                'map_frame': 'husky1_map',
            }],
            remappings=[
                ('/tf', 'tf'),
                ('/tf_static', 'tf_static'),
            ],
        )
    ])
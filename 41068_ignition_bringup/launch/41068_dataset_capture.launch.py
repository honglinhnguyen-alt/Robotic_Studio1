from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot = LaunchConfiguration('robot')
    interval_seconds = LaunchConfiguration('interval_seconds')
    max_images = LaunchConfiguration('max_images')
    output_directory = LaunchConfiguration('output_directory')

    return LaunchDescription([
        DeclareLaunchArgument(
            'robot',
            default_value='husky1',
            description='Robot namespace. For this dataset use husky1.',
        ),
        DeclareLaunchArgument(
            'interval_seconds',
            default_value='4.5',
            description='Wall-clock seconds between saved images.',
        ),
        DeclareLaunchArgument(
            'max_images',
            default_value='200',
            description='Stop automatically after this many images.',
        ),
        DeclareLaunchArgument(
            'output_directory',
            default_value='~/41068_ws/yolo_dataset',
            description='Base directory for dataset capture sessions.',
        ),
        Node(
            package='41068_ignition_bringup',
            executable='capture_yolo_dataset.py',
            namespace=robot,
            name='yolo_dataset_capture',
            output='screen',
            parameters=[{
                'image_topic': 'camera/image',
                'interval_seconds': interval_seconds,
                'max_images': max_images,
                'output_directory': output_directory,
                'jpeg_quality': 95,
            }],
        ),
    ])
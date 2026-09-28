"""Launch the complete thermal-candidate to RGB-YOLO perception pipeline."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    config_file = LaunchConfiguration('config_file')
    model_path = LaunchConfiguration('model_path')
    hot_threshold_k = LaunchConfiguration('hot_threshold_k')
    min_hot_area_px = LaunchConfiguration('min_hot_area_px')
    yolo_confidence = LaunchConfiguration('yolo_confidence')
    thermal_retrigger_sec = LaunchConfiguration('thermal_retrigger_sec')
    verification_frames = LaunchConfiguration('verification_frames')
    verification_min_votes = LaunchConfiguration('verification_min_votes')
    device = LaunchConfiguration('device')
    auto_verify_on_heat = LaunchConfiguration('auto_verify_on_heat')

    package_share = FindPackageShare('41068_ignition_bringup')
    default_config = PathJoinSubstitution([
        package_share,
        'config',
        'perception_thermal_yolo.yaml',
    ])
    default_model = PathJoinSubstitution([
        package_share,
        'weights',
        'my_model.pt',
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'namespace',
            default_value='husky1',
            description='Robot namespace.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use Gazebo simulation time.',
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Thermal and YOLO parameter file.',
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value=default_model,
            description='Path to the trained YOLO weights.',
        ),
        DeclareLaunchArgument(
            'hot_threshold_k',
            default_value='315.0',
            description='Minimum temperature for a heat candidate in kelvin.',
        ),
        DeclareLaunchArgument(
            'min_hot_area_px',
            default_value='30',
            description='Minimum connected hot-region area in thermal pixels.',
        ),
        DeclareLaunchArgument(
            'yolo_confidence',
            default_value='0.50',
            description='Minimum YOLO confidence.',
        ),
        DeclareLaunchArgument(
            'thermal_retrigger_sec',
            default_value='0.0',
            description=(
                'Repeat YOLO while heat remains visible; 0 means once until '
                'the heat disappears.'
            ),
        ),
        DeclareLaunchArgument(
            'verification_frames',
            default_value='5',
            description='Number of fresh RGB frames in each YOLO vote window.',
        ),
        DeclareLaunchArgument(
            'verification_min_votes',
            default_value='3',
            description='Votes required for one class to be accepted.',
        ),
        DeclareLaunchArgument(
            'auto_verify_on_heat',
            default_value='true',
            description=(
                'true: YOLO votes whenever heat appears. false: YOLO runs only '
                'when perception/verify_now is called (mission mode).'
            ),
        ),
        DeclareLaunchArgument(
            'device',
            default_value='',
            description='Ultralytics device override: cpu, 0, etc.',
        ),
        Node(
            package='41068_ignition_bringup',
            executable='thermal_candidate_detector.py',
            name='thermal_candidate_detector',
            namespace=namespace,
            output='screen',
            parameters=[
                config_file,
                {
                    'use_sim_time': ParameterValue(
                        use_sim_time,
                        value_type=bool,
                    ),
                    'hot_threshold_k': ParameterValue(
                        hot_threshold_k,
                        value_type=float,
                    ),
                    'min_hot_area_px': ParameterValue(
                        min_hot_area_px,
                        value_type=int,
                    ),
                },
            ],
        ),
        Node(
            package='41068_ignition_bringup',
            executable='animal_yolo_detector.py',
            name='animal_yolo_detector',
            namespace=namespace,
            output='screen',
            parameters=[
                config_file,
                {
                    'use_sim_time': ParameterValue(
                        use_sim_time,
                        value_type=bool,
                    ),
                    'model_path': model_path,
                    'confidence': ParameterValue(
                        yolo_confidence,
                        value_type=float,
                    ),
                    'thermal_retrigger_sec': ParameterValue(
                        thermal_retrigger_sec,
                        value_type=float,
                    ),
                    'verification_frames': ParameterValue(
                        verification_frames,
                        value_type=int,
                    ),
                    'verification_min_votes': ParameterValue(
                        verification_min_votes,
                        value_type=int,
                    ),
                    'device': ParameterValue(device, value_type=str),
                    'auto_verify_on_heat': ParameterValue(
                        auto_verify_on_heat,
                        value_type=bool,
                    ),
                },
            ],
        ),
    ])

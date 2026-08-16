from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    image_topic = LaunchConfiguration('image_topic')
    model_path = LaunchConfiguration('model_path')
    confidence = LaunchConfiguration('confidence')
    iou = LaunchConfiguration('iou')
    imgsz = LaunchConfiguration('imgsz')
    continuous = LaunchConfiguration('continuous')
    max_inference_hz = LaunchConfiguration('max_inference_hz')
    device = LaunchConfiguration('device')

    default_model = PathJoinSubstitution([
        FindPackageShare('41068_ignition_bringup'),
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
            'image_topic',
            default_value='camera/image',
            description='RGB image topic relative to the robot namespace.',
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value=default_model,
            description='Path to the trained YOLO .pt weights.',
        ),
        DeclareLaunchArgument(
            'confidence',
            default_value='0.50',
            description='YOLO confidence threshold.',
        ),
        DeclareLaunchArgument(
            'iou',
            default_value='0.70',
            description='YOLO NMS IoU threshold.',
        ),
        DeclareLaunchArgument(
            'imgsz',
            default_value='640',
            description='YOLO inference image size.',
        ),
        DeclareLaunchArgument(
            'continuous',
            default_value='True',
            description='Run continuous RGB detection while surveying.',
        ),
        DeclareLaunchArgument(
            'max_inference_hz',
            default_value='10.0',
            description='Maximum YOLO inference rate.',
        ),
        DeclareLaunchArgument(
            'device',
            default_value='',
            description='Ultralytics device override, e.g. cpu, 0. Empty = auto.',
        ),

        Node(
            package='41068_ignition_bringup',
            executable='animal_yolo_detector.py',
            name='animal_yolo_detector',
            namespace=namespace,
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'image_topic': image_topic,
                'model_path': model_path,
                'confidence': confidence,
                'iou': iou,
                'imgsz': imgsz,
                'continuous': continuous,
                'max_inference_hz': max_inference_hz,
                'device': device,
                'publish_annotated_image': True,
            }],
        ),
    ])
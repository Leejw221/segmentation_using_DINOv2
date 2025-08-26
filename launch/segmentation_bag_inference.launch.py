from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for bag-based DINOv2 segmentation
    Optimized for processing pre-recorded bag files with GPU acceleration
    """
    
    # Launch arguments with defaults optimized for bag processing
    backbone_size_arg = DeclareLaunchArgument(
        'backbone_size',
        default_value='base',
        choices=['base', 'giant'],
        description='DINOv2 backbone size (base=fast, giant=accurate)'
    )
    
    head_type_arg = DeclareLaunchArgument(
        'head_type', 
        default_value='multiscale',
        choices=['linear', 'multiscale'],
        description='Segmentation head type (linear=fast, multiscale=accurate)'
    )
    
    resolution_arg = DeclareLaunchArgument(
        'resolution',
        default_value='518', 
        description='Input resolution (must be multiple of 14, higher=more accurate)'
    )
    
    filter_mode_arg = DeclareLaunchArgument(
        'filter_mode',
        default_value='all_classes',
        choices=['all_classes', 'lab_only'],
        description='Object filtering mode'
    )
    
    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/camera/color/image_raw',
        description='Input image topic from bag file (original bag topic name)'
    )
    
    bag_number_arg = DeclareLaunchArgument(
        'bag_number',
        default_value='4',
        description='Bag file number to process (e.g., 4 for 8.20.4.tar.gz)'
    )
    
    # Startup info
    startup_info = LogInfo(
        msg=[
            '🚀 Starting DINOv2 Bag Segmentation\n',
            '📁 Bag number: ', LaunchConfiguration('bag_number'), '\n',
            '🔧 Backbone: ', LaunchConfiguration('backbone_size'), '\n', 
            '🧠 Head: ', LaunchConfiguration('head_type'), '\n',
            '📐 Resolution: ', LaunchConfiguration('resolution'), '\n',
            '🎯 Filter: ', LaunchConfiguration('filter_mode'), '\n',
            '📺 Input topic: ', LaunchConfiguration('input_topic'), '\n',
            '💡 GPU acceleration enabled if available\n',
            '📊 Results published to: /dinov2/bag_segmentation_result\n',
            '--------------------------------------------------\n',
            '🎮 Usage Instructions:\n',
            '1. Start bag playback in another terminal:\n',
            '   ./scripts/extract_bag.sh ', LaunchConfiguration('bag_number'), '\n',
            '   ros2 bag play [extracted_path] --remap /camera/color/image_raw:', LaunchConfiguration('input_topic'), '\n',
            '2. View results:\n',
            '   rqt_image_view (select /dinov2/bag_segmentation_result)\n',
            '3. Monitor performance in this terminal\n',
            '--------------------------------------------------'
        ]
    )
    
    # Main segmentation node
    segmentation_node = Node(
        package='dinov2_ros_segmentation',
        executable='dinov2_bag_segmentation_node',
        name='dinov2_bag_segmentation_node',
        output='screen',
        parameters=[{
            'backbone_size': LaunchConfiguration('backbone_size'),
            'head_type': LaunchConfiguration('head_type'),
            'resolution': LaunchConfiguration('resolution'),
            'filter_mode': LaunchConfiguration('filter_mode'),
            'input_topic': LaunchConfiguration('input_topic'),
        }],
        # GPU optimization environment
        additional_env={'CUDA_VISIBLE_DEVICES': '0'}  # Use first GPU
    )
    
    return LaunchDescription([
        # Arguments
        backbone_size_arg,
        head_type_arg, 
        resolution_arg,
        filter_mode_arg,
        input_topic_arg,
        bag_number_arg,
        
        # Actions
        startup_info,
        segmentation_node,
    ])
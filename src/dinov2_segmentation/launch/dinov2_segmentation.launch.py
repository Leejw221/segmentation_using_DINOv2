#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('dinov2_segmentation')
    
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.join(pkg_dir, 'models', 'best_model.pth'),
        description='Path to trained DINOv2 segmentation model'
    )
    
    num_classes_arg = DeclareLaunchArgument(
        'num_classes',
        default_value='150',
        description='Number of segmentation classes'
    )
    
    visualization_alpha_arg = DeclareLaunchArgument(
        'visualization_alpha',
        default_value='0.6',
        description='Alpha value for segmentation overlay'
    )
    
    auto_demo_arg = DeclareLaunchArgument(
        'auto_demo',
        default_value='true',
        description='Enable automatic demo with sample images'
    )
    
    demo_interval_arg = DeclareLaunchArgument(
        'demo_interval',
        default_value='5.0',
        description='Interval between demo images in seconds'
    )
    
    # DINOv2 Segmentation Node
    segmentation_node = Node(
        package='dinov2_segmentation',
        executable='ros2_segmentation_node',
        name='dinov2_segmentation',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'num_classes': LaunchConfiguration('num_classes'),
            'visualization_alpha': LaunchConfiguration('visualization_alpha'),
            'auto_demo': LaunchConfiguration('auto_demo'),
            'demo_interval': LaunchConfiguration('demo_interval'),
        }],
        remappings=[
            # You can remap topics here if needed
            # ('/dinov2/input_image', '/camera/image_raw'),  # Example for real camera
        ]
    )
    
    return LaunchDescription([
        model_path_arg,
        num_classes_arg,
        visualization_alpha_arg,
        auto_demo_arg,
        demo_interval_arg,
        segmentation_node,
    ])
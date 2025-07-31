#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """
    Simple demo launch file for quick testing
    """
    # Get package directory
    pkg_dir = get_package_share_directory('dinov2_segmentation')
    
    # Demo configuration - optimized for quick testing
    demo_node = Node(
        package='dinov2_segmentation',
        executable='ros2_segmentation_node',
        name='dinov2_demo',
        output='screen',
        parameters=[{
            'model_path': os.path.join(pkg_dir, 'models', 'best_model.pth'),
            'num_classes': 150,
            'visualization_alpha': 0.7,
            'auto_demo': True,
            'demo_interval': 3.0,  # Faster demo for testing
        }]
    )
    
    return LaunchDescription([
        demo_node,
    ])
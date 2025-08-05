#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """
    Launch DINOv2 segmentation node with automatic rqt visualization
    """
    # Get package directory
    pkg_dir = get_package_share_directory('dinov2_segmentation')
    
    # DINOv2 segmentation node
    dinov2_node = Node(
        package='dinov2_segmentation',
        executable='ros2_segmentation_node',
        name='dinov2_segmentation',
        output='screen',
        parameters=[{
            'model_path': os.path.join(pkg_dir, 'models', 'dinov2_trained_model', 'best_model.pth'),
            'num_classes': 151,  # ADE20K classes + background
            'visualization_alpha': 0.6,
            'auto_demo': False,  # 자동 데모 비활성화
            'demo_interval': 5.0,
        }]
    )
    
    # rqt 자동 실행 (3초 후)
    rqt_node = ExecuteProcess(
        cmd=['bash', '-c', 'sleep 3 && rqt'],
        output='screen',
        shell=False
    )
    
    return LaunchDescription([
        dinov2_node,
        rqt_node,
    ])
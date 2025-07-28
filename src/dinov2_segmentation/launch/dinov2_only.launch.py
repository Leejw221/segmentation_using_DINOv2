#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # DINOv2 Segmentation Node만 실행
        Node(
            package='dinov2_segmentation',
            executable='rise2_seg_node',
            name='dinov2_segmentation',
            output='screen',
            parameters=[],
        ),
    ])
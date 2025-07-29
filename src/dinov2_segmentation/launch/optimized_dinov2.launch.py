#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='dinov2_segmentation',
            executable='optimized_rise2_seg_node',
            name='optimized_dinov2_segmentation',
            output='screen',
            parameters=[
                {'use_sim_time': False}
            ],
            remappings=[
                ('/dinov2/segmentation_result', '/optimized_dinov2/segmentation_result'),
                ('/dinov2/original_image', '/optimized_dinov2/original_image'),
                ('/dinov2/image_url', '/optimized_dinov2/image_url'),
                ('/dinov2/image_file', '/optimized_dinov2/image_file')
            ]
        )
    ])

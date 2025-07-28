#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction
import os

def generate_launch_description():
    return LaunchDescription([
        # DINOv2 Segmentation Node
        Node(
            package='dinov2_segmentation',
            executable='rise2_seg_node',
            name='dinov2_segmentation',
            output='screen',
            parameters=[],
        ),
        
        # RQT Image View를 3초 후에 실행 (노드가 준비될 시간)
        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    cmd=['rqt', '--standalone', 'rqt_image_view'],
                    output='screen'
                )
            ]
        ),
        
        # 첫 번째 테스트 이미지를 5초 후에 전송
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='dinov2_segmentation',
                    executable='file_publisher',
                    name='test_image_1',
                    arguments=[os.path.expanduser('~/dinov2_ws/test_images/segmentation_demo.jpg')],
                    output='screen',
                )
            ]
        ),
        
        # 두 번째 테스트 이미지를 10초 후에 전송
        TimerAction(
            period=10.0,
            actions=[
                Node(
                    package='dinov2_segmentation',
                    executable='file_publisher',
                    name='test_image_2',
                    arguments=[os.path.expanduser('~/dinov2_ws/test_images/outdoor_scene.jpg')],
                    output='screen',
                )
            ]
        ),
    ])
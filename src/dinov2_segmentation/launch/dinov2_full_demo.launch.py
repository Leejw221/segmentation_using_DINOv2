#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction
import os

def generate_launch_description():
    return LaunchDescription([
        # DINOv2 Segmentation Node 실행
        Node(
            package='dinov2_segmentation',
            executable='rise2_seg_node',
            name='dinov2_segmentation',
            output='screen',
            parameters=[],
        ),
        
        # rqt를 2초 후에 실행 (일반 rqt - 사용자가 플러그인 추가)
        TimerAction(
            period=2.0,
            actions=[
                ExecuteProcess(
                    cmd=['rqt'],
                    output='screen',
                    shell=True
                )
            ]
        ),
        
        # 두 번째 rqt 창을 3초 후에 실행 (여러 창 동시 확인 가능)
        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    cmd=['rqt'],
                    output='screen',
                    shell=True
                )
            ]
        ),
        
        # RQT Graph도 실행 (노드 연결 상태 확인용)
        TimerAction(
            period=4.0,
            actions=[
                ExecuteProcess(
                    cmd=['rqt_graph'],
                    output='screen',
                    shell=True
                )
            ]
        ),
        
        # 첫 번째 테스트 이미지 전송
        TimerAction(
            period=6.0,
            actions=[
                Node(
                    package='dinov2_segmentation',
                    executable='file_publisher',
                    name='demo_image_1',
                    arguments=[os.path.expanduser('~/dinov2_ws/test_images/segmentation_demo.jpg')],
                    output='screen',
                )
            ]
        ),
        
        # 두 번째 테스트 이미지 전송
        TimerAction(
            period=12.0,
            actions=[
                Node(
                    package='dinov2_segmentation',
                    executable='file_publisher',
                    name='demo_image_2',
                    arguments=[os.path.expanduser('~/dinov2_ws/test_images/outdoor_scene.jpg')],
                    output='screen',
                )
            ]
        ),
        
        # 세 번째로 다시 첫 번째 이미지 (반복 테스트)
        TimerAction(
            period=18.0,
            actions=[
                Node(
                    package='dinov2_segmentation',
                    executable='file_publisher',
                    name='demo_image_3',
                    arguments=[os.path.expanduser('~/dinov2_ws/test_images/segmentation_demo.jpg')],
                    output='screen',
                )
            ]
        ),
    ])
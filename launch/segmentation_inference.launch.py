#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch.substitutions import PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    DINOv2 RealSense segmentation launch file (RealSense only mode)
    """
    
    # Declare launch arguments
    backbone_size_arg = DeclareLaunchArgument(
        'backbone_size',
        default_value='base',
        description='DINOv2 backbone size: small, base, large, giant'
    )
    
    head_type_arg = DeclareLaunchArgument(
        'head_type',
        default_value='multiscale',
        description='Head type: linear (faster) or multiscale (BNHead - more accurate)'
    )
    
    filter_mode_arg = DeclareLaunchArgument(
        'filter_mode',
        default_value='all_classes',
        description='Filter mode: all_classes or lab_only'
    )
    
    output_mode_arg = DeclareLaunchArgument(
        'output_mode',
        default_value='both',
        description='Output mode: 2d, 3d, or both'
    )
    
    # DINOv2 segmentation node
    dinov2_node = Node(
        package='dinov2_ros_segmentation',
        executable='dinov2_segmentation_node',
        name='dinov2_segmentation',
        output='screen',
        parameters=[{
            'backbone_size': LaunchConfiguration('backbone_size'),
            'head_type': LaunchConfiguration('head_type'),
            'filter_mode': LaunchConfiguration('filter_mode'),
            'output_mode': LaunchConfiguration('output_mode'),
            'dataset': 'ade20k',
            'resolution': 518,  # Multiple of 14 for DINOv2 patches
        }]
    )
    
    # RealSense camera driver
    realsense_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('realsense2_camera'), '/launch/rs_launch.py'
        ]),
        launch_arguments={
            'enable_depth': 'true',  # Enable depth for 3D mode
            'enable_infra1': 'false',
            'enable_infra2': 'false',
            'enable_color': 'true',
            'rgb_camera.color_profile': '640,480,60',  # USB 3.0 optimal
            'depth_module.depth_profile': '640,480,60',  # USB 3.0 optimal
            'align_depth.enable': 'true'  # Align depth to color
        }.items()
    )
    
    return LaunchDescription([
        backbone_size_arg,
        head_type_arg,
        filter_mode_arg,
        output_mode_arg,
        dinov2_node,
        realsense_camera,
    ])
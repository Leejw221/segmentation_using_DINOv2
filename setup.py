from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dinov2_ros_segmentation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'torchvision', 
        'opencv-python',
        'pillow',
        'numpy',
        'requests',
    ],
    zip_safe=True,
    maintainer='leejungwook',
    maintainer_email='leeju0917@gmail.com',
    description='DINOv2 official semantic segmentation ROS2 package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dinov2_segmentation_node = dinov2_ros_segmentation.dinov2_segmentation_node:main',
            'dinov2_bag_segmentation_node = dinov2_ros_segmentation.dinov2_bag_segmentation_node:main',
        ],
    },
)

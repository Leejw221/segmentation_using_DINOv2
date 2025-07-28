from setuptools import setup
import os
from glob import glob

package_name = 'dinov2_segmentation'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch 파일들 추가
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='leejungwook',
    maintainer_email='your_email@example.com',
    description='DINOv2 segmentation for mobile manipulator research',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'static_seg_node = dinov2_segmentation.static_seg_node:main',
            'rise2_seg_node = dinov2_segmentation.rise2_seg_node:main',
            'url_publisher = dinov2_segmentation.url_publisher:main',
            'file_publisher = dinov2_segmentation.file_publisher:main',
            'image_test_node = dinov2_segmentation.image_test_node:main',
            'test_node = dinov2_segmentation.test_node:main',
        ],
    },
)
from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'dinov2_segmentation'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Launch files
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
        
        # Configuration files
        (os.path.join('share', package_name, 'configs'), 
         glob('configs/*.yaml')),
        
        # Models directory (for trained weights)
        (os.path.join('share', package_name, 'models'), 
         []),  # Empty for now, models will be added after training
    ],
    install_requires=[
        'setuptools',
        'torch',
        'torchvision',
        'transformers',
        'numpy',
        'opencv-python',
        'pillow',
        'requests',
        'pyyaml',
        'tqdm',
        'scikit-learn',
        'albumentations',
        'tensorboard',
    ],
    zip_safe=True,
    maintainer='leejungwook',
    maintainer_email='your_email@example.com',
    description='DINOv2 based semantic segmentation for ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # ROS2 nodes
            'ros2_segmentation_node = dinov2_segmentation.inference.ros2_segmentation_node:main',
            
            # Training scripts
            'train_segmentation = dinov2_segmentation.training.train_segmentation:main',
            
            # Inference scripts
            'inference = dinov2_segmentation.inference.inference:main',
        ],
    },
)
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys

class ImageFilePublisher(Node):
    def __init__(self):
        super().__init__('image_file_publisher')
        
        self.publisher = self.create_publisher(String, '/dinov2/image_file', 10)
        
        self.get_logger().info('Image File Publisher initialized!')
        self.get_logger().info('Usage: ros2 run dinov2_segmentation file_publisher <image_file_path>')
    
    def publish_file_path(self, file_path):
        msg = String()
        msg.data = file_path
        self.publisher.publish(msg)
        self.get_logger().info(f'Published file path: {file_path}')

def main(args=None):
    rclpy.init(args=args)
    
    if len(sys.argv) < 2:
        print("Usage: ros2 run dinov2_segmentation file_publisher <image_file_path>")
        print("Example:")
        print("  ros2 run dinov2_segmentation file_publisher ~/dinov2_ws/test_images/segmentation_demo.jpg")
        return
    
    file_path = sys.argv[1]
    node = ImageFilePublisher()
    
    # 파일 경로 발행
    node.publish_file_path(file_path)
    
    # 잠시 대기 후 종료
    rclpy.spin_once(node, timeout_sec=1.0)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys

class ImageURLPublisher(Node):
    def __init__(self):
        super().__init__('image_url_publisher')
        
        self.publisher = self.create_publisher(String, '/dinov2/image_url', 10)
        
        self.get_logger().info('Image URL Publisher initialized!')
        self.get_logger().info('Usage: ros2 run dinov2_segmentation url_publisher <image_url>')
    
    def publish_url(self, url):
        msg = String()
        msg.data = url
        self.publisher.publish(msg)
        self.get_logger().info(f'Published URL: {url}')

def main(args=None):
    rclpy.init(args=args)
    
    if len(sys.argv) < 2:
        print("Usage: ros2 run dinov2_segmentation url_publisher <image_url>")
        print("Example URLs:")
        print("  https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg")
        print("  https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=640")
        return
    
    url = sys.argv[1]
    node = ImageURLPublisher()
    
    # URL 발행
    node.publish_url(url)
    
    # 잠시 대기 후 종료
    rclpy.spin_once(node, timeout_sec=1.0)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

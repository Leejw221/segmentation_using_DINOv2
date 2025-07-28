#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageTestNode(Node):
    def __init__(self):
        super().__init__('image_test_node')
        
        self.bridge = CvBridge()
        
        # 테스트 이미지 발행자
        self.test_publisher = self.create_publisher(
            Image,
            '/test/simple_image',
            10)
        
        # 1초마다 간단한 테스트 이미지 발행
        self.timer = self.create_timer(1.0, self.publish_test_image)
        
        self.counter = 0
        self.get_logger().info("Image Test Node initialized!")
        self.get_logger().info("Publishing test images to /test/simple_image")
    
    def publish_test_image(self):
        """간단한 컬러 이미지 생성 및 발행"""
        # 320x240 크기의 컬러 이미지 생성
        height, width = 240, 320
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 색상 변화 (빨강 -> 초록 -> 파랑 순환)
        color_cycle = self.counter % 3
        if color_cycle == 0:
            image[:, :] = [0, 0, 255]  # 빨강 (BGR)
        elif color_cycle == 1:
            image[:, :] = [0, 255, 0]  # 초록
        else:
            image[:, :] = [255, 0, 0]  # 파랑
        
        # 중앙에 흰색 텍스트 추가
        text = f"Test {self.counter}"
        cv2.putText(image, text, (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # ROS 메시지로 변환 및 발행
        msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "test_frame"
        
        self.test_publisher.publish(msg)
        self.get_logger().info(f"Published test image {self.counter} (color: {color_cycle})")
        
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = ImageTestNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
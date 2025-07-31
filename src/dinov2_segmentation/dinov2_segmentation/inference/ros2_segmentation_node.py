#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from PIL import Image as PILImage
import requests
import io
from ament_index_python.packages import get_package_share_directory

from .inference import DINOv2SegmentationInference


class DINOv2SegmentationNode(Node):
    """
    ROS2 Node for DINOv2 Segmentation
    """
    def __init__(self):
        super().__init__('dinov2_segmentation_node')
        
        # ROS2 setup
        self.bridge = CvBridge()
        
        # Publishers
        self.result_publisher = self.create_publisher(
            Image, '/dinov2/segmentation_result', 10)
        self.original_publisher = self.create_publisher(
            Image, '/dinov2/original_image', 10)
        self.confidence_publisher = self.create_publisher(
            Image, '/dinov2/confidence_map', 10)
        
        # Subscribers
        self.url_subscription = self.create_subscription(
            String, '/dinov2/image_url', self.url_callback, 10)
        self.file_subscription = self.create_subscription(
            String, '/dinov2/image_file', self.file_callback, 10)
        self.image_subscription = self.create_subscription(
            Image, '/dinov2/input_image', self.image_callback, 10)
        
        # Parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('num_classes', 150)
        self.declare_parameter('visualization_alpha', 0.6)
        self.declare_parameter('auto_demo', True)
        self.declare_parameter('demo_interval', 5.0)
        
        # Get parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        num_classes = self.get_parameter('num_classes').get_parameter_value().integer_value
        self.alpha = self.get_parameter('visualization_alpha').get_parameter_value().double_value
        auto_demo = self.get_parameter('auto_demo').get_parameter_value().bool_value
        demo_interval = self.get_parameter('demo_interval').get_parameter_value().double_value
        
        # Load inference model
        self.get_logger().info("Loading DINOv2 segmentation model...")
        try:
            if not model_path:
                # Try to find model in package share directory
                package_share = get_package_share_directory('dinov2_segmentation')
                model_path = os.path.join(package_share, 'models', 'best_model.pth')
            
            if not os.path.exists(model_path):
                self.get_logger().error(f"Model not found at: {model_path}")
                self.get_logger().info("Please train a model first or specify correct model_path parameter")
                return
            
            self.inference = DINOv2SegmentationInference(
                model_path=model_path,
                num_classes=num_classes
            )
            
            self.get_logger().info(f"Model loaded successfully from: {model_path}")
            self.get_logger().info(f"Number of classes: {num_classes}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {str(e)}")
            return
        
        # Sample images for demo
        self.sample_urls = [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg",
            "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=640",
            "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=640"
        ]
        self.url_index = 0
        
        # Auto demo timer
        if auto_demo:
            self.timer = self.create_timer(demo_interval, self.demo_callback)
            self.get_logger().info(f"Auto demo enabled with {demo_interval}s interval")
        
        self.get_logger().info("DINOv2 Segmentation Node initialized!")
        self.get_logger().info("Available topics:")
        self.get_logger().info("  Subscribe: /dinov2/image_url (String)")
        self.get_logger().info("  Subscribe: /dinov2/image_file (String)")  
        self.get_logger().info("  Subscribe: /dinov2/input_image (Image)")
        self.get_logger().info("  Publish: /dinov2/segmentation_result (Image)")
        self.get_logger().info("  Publish: /dinov2/original_image (Image)")
        self.get_logger().info("  Publish: /dinov2/confidence_map (Image)")
    
    def process_image(self, pil_image):
        """
        Process image with DINOv2 segmentation
        
        Args:
            pil_image: PIL Image object
        """
        if pil_image is None:
            return
        
        try:
            self.get_logger().info(f"Processing image: {pil_image.size}")
            
            # Perform segmentation
            seg_map, visualization, confidence = self.inference.predict_and_visualize(
                pil_image, alpha=self.alpha
            )
            
            # Convert PIL to OpenCV format
            image_array = np.array(pil_image)
            original_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Publish original image
            try:
                original_msg = self.bridge.cv2_to_imgmsg(original_bgr, "bgr8")
                original_msg.header.stamp = self.get_clock().now().to_msg()
                original_msg.header.frame_id = "camera_frame"
                self.original_publisher.publish(original_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish original image: {str(e)}")
            
            # Publish segmentation result
            try:
                result_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
                result_msg = self.bridge.cv2_to_imgmsg(result_bgr, "bgr8")
                result_msg.header.stamp = self.get_clock().now().to_msg()
                result_msg.header.frame_id = "camera_frame"
                self.result_publisher.publish(result_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish segmentation result: {str(e)}")
            
            # Publish confidence map
            try:
                # Normalize confidence to 0-255 range
                conf_normalized = (confidence * 255).astype(np.uint8)
                conf_msg = self.bridge.cv2_to_imgmsg(conf_normalized, "mono8")
                conf_msg.header.stamp = self.get_clock().now().to_msg()
                conf_msg.header.frame_id = "camera_frame"
                self.confidence_publisher.publish(conf_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish confidence map: {str(e)}")
            
            # Print statistics
            stats = self.inference.get_class_statistics(seg_map)
            top_classes = sorted(stats.items(), key=lambda x: x[1]['percentage'], reverse=True)[:5]
            
            self.get_logger().info("Top 5 detected classes:")
            for class_id, stat in top_classes:
                self.get_logger().info(f"  Class {class_id}: {stat['percentage']:.1f}%")
            
            self.get_logger().info("✅ Segmentation completed and published!")
            
        except Exception as e:
            self.get_logger().error(f"Image processing failed: {str(e)}")
    
    def download_image(self, url):
        """Download image from URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            pil_image = PILImage.open(io.BytesIO(response.content))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            return pil_image
        except Exception as e:
            self.get_logger().error(f"Failed to download image from {url}: {str(e)}")
            return None
    
    def url_callback(self, msg):
        """Handle URL input"""
        url = msg.data
        self.get_logger().info(f"Processing image from URL: {url}")
        pil_image = self.download_image(url)
        self.process_image(pil_image)
    
    def file_callback(self, msg):
        """Handle file path input"""
        file_path = msg.data
        self.get_logger().info(f"Processing image from file: {file_path}")
        try:
            pil_image = PILImage.open(file_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            self.process_image(pil_image)
        except Exception as e:
            self.get_logger().error(f"Failed to load image from {file_path}: {str(e)}")
    
    def image_callback(self, msg):
        """Handle ROS Image input"""
        self.get_logger().info("Processing ROS Image message")
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Convert to PIL Image
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
            self.process_image(pil_image)
        except Exception as e:
            self.get_logger().error(f"Failed to process ROS Image: {str(e)}")
    
    def demo_callback(self):
        """Auto demo callback"""
        if self.url_index < len(self.sample_urls):
            url = self.sample_urls[self.url_index]
            self.get_logger().info(f"🎬 Auto demo {self.url_index + 1}/{len(self.sample_urls)}: {url}")
            pil_image = self.download_image(url)
            self.process_image(pil_image)
            self.url_index += 1
        else:
            self.url_index = 0  # Reset for continuous demo


def main(args=None):
    rclpy.init(args=args)
    
    node = DINOv2SegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
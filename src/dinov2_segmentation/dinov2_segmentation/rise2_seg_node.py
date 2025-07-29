#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from PIL import Image as PILImage
import torchvision.transforms as transforms
import io
from transformers import AutoModel, AutoImageProcessor

class DINOv2ForSegmentation(nn.Module):
    """학습된 DINOv2 Segmentation 모델 클래스"""
    def __init__(self, num_classes=150):
        super().__init__()
        # DINOv2 backbone (frozen)
        self.dinov2 = AutoModel.from_pretrained("facebook/dinov2-base")
        for param in self.dinov2.parameters():
            param.requires_grad = False
        
        # Segmentation head (trainable)
        self.classifier = nn.Conv2d(768, num_classes, kernel_size=1)
        
    def forward(self, pixel_values):
        # DINOv2 feature extraction
        outputs = self.dinov2(pixel_values)
        features = outputs.last_hidden_state[:, 1:]  # Remove CLS token
        
        # Reshape to 2D
        batch_size, seq_len, hidden_size = features.shape
        height = width = int(seq_len ** 0.5)  # 16x16 for base model
        features = features.reshape(batch_size, height, width, hidden_size)
        features = features.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Classification
        logits = self.classifier(features)
        
        # Upsample to original size
        logits = torch.nn.functional.interpolate(
            logits, size=(224, 224), mode='bilinear', align_corners=False
        )
        return logits

class LearnedDINOv2SegmentationNode(Node):
    def __init__(self):
        super().__init__('learned_dinov2_segmentation_node')
        
        # ROS2 설정
        self.bridge = CvBridge()
        
        # Publishers
        self.result_publisher = self.create_publisher(
            Image,
            '/dinov2/segmentation_result',
            10)
        
        self.original_publisher = self.create_publisher(
            Image,
            '/dinov2/original_image',
            10)
        
        # Subscribers
        self.url_subscription = self.create_subscription(
            String,
            '/dinov2/image_url',
            self.url_callback,
            10)
        
        self.file_subscription = self.create_subscription(
            String,
            '/dinov2/image_file',
            self.file_callback,
            10)
        
        # 학습된 DINOv2 모델 로드
        self.get_logger().info("Loading trained DINOv2 segmentation model...")
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 모델 경로 설정
            model_path = "/home/leejungwook/Downloads/dinov2_trained_model"
            model_weights_path = f"{model_path}/dinov2_segmentation_trained_model.pth"
            processor_path = f"{model_path}/dinov2_segmentation_trained"
            
            # 모델 생성 및 가중치 로드 (학습된 모델과 동일한 클래스 수)
            self.model = DINOv2ForSegmentation(num_classes=104)  # 학습된 모델 클래스 수
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            # Processor 로드
            self.processor = AutoImageProcessor.from_pretrained(processor_path)
            
            # 클래스 정보 (학습된 모델 기준)
            self.num_classes = 104
            
            self.get_logger().info(f"Trained model loaded successfully on {self.device}")
            self.get_logger().info(f"Number of classes: {self.num_classes}")
        except Exception as e:
            self.get_logger().error(f"Failed to load trained model: {str(e)}")
            return
        
        # 타이머로 샘플 이미지 처리
        self.timer = self.create_timer(3.0, self.process_sample_image)
        
        # 샘플 이미지 URL들
        self.sample_urls = [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg",
            "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=640",
            "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=640"
        ]
        self.url_index = 0
        
        self.get_logger().info("Learned DINOv2 Segmentation Node initialized!")
        self.get_logger().info("Using trained segmentation model instead of K-means clustering!")
    
    def download_image(self, url):
        """웹에서 이미지 다운로드"""
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
    
    def perform_learned_segmentation(self, pil_image):
        """학습된 모델로 segmentation 수행"""
        try:
            # 이미지 전처리
            inputs = self.processor(pil_image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            
            # 추론
            with torch.no_grad():
                outputs = self.model(pixel_values)
                predictions = torch.argmax(outputs, dim=1)  # [B, H, W]
            
            # CPU로 이동 및 numpy 변환
            seg_map = predictions.squeeze(0).cpu().numpy()  # [H, W]
            
            self.get_logger().info(f"Segmentation completed. Shape: {seg_map.shape}")
            self.get_logger().info(f"Unique classes found: {np.unique(seg_map)[:10]}...")  # 처음 10개만 표시
            
            return seg_map
            
        except Exception as e:
            self.get_logger().error(f"Learned segmentation failed: {str(e)}")
            return None
    
    def create_colored_segmentation(self, seg_map):
        """Segmentation 결과에 색상 적용"""
        h, w = seg_map.shape
        colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 각 클래스마다 고유한 색상 할당
        np.random.seed(42)  # 일관된 색상
        
        unique_classes = np.unique(seg_map)
        for class_id in unique_classes:
            if class_id == 0:  # 배경은 검은색
                continue
            
            mask = seg_map == class_id
            if np.any(mask):
                # 클래스별 고유 색상 생성
                color = [
                    int((class_id * 50) % 255),
                    int((class_id * 100) % 255), 
                    int((class_id * 150) % 255)
                ]
                colored_seg[mask] = color
        
        return colored_seg
    
    def process_image(self, pil_image):
        """이미지 처리 메인 함수"""
        if pil_image is None:
            return
        
        # PIL Image를 numpy array로 변환
        image_array = np.array(pil_image)
        
        self.get_logger().info(f"Processing image of shape: {image_array.shape}")
        
        # 1. 원본 이미지 발행
        original_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        if original_bgr.dtype != np.uint8:
            original_bgr = original_bgr.astype(np.uint8)
        
        if not original_bgr.flags['C_CONTIGUOUS']:
            original_bgr = np.ascontiguousarray(original_bgr)
        
        try:
            original_msg = self.bridge.cv2_to_imgmsg(original_bgr, "bgr8")
            original_msg.header.stamp = self.get_clock().now().to_msg()
            original_msg.header.frame_id = "camera_frame"
            self.original_publisher.publish(original_msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish original image: {str(e)}")
        
        # 2. 학습된 모델로 segmentation 수행
        self.get_logger().info("Performing learned segmentation...")
        seg_map = self.perform_learned_segmentation(pil_image)
        if seg_map is None:
            return
        
        # 3. 원본 이미지 크기로 resize
        orig_h, orig_w = image_array.shape[:2]
        seg_map_resized = cv2.resize(
            seg_map.astype(np.float32), 
            (orig_w, orig_h), 
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
        
        # 4. 색상 적용
        colored_seg = self.create_colored_segmentation(seg_map_resized)
        
        # 5. 원본과 결과 합성
        alpha = 0.6
        overlay = cv2.addWeighted(
            original_bgr, 1-alpha, 
            colored_seg, alpha, 0
        )
        
        # 이미지 형식 검증
        if overlay.dtype != np.uint8:
            overlay = overlay.astype(np.uint8)
        
        if not overlay.flags['C_CONTIGUOUS']:
            overlay = np.ascontiguousarray(overlay)
        
        # 6. 결과 발행
        try:
            result_msg = self.bridge.cv2_to_imgmsg(overlay, "bgr8")
            result_msg.header.stamp = self.get_clock().now().to_msg()
            result_msg.header.frame_id = "camera_frame"
            self.result_publisher.publish(result_msg)
            
            self.get_logger().info(f"Published segmentation result: {overlay.shape}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish result: {str(e)}")
            return
        
        self.get_logger().info("✅ Learned segmentation completed and published!")
    
    def url_callback(self, msg):
        """URL을 받아서 이미지 처리"""
        url = msg.data
        self.get_logger().info(f"Processing image from URL: {url}")
        
        pil_image = self.download_image(url)
        self.process_image(pil_image)
    
    def file_callback(self, msg):
        """로컬 파일을 받아서 이미지 처리"""
        file_path = msg.data
        self.get_logger().info(f"Processing image from file: {file_path}")
        
        try:
            pil_image = PILImage.open(file_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            self.process_image(pil_image)
            
        except Exception as e:
            self.get_logger().error(f"Failed to load image from {file_path}: {str(e)}")
    
    def process_sample_image(self):
        """샘플 이미지 처리 (타이머 콜백)"""
        if self.url_index < len(self.sample_urls):
            url = self.sample_urls[self.url_index]
            self.get_logger().info(f"Auto-processing sample image {self.url_index + 1}: {url}")
            
            pil_image = self.download_image(url)
            self.process_image(pil_image)
            
            self.url_index += 1
            
            if self.url_index == 1:
                self.timer.cancel()
                self.timer = self.create_timer(10.0, self.process_sample_image)
                self.get_logger().info("Timer interval changed to 10 seconds")
                
        else:
            self.url_index = 0
            self.get_logger().info("All sample images processed. Restarting...")

def main(args=None):
    rclpy.init(args=args)
    node = LearnedDINOv2SegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

# Ultra-Fast DINOv2 Segmentation Node
# 실제로 small 모델 + 추가 최적화 기술들

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
import time
from PIL import Image as PILImage
import requests
import io

class UltraFastDINOv2Segmentation(nn.Module):
    """초고속 DINOv2 Segmentation - Small 모델 + 최적화"""
    def __init__(self, num_classes=150):
        super().__init__()
        # Small 모델 사용 (22M vs 86M parameters)
        self.dinov2 = AutoModel.from_pretrained("facebook/dinov2-small")
        
        # 모델 동결
        for param in self.dinov2.parameters():
            param.requires_grad = False
        
        # 경량화된 segmentation head
        self.classifier = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=1),  # 채널 축소
            nn.ReLU(inplace=True),
            nn.Conv2d(192, num_classes, kernel_size=1)
        )
        
    def forward(self, pixel_values):
        # DINOv2 feature extraction
        outputs = self.dinov2(pixel_values)
        features = outputs.last_hidden_state[:, 1:]  # Remove CLS token
        
        # Reshape to 2D (14x14 for small model)
        batch_size, seq_len, hidden_size = features.shape
        height = width = int(seq_len ** 0.5)  # 14x14
        features = features.reshape(batch_size, height, width, hidden_size)
        features = features.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Classification
        logits = self.classifier(features)
        
        # 더 작은 출력 크기 (속도 향상)
        logits = torch.nn.functional.interpolate(
            logits, size=(96, 96), mode='bilinear', align_corners=False
        )
        return logits

class UltraFastSegmentationNode(Node):
    def __init__(self):
        super().__init__('ultra_fast_segmentation_node')
        
        self.bridge = CvBridge()
        
        # Publishers
        self.result_publisher = self.create_publisher(
            Image, '/ultra_fast/segmentation_result', 10)
        self.original_publisher = self.create_publisher(
            Image, '/ultra_fast/original_image', 10)
        
        # Subscribers  
        self.url_subscription = self.create_subscription(
            String, '/ultra_fast/image_url', self.url_callback, 10)
        self.file_subscription = self.create_subscription(
            String, '/ultra_fast/image_file', self.file_callback, 10)
        
        # 초고속 모델 설정
        self.setup_ultra_fast_model()
        
        # 성능 측정
        self.processing_times = []
        
        # 테스트 타이머
        self.timer = self.create_timer(3.0, self.process_sample_image)
        self.sample_urls = [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg"
        ]
        self.url_index = 0
        
    def setup_ultra_fast_model(self):
        """초고속 모델 설정"""
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.get_logger().info(f"🚀 Ultra-fast mode on: {self.device}")
            
            # 새로운 small 모델 생성 (기존 가중치 무시하고 랜덤 초기화)
            self.model = UltraFastDINOv2Segmentation(num_classes=50)  # 클래스 수도 축소
            self.model.to(self.device)
            self.model.eval()
            
            # 강력한 최적화
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model, mode='max-autotune')
                self.get_logger().info("🔥 Model compiled with max-autotune mode")
            
            # 빠른 processor 설정
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            
            self.get_logger().info("⚡ Ultra-fast model ready!")
            
        except Exception as e:
            self.get_logger().error(f"Ultra-fast setup failed: {e}")
    
    def perform_ultra_fast_segmentation(self, pil_image):
        """초고속 segmentation"""
        start_time = time.time()
        
        try:
            # 더 작은 크기로 aggressive 리사이징
            width, height = pil_image.size
            if max(width, height) > 256:  # 더 작게!
                if width > height:
                    new_width, new_height = 256, int(height * 256 / width)
                else:
                    new_width, new_height = int(width * 256 / height), 256
                pil_image = pil_image.resize((new_width, new_height), PILImage.LANCZOS)
                self.get_logger().info(f"⚡ Resized: {width}x{height} → {new_width}x{new_height}")
            
            # 최적화된 전처리
            inputs = self.processor(pil_image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            
            # 초고속 추론
            with torch.no_grad():
                outputs = self.model(pixel_values)
                predictions = torch.argmax(outputs, dim=1)
            
            seg_map = predictions.squeeze(0).cpu().numpy()
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            if len(self.processing_times) > 5:
                self.processing_times = self.processing_times[-5:]
            avg_time = np.mean(self.processing_times)
            
            self.get_logger().info(f"⚡ Ultra-fast segmentation: {processing_time:.3f}s (avg: {avg_time:.3f}s)")
            
            return seg_map
            
        except Exception as e:
            self.get_logger().error(f"Ultra-fast segmentation failed: {e}")
            return None
    
    def create_simple_colored_segmentation(self, seg_map):
        """단순화된 색상 적용"""
        # 미리 정의된 색상 팔레트 (계산 시간 절약)
        palette = np.array([
            [0, 0, 0],      # 배경
            [255, 0, 0],    # 빨강
            [0, 255, 0],    # 초록  
            [0, 0, 255],    # 파랑
            [255, 255, 0],  # 노랑
            [255, 0, 255],  # 마젠타
            [0, 255, 255],  # 시안
            [128, 0, 0],    # 어두운 빨강
            [0, 128, 0],    # 어두운 초록
            [0, 0, 128],    # 어두운 파랑
        ], dtype=np.uint8)
        
        # 클래스 수 제한
        seg_map_clamped = np.clip(seg_map, 0, len(palette)-1)
        
        return palette[seg_map_clamped]
    
    def process_image(self, pil_image):
        """초고속 이미지 처리"""
        if pil_image is None:
            return
        
        start_total = time.time()
        
        # 원본 발행
        image_array = np.array(pil_image)
        original_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        try:
            original_msg = self.bridge.cv2_to_imgmsg(original_bgr, "bgr8")
            original_msg.header.stamp = self.get_clock().now().to_msg()
            self.original_publisher.publish(original_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish original: {e}")
            return
        
        # 초고속 segmentation
        seg_map = self.perform_ultra_fast_segmentation(pil_image)
        if seg_map is None:
            return
        
        # 원본 크기로 빠른 리사이즈
        orig_h, orig_w = image_array.shape[:2]
        if seg_map.shape != (orig_h, orig_w):
            seg_map = cv2.resize(seg_map.astype(np.float32), (orig_w, orig_h), 
                               interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        
        # 단순화된 색상 적용
        colored_seg = self.create_simple_colored_segmentation(seg_map)
        
        # 결과 합성
        overlay = cv2.addWeighted(original_bgr, 0.4, colored_seg, 0.6, 0)
        
        # 결과 발행
        try:
            result_msg = self.bridge.cv2_to_imgmsg(overlay, "bgr8")
            result_msg.header.stamp = self.get_clock().now().to_msg()
            self.result_publisher.publish(result_msg)
            
            total_time = time.time() - start_total
            self.get_logger().info(f"🚀 TOTAL ultra-fast processing: {total_time:.3f}s")
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish result: {e}")
    
    def download_image(self, url):
        """이미지 다운로드"""
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            pil_image = PILImage.open(io.BytesIO(response.content))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            return pil_image
        except Exception as e:
            self.get_logger().error(f"Download failed: {e}")
            return None
    
    def url_callback(self, msg):
        pil_image = self.download_image(msg.data)
        self.process_image(pil_image)
    
    def file_callback(self, msg):
        try:
            pil_image = PILImage.open(msg.data)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            self.process_image(pil_image)
        except Exception as e:
            self.get_logger().error(f"File load failed: {e}")
    
    def process_sample_image(self):
        """샘플 처리"""
        if self.url_index < len(self.sample_urls):
            self.get_logger().info(f"🚀 Ultra-fast test {self.url_index + 1}")
            pil_image = self.download_image(self.sample_urls[self.url_index])
            self.process_image(pil_image)
            self.url_index += 1
        else:
            self.url_index = 0

def main(args=None):
    rclpy.init(args=args)
    node = UltraFastSegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

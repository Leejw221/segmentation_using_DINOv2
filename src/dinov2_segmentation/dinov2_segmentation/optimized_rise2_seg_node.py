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
import time
import gc

class OptimizedDINOv2ForSegmentation(nn.Module):
    """최적화된 DINOv2 Segmentation 모델"""
    def __init__(self, num_classes=150, use_small_model=False):  # 기본값 False로 변경
        super().__init__()
        # 더 작은 모델 사용
        model_name = "facebook/dinov2-small" if use_small_model else "facebook/dinov2-base"
        self.dinov2 = AutoModel.from_pretrained(model_name)
        
        # 모델 동결
        for param in self.dinov2.parameters():
            param.requires_grad = False
        
        # Segmentation head
        hidden_size = 384 if use_small_model else 768
        self.classifier = nn.Conv2d(hidden_size, num_classes, kernel_size=1)
        
        # 최적화를 위한 플래그
        self.use_small_model = use_small_model
        
    def forward(self, pixel_values):
        # DINOv2 feature extraction
        outputs = self.dinov2(pixel_values)
        features = outputs.last_hidden_state[:, 1:]  # Remove CLS token
        
        # Reshape to 2D
        batch_size, seq_len, hidden_size = features.shape
        height = width = int(seq_len ** 0.5)  # 14x14 for small, 16x16 for base
        features = features.reshape(batch_size, height, width, hidden_size)
        features = features.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Classification
        logits = self.classifier(features)
        
        # 더 작은 크기로 업샘플링 (성능 향상)
        output_size = 224 if not self.use_small_model else 112  # base 모델용으로 수정
        logits = torch.nn.functional.interpolate(
            logits, size=(output_size, output_size), mode='bilinear', align_corners=False
        )
        return logits

class OptimizedDINOv2SegmentationNode(Node):
    def __init__(self):
        super().__init__('optimized_dinov2_segmentation_node')
        
        # ROS2 설정
        self.bridge = CvBridge()
        
        # Publishers
        self.result_publisher = self.create_publisher(
            Image, '/dinov2/segmentation_result', 10)
        self.original_publisher = self.create_publisher(
            Image, '/dinov2/original_image', 10)
        
        # Subscribers
        self.url_subscription = self.create_subscription(
            String, '/dinov2/image_url', self.url_callback, 10)
        self.file_subscription = self.create_subscription(
            String, '/dinov2/image_file', self.file_callback, 10)
        
        # 최적화된 모델 로드
        self.get_logger().info("Loading optimized DINOv2 segmentation model...")
        self.setup_optimized_model()
        
        # 성능 모니터링
        self.processing_times = []
        
        # 샘플 이미지 처리 타이머 (더 긴 간격)
        self.timer = self.create_timer(5.0, self.process_sample_image)
        self.sample_urls = [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg"
        ]
        self.url_index = 0
        
        self.get_logger().info("Optimized DINOv2 Segmentation Node initialized!")
    
    def setup_optimized_model(self):
        """최적화된 모델 설정"""
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.get_logger().info(f"Using device: {self.device}")
            
            # 모델 경로
            model_path = "/home/leejungwook/Downloads/dinov2_trained_model"
            model_weights_path = f"{model_path}/dinov2_segmentation_trained_model.pth"
            processor_path = f"{model_path}/dinov2_segmentation_trained"
            
            # 기존 base 모델 유지 (가중치 호환성을 위해)
            self.model = OptimizedDINOv2ForSegmentation(
                num_classes=104, 
                use_small_model=False  # base 모델 유지
            )
            
            # 가중치 로드 
            try:
                state_dict = torch.load(model_weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.get_logger().info("Loaded complete model weights")
                    
            except Exception as e:
                self.get_logger().warning(f"Could not load weights: {e}, using random initialization")
            
            self.model.to(self.device)
            self.model.eval()
            
            # 모델 컴파일 (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model)
                    self.get_logger().info("Model compiled with torch.compile")
                except:
                    self.get_logger().info("torch.compile not available")
            
            # Processor 로드
            self.processor = AutoImageProcessor.from_pretrained(processor_path)
            
            # 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.get_logger().info("Optimized model loaded successfully")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load optimized model: {str(e)}")
            return
    
    def perform_optimized_segmentation(self, pil_image):
        """최적화된 segmentation 수행"""
        start_time = time.time()
        
        try:
            # 이미지 크기 최적화 (원본 크기가 너무 크면 리사이즈)
            width, height = pil_image.size
            if max(width, height) > 512:
                # 큰 이미지는 512로 리사이즈
                if width > height:
                    new_width, new_height = 512, int(height * 512 / width)
                else:
                    new_width, new_height = int(width * 512 / height), 512
                pil_image = pil_image.resize((new_width, new_height), PILImage.LANCZOS)
                self.get_logger().info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            # 이미지 전처리
            inputs = self.processor(pil_image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            
            # 추론 (no_grad 사용)
            with torch.no_grad():
                outputs = self.model(pixel_values)
                predictions = torch.argmax(outputs, dim=1)
            
            # CPU로 이동 및 numpy 변환
            seg_map = predictions.squeeze(0).cpu().numpy()
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # 최근 5개 평균 처리 시간 출력
            if len(self.processing_times) > 5:
                self.processing_times = self.processing_times[-5:]
            avg_time = np.mean(self.processing_times)
            
            self.get_logger().info(f"Segmentation completed in {processing_time:.2f}s (avg: {avg_time:.2f}s)")
            self.get_logger().info(f"Output shape: {seg_map.shape}, unique classes: {len(np.unique(seg_map))}")
            
            return seg_map
            
        except Exception as e:
            self.get_logger().error(f"Optimized segmentation failed: {str(e)}")
            return None
        finally:
            # 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def create_colored_segmentation(self, seg_map):
        """빠른 색상 적용"""
        h, w = seg_map.shape
        
        # 실제 클래스 개수에 맞춰 색상 생성
        max_class_id = int(seg_map.max())
        num_colors_needed = max_class_id + 1
        
        # 색상 팔레트 생성 (충분한 크기로)
        np.random.seed(42)
        colors = np.random.randint(0, 255, (num_colors_needed, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # 배경은 검은색
        
        # 벡터화된 색상 매핑
        colored_seg = colors[seg_map]
        
        return colored_seg
    
    def process_image(self, pil_image):
        """최적화된 이미지 처리"""
        if pil_image is None:
            return
        
        start_total = time.time()
        
        # PIL Image를 numpy array로 변환
        image_array = np.array(pil_image)
        self.get_logger().info(f"Processing image: {image_array.shape}")
        
        # 1. 원본 이미지 발행
        original_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        try:
            original_msg = self.bridge.cv2_to_imgmsg(original_bgr, "bgr8")
            original_msg.header.stamp = self.get_clock().now().to_msg()
            self.original_publisher.publish(original_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish original: {str(e)}")
            return
        
        # 2. 최적화된 segmentation
        seg_map = self.perform_optimized_segmentation(pil_image)
        if seg_map is None:
            return
        
        # 3. 원본 크기로 리사이즈
        orig_h, orig_w = image_array.shape[:2]
        if seg_map.shape != (orig_h, orig_w):
            seg_map = cv2.resize(
                seg_map.astype(np.float32), 
                (orig_w, orig_h), 
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
        
        # 4. 빠른 색상 적용
        colored_seg = self.create_colored_segmentation(seg_map)
        
        # 5. 결과 합성
        alpha = 0.6
        overlay = cv2.addWeighted(original_bgr, 1-alpha, colored_seg, alpha, 0)
        
        # 6. 결과 발행
        try:
            result_msg = self.bridge.cv2_to_imgmsg(overlay, "bgr8")
            result_msg.header.stamp = self.get_clock().now().to_msg()
            self.result_publisher.publish(result_msg)
            
            total_time = time.time() - start_total
            self.get_logger().info(f"✅ Total processing time: {total_time:.2f}s")
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish result: {str(e)}")
    
    def download_image(self, url):
        """이미지 다운로드"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            pil_image = PILImage.open(io.BytesIO(response.content))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            return pil_image
        except Exception as e:
            self.get_logger().error(f"Download failed: {str(e)}")
            return None
    
    def url_callback(self, msg):
        """URL 콜백"""
        url = msg.data
        self.get_logger().info(f"Processing URL: {url}")
        pil_image = self.download_image(url)
        self.process_image(pil_image)
    
    def file_callback(self, msg):
        """파일 콜백"""
        file_path = msg.data
        self.get_logger().info(f"Processing file: {file_path}")
        try:
            pil_image = PILImage.open(file_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            self.process_image(pil_image)
        except Exception as e:
            self.get_logger().error(f"File load failed: {str(e)}")
    
    def process_sample_image(self):
        """샘플 이미지 처리"""
        if self.url_index < len(self.sample_urls):
            url = self.sample_urls[self.url_index]
            self.get_logger().info(f"Auto-processing sample {self.url_index + 1}")
            pil_image = self.download_image(url)
            self.process_image(pil_image)
            self.url_index += 1
        else:
            self.url_index = 0

def main(args=None):
    rclpy.init(args=args)
    node = OptimizedDINOv2SegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

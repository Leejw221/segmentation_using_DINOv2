#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from transformers import Dinov2Model, Dinov2Processor
from sklearn.cluster import KMeans
import requests
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import io
import base64

class StaticDINOv2SegmentationNode(Node):
    def __init__(self):
        super().__init__('static_dinov2_segmentation_node')
        
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
        
        # Subscribers (URL을 받기 위한)
        self.url_subscription = self.create_subscription(
            String,
            '/dinov2/image_url',
            self.url_callback,
            10)
        
        # DINOv2 모델 로드
        self.get_logger().info("Loading DINOv2 model...")
        try:
            self.processor = Dinov2Processor.from_pretrained('facebook/dinov2-base')
            self.model = Dinov2Model.from_pretrained('facebook/dinov2-base')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.get_logger().info(f"DINOv2 model loaded successfully on {self.device}")
        except Exception as e:
            self.get_logger().error(f"Failed to load DINOv2 model: {str(e)}")
            return
        
        # Segmentation 설정
        self.n_clusters = 6  # 클러스터 수
        
        # 타이머로 샘플 이미지 처리 (테스트용)
        self.timer = self.create_timer(10.0, self.process_sample_image)
        
        # 샘플 이미지 URL들
        self.sample_urls = [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg",
            "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=640",
            "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=640"
        ]
        self.url_index = 0
        
        self.get_logger().info("Static DINOv2 Segmentation Node initialized!")
        self.get_logger().info("Publish image URL to /dinov2/image_url or wait for automatic processing")
    
    def download_image(self, url):
        """웹에서 이미지 다운로드"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # PIL Image로 변환
            pil_image = PILImage.open(io.BytesIO(response.content))
            
            # RGB로 변환 (RGBA인 경우 등)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Numpy array로 변환
            image_array = np.array(pil_image)
            
            return image_array
            
        except Exception as e:
            self.get_logger().error(f"Failed to download image from {url}: {str(e)}")
            return None
    
    def extract_features(self, image):
        """DINOv2로부터 feature 추출"""
        try:
            # PIL Image로 변환 (transformers가 PIL을 기대함)
            if isinstance(image, np.ndarray):
                pil_image = PILImage.fromarray(image)
            else:
                pil_image = image
            
            # 이미지 전처리
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Feature 추출
            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state  # [1, num_patches, hidden_dim]
            
            return features.cpu().numpy()
            
        except Exception as e:
            self.get_logger().error(f"Feature extraction failed: {str(e)}")
            return None
    
    def perform_segmentation(self, image, features):
        """K-means clustering으로 segmentation 수행"""
        try:
            h, w = image.shape[:2]
            
            # Feature reshape
            features = features.squeeze(0)  # [num_patches, hidden_dim]
            
            # DINOv2는 14x14 patch 사용
            # 이미지 크기에 따라 patch 수 계산
            patch_size = 14
            # Processor가 resize하므로 실제 patch 수는 고정
            # DINOv2-base: 224x224 -> 16x16 patches (CLS token 제외하면 256개)
            num_patches_per_side = int(np.sqrt(features.shape[0]))
            
            # K-means clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Segmentation map 생성
            seg_map = cluster_labels.reshape(num_patches_per_side, num_patches_per_side)
            
            # 원본 이미지 크기로 resize
            seg_map_resized = cv2.resize(
                seg_map.astype(np.float32), 
                (w, h), 
                interpolation=cv2.INTER_NEAREST
            )
            
            return seg_map_resized.astype(np.uint8)
            
        except Exception as e:
            self.get_logger().error(f"Segmentation failed: {str(e)}")
            return None
    
    def create_colored_segmentation(self, seg_map):
        """Segmentation 결과에 색상 적용"""
        # 각 클러스터에 대한 색상 정의 (BGR 형식)
        colors = [
            [255, 0, 0],    # 빨강
            [0, 255, 0],    # 초록
            [0, 0, 255],    # 파랑
            [255, 255, 0],  # 노랑
            [255, 0, 255],  # 마젠타
            [0, 255, 255],  # 시안
            [128, 128, 128], # 회색
            [255, 128, 0],  # 주황
            [128, 0, 255],  # 보라
            [0, 128, 255],  # 하늘
        ]
        
        h, w = seg_map.shape
        colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i in range(self.n_clusters):
            mask = seg_map == i
            if i < len(colors):
                colored_seg[mask] = colors[i]
            else:
                # 랜덤 색상 생성
                colored_seg[mask] = np.random.randint(0, 255, 3)
        
        return colored_seg
    
    def process_image(self, image_array):
        """이미지 처리 메인 함수"""
        if image_array is None:
            return
        
        self.get_logger().info(f"Processing image of shape: {image_array.shape}")
        
        # 1. 원본 이미지 발행
        original_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        original_msg = self.bridge.cv2_to_imgmsg(original_bgr, "bgr8")
        self.original_publisher.publish(original_msg)
        
        # 2. DINOv2 feature 추출
        self.get_logger().info("Extracting features with DINOv2...")
        features = self.extract_features(image_array)
        if features is None:
            return
        
        # 3. Segmentation 수행
        self.get_logger().info("Performing segmentation...")
        seg_map = self.perform_segmentation(image_array, features)
        if seg_map is None:
            return
        
        # 4. 색상 적용
        colored_seg = self.create_colored_segmentation(seg_map)
        
        # 5. 원본과 결과 합성
        alpha = 0.6
        overlay = cv2.addWeighted(
            original_bgr, 1-alpha, 
            colored_seg, alpha, 0
        )
        
        # 6. 결과 발행
        result_msg = self.bridge.cv2_to_imgmsg(overlay, "bgr8")
        self.result_publisher.publish(result_msg)
        
        self.get_logger().info("Segmentation completed and published!")
    
    def url_callback(self, msg):
        """URL을 받아서 이미지 처리"""
        url = msg.data
        self.get_logger().info(f"Processing image from URL: {url}")
        
        # 이미지 다운로드
        image_array = self.download_image(url)
        
        # 이미지 처리
        self.process_image(image_array)
    
    def process_sample_image(self):
        """샘플 이미지 처리 (타이머 콜백)"""
        if self.url_index < len(self.sample_urls):
            url = self.sample_urls[self.url_index]
            self.get_logger().info(f"Auto-processing sample image {self.url_index + 1}: {url}")
            
            # 이미지 다운로드 및 처리
            image_array = self.download_image(url)
            self.process_image(image_array)
            
            self.url_index += 1
        else:
            # 모든 샘플 처리 완료 후 타이머 중지
            self.timer.cancel()
            self.get_logger().info("All sample images processed. Send URL to /dinov2/image_url for more processing.")

def main(args=None):
    rclpy.init(args=args)
    node = StaticDINOv2SegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

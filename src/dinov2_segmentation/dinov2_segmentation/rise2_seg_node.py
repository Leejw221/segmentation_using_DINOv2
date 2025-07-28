#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import requests
from PIL import Image as PILImage
import torchvision.transforms as transforms
import io
from transformers import AutoModel

class RISE2StyleDINOv2SegmentationNode(Node):
    def __init__(self):
        super().__init__('rise2_dinov2_segmentation_node')
        
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
        
        # 로컬 파일을 받기 위한 subscriber 추가
        self.file_subscription = self.create_subscription(
            String,
            '/dinov2/image_file',
            self.file_callback,
            10)
        
        # DINOv2 모델 로드 (RISE-2 정확한 방식)
        self.get_logger().info("Loading DINOv2 model via AutoModel (RISE-2 style)...")
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # RISE-2와 동일한 방식: AutoModel.from_pretrained
            # HuggingFace에서 공식 DINOv2 모델 로드
            model_name = "facebook/dinov2-base"  # 또는 "facebook/dinov2-small", "facebook/dinov2-large"
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # 모델 설정 정보
            self.patch_size = self.model.config.patch_size  # 14
            self.hidden_size = self.model.config.hidden_size  # 768
            
            # 이미지 전처리 변환 정의 (DINOv2 표준)
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            self.get_logger().info(f"DINOv2 model loaded successfully on {self.device}")
            self.get_logger().info(f"Model config - patch_size: {self.patch_size}, hidden_size: {self.hidden_size}")
        except Exception as e:
            self.get_logger().error(f"Failed to load DINOv2 model: {str(e)}")
            return
        
        # Segmentation 설정
        self.n_clusters = 6  # 클러스터 수
        
        # 타이머로 샘플 이미지 처리 (첫 번째는 빠르게, 이후는 10초 간격)
        self.timer = self.create_timer(3.0, self.process_sample_image)  # 첫 실행 3초 후
        
        # 샘플 이미지 URL들
        self.sample_urls = [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg",
            "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=640",
            "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=640"
        ]
        self.url_index = 0
        
        self.get_logger().info("RISE-2 Style DINOv2 Segmentation Node initialized!")
        self.get_logger().info("Send URL to /dinov2/image_url or file path to /dinov2/image_file")
        self.get_logger().info("Or wait for automatic sample processing")
    
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
            
            return pil_image
            
        except Exception as e:
            self.get_logger().error(f"Failed to download image from {url}: {str(e)}")
            return None
    
    def extract_features(self, pil_image):
        """DINOv2로부터 feature 추출 (RISE-2 정확한 방식)"""
        try:
            # 이미지 전처리
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # RISE-2와 동일한 feature 추출
            with torch.no_grad():
                # AutoModel forward -> last_hidden_state 사용
                outputs = self.model(input_tensor)
                
                # RISE-2 스타일: last_hidden_state에서 CLS token 제거
                # outputs.last_hidden_state shape: [B, 197, hidden_size]
                # [:, 1:] -> CLS token(첫 번째) 제거하고 patch tokens만 사용
                patch_features = outputs.last_hidden_state[:, 1:]  # [B, 196, hidden_size]
                
                self.get_logger().info(f"RISE-2 style features shape: {patch_features.shape}")
                self.get_logger().info(f"Expected: [1, 196, {self.hidden_size}]")
            
            return patch_features.cpu().numpy()
            
        except Exception as e:
            self.get_logger().error(f"Feature extraction failed: {str(e)}")
            return None
    
    def perform_segmentation(self, image, features):
        """K-means clustering으로 segmentation 수행 (RISE-2 스타일)"""
        try:
            # 원본 이미지 크기
            orig_h, orig_w = np.array(image).shape[:2]
            
            # Feature reshape: [1, 196, hidden_size] -> [196, hidden_size]
            features = features.squeeze(0)  # [196, hidden_size]
            self.get_logger().info(f"Features for clustering: {features.shape}")
            
            # RISE-2에서 사용하는 patch grid 크기 계산
            # 224x224 이미지, patch_size=14 -> 16x16=256??? 아니 14x14=196
            H, W = 224, 224  # 입력 이미지 크기
            grid_H, grid_W = H // self.patch_size, W // self.patch_size  # 16, 16
            self.get_logger().info(f"Grid size: {grid_H}x{grid_W} = {grid_H*grid_W}")
            
            # K-means clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Segmentation map 생성 
            # DINOv2는 14x14 패치를 사용하므로
            patch_grid_size = int(np.sqrt(len(cluster_labels)))  # 14
            seg_map = cluster_labels.reshape(patch_grid_size, patch_grid_size)
            
            self.get_logger().info(f"Segmentation map shape: {seg_map.shape}")
            
            # 원본 이미지 크기로 resize
            seg_map_resized = cv2.resize(
                seg_map.astype(np.float32), 
                (orig_w, orig_h), 
                interpolation=cv2.INTER_NEAREST
            )
            
            return seg_map_resized.astype(np.uint8)
            
        except Exception as e:
            self.get_logger().error(f"Segmentation failed: {str(e)}")
            return None
    
    def create_colored_segmentation(self, seg_map):
        """Segmentation 결과에 색상 적용 (HuggingFace 스타일 랜덤)"""
        h, w = seg_map.shape
        colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
        
        # HuggingFace 예제처럼 각 클러스터마다 랜덤 색상 생성
        # np.random.seed(42)  # 일관된 색상: 주석 해제하면 매번 같은 색상
        # 주석 처리하면 매번 다른 랜덤 색상
        
        for i in range(self.n_clusters):
            mask = seg_map == i
            if np.any(mask):  # 해당 클러스터가 존재하는 경우만
                # 밝고 선명한 랜덤 색상 생성 (BGR 순서)
                random_color = [
                    np.random.randint(50, 255),   # B
                    np.random.randint(50, 255),   # G  
                    np.random.randint(50, 255)    # R
                ]
                colored_seg[mask] = random_color
        
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
        
        # 이미지 형식 검증
        if original_bgr.dtype != np.uint8:
            original_bgr = original_bgr.astype(np.uint8)
        
        if not original_bgr.flags['C_CONTIGUOUS']:
            original_bgr = np.ascontiguousarray(original_bgr)
        
        try:
            original_msg = self.bridge.cv2_to_imgmsg(original_bgr, "bgr8")
            # 헤더 정보 추가
            original_msg.header.stamp = self.get_clock().now().to_msg()
            original_msg.header.frame_id = "camera_frame"
            self.original_publisher.publish(original_msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish original image: {str(e)}")
        
        # 2. DINOv2 feature 추출 (RISE-2 방식)
        self.get_logger().info("Extracting features with DINOv2...")
        features = self.extract_features(pil_image)
        if features is None:
            return
        
        # 3. Segmentation 수행
        self.get_logger().info("Performing segmentation...")
        seg_map = self.perform_segmentation(pil_image, features)
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
        
        # 이미지 형식 검증 및 수정
        if overlay.dtype != np.uint8:
            overlay = overlay.astype(np.uint8)
        
        # 이미지 크기 및 채널 확인
        if len(overlay.shape) != 3 or overlay.shape[2] != 3:
            self.get_logger().error(f"Invalid image shape: {overlay.shape}")
            return
        
        # 이미지가 연속적인 메모리인지 확인 (cv_bridge 요구사항)
        if not overlay.flags['C_CONTIGUOUS']:
            overlay = np.ascontiguousarray(overlay)
        
        # 6. 결과 발행
        try:
            result_msg = self.bridge.cv2_to_imgmsg(overlay, "bgr8")
            # 헤더 정보 추가
            result_msg.header.stamp = self.get_clock().now().to_msg()
            result_msg.header.frame_id = "camera_frame"
            self.result_publisher.publish(result_msg)
            
            self.get_logger().info(f"Published image: {overlay.shape}, dtype: {overlay.dtype}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to convert/publish image: {str(e)}")
            return
        
        self.get_logger().info("Segmentation completed and published!")
    
    def url_callback(self, msg):
        """URL을 받아서 이미지 처리"""
        url = msg.data
        self.get_logger().info(f"Processing image from URL: {url}")
        
        # 이미지 다운로드
        pil_image = self.download_image(url)
        
        # 이미지 처리
        self.process_image(pil_image)
    
    def file_callback(self, msg):
        """로컬 파일을 받아서 이미지 처리"""
        file_path = msg.data
        self.get_logger().info(f"Processing image from file: {file_path}")
        
        try:
            # 로컬 파일에서 이미지 로드
            pil_image = PILImage.open(file_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 이미지 처리
            self.process_image(pil_image)
            
        except Exception as e:
            self.get_logger().error(f"Failed to load image from {file_path}: {str(e)}")
    
    def process_sample_image(self):
        """샘플 이미지 처리 (타이머 콜백) - 무한 반복"""
        if self.url_index < len(self.sample_urls):
            url = self.sample_urls[self.url_index]
            self.get_logger().info(f"Auto-processing sample image {self.url_index + 1}: {url}")
            
            # 이미지 다운로드 및 처리
            pil_image = self.download_image(url)
            self.process_image(pil_image)
            
            self.url_index += 1
            
            # 첫 번째 이미지 처리 후 타이머 간격을 10초로 변경
            if self.url_index == 1:
                self.timer.cancel()
                self.timer = self.create_timer(10.0, self.process_sample_image)
                self.get_logger().info("Timer interval changed to 10 seconds")
                
        else:
            # 모든 샘플 처리 완료 후 처음부터 다시 시작 (무한 반복)
            self.url_index = 0
            self.get_logger().info("All sample images processed. Restarting from first image...")

def main(args=None):
    rclpy.init(args=args)
    node = RISE2StyleDINOv2SegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

"""
추론 테스트 스크립트 - GPU 없이 더미 모델로 테스트
"""

import torch
import numpy as np
from PIL import Image
import cv2
import sys
import os

# 패키지 모듈 import
sys.path.append('/home/leejungwook/dinov2_ws/src/dinov2_segmentation')
from dinov2_segmentation.training.model import DINOv2ForSegmentation
from dinov2_segmentation.inference.inference import DINOv2SegmentationInference

def create_dummy_model():
    """더미 모델 생성"""
    print("🤖 Creating dummy model...")
    
    # 작은 모델 생성 (클래스 수를 줄여서 빠르게)
    model = DINOv2ForSegmentation(num_classes=10, freeze_backbone=True)
    
    # 더미 가중치로 초기화 (실제 학습된 모델 없이)
    dummy_weights = model.state_dict()
    
    # 더미 모델 저장
    dummy_model_path = '/tmp/dummy_dinov2_model.pth'
    torch.save(dummy_weights, dummy_model_path)
    
    print(f"✅ Dummy model saved to: {dummy_model_path}")
    return dummy_model_path

def test_inference_with_dummy():
    """더미 모델로 추론 테스트"""
    print("\n🧪 Testing inference with dummy model...")
    
    # 더미 모델 생성
    model_path = create_dummy_model()
    
    try:
        # 추론 객체 생성
        inference = DINOv2SegmentationInference(
            model_path=model_path,
            num_classes=10,
            device='cpu'  # CPU 강제 사용
        )
        
        print("✅ Inference object created successfully")
        
        # 더미 이미지 생성
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(dummy_image)
        
        print("🖼️  Created dummy image (224x224)")
        
        # 추론 수행
        print("🔄 Performing segmentation...")
        seg_map, confidence = inference.predict(pil_image)
        
        print(f"✅ Segmentation completed!")
        print(f"   - Segmentation map shape: {seg_map.shape}")
        print(f"   - Confidence map shape: {confidence.shape}")
        print(f"   - Unique classes: {np.unique(seg_map)}")
        print(f"   - Confidence range: {confidence.min():.3f} - {confidence.max():.3f}")
        
        # 시각화 테스트
        print("🎨 Testing visualization...")
        visualization = inference.visualize_segmentation(pil_image, seg_map, alpha=0.6)
        
        print(f"✅ Visualization completed!")
        print(f"   - Visualization shape: {visualization.shape}")
        
        # 통계 테스트
        stats = inference.get_class_statistics(seg_map)
        print(f"✅ Statistics calculated for {len(stats)} classes")
        
        # 결과 저장
        output_path = '/tmp/dummy_segmentation_result.png'
        cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        print(f"💾 Result saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """모델 로딩 테스트"""
    print("\n📦 Testing model loading...")
    
    try:
        # DINOv2 허깅페이스 모델 로드 테스트 (인터넷 필요)
        print("🌐 Loading DINOv2 from HuggingFace...")
        from transformers import AutoModel, AutoImageProcessor
        
        model_name = "facebook/dinov2-base"
        dinov2 = AutoModel.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
        
        print("✅ DINOv2 backbone loaded successfully")
        print(f"   - Hidden size: {dinov2.config.hidden_size}")
        print(f"   - Patch size: {dinov2.config.patch_size}")
        
        # 더미 입력으로 테스트
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            outputs = dinov2(dummy_input)
            features = outputs.last_hidden_state
        
        print(f"✅ Forward pass successful")
        print(f"   - Feature shape: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    print("=== DINOv2 Segmentation Inference Test ===")
    print("Testing without GPU (CPU-only)\n")
    
    # 테스트 실행
    results = []
    
    # 1. 모델 로딩 테스트
    results.append(("Model Loading", test_model_loading()))
    
    # 2. 추론 테스트
    results.append(("Dummy Inference", test_inference_with_dummy()))
    
    # 결과 요약
    print("\n" + "="*50)
    print("📊 Test Results Summary:")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The package is working correctly.")
        print("\n📝 Next steps:")
        print("   1. Download ADE20K dataset for training")
        print("   2. Train the model on server with GPU")
        print("   3. Test ROS2 node with trained model")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
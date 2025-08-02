#!/usr/bin/env python3

"""
학습된 DINOv2 모델을 사용한 추론 테스트 스크립트
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

def test_trained_model():
    """학습된 모델로 추론 테스트"""
    print("🚀 Testing trained DINOv2 segmentation model...")
    
    # 모델 경로 설정
    model_path = '/home/leejungwook/dinov2_ws/install/dinov2_segmentation/share/dinov2_segmentation/models/dinov2_trained_model/best_model.pth'
    
    # 모델 파일 존재 확인
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"✅ Found model file: {model_path}")
    
    try:
        # 추론 객체 생성 (ADE20K: 151 classes - including background)
        print("📦 Loading trained model...")
        inference = DINOv2SegmentationInference(
            model_path=model_path,
            num_classes=151,  # ADE20K classes + background
            device='cpu'  # CPU 사용 (GPU 없는 환경)
        )
        
        print("✅ Trained model loaded successfully")
        
        # 테스트 이미지 경로들
        test_images = [
            '/home/leejungwook/dinov2_ws/test_images/segmentation_demo.jpg',
            '/home/leejungwook/dinov2_ws/test_images/outdoor_scene.jpg',
            '/home/leejungwook/dinov2_ws/test_images/street_scene.jpg'
        ]
        
        # 각 테스트 이미지에 대해 추론 수행
        successful_inferences = 0
        for i, image_path in enumerate(test_images):
            if not os.path.exists(image_path):
                print(f"⚠️  Test image not found: {image_path}")
                continue
            
            # 파일이 유효한 이미지인지 확인
            try:
                with Image.open(image_path) as img:
                    img.verify()
                Image.open(image_path)  # 다시 열어서 실제로 사용 가능한지 확인
            except Exception as e:
                print(f"⚠️  Invalid image file {image_path}: {e}")
                continue
                
            print(f"\n🖼️  Processing image {i+1}: {os.path.basename(image_path)}")
            
            try:
                # 추론 및 시각화
                seg_map, visualization, confidence = inference.predict_and_visualize(
                    image_path, alpha=0.6
                )
                successful_inferences += 1
            except Exception as e:
                print(f"⚠️  Failed to process {image_path}: {e}")
                continue
            
            print(f"✅ Segmentation completed!")
            print(f"   - Segmentation map shape: {seg_map.shape}")
            print(f"   - Confidence range: {confidence.min():.3f} - {confidence.max():.3f}")
            print(f"   - Unique classes found: {len(np.unique(seg_map))}")
            
            # 클래스 통계
            stats = inference.get_class_statistics(seg_map)
            dominant_classes = sorted(stats.items(), key=lambda x: x[1]['percentage'], reverse=True)[:5]
            
            print("   - Top 5 dominant classes:")
            for class_id, stat in dominant_classes:
                print(f"     Class {class_id}: {stat['percentage']:.1f}% ({stat['pixel_count']} pixels)")
            
            # 결과 저장
            output_path = f'/tmp/trained_model_result_{i+1}.png'
            cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            print(f"💾 Result saved to: {output_path}")
        
        print(f"\n📈 Successfully processed {successful_inferences} out of {len(test_images)} images")
        return successful_inferences > 0
        
    except Exception as e:
        print(f"❌ Trained model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_info():
    """모델 정보 확인"""
    print("\n📊 Checking model information...")
    
    model_path = '/home/leejungwook/dinov2_ws/install/dinov2_segmentation/share/dinov2_segmentation/models/dinov2_trained_model/best_model.pth'
    
    try:
        # 모델 체크포인트 로드
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print("✅ Model checkpoint loaded successfully")
        print(f"   - Checkpoint keys: {list(checkpoint.keys())}")
        
        # 모델 상태 정보 추출
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if 'epoch' in checkpoint:
                print(f"   - Training epoch: {checkpoint['epoch']}")
            if 'loss' in checkpoint:
                print(f"   - Training loss: {checkpoint['loss']:.4f}")
        else:
            state_dict = checkpoint
        
        # 모델 구조 정보
        layer_info = {}
        for key in state_dict.keys():
            layer_name = key.split('.')[0]
            if layer_name not in layer_info:
                layer_info[layer_name] = 0
            layer_info[layer_name] += 1
        
        print("   - Model layers:")
        for layer, count in layer_info.items():
            print(f"     {layer}: {count} parameters")
        
        # 총 파라미터 수
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"   - Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model info check failed: {e}")
        return False

def create_inference_script():
    """실제 사용을 위한 간단한 추론 스크립트 생성"""
    script_path = '/home/leejungwook/dinov2_ws/run_inference.py'
    
    script_content = '''#!/usr/bin/env python3

"""
DINOv2 Segmentation 간단 추론 스크립트
사용법: python run_inference.py <image_path> [output_path]
"""

import sys
import os
sys.path.append('/home/leejungwook/dinov2_ws/src/dinov2_segmentation')

from dinov2_segmentation.inference.inference import DINOv2SegmentationInference
import cv2

def main():
    if len(sys.argv) < 2:
        print("사용법: python run_inference.py <image_path> [output_path]")
        print("예시: python run_inference.py ~/dinov2_ws/test_images/segmentation_demo.jpg")
        return 1
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'segmentation_result.png'
    
    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
        return 1
    
    # 학습된 모델 경로
    model_path = '/home/leejungwook/dinov2_ws/install/dinov2_segmentation/share/dinov2_segmentation/models/dinov2_trained_model/best_model.pth'
    
    print("🚀 DINOv2 Segmentation 시작...")
    print(f"   입력 이미지: {image_path}")
    print(f"   출력 경로: {output_path}")
    
    try:
        # 추론 객체 생성
        inference = DINOv2SegmentationInference(
            model_path=model_path,
            num_classes=151,
            device='cpu'
        )
        
        # 추론 및 시각화
        seg_map, visualization, confidence = inference.predict_and_visualize(image_path)
        
        # 결과 저장
        cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        
        print("✅ Segmentation 완료!")
        print(f"   결과 저장: {output_path}")
        
        # 간단한 통계
        stats = inference.get_class_statistics(seg_map)
        print(f"   감지된 클래스 수: {len(stats)}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 추론 실패: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
'''
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 실행 권한 부여
    os.chmod(script_path, 0o755)
    
    print(f"📝 Simple inference script created: {script_path}")
    print("   사용법: python run_inference.py <image_path> [output_path]")
    
    return script_path

def main():
    print("=== DINOv2 Trained Model Inference Test ===")
    print("Testing with the trained best_model.pth\n")
    
    # 테스트 실행
    results = []
    
    # 1. 모델 정보 확인
    results.append(("Model Info Check", test_model_info()))
    
    # 2. 학습된 모델로 추론 테스트
    results.append(("Trained Model Inference", test_trained_model()))
    
    # 3. 간단한 추론 스크립트 생성
    script_path = create_inference_script()
    results.append(("Inference Script Creation", script_path is not None))
    
    # 결과 요약
    print("\n" + "="*50)
    print("📊 Test Results Summary:")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The trained model is ready for inference.")
        print("\n📝 Next steps:")
        print("   1. Use the simple inference script:")
        print("      python /home/leejungwook/dinov2_ws/run_inference.py <image_path>")
        print("   2. Integrate with ROS2 node for real-time processing")
        print("   3. Test with RealSense camera input")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
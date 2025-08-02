#!/usr/bin/env python3

"""
미학습 이미지 테스트 스크립트
ADE20K와 다른 스타일의 이미지들로 모델 일반화 성능 테스트
"""

import os
import sys
sys.path.append('/home/leejungwook/dinov2_ws/src/dinov2_segmentation')

from dinov2_segmentation.inference.inference import DINOv2SegmentationInference
import cv2

def test_unseen_images():
    """미학습 이미지들로 segmentation 테스트"""
    
    # 학습된 모델 로드
    model_path = '/home/leejungwook/dinov2_ws/install/dinov2_segmentation/share/dinov2_segmentation/models/dinov2_trained_model/best_model.pth'
    
    print("🚀 미학습 이미지 Segmentation 테스트")
    print("="*50)
    
    try:
        inference = DINOv2SegmentationInference(
            model_path=model_path,
            num_classes=151,
            device='cpu'
        )
        print("✅ 모델 로드 성공")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 기존 test_images (ADE20K와 다른 스타일)
    test_images = [
        '/home/leejungwook/dinov2_ws/test_images/segmentation_demo.jpg',
        '/home/leejungwook/dinov2_ws/test_images/outdoor_scene.jpg'
    ]
    
    results = []
    
    for i, image_path in enumerate(test_images):
        if not os.path.exists(image_path):
            print(f"⚠️  이미지 없음: {image_path}")
            continue
            
        print(f"\n🖼️  테스트 {i+1}: {os.path.basename(image_path)}")
        
        try:
            # Segmentation 수행
            seg_map, visualization, confidence = inference.predict_and_visualize(image_path)
            
            # 결과 저장
            output_path = f'/tmp/unseen_test_{i+1}.png'
            cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            
            # 통계 계산
            stats = inference.get_class_statistics(seg_map)
            
            print(f"   - Segmentation 완료")
            print(f"   - 신뢰도 범위: {confidence.min():.3f} - {confidence.max():.3f}")
            print(f"   - 감지된 클래스 수: {len(stats)}")
            print(f"   - 결과 저장: {output_path}")
            
            # 주요 클래스 출력
            dominant_classes = sorted(stats.items(), key=lambda x: x[1]['percentage'], reverse=True)[:3]
            print("   - 주요 클래스:")
            for class_id, stat in dominant_classes:
                print(f"     Class {class_id}: {stat['percentage']:.1f}%")
                
            results.append({
                'image': os.path.basename(image_path),
                'classes': len(stats),
                'confidence_avg': confidence.mean(),
                'output': output_path
            })
            
        except Exception as e:
            print(f"   ❌ 처리 실패: {e}")
    
    # 요약 출력
    print(f"\n📊 미학습 이미지 테스트 요약")
    print("="*50)
    for result in results:
        print(f"📷 {result['image']}")
        print(f"   클래스 수: {result['classes']}")
        print(f"   평균 신뢰도: {result['confidence_avg']:.3f}")
        print(f"   결과: {result['output']}")
    
    print(f"\n💡 분석:")
    print("- 학습된 ADE20K 이미지와 비교해보세요")
    print("- 미학습 이미지는 신뢰도가 낮거나 부정확할 수 있습니다")
    print("- 이는 모델이 학습 데이터에 특화되어 있음을 의미합니다")

if __name__ == '__main__':
    test_unseen_images()
#!/usr/bin/env python3

"""
DINOv2 학습 환경 검증 스크립트
설치된 패키지들이 정상 작동하는지 확인
"""

import sys

def check_gpu():
    """GPU 및 CUDA 확인"""
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.version.cuda}")
            print(f"✅ GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"     Memory: {memory:.1f} GB")
        else:
            print("❌ CUDA 사용 불가 - GPU 학습이 제한됩니다")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch/CUDA 오류: {e}")
        return False

def check_transformers():
    """Transformers 및 DINOv2 확인"""
    try:
        from transformers import AutoModel, AutoImageProcessor
        print(f"✅ Transformers 설치됨")
        
        # DINOv2 모델 로드 테스트
        print("🔄 DINOv2 모델 로드 테스트...")
        model = AutoModel.from_pretrained("facebook/dinov2-base")
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        print("✅ DINOv2 모델 로드 성공")
        
        return True
    except Exception as e:
        print(f"❌ Transformers/DINOv2 오류: {e}")
        return False

def check_data_processing():
    """데이터 처리 패키지 확인"""
    packages = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("albumentations", "Albumentations"),
        ("sklearn", "Scikit-learn"),
    ]
    
    success = True
    for module, name in packages:
        try:
            exec(f"import {module}")
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} 누락")
            success = False
    
    return success

def check_training_tools():
    """학습 도구 확인"""
    packages = [
        ("torch.utils.tensorboard", "TensorBoard"),
        ("tqdm", "TQDM"),
        ("yaml", "PyYAML"),
        ("matplotlib", "Matplotlib"),
    ]
    
    success = True
    for module, name in packages:
        try:
            exec(f"import {module}")
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} 누락")
            success = False
    
    return success

def run_model_test():
    """간단한 모델 테스트"""
    try:
        import torch
        import torch.nn as nn
        from transformers import AutoModel
        
        print("🔄 모델 생성 테스트...")
        
        # 작은 모델 생성
        dinov2 = AutoModel.from_pretrained("facebook/dinov2-base")
        classifier = nn.Conv2d(768, 10, kernel_size=1)
        
        # 더미 입력으로 forward 테스트
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            features = dinov2(dummy_input).last_hidden_state[:, 1:]  # Remove CLS
            batch_size, seq_len, hidden_size = features.shape
            height = width = int(seq_len ** 0.5)
            features = features.reshape(batch_size, height, width, hidden_size)
            features = features.permute(0, 3, 1, 2)
            output = classifier(features)
        
        print(f"✅ 모델 테스트 성공 (출력 크기: {output.shape})")
        return True
        
    except Exception as e:
        print(f"❌ 모델 테스트 실패: {e}")
        return False

def main():
    print("=== DINOv2 학습 환경 검증 ===\n")
    
    tests = [
        ("GPU 및 CUDA", check_gpu),
        ("Transformers", check_transformers),
        ("데이터 처리", check_data_processing),
        ("학습 도구", check_training_tools),
        ("모델 테스트", run_model_test),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n📋 {name} 확인:")
        result = test_func()
        results.append(result)
        print()
    
    # 결과 요약
    passed = sum(results)
    total = len(results)
    
    print("=" * 40)
    print(f"결과: {passed}/{total} 테스트 통과")
    
    if passed == total:
        print("🎉 모든 테스트 통과! 학습 환경이 준비되었습니다.")
        return 0
    else:
        print("⚠️ 일부 테스트 실패. 누락된 패키지를 설치하세요.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3

"""
Import 테스트 스크립트
GPU 없이도 모든 모듈이 정상적으로 import되는지 확인
"""

import sys
import traceback

def test_import(module_name, description):
    """모듈 import 테스트"""
    try:
        exec(f"import {module_name}")
        print(f"✅ {description}: OK")
        return True
    except ImportError as e:
        print(f"❌ {description}: FAILED - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {description}: ERROR - {e}")
        return False

def main():
    print("=== DINOv2 Segmentation Import Test ===\n")
    
    # 기본 의존성 테스트
    basic_deps = [
        ("torch", "PyTorch"),
        ("torchvision", "Torchvision"),
        ("transformers", "Transformers"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("requests", "Requests"),
        ("yaml", "PyYAML"),
        ("tqdm", "TQDM"),
        ("sklearn", "Scikit-learn"),
        ("albumentations", "Albumentations"),
    ]
    
    print("1. Basic Dependencies:")
    success_count = 0
    for module, desc in basic_deps:
        if test_import(module, desc):
            success_count += 1
    
    print(f"\nBasic deps: {success_count}/{len(basic_deps)} passed\n")
    
    # ROS2 의존성 테스트
    print("2. ROS2 Dependencies:")
    ros_deps = [
        ("rclpy", "ROS2 Python Client"),
        ("sensor_msgs.msg", "Sensor Messages"),
        ("std_msgs.msg", "Standard Messages"),
        ("cv_bridge", "CV Bridge"),
        ("ament_index_python.packages", "Ament Index"),
    ]
    
    ros_success = 0
    for module, desc in ros_deps:
        if test_import(module, desc):
            ros_success += 1
    
    print(f"\nROS2 deps: {ros_success}/{len(ros_deps)} passed\n")
    
    # 우리 패키지 테스트
    print("3. Package Modules:")
    try:
        # 환경 설정
        import os
        os.chdir('/home/leejungwook/dinov2_ws')
        
        # 우리 모듈들 테스트
        package_modules = [
            ("dinov2_segmentation.training.model", "Training Model"),
            ("dinov2_segmentation.training.dataset", "Dataset Module"),
            ("dinov2_segmentation.training.utils", "Training Utils"),
            ("dinov2_segmentation.inference.inference", "Inference Module"),
        ]
        
        pkg_success = 0
        for module, desc in package_modules:
            if test_import(module, desc):
                pkg_success += 1
        
        print(f"\nPackage modules: {pkg_success}/{len(package_modules)} passed\n")
        
    except Exception as e:
        print(f"Package test failed: {e}")
        pkg_success = 0
    
    # 모델 생성 테스트 (GPU 없이)
    print("4. Model Creation Test (CPU):")
    try:
        from dinov2_segmentation.training.model import DINOv2ForSegmentation
        import torch
        
        # CPU에서 작은 모델 생성 테스트
        print("  Creating model...")
        model = DINOv2ForSegmentation(num_classes=10, freeze_backbone=True)
        print("  ✅ Model creation: OK")
        
        # 더미 입력으로 forward 테스트
        print("  Testing forward pass...")
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"  ✅ Forward pass: OK (output shape: {output.shape})")
        
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        traceback.print_exc()
    
    print("\n=== Test Summary ===")
    total_basic = len(basic_deps)
    total_ros = len(ros_deps)
    total_pkg = len(package_modules)
    
    if success_count == total_basic and ros_success == total_ros and pkg_success == total_pkg:
        print("🎉 All tests passed! Package is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check missing dependencies.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
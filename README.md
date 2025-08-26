# DINOv2 ROS2 Bag Segmentation

GPU 가속 DINOv2를 사용한 ROS2 bag 파일 기반 semantic segmentation 시스템

## 🚀 빠른 시작 (새로운 PC에서)

### 1단계: 자동 환경 설정
```bash
# 저장소 클론 후
cd dinov2_ros2_segmentation/src/dinov2_ros2_segmentation
./scripts/setup_environment.sh
```

### 2단계: 워크스페이스 환경 설정 (매번 새 터미널에서)
```bash
source setup_workspace.sh
```

### 3단계: Bag 파일로 테스트
```bash
# Bag 파일 추출
./scripts/extract_bag.sh 4

# GPU 가속 segmentation 실행 (터미널 1)
ros2 launch dinov2_ros_segmentation segmentation_bag_inference.launch.py bag_number:=4

# Bag 재생 (터미널 2)
ros2 bag play [추출된_경로] --rate 1.0

# 결과 확인 (터미널 3)
rqt_image_view
```

## 📋 시스템 요구사항

- **OS**: Ubuntu 24.04
- **ROS2**: Jazzy
- **Python**: 3.12
- **GPU**: NVIDIA (CUDA 12.1 이상) - 선택사항
- **Conda**: Miniconda 또는 Anaconda

## 🎯 주요 기능

- ✅ **GPU 가속 처리**: RTX 5080에서 13+ FPS
- ✅ **실시간 segmentation**: ADE20k 모델 기반
- ✅ **Bag 파일 지원**: 다양한 8.20.x bag 파일 처리
- ✅ **자동화 스크립트**: 환경 설정 및 테스트 자동화
- ✅ **포터블**: 절대경로 없이 어떤 PC에서도 실행 가능

## 📊 성능

- **처리 속도**: 13.6 FPS (GPU 모드)
- **해상도**: 1280x720 지원
- **GPU 메모리**: ~2GB 사용
- **감지 객체**: 150개 ADE20k 클래스

## 🛠️ 고급 사용법

### 다양한 설정 옵션
```bash
# Linear head (더 빠른 처리)
ros2 launch dinov2_ros_segmentation segmentation_bag_inference.launch.py head_type:=linear

# 해상도 조정
ros2 launch dinov2_ros_segmentation segmentation_bag_inference.launch.py resolution:=384

# 연구실 객체만 필터링
ros2 launch dinov2_ros_segmentation segmentation_bag_inference.launch.py filter_mode:=lab_only
```

### 다른 bag 파일 테스트
```bash
# 사용 가능한 파일들: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
./scripts/extract_bag.sh 13  # 큰 파일 (23GB)
./scripts/quick_test.sh 1    # 빠른 테스트
```

## 📁 프로젝트 구조

```
dinov2_ros2_segmentation/
├── dinov2_ros_segmentation/           # ROS2 패키지
│   ├── dinov2_segmentation_node.py   # 원본 RealSense 노드
│   └── dinov2_bag_segmentation_node.py  # GPU 최적화 Bag 노드
├── launch/
│   ├── segmentation_inference.launch.py     # RealSense용
│   └── segmentation_bag_inference.launch.py # Bag용
├── scripts/
│   ├── setup_environment.sh         # 환경 설정 자동화
│   ├── extract_bag.sh              # Bag 파일 추출
│   └── quick_test.sh               # 빠른 테스트
├── data_collect/                   # Bag 파일들
│   ├── 8.20.1.tar.gz              # 5.9GB
│   ├── 8.20.4.tar.gz              # 3.6GB
│   └── 8.20.13.tar.gz             # 23GB
├── setup_workspace.sh             # 워크스페이스 환경 설정
└── README.md                      # 이 파일
```

## 🐛 문제 해결

### GPU 인식 안됨
```bash
# CUDA 확인
nvidia-smi
# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 패키지 빌드 실패
```bash
# 종속성 재설치
./scripts/setup_environment.sh
# 수동 빌드
colcon build --packages-select dinov2_ros_segmentation
```

### Bag 파일 추출 실패
```bash
# 디스크 공간 확인
df -h /tmp
# 수동 추출
tar -tf data_collect/8.20.4.tar.gz | head -5
```

## 📞 지원

- 📄 **자세한 로그**: `Project Progress.md` 참고
- 🔧 **환경 설정**: `setup_environment.sh` 실행
- 📊 **성능 모니터링**: 터미널 로그에서 FPS 확인

---

💡 **Tip**: 처음 실행시 PyTorch 모델 다운로드로 시간이 걸릴 수 있습니다.
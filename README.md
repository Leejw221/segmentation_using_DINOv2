# DINOv2 Segmentation for ROS2

<<<<<<< HEAD
DINOv2 기반 semantic segmentation을 위한 ROS2 패키지입니다. ADE20K 데이터셋을 사용하여 150개 클래스에 대한 segmentation을 수행합니다.

## 특징

- **DINOv2 backbone**: Facebook의 최신 self-supervised vision transformer 사용
- **ADE20K dataset**: 150개 클래스의 scene parsing 데이터셋 지원
- **ROS2 integration**: 실시간 이미지 처리를 위한 ROS2 노드
- **유연한 구조**: 학습과 추론이 분리된 모듈식 설계
- **RISE-2 inspired**: 로봇 연구에 최적화된 구조

## 시스템 요구사항

- ROS2 Humble
- Python 3.8+
- PyTorch 1.12+
- CUDA (권장, CPU도 지원)

## 설치

```bash
# 워크스페이스로 이동
cd ~/dinov2_ws

# 의존성 설치 (pip)
pip install torch torchvision transformers numpy opencv-python pillow requests pyyaml tqdm scikit-learn albumentations tensorboard

# 패키지 빌드
colcon build --packages-select dinov2_segmentation

# 환경 설정
source install/setup.bash
```

## 사용법

### 1. 데이터 준비 (학습용)

ADE20K 데이터셋을 다운로드하고 다음과 같은 구조로 정리:

```
ade20k/
├── images/
│   ├── training/
│   └── validation/
└── annotations/
    ├── training/
    └── validation/
```

### 2. 모델 학습

```bash
# 기본 설정으로 학습
ros2 run dinov2_segmentation train_segmentation --data_root /path/to/ade20k

# 커스텀 설정으로 학습  
ros2 run dinov2_segmentation train_segmentation \
    --data_root /path/to/ade20k \
    --config configs/config.yaml \
    --output_dir models/my_experiment
```

### 3. ROS2 노드 실행

```bash
# 기본 실행 (자동 데모 포함)
ros2 launch dinov2_segmentation demo.launch.py

# 커스텀 모델로 실행
ros2 launch dinov2_segmentation dinov2_segmentation.launch.py \
    model_path:=/path/to/your/model.pth

# 파라미터 조정
ros2 launch dinov2_segmentation dinov2_segmentation.launch.py \
    model_path:=/path/to/model.pth \
    num_classes:=150 \
    visualization_alpha:=0.7 \
    auto_demo:=false
```

### 4. 이미지 전송

#### URL로 이미지 전송:
```bash
ros2 topic pub /dinov2/image_url std_msgs/msg/String \
  "data: 'https://example.com/image.jpg'"
```

#### 로컬 파일로 이미지 전송:
```bash
ros2 topic pub /dinov2/image_file std_msgs/msg/String \
  "data: '/path/to/image.jpg'"
```

#### ROS Image 메시지로 전송:
```bash
# 다른 카메라 노드에서 이미지를 받는 경우
ros2 topic pub /dinov2/input_image sensor_msgs/msg/Image [...]
```

### 5. 결과 확인

```bash
# 시각화 도구 실행
rqt_image_view

# 토픽 선택:
# - /dinov2/original_image: 원본 이미지
# - /dinov2/segmentation_result: segmentation 결과 (오버레이)
# - /dinov2/confidence_map: 신뢰도 맵
```

## 토픽 구조
=======
DINOv2 기반 semantic segmentation을 위한 ROS2 패키지입니다. ADE20K 데이터셋을 사용하여 151개 클래스에 대한 segmentation을 수행합니다.

## 🎯 프로젝트 개요

- **DINOv2 backbone**: Facebook의 self-supervised vision transformer 사용
- **ADE20K dataset**: 150개 클래스 + background (총 151개)
- **ROS2 integration**: 실시간 이미지 처리를 위한 ROS2 노드
- **무작위 데모**: 20,210개 훈련 이미지 중 무작위 선택하여 성능 검증
- **Mobile Manipulator 연구**: RealSense 카메라 연동 준비

## 📁 프로젝트 구조

```
dinov2_ws/
├── README.md                    # 이 파일
├── DINOv2_Project_Status_0802.md  # 프로젝트 현황 문서
└── src/dinov2_segmentation/
    ├── dinov2_segmentation/      # 패키지 소스코드
    │   ├── training/             # 학습 관련 코드
    │   │   ├── model.py          # DINOv2 segmentation 모델
    │   │   ├── dataset.py        # ADE20K 데이터셋 로더
    │   │   ├── train_segmentation.py  # 학습 스크립트
    │   │   └── utils.py          # 유틸리티 함수
    │   └── inference/            # 추론 관련 코드
    │       ├── inference.py      # 추론 클래스
    │       └── ros2_segmentation_node.py  # ROS2 노드
    ├── datasets/                 # ADE20K 데이터셋
    │   └── ADEChallengeData2016/
    ├── models/                   # 학습된 모델 저장소
    ├── test/                     # 테스트 파일들
    ├── configs/                  # 설정 파일
    ├── launch/                   # ROS2 launch 파일들
    └── setup_dinov2_training.sh  # 훈련 환경 설정 스크립트
```

## 🔧 시스템 요구사항

- **OS**: Ubuntu 22.04
- **ROS2**: Humble
- **Python**: 3.8+
- **PyTorch**: 1.12+
- **CUDA**: 선택사항 (CPU도 지원)

## 📦 설치 및 설정

### 1. 의존성 설치

```bash
cd ~/dinov2_ws/src/dinov2_segmentation
chmod +x setup_dinov2_training.sh
./setup_dinov2_training.sh
```

또는 수동 설치:
```bash
pip install torch torchvision transformers numpy opencv-python pillow requests pyyaml tqdm scikit-learn albumentations tensorboard
```

### 2. 패키지 빌드

```bash
cd ~/dinov2_ws
colcon build --packages-select dinov2_segmentation
source install/setup.bash
```

## 🚀 사용법

### 1. 빠른 시작 (데모 실행)

```bash
cd ~/dinov2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch dinov2_segmentation demo.launch.py
```

**특징:**
- 자동으로 ADE20K 훈련 이미지 중 무작위 선택
- 3초마다 새로운 이미지 처리
- 151 클래스 segmentation 수행

### 2. 시각화 확인

별도 터미널에서:
```bash
source /opt/ros/humble/setup.bash
rqt
```

**rqt에서 토픽 선택:**
- `Plugins` → `Visualization` → `Image View`
- 토픽 선택:
  - `/dinov2/original_image`: 원본 이미지
  - `/dinov2/segmentation_result`: Segmentation 결과 (오버레이)
  - `/dinov2/confidence_map`: 예측 신뢰도 맵

### 3. 커스텀 이미지 처리

#### 로컬 파일:
```bash
ros2 topic pub /dinov2/image_file std_msgs/msg/String \"data: '/path/to/your/image.jpg'\"
```

#### 웹 URL:
```bash
ros2 topic pub /dinov2/image_url std_msgs/msg/String \"data: 'https://example.com/image.jpg'\"
```

## 🔍 토픽 구조
>>>>>>> origin/main

### Subscribe 토픽:
- `/dinov2/image_url` (std_msgs/String): 웹 이미지 URL
- `/dinov2/image_file` (std_msgs/String): 로컬 파일 경로  
- `/dinov2/input_image` (sensor_msgs/Image): ROS 이미지 메시지

### Publish 토픽:
- `/dinov2/original_image` (sensor_msgs/Image): 원본 이미지
- `/dinov2/segmentation_result` (sensor_msgs/Image): Segmentation 결과
- `/dinov2/confidence_map` (sensor_msgs/Image): 예측 신뢰도 맵

<<<<<<< HEAD
## 구조

```
dinov2_segmentation/
├── dinov2_segmentation/
│   ├── training/              # 학습 관련 코드
│   │   ├── model.py          # DINOv2 segmentation 모델
│   │   ├── dataset.py        # ADE20K 데이터셋 로더
│   │   ├── train_segmentation.py  # 학습 스크립트
│   │   └── utils.py          # 학습 유틸리티
│   └── inference/            # 추론 관련 코드
│       ├── inference.py      # 추론 클래스
│       └── ros2_segmentation_node.py  # ROS2 노드
├── configs/
│   └── config.yaml           # 설정 파일
├── launch/
│   ├── dinov2_segmentation.launch.py  # 메인 런치 파일
│   └── demo.launch.py        # 데모 런치 파일
└── models/                   # 학습된 모델 저장소
```

## 파라미터

### ROS2 노드 파라미터:
- `model_path`: 학습된 모델 파일 경로
- `num_classes`: 클래스 개수 (기본: 150)
- `visualization_alpha`: 오버레이 투명도 (기본: 0.6)
- `auto_demo`: 자동 데모 활성화 (기본: true)
- `demo_interval`: 데모 간격 초단위 (기본: 5.0)

### 학습 파라미터 (config.yaml):
- `batch_size`: 배치 크기 (기본: 8)
- `learning_rate`: 학습률 (기본: 1e-4)
- `num_epochs`: 에포크 수 (기본: 50)
- `image_size`: 입력 이미지 크기 (기본: 224)

## 예제

### 빠른 테스트:
```bash
# 1. 빌드
colcon build --packages-select dinov2_segmentation

# 2. 데모 실행 (자동으로 샘플 이미지 처리)
ros2 launch dinov2_segmentation demo.launch.py

# 3. 결과 확인
rqt_image_view
```

### 카스텀 이미지 테스트:
```bash
# 터미널 1: 노드 실행
ros2 launch dinov2_segmentation demo.launch.py

# 터미널 2: 이미지 전송
ros2 topic pub /dinov2/image_file std_msgs/msg/String \
  "data: '/home/user/test_image.jpg'"
```

## 트러블슈팅

### 모델을 찾을 수 없는 경우:
```bash
# 모델 경로 확인
ls ~/dinov2_ws/install/dinov2_segmentation/share/dinov2_segmentation/models/

# 모델 경로 직접 지정
ros2 launch dinov2_segmentation dinov2_segmentation.launch.py \
    model_path:=/absolute/path/to/your/model.pth
```

### GPU 메모리 부족:
- `config.yaml`에서 `batch_size` 줄이기
- 더 작은 이미지 크기 사용 (`image_size: 224 → 196`)

### 의존성 오류:
```bash
# PyTorch 재설치
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Transformers 업데이트
pip install --upgrade transformers
```

## 개발 계획

- [ ] Realsense 카메라 연동
- [ ] 실시간 성능 최적화
- [ ] 커스텀 데이터셋 지원
- [ ] 모바일 로봇 내비게이션 연동

## 라이센스

MIT License

## 참고자료

- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [ADE20K Dataset](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
- [RISE-2 Project](https://github.com/rise-lab/rise2)
=======
## ⚙️ 파라미터 설정

### ROS2 노드 파라미터:
- `model_path`: 학습된 모델 파일 경로
- `num_classes`: 클래스 개수 (기본: 151)
- `visualization_alpha`: 오버레이 투명도 (기본: 0.6)
- `auto_demo`: 자동 데모 활성화 (기본: true)
- `demo_interval`: 데모 간격 초단위 (기본: 3.0)

### 커스텀 실행:
```bash
ros2 launch dinov2_segmentation demo.launch.py \\
    model_path:=/path/to/your/model.pth \\
    num_classes:=151 \\
    visualization_alpha:=0.7 \\
    demo_interval:=5.0
```

## 🧪 테스트

### 테스트 파일 실행:
```bash
cd ~/dinov2_ws/src/dinov2_segmentation/test
python test_imports.py          # 의존성 확인
python test_inference.py        # 추론 테스트
python test_trained_model.py    # 모델 로딩 테스트
python validate_training_env.py # 훈련 환경 검증
```

## 🔧 트러블슈팅

### 모델 파일 문제:
```bash
# 모델 파일 확인
ls ~/dinov2_ws/install/dinov2_segmentation/share/dinov2_segmentation/models/

# 모델 직접 지정
ros2 launch dinov2_segmentation demo.launch.py \\
    model_path:=/absolute/path/to/model.pth
```

### PyTorch 관련 오류:
```bash
# PyTorch 재설치 (CUDA 지원)
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU만 사용하는 경우
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 토픽 확인:
```bash
# 토픽 목록
ros2 topic list | grep dinov2

# 토픽 정보
ros2 topic info /dinov2/segmentation_result

# 토픽 데이터 확인
ros2 topic echo /dinov2/segmentation_result --once
```

## 📊 성능 정보

- **모델 크기**: ~80MB (best_model.pth)
- **추론 속도**: ~100ms/이미지 (CPU), ~50ms/이미지 (GPU)
- **메모리 사용량**: ~1.2GB (모델 로딩시)
- **지원 이미지 크기**: 임의 크기 (내부적으로 224x224로 리사이즈)

## 🎯 개발 계획

- [x] ADE20K 데이터셋 학습 완료
- [x] ROS2 노드 구현 완료
- [x] 무작위 이미지 데모 구현
- [ ] RealSense D435 카메라 연동
- [ ] 실시간 성능 최적화
- [ ] 모바일 로봇 내비게이션 연동

## 📋 현재 상태 (2025.08.02)

✅ **완료된 기능:**
- DINOv2 모델 ADE20K 데이터셋으로 학습 완료
- ROS2 환경에서 실시간 segmentation 동작
- 20,210개 훈련 이미지 중 무작위 선택 데모
- rqt를 통한 시각화 시스템 구축

🔄 **진행 중:**
- 월요일 교수님 미팅 준비 (성능 검증 및 피드백)
- RealSense 카메라 연동 계획 수립

## 📚 참고자료

- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [ADE20K Dataset](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)

## 📄 라이센스

MIT License
>>>>>>> origin/main

# DINOv2 Segmentation for ROS2

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

### Subscribe 토픽:
- `/dinov2/image_url` (std_msgs/String): 웹 이미지 URL
- `/dinov2/image_file` (std_msgs/String): 로컬 파일 경로  
- `/dinov2/input_image` (sensor_msgs/Image): ROS 이미지 메시지

### Publish 토픽:
- `/dinov2/original_image` (sensor_msgs/Image): 원본 이미지
- `/dinov2/segmentation_result` (sensor_msgs/Image): Segmentation 결과
- `/dinov2/confidence_map` (sensor_msgs/Image): 예측 신뢰도 맵

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
#!/bin/bash

# DINOv2 ROS2 Bag Segmentation 환경 설정 자동화 스크립트
# 다른 PC에서도 바로 사용할 수 있도록 모든 종속성과 가상환경을 설정

set -e  # 오류 발생시 스크립트 중단

echo "🚀 DINOv2 ROS2 Bag Segmentation 환경 설정 시작"
echo "===================================================="

# 현재 스크립트 위치 기반으로 프로젝트 루트 찾기
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE_DIR="$(dirname "$(dirname "$(dirname "$PROJECT_DIR")")")"

echo "📁 프로젝트 디렉토리: $PROJECT_DIR"
echo "🏠 워크스페이스: $WORKSPACE_DIR"

# 시스템 요구사항 확인
echo ""
echo "1️⃣ 시스템 요구사항 확인 중..."

# CUDA 확인
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU 감지됨"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
else
    echo "⚠️ NVIDIA GPU가 감지되지 않음 (CPU 모드로 실행됩니다)"
fi

# Conda 확인
if command -v conda &> /dev/null; then
    echo "✅ Conda 이미 설치됨"
else
    echo "❌ Conda가 설치되지 않음"
    echo "다음 명령어로 Miniconda를 설치해주세요:"
    echo "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# ROS2 Jazzy 확인
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    echo "✅ ROS2 Jazzy 감지됨"
else
    echo "❌ ROS2 Jazzy가 설치되지 않음"
    echo "다음 명령어로 ROS2 Jazzy를 설치해주세요:"
    echo "https://docs.ros.org/en/jazzy/Installation.html"
    exit 1
fi

# 2. Conda 가상환경 생성
echo ""
echo "2️⃣ Conda 가상환경 설정 중..."

ENV_NAME="dinov2_segmentation"

# 기존 환경 확인 및 제거
if conda env list | grep -q "^$ENV_NAME "; then
    echo "⚠️ 기존 '$ENV_NAME' 환경 감지됨. 재생성합니다."
    read -p "기존 환경을 삭제하고 새로 만들시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME -y
    else
        echo "기존 환경을 유지합니다."
    fi
fi

if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "🔧 새로운 conda 환경 '$ENV_NAME' 생성 중..."
    conda create -n $ENV_NAME python=3.12 -y
    echo "✅ Conda 환경 생성 완료"
fi

# 3. 가상환경 활성화 및 패키지 설치
echo ""
echo "3️⃣ Python 패키지 설치 중..."

# Conda 환경 활성화
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "📦 PyTorch 및 CUDA 패키지 설치..."
# CUDA 12.1 호환 PyTorch 설치
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo "📦 ROS2 Python 패키지 설치..."
pip install \
    opencv-python \
    pillow \
    numpy \
    rclpy \
    sensor-msgs \
    std-msgs \
    geometry-msgs \
    nav-msgs \
    tf2-msgs \
    cv-bridge \
    message-filters

echo "📦 추가 도구 설치..."
pip install \
    requests \
    setuptools \
    colcon-common-extensions

# 4. ROS2 패키지 빌드
echo ""
echo "4️⃣ ROS2 패키지 빌드 중..."

cd "$WORKSPACE_DIR"

# ROS2 환경 소싱
source /opt/ros/jazzy/setup.bash

# 패키지 빌드
echo "🔨 colcon build 실행 중..."
colcon build --packages-select dinov2_ros_segmentation

if [ $? -eq 0 ]; then
    echo "✅ ROS2 패키지 빌드 성공"
else
    echo "❌ ROS2 패키지 빌드 실패"
    exit 1
fi

# 5. setup_workspace.sh 확인
echo ""
echo "5️⃣ setup_workspace.sh 확인..."

SETUP_SCRIPT="$PROJECT_DIR/setup_workspace.sh"
if [ ! -f "$SETUP_SCRIPT" ]; then
    echo "⚠️ setup_workspace.sh 없음. 기본 파일 생성..."
    cat > "$SETUP_SCRIPT" << 'EOF'
#!/bin/bash
# 간단한 워크스페이스 환경 설정 (매번 새 터미널에서 실행)

eval "$(conda shell.bash hook)"
conda activate dinov2_segmentation
source /opt/ros/jazzy/setup.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$WORKSPACE_DIR" && source install/setup.bash

echo "✅ 환경 설정 완료! 현재 위치: $(pwd)"
EOF
    chmod +x "$SETUP_SCRIPT"
fi

echo "✅ setup_workspace.sh 준비됨"

# 6. 테스트 실행
echo ""
echo "6️⃣ 설치 테스트 중..."

# PyTorch 및 CUDA 테스트
python -c "
import torch
print(f'✅ PyTorch 버전: {torch.__version__}')
if torch.cuda.is_available():
    print(f'✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}')
    print(f'✅ CUDA 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    print('⚠️ CUDA 사용 불가 (CPU 모드)')
"

# ROS2 패키지 테스트
source install/setup.bash
if ros2 pkg list | grep -q dinov2_ros_segmentation; then
    echo "✅ ROS2 패키지 정상 등록됨"
else
    echo "❌ ROS2 패키지 등록 실패"
    exit 1
fi

# 7. 설치 완료 및 사용법 안내
echo ""
echo "🎉 설치 완료!"
echo "===================================================="
echo ""
echo "📋 사용법:"
echo ""
echo "1️⃣ 매번 새 터미널에서 환경 설정:"
echo "   source $PROJECT_DIR/setup_workspace.sh"
echo ""
echo "2️⃣ Bag 파일 추출 및 테스트:"
echo "   cd $PROJECT_DIR"
echo "   ./scripts/extract_bag.sh 4"
echo ""
echo "3️⃣ GPU 최적화 segmentation 실행:"
echo "   # 터미널 1:"
echo "   ros2 launch dinov2_ros_segmentation segmentation_bag_inference.launch.py bag_number:=4"
echo ""
echo "   # 터미널 2:"
echo "   ros2 bag play [추출된_경로] --rate 1.0"
echo ""
echo "   # 터미널 3:"
echo "   rqt_image_view"
echo ""
echo "📁 주요 파일 위치:"
echo "   - 설정 스크립트: $SETUP_SCRIPT"
echo "   - 프로젝트 루트: $PROJECT_DIR"
echo "   - 워크스페이스: $WORKSPACE_DIR"
echo ""
echo "🔗 자세한 사용법은 Project Progress.md를 참고하세요."
echo "✅ 모든 설정이 완료되었습니다!"
#!/bin/bash

# DINOv2 학습 환경 설치 스크립트
# 사용법: bash setup_dinov2_training.sh

echo "🚀 DINOv2 학습 환경 설정 시작..."

# 0. Conda 초기화 확인
if ! command -v conda &> /dev/null; then
    echo "❌ Conda가 설치되지 않았습니다. Miniconda/Anaconda를 먼저 설치하세요."
    exit 1
fi

# Conda 초기화
eval "$(conda shell.bash hook)"

# 1. Conda 환경 생성
echo "📦 Conda 환경 생성..."
conda create -n dinov2_training python=3.10 -y

echo "✅ 환경이 생성되었습니다. 이제 수동으로 활성화하고 패키지를 설치하세요:"
echo ""
echo "🎯 다음 명령어들을 순서대로 실행하세요:"
echo "1. conda activate dinov2_training"
echo "2. bash install_packages.sh"
echo ""

# 패키지 설치 스크립트 생성
cat > install_packages.sh << 'EOF'
#!/bin/bash

echo "📦 패키지 설치 시작..."

# 현재 conda 환경 확인
if [[ "$CONDA_DEFAULT_ENV" != "dinov2_training" ]]; then
    echo "❌ dinov2_training 환경이 활성화되지 않았습니다."
    echo "   conda activate dinov2_training 를 먼저 실행하세요."
    exit 1
fi

# 2. PyTorch 설치 (CUDA 12.4 호환)
echo "🔥 PyTorch 설치..."
# CUDA 12.4는 CUDA 12.1 버전과 호환됩니다
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Transformers 설치
echo "🤗 Transformers 설치..."
pip install transformers==4.35.0

# 4. 기본 패키지 설치
echo "📊 기본 패키지 설치..."
conda install -c conda-forge \
    pillow \
    opencv \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    tensorboard \
    tqdm \
    pyyaml \
    -y

# 5. Albumentations 설치 (pip 권장)
echo "🖼️ Albumentations 설치..."
pip install albumentations==1.3.1

# 6. 추가 유틸리티
echo "🛠️ 추가 도구 설치..."
pip install \
    wandb \
    timm \
    segmentation-models-pytorch \
    torchmetrics

# 7. Jupyter 환경 (선택사항)
echo "📓 Jupyter 환경 설치..."
conda install -c conda-forge jupyter jupyterlab -y

# 8. 설치 확인
echo "✅ 설치 확인..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')

import transformers
print(f'Transformers: {transformers.__version__}')

import albumentations
print(f'Albumentations: {albumentations.__version__}')

print('🎉 모든 패키지가 성공적으로 설치되었습니다!')
"
EOF

chmod +x install_packages.sh
echo "🔧 환경 설정 준비 완료!"

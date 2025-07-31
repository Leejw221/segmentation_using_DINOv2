# 🚀 서버 PC 이전 가이드

## 📋 현재 상황
- **로컬 PC**: 개발 및 코드 작성 완료
- **서버 PC**: CUDA 12.4 설치됨, GPU 학습 환경
- **목표**: DINOv2 학습을 서버에서 실행

## 🔄 이전 과정

### 1. **GitHub에 코드 커밋 및 푸시**

```bash
# 현재 디렉토리로 이동
cd /home/leejungwook/dinov2_ws/src/dinov2_segmentation

# 모든 변경 사항 추가
git add .

# 커밋 메시지와 함께 커밋
git commit -m "Add DINOv2 training infrastructure for CUDA 12.4

- Add complete training pipeline with model, dataset, utils
- Add inference module with ROS2 integration
- Add conda environment setup for CUDA 12.4
- Add validation and testing scripts
- Ready for server deployment"

# GitHub에 푸시
git push origin main
```

### 2. **서버 PC에서 클론**

```bash
# 서버 PC에서 실행
git clone https://github.com/Leejw221/segmentation_using_DINOv2.git
cd segmentation_using_DINOv2
```

### 3. **서버에서 환경 설정**

```bash
# 스크립트 실행 권한 부여
chmod +x setup_dinov2_training.sh

# 1단계: conda 환경 생성
bash setup_dinov2_training.sh

# 2단계: 환경 활성화 및 패키지 설치
conda activate dinov2_training
bash install_packages.sh
```

### 4. **GPU 및 환경 확인**

```bash
# 환경 검증
python validate_training_env.py

# GPU 확인
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## 🎯 서버 PC에서의 학습 실행

### **데이터셋 준비**
```bash
# ADE20K 데이터셋 다운로드 및 준비
mkdir -p data/ADE20K
# 데이터셋을 data/ADE20K 디렉토리에 배치
```

### **학습 시작**
```bash
conda activate dinov2_training
cd dinov2_segmentation

# 학습 실행
python -m training.train_segmentation \
    --config configs/train_config.yaml \
    --data_root data/ADE20K \
    --output_dir outputs \
    --num_epochs 50 \
    --batch_size 8 \
    --learning_rate 1e-4
```

## ⚡ 서버 최적화 팁

### **GPU 메모리 최적화**
```python
# 학습 시 배치 크기 조정
# GPU 메모리에 따라 batch_size 조정:
# - 8GB VRAM: batch_size=4
# - 16GB VRAM: batch_size=8
# - 24GB+ VRAM: batch_size=16
```

### **멀티 GPU 사용 (가능한 경우)**
```bash
# DataParallel 사용
python -m training.train_segmentation --multi_gpu

# 또는 DistributedDataParallel
torchrun --nproc_per_node=2 -m training.train_segmentation
```

## 🔧 CUDA 12.4 호환성 확인

현재 설정:
- **PyTorch**: pytorch-cuda=12.1 (CUDA 12.4와 호환)
- **CUDA Toolkit**: 12.1 (하위 호환성으로 CUDA 12.4와 작동)
- **cuDNN**: 자동으로 적절한 버전 설치

## 📊 모니터링 도구

### **TensorBoard**
```bash
# 학습 중 다른 터미널에서
tensorboard --logdir outputs/logs --port 6006
```

### **Weights & Biases (선택사항)**
```bash
# wandb 로그인 후 사용
wandb login
```

## 🚨 주의사항

1. **방화벽**: 서버의 포트(6006 등)가 열려있는지 확인
2. **권한**: 데이터 디렉토리에 쓰기 권한 확인
3. **디스크 공간**: 최소 100GB 여유 공간 필요
4. **메모리**: 시스템 RAM 16GB+ 권장

## 📞 문제 해결

### **CUDA 오류 시**
```bash
# CUDA 버전 확인
nvcc --version
nvidia-smi

# PyTorch CUDA 호환성 재확인
python -c "import torch; print(torch.version.cuda)"
```

### **메모리 부족 시**
- 배치 크기 감소
- Gradient checkpointing 활성화
- Mixed precision 사용

이제 GitHub에 푸시하고 서버로 이전할 준비가 완료되었습니다! 🚀

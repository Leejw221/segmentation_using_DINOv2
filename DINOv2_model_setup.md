# DINOv2 Segmentation Model Setup

## 📋 프로젝트 개요
- **목표**: ADE20K 데이터셋으로 DINOv2 기반 semantic segmentation 구현
- **접근**: DINOv2 논문과 동일한 설정으로 2-phase 학습 진행
- **하드웨어**: RTX 3090 × 2 GPU (24GB each)

---

## 🔍 DINOv2 논문 원본 설정 (Linear Probing + Multi-Scale)

### Architecture
- **Backbone**: DINOv2-base (ViT-B/14)
- **Input Size**: 640×640 pixels
- **Head**: Simple BNHead (BatchNorm + 1×1 Conv)
- **Feature**: last_hidden_state (CLS token 제외)

### Training Hyperparameters
- **Total Iterations**: 40,000
- **Optimizer**: AdamW
- **Learning Rate**: 1e-3
- **Weight Decay**: 1e-4
- **Betas**: (0.9, 0.999)
- **Batch Size**: 16 (2 per GPU × 8 GPUs)
- **Scheduler**: Poly (power=1.0, min_lr=0.0)
- **Warmup**: Linear warmup (1,500 iterations, ratio=1e-6)

### Multi-Scale Settings
- **Training**: Single scale (640×640)
- **Inference**: Multi-scale ratios [1.0, 1.32, 1.73, 2.28, 3.0]
- **Test Augmentation**: Flip augmentation enabled
- **Sliding Window**: 640×640 crop, 320×320 stride

### Data Augmentation
- Multi-scale crop with ratio range (1.0, 3.0)
- Random crop to 640×640
- Random horizontal flip (50% probability)
- Photometric distortion

### Performance Results
- **Linear Probing**: 47.3 mIoU
- **Multi-Scale (+ms)**: 53.0 mIoU (5.7%p improvement)

---

## 🎯 현재 구현 설정 (2-Phase Training)

### 하드웨어 환경
```bash
GPU 0: NVIDIA GeForce RTX 3090 (24GB)
GPU 1: NVIDIA GeForce RTX 3090 (24GB)
Total: 48GB GPU Memory
```

### Architecture (DINOv2와 동일)
- **Backbone**: DINOv2-base (facebook/dinov2-base)
- **Input Size**: 640×640 pixels  
- **Head**: Simple BNHead (BatchNorm + 1×1 Conv)
- **Classes**: 150 (ADE20K)
- **Feature**: last_hidden_state (CLS token 제외)

### Training Strategy (2-Phase)

#### **Phase 1: Linear Probing (25%)**
- **Iterations**: 10,000 (약 2 epochs)
- **Backbone**: Frozen ❄️
- **Learning Rate**: 1e-3
- **Optimizer**: AdamW
- **Weight Decay**: 1e-4
- **Warmup**: 1,500 iterations (linear)

#### **Phase 2: Fine-tuning (75%)**
- **Iterations**: 30,000 (약 6 epochs)  
- **Backbone**: Unfrozen 🔥
- **Learning Rate**: 5e-4 (Phase1의 절반)
- **Optimizer**: AdamW
- **Weight Decay**: 1e-4
- **Scheduler**: Poly (power=1.0)

### Multi-GPU Training
- **Batch Size**: 4 total (2 per GPU × 2 GPUs)
- **Workers**: 6 per GPU
- **Backend**: NCCL (DistributedDataParallel)
- **Synchronization**: SyncBatchNorm

### Inference Settings
- **Method**: Multi-scale inference
- **Scale Ratios**: [1.0, 1.32, 1.73, 2.28, 3.0]
- **Test Augmentation**: Horizontal flip enabled
- **Output**: Ensemble of multi-scale predictions

### Dataset Configuration
- **Dataset**: ADE20K (20,210 train, 2,000 val)
- **Original Resolution**: Variable (256×256 ~ 683×512)
- **Input Resize**: 640×640 (match DINOv2)
- **Classes**: 150 semantic categories

---

## 📊 Epoch vs Iteration 계산

### Training Data
- **Training Images**: 20,210
- **Batch Size**: 4
- **Iterations per Epoch**: 20,210 ÷ 4 = 5,052.5

### Phase Breakdown  
- **Total Iterations**: 40,000
- **Total Epochs**: 40,000 ÷ 5,052.5 ≈ 8 epochs
- **Phase 1**: 10,000 iterations ≈ 2 epochs (linear probing)
- **Phase 2**: 30,000 iterations ≈ 6 epochs (fine-tuning)

---

## 🔄 기존 구현에서 변경사항

### 제거할 요소
- ❌ LoRA fine-tuning 관련 코드
- ❌ RISE-2 방식 구현
- ❌ Complex conv head (512→256→128→classes)
- ❌ Epoch 기반 학습 (160 epochs)
- ❌ 높은 learning rate (1e-2)

### 추가할 요소
- ✅ Simple BNHead (BatchNorm + 1×1 Conv)
- ✅ Multi-GPU DDP training
- ✅ Iteration 기반 학습 (40,000 iterations)
- ✅ Multi-scale inference
- ✅ Linear warmup scheduler
- ✅ Poly learning rate decay

---

## 🎯 예상 성능 목표

### DINOv2 Paper Baseline
- **Linear Probing**: 47.3 mIoU
- **Multi-Scale (+ms)**: 53.0 mIoU

### 현재 구현 목표
- **Phase 1 완료**: ~47% mIoU (DINOv2 linear probing 수준)
- **Phase 2 완료**: ~50-53% mIoU (fine-tuning 효과)
- **Multi-scale 적용**: +3-5%p 추가 향상 기대

### 비교 기준
- **현재 최고 성능**: 32.78% mIoU (Phase 2, epoch 94)
- **목표 향상도**: +15-20%p (DINOv2 설정 적용 효과)

---

## 📁 수정할 파일 목록

### 설정 파일
- `configs/config.yaml` - DINOv2 hyperparameters로 전면 수정

### 모델 파일  
- `dinov2_segmentation/training/model.py` - Simple BNHead로 변경
- `dinov2_segmentation/training/dataset.py` - 640×640 resize 적용

### 학습 파일
- `dinov2_segmentation/training/train_segmentation.py` - DDP + iteration 기반으로 전면 수정

### 제거할 파일
- `dinov2_segmentation/training/lora.py` - LoRA 관련 코드 제거
- `dinov2_segmentation/training/warmup_scheduler.py` - 기존 warmup 제거

---

## 🚀 실행 계획

1. **설정 파일 수정** - DINOv2 hyperparameters 적용
2. **모델 구조 변경** - Simple BNHead 구현  
3. **학습 스크립트 재작성** - DDP + 2-phase iteration 기반
4. **데이터셋 수정** - 640×640 입력 크기 적용
5. **Multi-scale inference 구현** - 최대 성능 달성
6. **실험 실행** - 2 GPU 병렬 학습
7. **성능 평가** - DINOv2 paper와 비교

**목표**: DINOv2 논문과 동일한 조건에서 2-phase 학습의 효과 검증

---

*Generated: 2025-08-04*
*Hardware: RTX 3090 × 2*  
*Target: ADE20K Semantic Segmentation*

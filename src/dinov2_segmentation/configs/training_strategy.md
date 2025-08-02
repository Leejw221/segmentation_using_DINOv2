# Progressive Training Strategy for DINOv2 Segmentation

# Stage 1: Linear Probing (5-10 epochs)
# - freeze_backbone: true
# - learning_rate: 0.001
# - image_size: 224
# - Focus on learning basic segmentation head

# Stage 2: Fine-tuning with low resolution (10-20 epochs) 
# - freeze_backbone: false
# - learning_rate: 0.00005
# - image_size: 336
# - Gradually unfreeze backbone layers

# Stage 3: Full fine-tuning with high resolution (20-30 epochs)
# - freeze_backbone: false  
# - learning_rate: 0.00002
# - image_size: 518
# - Full network training with high resolution

### 권장 실행 순서:

1. **데이터 검증**:
   ```bash
   # Ground truth 마스크 확인
   ls ~/dinov2_ws/src/dinov2_segmentation/datasets/ADEChallengeData2016/annotations/training/ | head -10
   ```

2. **Stage 1 학습** (Linear Probing):
   - config_stage1.yaml 사용
   - 빠른 수렴으로 기본 성능 확인

3. **Stage 2-3 학습** (Progressive Fine-tuning):
   - config_improved.yaml 사용  
   - 점진적 해상도 증가

### 추가 개선 사항:

1. **Data Augmentation 강화**:
   - MixUp, CutMix 적용
   - Multi-scale training
   
2. **Test-time Augmentation**:
   - Multi-scale inference
   - Horizontal flip TTA
   
3. **Post-processing**:
   - CRF (Conditional Random Fields) 적용
   - Connected components filtering

### 성능 평가 지표:

- **mIoU (mean Intersection over Union)**: 전체 성능
- **Per-class IoU**: 개별 클래스 성능  
- **Pixel Accuracy**: 픽셀 단위 정확도
- **Frequency Weighted IoU**: 클래스 빈도 고려

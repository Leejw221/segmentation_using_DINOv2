# DINOv2 ROS2 Segmentation 사용법

## 🚀 빠른 시작 (Launch 파일 사용)

### 1️⃣ 깔끔한 노드 실행 (추천)
```bash
cd ~/dinov2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch dinov2_segmentation dinov2_simple.launch.py
```
- DINOv2 노드만 실행
- 10초마다 자동 샘플 이미지 처리 (무한 반복)
- rqt는 수동으로 실행 필요

### 2️⃣ 노드만 실행 (동일)
```bash
ros2 launch dinov2_segmentation dinov2_only.launch.py
```
- 위와 동일한 기능

### 3️⃣ 완전한 데모 (다중 창 + rqt 자동)
```bash
ros2 launch dinov2_segmentation dinov2_full_demo.launch.py
```
- DINOv2 노드 + 자동 rqt 실행
- 다중 창으로 시각화

## 🎯 수동 실행 (상세 제어)

### 터미널 1: 노드 실행
```bash
cd ~/dinov2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 run dinov2_segmentation rise2_seg_node
```

### 터미널 2: 시각화 (선택사항)
```bash
source /opt/ros/humble/setup.bash
rqt
```

**rqt에서 이미지 확인 방법:**
1. `Plugins` → `Visualization` → `Image View` 선택
2. Topic 드롭다운에서 토픽 선택:
   - `/dinov2/original_image` (원본 이미지)
   - `/dinov2/segmentation_result` (segmentation 결과)
3. 이미지가 10초마다 자동 업데이트됨

### 터미널 3: 이미지 처리
```bash
cd ~/dinov2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash

# 로컬 파일 처리 (추천)
ros2 run dinov2_segmentation file_publisher ~/dinov2_ws/test_images/segmentation_demo.jpg

# 웹 URL 처리
ros2 run dinov2_segmentation url_publisher "https://your-image-url.jpg"
```

## 📁 테스트 이미지

준비된 테스트 이미지들:
- `~/dinov2_ws/test_images/segmentation_demo.jpg` - 표준 segmentation 데모
- `~/dinov2_ws/test_images/outdoor_scene.jpg` - 복잡한 실외 장면

## 🔍 토픽 확인

```bash
# 토픽 목록
ros2 topic list | grep dinov2

# 결과 토픽 데이터 확인
ros2 topic echo /dinov2/segmentation_result --once

# 토픽 정보
ros2 topic info /dinov2/segmentation_result
```

## 📊 시스템 상태

- **모델**: `facebook/dinov2-base` (AutoModel)
- **패치 그리드**: 16x16 (256 patches)
- **Feature 차원**: 768
- **클러스터 수**: 6개 색상으로 segmentation

## 🎨 결과 해석

- **원본 이미지**: `/dinov2/original_image`
- **Segmentation 결과**: `/dinov2/segmentation_result` 
  - 6가지 색상으로 구분된 영역들
  - 각 색상은 유사한 특성을 가진 이미지 영역을 나타냄
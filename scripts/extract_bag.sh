#!/bin/bash

# ROS2 Bag 추출 및 재생 스크립트
# 사용법: ./extract_bag.sh [숫자]
# 예시: ./extract_bag.sh 4  -> 8.20.4.tar.gz 파일을 추출하고 재생

# 파라미터 확인
if [ -z "$1" ]; then
    echo "사용법: $0 [숫자]"
    echo "예시: $0 4  -> 8.20.4.tar.gz 파일 처리"
    echo ""
    echo "사용 가능한 파일들:"
    ls $DATA_DIR/8.20.*.tar.gz 2>/dev/null | sed 's/.*8\.20\.\([0-9]*\)\.tar\.gz/  \1/' | sort -n
    exit 1
fi

BAG_NUM=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data_collect"
BAG_FILE="$DATA_DIR/8.20.$BAG_NUM.tar.gz"
EXTRACT_DIR="/tmp/bag_8_20_$BAG_NUM"

# 파일 존재 확인
if [ ! -f "$BAG_FILE" ]; then
    echo "❌ 파일이 존재하지 않습니다: $BAG_FILE"
    echo ""
    echo "사용 가능한 파일들:"
    ls $DATA_DIR/8.20.*.tar.gz 2>/dev/null | sed 's/.*8\.20\.\([0-9]*\)\.tar\.gz/  \1/' | sort -n
    exit 1
fi

echo "📦 ROS2 Bag 파일 처리: 8.20.$BAG_NUM.tar.gz"
echo "📁 소스 파일: $BAG_FILE"
echo "📂 추출 위치: $EXTRACT_DIR"
echo ""

# 이전 추출 디렉토리 정리
if [ -d "$EXTRACT_DIR" ]; then
    echo "🧹 이전 추출 폴더 제거 중..."
    rm -rf "$EXTRACT_DIR"
fi

# 새 디렉토리 생성
mkdir -p "$EXTRACT_DIR"

# 파일 크기 표시
FILE_SIZE=$(du -h "$BAG_FILE" | cut -f1)
echo "📊 파일 크기: $FILE_SIZE"

# 압축 해제
echo "🔧 압축 해제 중... (시간이 걸릴 수 있습니다)"
echo "⏳ 진행 상황:"

# tar 명령어에 진행상황 표시 추가
cd "$EXTRACT_DIR"
if tar -xf "$BAG_FILE" --checkpoint=1000 --checkpoint-action=dot 2>/dev/null; then
    echo ""
    echo "✅ 압축 해제 완료!"
else
    echo ""
    echo "❌ 압축 해제 실패"
    exit 1
fi

# bag 정보 확인
echo ""
echo "📋 ROS2 Bag 정보 확인 중..."
cd "$EXTRACT_DIR"

# 첫 번째 하위 디렉토리를 찾아서 이동 (압축 파일 구조에 따라)
BAG_PATH="$EXTRACT_DIR"
if [ -d "$(find $EXTRACT_DIR -mindepth 1 -maxdepth 1 -type d | head -1)" ]; then
    BAG_PATH="$(find $EXTRACT_DIR -mindepth 1 -maxdepth 1 -type d | head -1)"
    echo "📁 Bag 경로: $BAG_PATH"
fi

# bag info 실행
ros2 bag info "$BAG_PATH"

echo ""
echo "🚀 실행 명령어들:"
echo ""
echo "1️⃣ Bag 재생 (터미널 1에서 실행):"
echo "   ros2 bag play \"$BAG_PATH\""
echo ""
echo "2️⃣ Segmentation 실행 (터미널 2에서 실행):"
echo "   conda activate dinov2_segmentation"
echo "   source /opt/ros/jazzy/setup.bash"
echo "   cd /home/moai/jungwook_ws/segmentation_using_DINOv2"  
echo "   source install/setup.bash"
echo "   ros2 launch dinov2_ros_segmentation segmentation_inference.launch.py"
echo ""
echo "3️⃣ 결과 확인 (터미널 3에서 실행):"
echo "   conda activate dinov2_segmentation"
echo "   source /opt/ros/jazzy/setup.bash"
echo "   rqt_image_view"
echo ""
echo "💾 추출된 폴더: $EXTRACT_DIR"
echo "🗑️  정리하려면: rm -rf \"$EXTRACT_DIR\""
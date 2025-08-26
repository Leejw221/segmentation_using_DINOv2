#!/bin/bash

# 빠른 테스트 스크립트 - Bag 재생과 Segmentation을 한 번에 실행
# 사용법: ./quick_test.sh [숫자]
# 예시: ./quick_test.sh 4

if [ -z "$1" ]; then
    echo "사용법: $0 [숫자]"
    echo "예시: $0 4  -> 8.20.4.tar.gz 파일로 테스트"
    echo ""
    echo "사용 가능한 파일들:"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
    DATA_DIR="$PROJECT_DIR/data_collect"
    ls $DATA_DIR/8.20.*.tar.gz 2>/dev/null | sed 's/.*8\.20\.\([0-9]*\)\.tar\.gz/  \1/' | sort -n
    exit 1
fi

BAG_NUM=$1
SCRIPT_DIR="$(dirname "$0")"

echo "🚀 빠른 테스트 모드: 8.20.$BAG_NUM"
echo ""

# 1. Bag 추출
echo "1️⃣ Bag 파일 추출 중..."
$SCRIPT_DIR/extract_bag.sh $BAG_NUM

if [ $? -ne 0 ]; then
    echo "❌ Bag 추출 실패"
    exit 1
fi

BAG_PATH="/tmp/bag_8_20_$BAG_NUM"
if [ -d "$(find $BAG_PATH -mindepth 1 -maxdepth 1 -type d | head -1)" ]; then
    BAG_PATH="$(find $BAG_PATH -mindepth 1 -maxdepth 1 -type d | head -1)"
fi

echo ""
echo "2️⃣ 테스트 실행 준비 완료!"
echo ""
echo "📋 다음 단계로 진행하세요:"
echo ""
echo "🟢 터미널 1 (Bag 재생):"
echo "   ros2 bag play \"$BAG_PATH\""
echo ""
echo "🔵 터미널 2 (Segmentation - 환경 설정 필요):"
echo "   conda activate dinov2_segmentation"
echo "   source /opt/ros/jazzy/setup.bash"
echo "   cd /home/moai/jungwook_ws/segmentation_using_DINOv2"
echo "   source install/setup.bash"
echo "   ros2 launch dinov2_ros_segmentation segmentation_inference.launch.py"
echo ""
echo "🟡 터미널 3 (결과 확인):"
echo "   conda activate dinov2_segmentation && source /opt/ros/jazzy/setup.bash"
echo "   rqt_image_view"
echo ""
echo "💡 팁: 모든 터미널에서 Ctrl+C로 중단 가능"
#!/bin/bash
# 간단한 워크스페이스 환경 설정 (매번 새 터미널에서 실행)

# Conda 환경 활성화
eval "$(conda shell.bash hook)"
conda activate dinov2_segmentation

# ROS2 환경 소싱
source /opt/ros/jazzy/setup.bash

# 워크스페이스 찾기 및 소싱
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$WORKSPACE_DIR"
source install/setup.bash

echo "✅ 환경 설정 완료!"
echo "📂 현재 위치: $(pwd)"
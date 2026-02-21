#!/bin/bash
# Bullet3リポジトリから取得したLaikagoのURDFを表示するスクリプト
# Z-up版を使用（正しい向き、元のLaikagoと同じ色設定）

URDF_FILE="${1:-laikago_toes_zup_final.urdf}"

echo "Bullet3のLaikago URDF (Z-up版、最終版) を表示します..."
echo "使用するURDF: $URDF_FILE"
docker exec -it genesis_tensorboard bash -c "cd /workspace/bullet3/examples/pybullet/gym/pybullet_data/laikago && gs view $URDF_FILE"


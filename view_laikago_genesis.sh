#!/bin/bash
# Genesisプロジェクト内のLaikagoのURDFを表示するスクリプト
# これが最も信頼できるソースです

URDF_FILE="${1:-laikago_toes_zup.urdf}"

echo "Genesisプロジェクト内のLaikago URDFを表示します..."
echo "使用するURDF: $URDF_FILE"
docker exec -it genesis_tensorboard bash -c "cd /workspace/Genesis/genesis/assets/urdf/laikago/urdf && gs view $URDF_FILE"





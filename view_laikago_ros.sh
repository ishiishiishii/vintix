#!/bin/bash
# ROSパッケージから取得したLaikagoのURDFを表示するスクリプト
# READMEに記載されているlaikago_descriptionパッケージを使用

echo "ROSパッケージのLaikago URDFを表示します..."
echo "（READMEのlaikago_descriptionパッケージから）"
docker exec -it genesis_tensorboard bash -c "cd /workspace/laikago_ros/laikago_description && gs view laikago_genesis.urdf"


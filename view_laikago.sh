#!/bin/bash
# LaikagoのURDFを表示するスクリプト

# Dockerコンテナに入ってLaikagoのURDFを表示
docker exec -it genesis_tensorboard bash -c "cd /workspace/Genesis/genesis/assets/urdf/laikago/urdf && gs view laikago.urdf"



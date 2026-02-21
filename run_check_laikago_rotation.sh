#!/bin/bash
# Laikago回転テストツールを実行するスクリプト
# 対話的な入力を受け付けるため、-itフラグを使用

docker exec -it genesis_tensorboard bash -c "cd /workspace/Genesis/examples/locomotion && python check_laikago_rotation.py"




#!/bin/bash
# UnitreeA1のURDFを可視化するスクリプト

cd /home/kawa37/genesis_project

# 仮想環境を有効化（venv39を使用）
source venv39/bin/activate

# URDFファイルのパス
URDF_FILE="Genesis/genesis/assets/urdf/unitree_a1/urdf/a1.urdf"

# Genesisのviewコマンドを実行
cd Genesis
python3 -m genesis._main view "$URDF_FILE" "$@"

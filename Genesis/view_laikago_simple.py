#!/usr/bin/env python3
"""
Laikago URDFビューアー（シンプル版）
gs viewコマンドの代わりに使用
"""
import sys
import os
sys.path.insert(0, '/workspace/Genesis')

import genesis as gs

# アセットディレクトリを確認
assets_dir = gs.utils.get_assets_dir()
print(f"Assets directory: {assets_dir}")

urdf_file = sys.argv[1] if len(sys.argv) > 1 else "urdf/laikago/urdf/laikago_toes_zup.urdf"
print(f"Loading URDF: {urdf_file}")

# gs viewコマンドを直接呼び出す
from genesis._main import view
view(urdf_file, collision=False, rotate=False, scale=1.0, show_link_frame=False)


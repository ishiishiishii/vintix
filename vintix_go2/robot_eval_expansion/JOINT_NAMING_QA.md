# 関節・足リンク命名 QA（#16・CPU）

立位レンダ前の準備。Genesis / GPU は使わない。

| ロボット | Go2 名でそのまま描ける | 別名スキーム | 備考 |
|----------|------------------------|--------------|------|
| go1 | yes | （Unitree 系） | GO2 既定で可 |
| go2 | yes | （Unitree 系） | GO2 既定で可 |
| aliengo | yes | （Unitree 系） | GO2 既定で可 |
| b1 | yes | （Unitree 系） | GO2 既定で可 |
| anymalb | **no** | ok | Env の joint_names を render に渡す |
| anymalc | **no** | ok | Env の joint_names を render に渡す |
| spot | **no** | ok | Env の joint_names を render に渡す |
| laikago | **no** | ok | Env の joint_names を render に渡す |

## レンダ時の注意

- **Unitree 系（go1/go2/aliengo/b1）:** `GO2_JOINT_NAMES` で立位可
- **ANYmal-B/C:** `LF/RF/LH/RH` × `HAA/HFE/KFE`（Env 既存）
- **Spot:** `fl/fr/hl/hr` × `hx/hy/kn`（足リンクは `*_lleg`）
- **Laikago:** `*_hip_motor_2_chassis_joint` 等の長い名前

再現: `python vintix_go2/robot_eval_expansion/check_joint_names_cpu.py`

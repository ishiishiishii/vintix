# GenLoco → Genesis ラッパ

## 公式との差分

| 項目 | 公式 GenLoco | 本リポラッパ |
|------|--------------|--------------|
| Sim | PyBullet（`motion_imitation`） | Genesis（評価入口は未接続。CPU 次元チェックのみ） |
| 観測 | phase_only: Motor×15 + IMU6×15 + Act×15 + phase = **451** | 同寸法。IMU の R/P は projected gravity から近似、Y=0 |
| 行動 | 12 DoF 位置オフセット | actor mean 12 |
| タスク | モーション模倣（pace / spin / inverse） | 速度追従とは別定義。比較指標は揃え直しが必要 |

## Pretrained（3 zip・入力同一）

`third_party/GenLoco/motion_imitation/data/policies/`

- `morphology_generator_pace_model.zip`
- `morphology_generator_spin_model.zip`
- `morphology_generator_inverse.zip`

Stable-Baselines PPO1 形式。`parameters` NPZ:

- `pi_fc0` (451, 1024) → ReLU → `pi_fc1` (1024, 512) → ReLU → `pi` (512, 12)

## 接続ステップ

1. ~~次元を zip から確定~~（CPU スモーク通過）
2. ~~評価入口 dry-run~~（`run_genloco_genesis_eval.py --dry_run`）
3. Genesis Go2 env から `GenesisToGenlocoObs.step` を呼ぶ `--run`（GPU 空き後）
4. 公式 PyBullet 再生と比較し、差分を記録（`EVAL_PROTOCOL.md`）
5. 本線指標（完走・速度追従）へマッピングする評価スクリプト

## ファイル

| ファイル | 役割 |
|----------|------|
| `obs_adapter.py` | Genesis 45 → GenLoco 451（履歴＋位相） |
| `policy_loader.py` | zip NPZ 次元 assert + 最小 actor |
| `smoke_cpu_check.py` | GPU 不要スモーク |
| `run_genloco_genesis_eval.py` | 評価入口（既定 dry-run / `--run` は GPU 待ち） |

```bash
cd /workspace/vintix_go2/general_locomotion_baselines
python wrappers/genloco/smoke_cpu_check.py
python wrappers/genloco/run_genloco_genesis_eval.py --dry_run
```

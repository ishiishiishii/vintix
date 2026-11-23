# GenLoco → Genesis ラッパ

## 公式との差分

| 項目 | 公式 GenLoco | 本リポラッパ |
|------|--------------|--------------|
| Sim | PyBullet（`motion_imitation`） | Genesis（`--run` 配線済。近似 obs） |
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
3. ~~Genesis Go2 `--run`~~（2026-11-23: pace zip、4×500 → completion 0%、mean len 100）
4. 公式 PyBullet 再生と比較し、差分を記録（`EVAL_PROTOCOL.md`）
5. 本線指標（完走・速度追従）へマッピングする評価スクリプト

## 初回 `--run` 結果（pace・近似 obs）

- JSON: `results/genloco_genesis/genloco_pace_go2_ne4_s500.json`
- completion **0%**、mean episode length **100**、mean reward/step **≈0.004**
- MorFiC（mean len ≈11.6）より長く立つが、完走は同様に 0%
- エピソード長が時間とともに伸びる傾向（履歴バッファ充填の可能性）あり → 未切り分け

## ファイル

| ファイル | 役割 |
|----------|------|
| `obs_adapter.py` | Genesis 45 → GenLoco 451（履歴＋位相） |
| `policy_loader.py` | zip NPZ 次元 assert + `GenlocoActorPolicy` |
| `smoke_cpu_check.py` | GPU 不要スモーク |
| `run_genloco_genesis_eval.py` | 評価入口（`--dry_run` / `--run`） |

```bash
cd /workspace/vintix_go2/general_locomotion_baselines
python wrappers/genloco/smoke_cpu_check.py
python wrappers/genloco/run_genloco_genesis_eval.py --dry_run
python wrappers/genloco/run_genloco_genesis_eval.py --run --num_envs 4 --max_steps 500 --policy pace
```

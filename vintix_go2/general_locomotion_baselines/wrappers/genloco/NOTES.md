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
4. ~~公式に近い PyBullet 再生~~（`run_genloco_pybullet_eval.py`・TF 不要）
5. 本線指標（完走・速度追従）へ対応づける

## 公式に近い PyBullet 結果（pace・Go1・2026-11-23）

- JSON: `results/genloco_pybullet/genloco_pace_go1_ep3_s300.json`
- 3 エピソードとも **300/300 ステップ生存**、平均 return ≈254
- シミュレータと観測は GenLoco 公式。行動は zip の平均方策（TensorFlow なし）

## Genesis `--run` 結果（pace・観測近似・Go2）

- JSON: `results/genloco_genesis/genloco_pace_go2_ne4_s500.json`
- 完走率 **0%**、平均エピソード長 **100**
- 公式 PyBullet では歩けるのにこちらは歩けない → 配線・観測・タスク差が主因候補

## ファイル

| ファイル | 役割 |
|----------|------|
| `obs_adapter.py` | Genesis 45 → GenLoco 451（履歴＋位相） |
| `policy_loader.py` | zip NPZ 次元確認 + `GenlocoActorPolicy` |
| `smoke_cpu_check.py` | GPU 不要の寸法確認 |
| `run_genloco_pybullet_eval.py` | **推奨:** 公式に近い PyBullet 評価 |
| `run_genloco_genesis_eval.py` | Genesis 参考評価（`--dry_run` / `--run`） |

```bash
cd /workspace/vintix_go2/general_locomotion_baselines
python wrappers/genloco/smoke_cpu_check.py
python wrappers/genloco/run_genloco_pybullet_eval.py --robot go1 --policy pace --num_episodes 3
python wrappers/genloco/run_genloco_genesis_eval.py --dry_run
```

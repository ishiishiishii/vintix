# URDF 質量 QA（#16）

日付: 2026-11-18  
目的: 導入 URDF の Σmass を公式スペック・文献（ManyQuadrupeds 等）と照合し、Env 追加前の差分を記録する。

## 要約

| ロボット | URDF Σmass | 公式目安 | ManyQ 目安 | 判定 |
|----------|------------|----------|------------|------|
| Aliengo | **24.94 kg** | ~21.5 kg（製品カタログ） | ~20.6–21 kg | **差分あり**（+3–4 kg）。sim 用として採用し、DR 質量レンジで吸収する想定 |
| B1 | **62.58 kg** | ~50 kg（製品カタログ） | ~52.7 kg | **差分あり**（+10–12 kg）。ペイロード／ケーブル無しモデル差の可能性。要メモ付き採用 |
| ANYmal-B | **30.42 kg** | — | ~30 kg | **一致**（simple_description） |
| Spot (RAI simple) | **50.34 kg** | ~32.5 kg | ~32–50 系 | **差分あり**（Isaac 由来慣性）。ライセンス OK |
| ANYmal-C（既存） | 52.14 kg | — | ~52.1 kg | 一致 |
| Laikago（既存） | 25.17 kg | — | ~25 kg | 一致 |

公式カタログ値はメーカー公開スペックの概数。URDF は `unitree_ros` / `anymal_b_simple_description` 由来（各 `SOURCE.md`）。

## Aliengo

- 置き場: `Genesis/genesis/assets/urdf/aliengo/urdf/aliengo.urdf`
- trunk mass: 11.644 kg（DR フォールバック）
- 関節: Unitree 命名（`FR/FL/RR/RL` × hip/thigh/calf）。**thigh は `continuous`（角度 limit なし）**、hip/calf は revolute+limit
- Env: `Genesis/examples/locomotion/aliengo_env.py`
- 立位・PD（2026-11-23 修正）: MorFiC 準拠で `base_init_pos` z=0.40 / `base_height_target` 0.36 / kp=40。旧 z=0.55・kp=20 では PPO が即転倒のまま失敗
- 結論: 質量差は既知。高さ・PD は MorFiC 設定を優先


## B1

- 置き場: `Genesis/genesis/assets/urdf/b1/xacro/b1.urdf`（展開済み単一 URDF）
- trunk mass: 29.45 kg
- 関節: Unitree 12-DoF（すべて revolute）
- Env: `b1_env.py`（暫定 `base_init` z=0.65 / target 0.55）
- 結論: カタログより重い。タスク1ではそのまま導入し、タスク2で報酬・質量を詰める

## ANYmal-B

- 置き場: `Genesis/genesis/assets/urdf/anymal_b/urdf/anymal.urdf`
- base_inertia mass: 16.7935 kg
- 関節: ANYmal 命名（`LF/RF/LH/RH` × HAA/HFE/KFE）— C と同系
- Env: `anymalb_env.py`（暫定は ANYmal-C 角を流用、`base_init` z=0.50 / target 0.45）
- 結論: 質量 OK。simple_description のため視覚・衝突形状は簡易

## Spot（Boston Dynamics）

- 置き場: `Genesis/genesis/assets/urdf/spot/`（RAI `spot_simple` を flatten）
- ライセンス: **MIT（RAI）+ BSD-3（Clearpath 由来）** — 取り込み可
- Clearpath 素の `spot.urdf.xacro` は **inertial 無し**のため不採用
- Σmass ≈ **50.34 kg**（body 32.86 + 脚）。製品カタログ ~32.5 kg より重い（Isaac Sim 由来密度）
- Env: `spot_env.py` / `train.py -r spot`（関節 `fl_hx` 等、立位暫定）
- 結論: ライセンス OK・物理モデルは重い。レンダ後に高さ・角を詰める

## 残作業

- 立位レンダは 2026-11-23 に完了（`figures/standing/`）
- 仕様表の土台寸法（メッシュ依存）は必要なら後で埋める
- 報酬・専門家学習はタスク2（本線の後）

## 参照

- ManyQuadrupeds (arXiv:2310.10486) 質量・高さ表
- GenLoco / MorFiC 評価機体リスト → `ROBOT_SURVEY.md`

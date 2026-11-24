# #16 タスク2計画: 専門家（PPO）のみ — 未知ロボット評価用

更新: 2026-11-24（ユーザー訂正）。タスク1（URDF・Env・立位）は完了済み。

## 目的

拡充機体は **未知ロボットとしての評価ターゲット**。歩行できる専門家（PPO）と品質指標・動画があれば足りる。

## 必須ルール（2026-11-24 訂正）

- **Adaptation 用履歴（AD）収集は不要**（本線の Vintix 学習データではない）
- **指令速度・胴体高さは体格に合わせる**（[`PARAMETER_DESIGN.md`](PARAMETER_DESIGN.md)）
- 可能なら **Go2 同条件（\(v_x=0.5\)）の専門家も**別途取る（`--cmd_profile matched_go2`）
- 手順は:

1. PPO 学習（デフォルト=`body` 指令）  
   `train.py -r <robot> ...`  
   必要なら追加: `train.py -r <robot> --cmd_profile matched_go2 -e <name>-vx05 ...`
2. `eval_expert_quality.py`（完走・前進速度・RMSE）＋歩行動画 → Notion
3. 次機体へ（1 台ずつ）

※ 既取得の Aliengo / B1 / ANYmal-B ckpt は **matched (\(v_x=0.5\))** として扱う。body 指令の再学習は任意／優先度付き。

## 対象機体（優先順）

| 順 | 機体 | 状態 |
|----|------|------|
| 1 | Aliengo | PPO＋完走済。動画・速度 RMSE を補強 |
| 2 | B1 | PPO＋完走済。追従弱 → 要再チューニング／動画 |
| 3 | ANYmal-B | PPO 済（`model_300`）。品質評価＋動画が次 |
| 4 | Spot | matched (\(v_x=0.5\), h≈0.48) `model_300` 済。完走100%・速度比≈0.50（不合格）。**body** (\(v_x=0.9\), h=0.50) 学習中 |
| 5 | Laikago | PPO 未。Env 済 |

## 報酬・設定

- 新規報酬設計はしない。既存の `get_<robot>_cfgs()` を再利用
- 既定: `num_envs=4096`, `max_iterations=301`、seed=1
- 密保存（`save_interval=20`）は B1 教訓どおり維持

## 見積もり（1 機体）

| 段階 | GPU 時間（目安） |
|------|------------------|
| PPO 301 iter | 約 2–4 h |
| 品質評価＋動画 | 約 0.2–0.5 h |
| **合計 / 台** | **約 2–5 h**（AD なし） |

## ANYmal-B

- [x] PPO `model_300.pt`（最終平均長 ≈917、ゲート PASS）
- [x] `eval_expert_quality.py` ＋動画（2026-11-24）
  - 完走 **100%**・平均 \(v_x\)≈**0**（指令 0.5 の 0.01%）→ **速度不合格**
  - 動画でもほぼ立位のまま → **要再チューニング**
- AD: **中止**（方針変更）

## Spot / Laikago

- PPO → 品質評価＋動画のみ（AD 連鎖スクリプトは使わない）
- Spot matched: `logs/spot-walking/model_300.pt` + `results/spot_expert_quality.json`（完走 PASS / 速度 FAIL）
- Spot body: `logs/spot-walking-body/`（PARAMETER_DESIGN 再監査 OK 後に起動）
- Laikago: Spot body 後に PPO（h/vx は信頼度「弱」→失敗時再チューン）
- ANYmal-B 速度不合格の再チューンは Spot body 後でも可

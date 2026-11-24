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
| 1 | Aliengo | **合格**（完走100%・指令比≈99%） |
| 2 | B1 | body（0.60）速度不合格 → **mid 0.50 合格**（完走100%・指令比≈60%） |
| 3 | ANYmal-B | matched/body 立位 → **kp40 合格**（完走100%・指令比≈95%） |
| 4 | Spot | matched/body 不合格 → **mid 0.7 合格**（完走100%・指令比≈99%） |
| 5 | Laikago | 初回／term20 とも崩壊 → **高さ罰 -10 再学習中**（`logs/laikago-walking-bh10/`） |

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

- [x] matched PPO `model_300.pt` → 完走100%・前進≈0（不合格）
- [x] body PPO（\(h=0.50\), \(v_x=0.60\), kp=20）→ 完走100%・前進≈0.0035（不合格・`results/anymalb_body_expert_quality.json`）
- [x] kp40 PPO（高さ・指令は body のまま）→ **合格**（完走100%・指令比≈94.6%・`results/anymalb_kp40_expert_quality.json`）
- AD: **中止**（方針変更）

## Spot / Laikago

- PPO → 品質評価＋動画のみ（AD 連鎖スクリプトは使わない）
- Spot matched: `logs/spot-walking/model_300.pt` + `results/spot_expert_quality.json`（完走 PASS / 速度 FAIL）
- Spot body: `logs/spot-walking-body/model_300.pt` + `results/spot_body_expert_quality.json`（完走 FAIL / 速度比≈92%）
- Spot mid: `logs/spot-walking-mid07/model_300.pt` + `results/spot_mid07_expert_quality.json`（**完走・速度とも PASS**・指令比≈99%）
- Laikago 初回: `logs/laikago-walking/model_300.pt` + `results/laikago_expert_quality.json`（完走0%・学習崩壊）
- Laikago term20: `logs/laikago-walking-term20/model_300.pt` + `results/laikago_term20_expert_quality.json`（完走0%・平均長≈7・不合格）
- Laikago bh10: `logs/laikago-walking-bh10/`（高さ罰 -10・転倒判定 20° 維持・密保存・学習中）
- B1 body: `logs/b1-walking-body/model_300.pt` + `results/b1_body_expert_quality.json`（完走 PASS / 速度比≈49% FAIL）
- B1 mid: `logs/b1-walking-mid05/model_300.pt` + `results/b1_mid05_expert_quality.json`（**完走・速度とも PASS**・指令比≈60%）
- 合格は Aliengo / Spot mid / B1 mid / **ANYmal-B kp40** の 4 機。残りは Laikago 再チューン

# Spot 第1波: 起動前チェック → matched 結果

更新: 2026-11-24 05:20 UTC（autopilot）。拡充機体は **未知ロボット評価用**（Adaptation 履歴は不要）。

## 配線（現状）

| 項目 | 状態 |
|------|------|
| Env / `train.py -r spot` | 済（`spot_env.py`・`get_spot_cfgs`） |
| 立位レンダ | 済（`figures/standing/spot_standing.png`） |
| 質量 QA | Σ≈**50.3 kg**（RAI `spot_simple`）。公式カタログ ≈32.5 kg より重い → メモ付き採用（`MASS_QA.md`） |
| Adaptation 履歴 | **不要**（2026-11-24） |
| matched PPO（\(v_x=0.5\)） | **完了** → 完走 PASS / 速度 FAIL（≈50%） |
| body PPO（\(v_x=0.9\)） | **完了** → 完走 FAIL（0%）/ 速度比 ≈92%（`spot_body_expert_quality.json`） |

## Aliengo / B1 / ANYmal-B から持ち越すルール

1. **保存を密にする**（20 iter ごと）: `cfgs.pkl` 事前配置（共通 `train.py` は触らない）
2. **学習ログの最終平均長だけで合格にしない**: 短評価（決定論）で完走＋前進速度を確認
3. **ピーク近傍の重みを候補にする**: 最終不合格でもピーク ≥ 閾値ならフォールバックして短評価
4. **完走だけ・前進速度不足は不合格**（ANYmal-B≈0、Spot matched≈50%）

## 設定メモ（`get_spot_cfgs`・PARAMETER_DESIGN.md）

- **body（現行デフォルト）:** 目標高さ **0.50** / init z=**0.55** / \(v_x=0.90\)（BD Med）
- **matched:** `--cmd_vx 0.5`（または `--cmd_profile matched_go2`）。今回の第1波はこちら
- kp=**40** / kd=1.0、関節: `fl/fr/hl/hr` × `hx/hy/kn`
- URDF は RAI `spot_simple` のみ

## matched 結果（2026-11-24）

| 指標 | 値 | 判定 |
|------|-----|------|
| 完走率 | **100%**（10 env×1000） | PASS |
| 平均前進 \(v_x\) | ≈**0.249** m/s | — |
| 指令比（指令 0.5） | ≈**50%**（閾値 60%） | **FAIL** |
| RMSE | ≈0.312 | 記録 |
| 動画 | `figures/videos/spot_walk_model300.mp4` | 済 |

中間 ckpt（140–280）は決定論で完走 0%。最終 `model_300` が最良だが速度不足。

## 監視ポイント（body 学習中）

| 合図 | 目安 | 次の一手 |
|------|------|----------|
| 序盤の平均長 | ≈20→上がる | 正常（現状 iter≈87 で ≈16 → 遅れ気味） |
| 中盤ピーク後の急落 | B1 と同型 | 密保存＋ピーク ckpt で短評価 |
| 最終平均長 ≥200 | ゲート候補 | `eval_expert_quality --cmd_vx 0.9`＋動画 |
| 完走≥95% かつ \(v_x\)≥指令の60% | 合格 | 次機体（Laikago） |
| 完走だけ・速度不足 | 不合格 | 再チューニング（AD は取らない） |

## チェックリスト

- [x] ANYmal-B PPO 完了（品質は速度不合格 → 再チューニング待ち）
- [x] Spot matched PPO → 品質評価＋動画（速度不合格）
- [ ] Spot body PPO → 品質評価＋動画
- [x] Issue #16 / Notion に記録

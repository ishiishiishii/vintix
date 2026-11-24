# Spot 第1波: 起動前チェック（ANYmal-B の次）

更新: 2026-11-24 04:28 UTC（autopilot）。拡充機体は **未知ロボット評価用**（Adaptation 履歴は不要）。

## 配線（現状）

| 項目 | 状態 |
|------|------|
| Env / `train.py -r spot` | 済（`spot_env.py`・`get_spot_cfgs`） |
| 立位レンダ | 済（`figures/standing/spot_standing.png`） |
| 質量 QA | Σ≈**50.3 kg**（RAI `spot_simple`）。公式カタログ ≈32.5 kg より重い → メモ付き採用（`MASS_QA.md`） |
| Adaptation 履歴 | **不要**（2026-11-24） |
| 次 | PPO → `eval_expert_quality`＋動画 |

## Aliengo / B1 / ANYmal-B から持ち越すルール

1. **保存を密にする**（20 iter ごと）: `cfgs.pkl` 事前配置（共通 `train.py` は触らない）
2. **学習ログの最終平均長だけで合格にしない**: 短評価（決定論）で完走＋前進速度を確認
3. **ピーク近傍の重みを候補にする**: 最終不合格でもピーク ≥ 閾値ならフォールバックして短評価
4. **完走だけ・前進速度≈0 は不合格**（ANYmal-B `model_300` の教訓）

## 設定メモ（暫定・`get_spot_cfgs`）

- 初期高さ z=**0.55**、kp=**40** / kd=1.0
- 関節: `fl/fr/hl/hr` × `hx/hy/kn`（Boston Dynamics 系の命名）
- 立位角は暫定。だめなら: 目標高さを立位実測に合わせる → kp 調整 → 転倒判定の緩和、の順
- URDF は Clearpath 素（inertial 無し）は使わない。RAI `spot_simple` のみ

## 監視ポイント（学習中）

| 合図 | 目安 | 次の一手 |
|------|------|----------|
| 序盤の平均長 | ≈20→上がる | 正常 |
| 中盤ピーク後の急落 | B1 と同型 | 密保存＋ピーク ckpt で短評価 |
| 最終平均長 ≥200 | ゲート候補 | `eval_expert_quality`＋動画 |
| 完走≥95% かつ \(v_x\)≥指令の60% | 合格 | 次機体（Laikago） |
| 完走だけ・速度≈0 | 不合格 | 再チューニング（AD は取らない） |

## 着手前チェックリスト

- [x] ANYmal-B PPO 完了（品質は速度不合格 → 再チューニング待ち）
- [x] GPU が空いている（#22 MorFiC 済）
- [ ] Spot PPO → 品質評価＋動画
- [ ] Issue #16 / Notion に着手を記録

# Spot 第1波: 起動前チェック（ANYmal-B の次）

更新: 2026-11-24 01:58 UTC（autopilot）。GPU を使わずに固定した監視・配線メモ。
**いまは起動しない**（優先順は Aliengo → B1 → ANYmal-B → Spot）。ANYmal-B 第1波が終わるまで待機。

## 配線（現状）

| 項目 | 状態 |
|------|------|
| Env / `train.py -r spot` | 済（`spot_env.py`・`get_spot_cfgs`） |
| 立位レンダ | 済（`figures/standing/spot_standing.png`） |
| 質量 QA | Σ≈**50.3 kg**（RAI `spot_simple`）。公式カタログ ≈32.5 kg より重い → メモ付き採用（`MASS_QA.md`） |
| 履歴収集 `-r spot` | **未**（`collect_ad_data_parallel.py` に未追加。着手時に足す） |
| ANYmal-B 完了後の連鎖 | **未**（第1波完了後に `wait_after_anymalb_*` を書く） |

## Aliengo / B1 から持ち越すルール

1. **保存を密にする**（20 iter ごと）: `cfgs.pkl` 事前配置（共通 `train.py` は触らない）
2. **学習ログの最終平均長だけで不合格にしない**: 短評価（決定論）で歩行を確認
3. **ピーク近傍の重みを候補にする**: 最終不合格でもピーク ≥ 閾値ならフォールバックして履歴収集
4. **履歴なしの専門家は第1波完了としない**

## 設定メモ（暫定・`get_spot_cfgs`）

- 初期高さ z=**0.55**、kp=**40** / kd=1.0
- 関節: `fl/fr/hl/hr` × `hx/hy/kn`（Boston Dynamics 系の命名）
- 立位角は暫定。だめなら: 目標高さを立位実測に合わせる → kp 調整 → 転倒判定の緩和、の順
- URDF は Clearpath 素（inertial 無し）は使わない。RAI `spot_simple` のみ

## 監視ポイント（学習中・案）

| 合図 | 目安 | 次の一手 |
|------|------|----------|
| 序盤の平均長 | ≈20→上がる | 正常 |
| 中盤ピーク後の急落 | B1 と同型 | 密保存＋ピーク ckpt で短評価 |
| 最終平均長 ≥200 | ゲート直通 | `model_300` から履歴 |
| 最終 <200 だがピーク ≥200 | フォールバック | ピーク近傍 ckpt → 履歴 |
| ピークも <200 | 失敗 | 履歴は取らない。高さ / PD / 転倒を診断 |

## 着手前チェックリスト

- [ ] ANYmal-B 第1波（PPO + 履歴 + 短評価）が完了している
- [ ] GPU が空いている
- [ ] `collect_ad_data_parallel.py` に `-r spot` を追加
- [ ] `wait_after_anymalb_ad_start_spot.sh`（または同等）を用意
- [ ] Issue #16 / Notion に着手を記録

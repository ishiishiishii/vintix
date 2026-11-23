# #16 タスク2計画: 専門家（PPO）＋学習履歴（Adaptation 用データ）

更新: 2026-11-23 20:19（autopilot）。タスク1（URDF・Env・立位）は完了済み。

## 目的

拡充した評価機体でも、本線と同じく「専門家の歩行」と「上達の履歴」を用意する。
履歴なしのチェックポイントだけでは、のちの Transformer（Vintix）学習に使えない。

## 必須ルール（ユーザー合意）

専門家を用意するときは **学習履歴（Adaptation 用データ）の保存を忘れない**。
手順は常に:

1. PPO 学習（`Genesis/examples/locomotion/train.py -r <robot>`）
2. 学習済み重みから履歴収集（`vintix_go2/scripts/collect_ad_data_parallel.py`）
3. 完走率・報酬などの短い評価スモーク

## 対象機体（優先順）

| 順 | 機体 | 理由 |
|----|------|------|
| 1 | **Aliengo** | Unitree 系・関節名が Go2 に近く、Env 済 |
| 2 | B1 | Unitree 大型。Env 済 |
| 3 | ANYmal-B | 文献頻度高。関節マップ要確認 |
| 4 | Spot | Env 済。関節マップ要確認 |
| 5 | Laikago | Env 済。座標系注意（立位 QA 済） |

1 台ずつ完走させてから次へ（同時に複数台の本学習はしない）。

## 報酬・設定

- 新規報酬設計はしない。既存の `train.py` 内 `get_<robot>_cfgs()` を再利用
- 既定: `num_envs=4096`, `max_iterations=301`（本線と同型）
- 学習シードはまず **seed=1**。必要なら後で追加

## 学習履歴収集のギャップ（実装メモ）

`collect_ad_data_parallel.py` の `-r` に **`aliengo` を追加済み**（2026-11-23）。
デフォルト ckpt は `robot_eval_expansion/logs/aliengo-walking/model_300.pt`。
B1 / Spot / ANYmal-B は同様に後続で足す。

## 見積もり（1 機体）

| 段階 | GPU 時間（目安） |
|------|------------------|
| PPO 301 iter | 約 2–4 h |
| 履歴収集（並列） | 約 0.5–2 h |
| 評価スモーク | 約 0.2 h |
| **合計 / 台** | **約 3–6 h**（12 h ガード内） |

## 完了条件（Aliengo 第1波） — **完了 2026-11-23 18:54**

- [x] `logs/aliengo-walking/model_300.pt`（MorFiC URDF・平均長 1001）
- [x] Adaptation 用履歴: `data/aliengo_ad_parallel/`（10×h5+json、合計 **2,005,619** transitions）→ `results/aliengo_wave1_summary.json`
- [x] 短評価完走率 **100%**・mean return ≈22.6（`results/aliengo_smoke_model300.json`）
- [x] Issue #16 / Notion に数値とパスを記録

試行メモ: unitree URDF では高さ・PD・転倒判定いじりでも平均長 ≈10 止まり。**MorFiC URDF 切替が決定打**。

## 進行中（B1 第1波）

- [ ] PPO 試行:
  1. MorFiC URDF+kp90 → 平均長 ≈5（退避 `b1-walking_morfic_failed_*`）
  2. unitree+MorFiC角+kp90+term20 → 平均長 ≈12 で頭打ち（退避 `b1-walking_unitree_term20_plateau12_*`）
  3. unitree+Go2系角+kp40+z0.66+term20+高さ罰**-10** → **不合格**（最終 13.56 / ピーク 1001@83）
     - 退避: `b1-walking_go2like_bh10_collapsed_20261123T201710Z` / `results/b1_ad_quality_gate.json`
  4. **進行中:** 3 と同じ角・PD で高さ罰 **-2**（`logs/train_ppo/b1_seed1_bh2.out`、PID は `b1_latest.pid`）
     - `wait_collect_b1_ad.sh` に品質ゲート（最終平均長 ≥200 でないと履歴を取らない）
     - 序盤 iter≈0–7 の平均長 ≈23→38（試行3と同程度。高さ罰の寄与は小さくなった）
- [ ] Adaptation 用履歴（品質ゲート通過時のみ）
- [ ] 短評価スモーク

### 試行4のあと（これも最終で歩けない場合）

- 候補: 保存間隔を短くしてピーク ckpt を取る／目標高さを FK 実測に再合わせ／接触・慣性の診断
- 失敗 ckpt からは履歴を取らない（ゲート済み）

## やらないこと（今）

- Go1 以外の体スケール崖評価（保留）
- 報酬の全面再設計
- 6 機体同時学習

# #16 タスク2計画: 専門家（PPO）＋学習履歴（Adaptation 用データ）

更新: 2026-11-23（autopilot）。タスク1（URDF・Env・立位）は完了済み。

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

## 完了条件（Aliengo 第1波）

**進捗 2026-11-24 02:50:** MorFiC URDF で `model_300.pt` 完了。短評価完走率 100%。履歴収集中。


- [x] `logs/aliengo-walking/model_300.pt` へ向けて学習成功軌道（2026-11-23）
  - 1回目: unitree URDF + z=0.55/kp=20 → 平均長 ≈5。退避 `aliengo-walking_kp20_failed_20261123`
  - 2回目: unitree + z=0.40/kp=40 → ≈1.7。退避 `aliengo-walking_kp40_z40_failed_*`
  - 3回目案 z=0.55 は **棄却**（FK 実測 0.46 より高い）
  - 4回目: unitree + z=0.50/term20 → 平均長 ≈9–10 で頭打ち。退避 `aliengo-walking_unitree_urdf_failed_20261123`
  - **5回目（成功）:** Env 既定 URDF を **MorFiC `aliengo_description`** に切替。cfg: init z=**0.36** / 目標 **0.36** / kp=40 / 転倒判定 **10°**。ゼロ行動 80+ step。iter≈60 で平均エピソード長 **1001** に到達し以降維持。PID **1940460**、ログ `logs/train_ppo/aliengo_seed1_morfic_urdf.out`
- [ ] Adaptation 用履歴（h5）: 収集中（PID **1942647**、出力 `data/aliengo_ad_parallel/`、10 env × 200k step）
- [x] 短い評価で「歩ける」: 10 env × 1000 step で完走率 **100%**・mean return ≈22.6（`results/aliengo_smoke_model300.json`）
- [x] Issue #16 / Notion「評価ロボット拡充」に数値とパスを記録（2026-11-23 17:48）

## やらないこと（今）

- Go1 以外の体スケール崖評価（保留）
- 報酬の全面再設計
- 6 機体同時学習

# #16 タスク2計画: 専門家（PPO）＋学習履歴（Adaptation 用データ）

更新: 2026-11-23 22:45（autopilot）。タスク1（URDF・Env・立位）は完了済み。

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

`collect_ad_data_parallel.py` の `-r` に **`aliengo` / `b1` / `anymalb` を追加済み**（2026-11-23）。
デフォルト ckpt は各 `robot_eval_expansion/logs/<robot>-walking/model_300.pt`。
Spot は後続で足す。

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

- [x] PPO 試行（要点）:
  1. MorFiC URDF+kp90 → 平均長 ≈5（退避 `b1-walking_morfic_failed_*`）
  2. unitree+MorFiC角+kp90+term20 → 平均長 ≈12 で頭打ち（退避 `b1-walking_unitree_term20_plateau12_*`）
  3. unitree+Go2系角+kp40+z0.66+term20+高さ罰**-10** → 最終 13.56 / ピーク 1001@83（退避 `*_go2like_bh10_*`）
  4. 高さ罰 **-2**・301 iter → ピーク ≈793@81 の直後に崩壊（退避 `*_bh2_collapsed_*`）。解析: `results/b1_bh2_collapse_analysis.json`
  5. **早期停止 81 iter + 保存間隔 20**（`b1_seed1_earlystop80.out`）→ 学習ログ最終長 123（旧ゲート不合格）だが密 ckpt を確保
- [x] 短評価スモーク:
  - 10 env×1000: `model_40/60/80` 完走率 **100%**、return ≈4–6（`results/b1_smoke_ckpt_compare.json`）
  - 8 env×500 再スモーク: `model_40/80` 完走 100%（return **2.3 / 1.85**）、`model_60` は崩壊（`results/b1_ckpt_smoke_eval.json`）
- [ ] Adaptation 用履歴: **収集中**（PID **1961754**、**`model_40.pt`** → `data/b1_ad_parallel/`）
  - 合格条件は短評価完走。再スモークで return が高い `model_40` を採用
  - **進捗訂正（2026-11-23 22:45）:** ログの `Collected steps (sum/env)` は **全 env 合計**。前回の「≈30%」は誤読（合計を 20万で割っていた）。正しい進捗は合計 ≈**42万 / 200万**（≈**21%**）、env 平均 ≈4.2万 step、ε≈**0.42**、直近平均長 **≈902**、ETA ≈**03:53 UTC**（`results/b1_ad_collect_mid.json`）

### メモ（B1）

- 中盤崩壊型。密な保存＋短評価で歩行重みを取れる。ただし短評価プロトコルで `model_60` の合否が変わる → 複数プロトコルで確認する
- 速度追従（return）は Aliengo（≈22）より弱い（≈2–6）。履歴の質は収集完了後に短評価で再判断
- 次機体: ANYmal-B（Env 済・`get_anymalb_cfgs` あり。**`-r anymalb` と `wait_collect_anymalb_ad.sh` は配線済み**。PPO は B1 履歴完了後）
- 次がダメなら: 目標高さ FK 再合わせ／接触・慣性診断／追従重視の追加学習

## やらないこと（今）

- Go1 以外の体スケール崖評価（保留）
- 報酬の全面再設計
- 6 機体同時学習

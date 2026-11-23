# 評価対象ロボットの拡充（#16）

Issue: [#16](https://github.com/ishiishiishii/vintix/issues/16)  
稼働中 #18 ジョブには非介入。

## 目的
他論文で使われる評価ロボットを一通り導入する。報酬・PPO は本線実験後。

## 進捗（タスク1）
1. ~~文献調査~~ → [`ROBOT_SURVEY.md`](ROBOT_SURVEY.md)
2. ~~質量 QA~~ → [`MASS_QA.md`](MASS_QA.md)
3. ~~Env / URDF~~  
   - Aliengo / B1 / ANYmal-B / Spot（`train.py -r {aliengo,b1,anymalb,spot}`）
4. ~~関節命名 QA（CPU）~~ → [`JOINT_NAMING_QA.md`](JOINT_NAMING_QA.md) / `check_joint_names_cpu.py`
5. ~~立位レンダ準備（CPU）~~ → `prepare_standing_render.py` / `standing_render_manifest.json`
6. ~~立位レンダ `--run`~~ → `figures/standing/*_standing.png`（6 機体・2026-11-23）
7. ~~立位 QA 修正（aliengo / laikago）~~ → 下記実行ログ 05:19
8. ~~タスク1仕上げ~~（2026-11-23）
9. **タスク2 着手中**: Aliengo 専門家 PPO（301 iter）＋履歴収集配線。計画は [`TASK2_PLAN.md`](TASK2_PLAN.md)

## やらないこと（今）
- 6 機体同時の本学習（1 台ずつ）
- Go1 以外の体スケール崖評価（保留）

## 実行ログ

### 2026-11-23 15:43: タスク2 開始（Aliengo PPO）

- **実行内容:**
  - `collect_ad_data_parallel.py` に `-r aliengo` を追加
  - `python train.py -r aliengo -e aliengo-walking --log_dir robot_eval_expansion/logs/aliengo-walking --max_iterations 301 --seed 1`
- **結果:** 学習起動（GPU）。ログ: `logs/train_ppo/aliengo_seed1.out`
- **次:** （下記 16:16 で失敗確定 → 再学習）

### 2026-11-23 16:16（autopilot）: Aliengo PPO 1回目失敗 → 設定修正・再学習待ち

- **実行内容:** 学習ログを集計。平均エピソード長 ≈**5**・報酬 ≈-0.03 のまま。MorFiC 準拠へ `get_aliengo_cfgs` を修正
- **結果:** 1回目 ckpt は**使わない**
- **次:** （下記 01:19 で再学習起動）

### 2026-11-24 01:19: kp40 再学習を起動

- **実行内容:** 失敗分を `logs/aliengo-walking_kp20_failed_20261123` へ退避し、同パス `logs/aliengo-walking` で再学習（pid 1933196）
- **結果:** cfg 確認済み（kp=40, z=0.40）。序盤の平均エピソード長 ≈1.6（要観察）
- **次:** エピソード長の伸びを 30 分ごとに確認。歩けるまで履歴収集しない

### 2026-11-19 09:04（autopilot）: 立位レンダ前の関節命名 QA（CPU）

- **実行内容:** URDF を XML パースし、Go2 既定名との一致／別名スキームを `JOINT_NAMING_QA.md` に整理
- **結果:** aliengo/b1 は Unitree 名でそのまま可。ANYmal-B/C・Spot・Laikago は明示マップ必須
- **次:** #18 完了後に `render_robot_comparison.py` で立位 PNG

### 2026-11-20 03:23（autopilot）: 立位レンダ起動スクリプト（CPU dry-run）

- **実行内容:** `prepare_standing_render.py` で 6 機体の joint alias を URDF 突合しマニフェスト化
- **結果:** 全 OK。`--run` は Unitree 系のみ即可。他は default angles 未配線で defer
- **次:** #21 解放後に `--run`

### 2026-11-23 03:56（autopilot）: 立位レンダ 6 機体完了

- **実行内容:** `#21` 完了後に `--run`。B1/ANYmal-B の `package://` メッシュ参照を相対パスへ修正し再実行
- **結果:** `figures/standing/{aliengo,b1,anymalb,anymalc,spot,laikago}_standing.png` 全成功（failed 0）
- **次:** タスク1のレンダ区切り。タスク2（報酬+PPO+AD）は本線後。#22 `--run` 実装へ

### 2026-11-23 05:19（autopilot）: aliengo / laikago 立位 QA 修正

- **実行内容:**
  - **Aliengo:** `calf.dae` が COLLADA unit=0.0254m（インチ）なのに Genesis が単位を無視 → URDF visual `scale="0.0254"`（4 脚）。`SOURCE.md` に注記
  - **Laikago:** Y-up URDF。`apply_standing_pose` 後の quat では逆さま → Env と同じ `base_init_quat=[0.5,0.5,0.5,0.5]` + 固定 z=0.50 の専用スクリプト `render_laikago_standing.py`
  - `prepare_standing_render.py` に `--force` / `--only`
- **結果:** 両機とも四足立位として認識可能な PNG。b1/anymalb/anymalc/spot は前回どおり OK → **タスク1 視覚 QA 完了**
- **次:** タスク2（報酬+PPO+AD）は本線後。#22 GenLoco `--run` / URMA RL-X

### 2026-11-23 15:40（autopilot）: タスク2計画を固定

- **実行内容:** `TASK2_PLAN.md` を追加（Aliengo 先頭、PPO→学習履歴→短評価、1 台ずつ）
- **結果:** `collect_ad_data_parallel.py` に Aliengo が未対応であるギャップを明記
- **次:** Aliengo 用の履歴収集配線 → PPO 本学習（GPU 空き時）


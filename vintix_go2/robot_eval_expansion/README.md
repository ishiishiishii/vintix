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
9. ~~タスク2 Aliengo 第1波~~（PPO + 履歴 + 短評価）
10. **タスク2 進行中（B1）**: **`model_40`** から履歴収集中（≈**58%**・PID 1961754）。完了後は `wait_after_b1_ad_start_anymalb.sh`（PID 1970393）が ANYmal-B PPO を自動起動（密保存・ピークフォールバック済み）。計画は [`TASK2_PLAN.md`](TASK2_PLAN.md) / [`ANYMALB_PREFLIGHT.md`](ANYMALB_PREFLIGHT.md)

## やらないこと（今）
- 6 機体同時の本学習（1 台ずつ）
- Go1 以外の体スケール崖評価（保留）

## 実行ログ

### 2026-11-24 00:52: B1 履歴≈58%・ANYmal-B に密保存とピークフォールバックを配線

- **分かったこと:** 正しい進捗 ≈**58%**（合計 ≈116万 / 200万）、ε≈**0.10**、直近平均長 **≈921**、直近報酬 ≈**+3.3**。収集は健全。B1 で効いた「密な保存＋ピーク重み」を次機体の連鎖に先入れした
- **実行内容:** `results/b1_ad_collect_mid.json` 更新。`wait_after_b1_ad_start_anymalb.sh` で `cfgs.pkl` 事前配置（保存間隔 20）。`wait_collect_anymalb_ad.sh` は最終平均長不足でもピーク近傍 ckpt へフォールバック。`ANYMALB_PREFLIGHT.md` 追加。稼働中 B1 収集（PID **1961754**）には未介入
- **次:** B1 履歴完了（ETA ≈**03:21 UTC**）→ 第1波サマリ → ANYmal-B。#22 は合意で close 可

### 2026-11-23 23:48: B1 履歴≈38%・報酬がプラスへ／完了後に ANYmal-B を連鎖起動

- **分かったこと:** 正しい進捗 ≈**38%**（合計 ≈76万 / 200万）、ε≈**0.24**、直近平均長 **≈892**、直近報酬 ≈**+1.74**（前回 −2.6 から符号が反転）。探索ノイズ低下とともに歩容が安定している。ETA ≈**03:43 UTC**
- **実行内容:** `results/b1_ad_collect_mid.json` を再集計。`wait_after_b1_ad_start_anymalb.sh` を起動（PID **1970393**）— B1 収集終了後に第1波サマリ書き出し → ANYmal-B PPO（301 iter）→ `wait_collect_anymalb_ad.sh`。稼働中の B1 収集（PID **1961754**）には未介入
- **次:** B1 履歴完了を待つ（連鎖スクリプトが次機体を起動）。#22 は合意で close 可

### 2026-11-23 22:45: B1 履歴進捗の誤読を訂正（実≈21%、残り≈5h）＋ANYmal-B 収集配線

- **分かったこと:** ログの `Collected steps (sum/env)` は **全環境の合計**。前回の「≈30%」は合計を 20万で割った誤読（実は当時 ≈3%）。現在は合計 ≈**42万 / 200万**（≈**21%**）、ε≈**0.42**、直近平均長 **≈902**、ETA ≈**03:53 UTC**
- **実行内容:** `results/b1_ad_collect_mid.json` を正しい定義で再集計。`collect_ad_data_parallel.py` に `-r anymalb`、待ちスクリプト `wait_collect_anymalb_ad.sh` を追加（PPO は未起動・GPU は B1 収集を継続）
- **次:** B1 履歴完了 → 短評価で第1波区切り → ANYmal-B PPO。#22 は合意で close 可

### 2026-11-23 21:41: B1 履歴は model_40・探索下でも平均長≈730

- **分かったこと:** 再スモーク（8×500）で `model_40` return **2.3** > `model_80` **1.85**（`model_60` は崩壊）。※進捗「≈30%」は後続 22:45 で誤読と判明
- **実行内容:** 中間集計 `results/b1_ad_collect_mid.json`。ドキュメントを `model_40`/PID **1961754** に同期。稼働中ジョブには未介入
- **次:** （22:45 で ETA を訂正）

### 2026-11-23 21:07: B1 早期停止完了・短評価100%・履歴収集開始

- **分かったこと:** 学習ログ最終平均長 **123** は旧ゲート不合格だが、`model_40/60/80` の短評価はいずれも完走率 **100%**（決定論）。学習中の「崩壊」は探索ノイズ下。return ≈4–6 で Aliengo（≈22）より弱い追従
- **実行内容:** 短評価 `results/b1_smoke_*.json`。当初 `model_80.pt` から収集起動 → 後続で `model_40` に切替（下記 21:41）
- **次:** 履歴完了 → B1 第1波区切り。追従が弱すぎれば追加診断

### 2026-11-23 20:46: B1 中盤ピーク取り逃し → 81 iter 早期停止

- **分かったこと:** 高さ罰-2でも歩行は一時的に成立するが、保存間隔100だと崩壊後の ckpt しか残らない
- **実行内容:** max_iterations=81・save_interval=20 で再起動（`b1_seed1_earlystop80.out`）
- **次:** `model_80` で履歴ゲート

### 2026-11-23 20:16: B1 3回目不合格（最終≈14）→ 高さ罰-2で4回目

- **分かったこと:** 品質ゲート FAIL（最終 13.56 / ピーク 1001@83）
- **実行内容:** 退避後、高さ罰 -10→-2 で再学習（`b1_seed1_bh2.out`）
- **次:** 最終平均長≥200 なら履歴

### 2026-11-23 19:45: B1 Go2系角は中盤で完走→崩壊（履歴ゲート追加）

- **分かったこと:** 3回目は「立てない」ではなく **学習の不安定**。iter **83** で平均エピソード長 **1001**、iter **89** で急落。その後も 15↔450 を往復（例: iter 167≈449 → 173≈112）。ピーク時の平均報酬 ≈**-10**（高さペナルティ支配）
- **実行内容:**
  - 集計を `results/b1_go2like_midtrain_analysis.json` に保存
  - `wait_collect_b1_ad.sh` に最終平均長 ≥200 の品質ゲートを追加し再起動（PID **1953361**）。失敗専門家からは履歴を取らない
  - 試行4案: 高さ報酬スケール -10→-2（現行学習には未適用）
- **結果:** 学習は継続中（PID **1951110**、残り ≈20 分）。ゲート待ち
- **次:** 最終長を見て履歴収集 or 試行4。#22 は合意で close 可

### 2026-11-23 19:15: B1 unitree+MorFiC角は平均長≈12で頭打ち → Go2系角で再学習

- **分かったこと:** unitree+kp90+term20 はゼロ行動は耐えるが PPO では平均長 ≈12 で停滞（MorFiC の ≈5 よりマシ）
- **実行内容:** 退避 `b1-walking_unitree_term20_plateau12_*`。Go2 系角・kp40・z=0.66・action_scale=0.25 で再起動（`b1_seed1_go2like.out`）
- **結果:** ゼロ行動 100 step OK。学習監視へ
- **次:** 平均エピソード長が伸びるか（30分間隔）

### 2026-11-23 19:08: Aliengo 第1波完了 → B1 は MorFiC 失敗後に unitree で再学習

- **分かったこと:** Aliengo 履歴は合計 **2,005,619** transitions（10 env）。B1 は Aliengo と逆で **MorFiC URDF が沈む**（ゼロ行動で胴体 z→0.22、学習は平均長 ≈5）。unitree の方が高さ維持が良い
- **実行内容:**
  - `results/aliengo_wave1_summary.json` で第1波を固定
  - 失敗 B1 を `logs/b1-walking_morfic_failed_*` へ退避
  - `b1_env.py` 既定を unitree に戻し、転倒判定 20°・高さ報酬 -10 で PPO 再起動（`logs/train_ppo/b1_seed1_unitree.out`）
  - `wait_collect_b1_ad.sh` で `model_300` 後の履歴収集を待機
- **結果:** Aliengo 第1波完了。B1 再学習中
- **次:** B1 のエピソード長が伸びるか監視。#22 は合意で close 可

### 2026-11-23 17:48（autopilot）: Aliengo PPO 完了 → 学習履歴収集を起動

- **分かったこと:** MorFiC URDF の PPO が **301 iter 完了**。最終平均エピソード長 **1001**・報酬 ≈**22.7**・線速度追従 ≈**0.99**。unitree URDF では設定いじりでも平均長 ≈10 止まりだった
- **実行内容:**
  - `model_300.pt` 確認。ゾンビ待ちを解除し Adaptation 用履歴収集を起動（PID **1942647**）
  - 出力 `data/aliengo_ad_parallel/`（10 env × 200k）。待ちスクリプトにゾンビ耐性を追加
  - #22 論文用主表 `general_locomotion_baselines/results/MAIN_TABLE.md` を固定
- **結果:** ckpt あり。履歴収集進行中。短評価は並行で完走率 100%（`results/aliengo_smoke_model300.json`）
- **次:** 履歴 h5 完了 → Aliengo 第1波区切り → B1。#22 は合意で close 可

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

### 2026-11-23 17:04（autopilot）: FK 実測高さ＋転倒判定緩和で再学習

- **実行内容:**
  - ゾンビ待ち解除。2回目失敗を退避（平均長 ≈1.7）
  - `robot_standing_pose` で胴体高さ ≈**0.4607 m**（足は床上 ≈1 cm）を実測 → 予定 z=0.55 を棄却
  - 20 iter スモーク（z=0.50）でも平均長 ≈**6**（高さだけでは不足）
  - 転倒判定 10°→**20°**、高さ報酬 -50→**-10** を追加し、`setsid` で 301 iter 再起動
- **結果:** 学習中（`logs/train_ppo/aliengo_latest.pid`）。失敗 ckpt からは履歴を取らない
- **次:** エピソード長が伸びるか監視 → 伸びたら履歴収集。#22 は完了条件案 `DONE_CRITERIA.md`

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


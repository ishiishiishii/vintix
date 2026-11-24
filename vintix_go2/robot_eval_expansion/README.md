> **2026-11-24 12:30 UTC:** ANYmal-B kp40 **合格**（完走100%・指令比≈**95%**）。合格は **4 機**。Laikago 転倒判定 20° 再学習中（PID **2020338**）。設計: `PARAMETER_DESIGN.md`。
# 評価対象ロボットの拡充（#16）

Issue: [#16](https://github.com/ishiishiishii/vintix/issues/16)  
稼働中ジョブには非介入。

## 目的
他論文で使われる評価ロボットを一通り導入し、**未知ターゲットとして評価**する。専門家 PPO と品質指標・動画まで。AD データは取らない。

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
9. ~~タスク2 Aliengo~~（PPO＋**品質合格**: 完走100%・\(v_x\)比≈99%・動画済）
10. ~~タスク2 B1~~: body（0.60）速度不合格 → **mid 0.50 合格**（完走100%・指令比≈60%）
11. ~~タスク2 ANYmal-B~~: matched/body 立位 → **kp40 合格**（完走100%・指令比≈95%）
12. ~~タスク2 Spot~~: matched/body 不合格 → **mid 0.7 合格**（完走100%・指令比≈99%）
13. **タスク2 Laikago**: 初回崩壊 → **term20 再学習中**（`logs/laikago-walking-term20/`）

## やらないこと（今）
- 拡充機体の Adaptation 履歴収集
- 6 機体同時の本学習（1 台ずつ）
- Go1 以外の体スケール崖評価（保留）

## 実行ログ

### 2026-11-24 12:30 UTC: ANYmal-B kp40 合格 → Laikago term20 起動

- **分かったこと:**
  - ANYmal-B kp40（\(h=0.50\), \(v_x=0.60\), kp=40, `model_300`）: 完走 **100%**、平均前進 ≈**0.568** m/s、指令比 ≈**94.6%**、RMSE≈0.073 → **完走・速度とも合格**（4 機目）
  - 関節の硬さ（PD）を上げるだけで立位から歩行へ抜けた。高さ・指令だけでは足りなかった
- **実行内容:** `eval_expert_quality.py --cmd_vx 0.60`＋動画。1 要因（転倒判定 10°→**20°**）で Laikago term20 PPO を密保存起動（PID **2020338**）
- **結果:** `results/anymalb_kp40_expert_quality.json`、`figures/videos/anymalb_kp40_walk_model300.mp4`。ログ `logs/train_ppo/laikago_term20_seed1.out` / `logs/laikago-walking-term20/`
- **次:** Laikago term20 完了 → 品質評価＋動画。不合格なら高さ罰／PD 1 要因

### 2026-11-24 11:22 UTC: ANYmal-B body も立位のまま → kp40 起動

- **分かったこと:**
  - ANYmal-B body（\(h=0.50\), \(v_x=0.60\), kp=20, `model_300`）: 完走 **100%**、平均前進 ≈**0.0035** m/s、指令比 ≈**0.6%** → **速度不合格**（ほぼ立位。matched と同型）
  - Spot/B1 の「巡航を抑える」は前進が出てから有効。前進≈0 では指令下げは無効
- **実行内容:** `eval_expert_quality.py --cmd_vx 0.60`＋動画。評価は学習時 `cfgs.pkl` を読むよう修正。1 要因 PD（kp 20→**40**, kd 0.5→**1.0**）で PPO を密保存起動（PID **2016379**）
- **結果:** `results/anymalb_body_expert_quality.json`、`figures/videos/anymalb_body_walk_model300.mp4`。ログ `logs/train_ppo/anymalb_kp40_seed1.out` / `logs/anymalb-walking-kp40/`
- **次:** kp40 完了 → 品質評価＋動画。不合格なら転倒判定／高さ罰 1 要因。合格なら Laikago 再チューン

### 2026-11-24 10:15 UTC: B1 mid 合格（指令比≈60%）→ ANYmal-B body 起動

- **分かったこと:**
  - B1 mid（\(h=0.55\), \(v_x=0.50\), `model_300`）: 完走 **100%**、平均前進 ≈**0.302** m/s、指令比 ≈**60.4%** → **完走・速度とも合格**（合格線ちょうど超え）
  - Spot mid と同型の「巡航を抑える」仮説が B1 でも成立。合格は **3 機目**
- **実行内容:** `eval_expert_quality.py --cmd_vx 0.50`＋動画。ANYmal-B body PPO（\(h=0.50\), \(v_x=0.60\)・密保存）を起動（PID **2012672**）
- **結果:** `results/b1_mid05_expert_quality.json`、`figures/videos/b1_mid05_walk_model300.mp4`。ログ `logs/train_ppo/anymalb_body_seed1.out` / `logs/anymalb-walking-body/`
- **次:** ANYmal-B body 完了 → 品質評価＋動画。不合格なら PD／高さ 1 要因。合格なら Laikago 再チューン

### 2026-11-24 09:05 UTC: B1 body は完走のみ・速度比≈49% → mid 0.50 起動

- **分かったこと:**
  - B1 body（\(h=0.55\), \(v_x=0.60\), `model_300`）: 完走 **100%**、平均前進 ≈**0.292** m/s、指令比 ≈**49%** → 速度不合格（合格線 60%）
  - 中間重み（40/80/120/200/280）も最大比 ≈**53%**（`model_200`）。どれも速度合格せず
  - Spot mid と同型の「完走は取れるが遅すぎ」パターン。絶対速度は ≈0.3 付近で頭打ち
- **実行内容:** `eval_expert_quality.py --cmd_vx 0.60`＋動画。1 要因（指令のみ 0.60→**0.50**）で mid PPO を密保存起動（PID **2008858**）
- **結果:** `results/b1_body_expert_quality.json`、`figures/videos/b1_body_walk_model300.mp4`。ログ `logs/train_ppo/b1_mid05_seed1.out` / `logs/b1-walking-mid05/`
- **次:** B1 mid 完了 → 品質評価＋動画。不合格なら PD／高さ 1 要因。合格なら ANYmal-B / Laikago 再チューン

### 2026-11-24 07:50 UTC: Spot mid 合格 → B1 body 再学習起動

- **分かったこと:**
  - Spot mid（指令 \(v_x=0.7\), `model_300`）: 完走 **100%**、平均前進 ≈**0.691** m/s、指令比 ≈**99%**、RMSE≈0.12 → **完走・速度とも合格**
  - matched（遅い）と body（転倒）のあいだの中間指令で両立できた。合格は Aliengo に続き **2 機目**
- **実行内容:** `eval_expert_quality.py --cmd_vx 0.7`＋動画。B1 body PPO（\(h=0.55\), \(v_x=0.60\)・密保存）を起動（PID **2004461**）
- **結果:** `results/spot_mid07_expert_quality.json`、`figures/videos/spot_mid07_walk_model300.mp4`。ログ `logs/train_ppo/b1_body_seed1.out` / `logs/b1-walking-body/`
- **次:** B1 body 完了 → 品質評価＋動画。続けて ANYmal-B / Laikago 再チューン

### 2026-11-24 07:10 UTC: Laikago 第1波不合格 → Spot mid（指令 0.7）起動

- **分かったこと:**
  - Laikago `model_300`（指令 0.4）: 完走 **0%**、平均長 ≈**7**、平均前進 ≈**-0.066** m/s（指令比 ≈**-16%**）→ 完走・速度とも不合格
  - 学習中も平均長 ≈5–11 のまま伸びず（学習崩壊）。第1波の初回パスは全機評価済み
  - 合格は引き続き Aliengo のみ
- **実行内容:** `eval_expert_quality.py --cmd_vx 0.4`＋動画。Spot mid PPO（`--cmd_vx 0.7`・密保存）を起動
- **結果:** `results/laikago_expert_quality.json`、`figures/videos/laikago_walk_model300.mp4`。ログ `logs/train_ppo/spot_mid07_seed1.out` / `logs/spot-walking-mid07/`
- **次:** Spot mid 完了 → 品質評価＋動画。不合格なら B1 / ANYmal-B / Laikago 再チューン（Laikago 候補: term20・高さ罰-10・kp）

### 2026-11-24 05:59 UTC: Spot body は速いがすぐ転倒 → Laikago PPO 起動

- **分かったこと:**
  - Spot body（\(v_x=0.9\), `model_300`）: 完走 **0%**、平均長 ≈**71**、平均前進 ≈**0.827** m/s（指令比 ≈**92%**）→ 速度は合格線超えだが完走失格
  - matched（完走100%・比≈50%）と body（完走0%・比≈92%）は対照的。「倒れないだけ」「速いだけ」のどちらも専門家としては不合格
- **実行内容:** `eval_expert_quality.py --cmd_vx 0.9`＋動画。密保存で Laikago PPO 起動（PID **1997767**）
- **結果:** `results/spot_body_expert_quality.json`、`figures/videos/spot_body_walk_model300.mp4`。ログ `logs/train_ppo/laikago_seed1.out`
- **次:** Laikago 完了 → 品質評価＋動画。Spot / B1 / ANYmal-B 再チューンは後続（候補: Spot 中間指令 0.7）

### 2026-11-24 05:20 UTC: Spot matched は完走のみ・速度不足 → body 学習中

- **分かったこと:**
  - Spot matched（\(v_x=0.5\), `model_300`）: 完走 **100%**、平均前進 ≈**0.249** m/s（指令比 ≈**50%**）→ 速度不合格（閾値 60%）
  - 密保存の中間重み（140–280）は決定論評価で完走 **0%**。学習ログのピークは探索ノイズ下で、推論では最終重みが最良
  - ANYmal-B（前進≈0）よりは動くが、Aliengo（比≈99%）には遠く、再チューニング候補
- **実行内容:** `eval_expert_quality.py`（`--cmd_vx 0.5`・動画）。body PPO（\(v_x=0.9\), h=0.50）は既起動のまま監視（PID **1993541**）。重複起動を 1 本停止済
- **結果:** `results/spot_expert_quality.json`、`figures/videos/spot_walk_model300.mp4`。`PARAMETER_DESIGN.md` / `SPOT_PREFLIGHT.md` 更新
- **次:** body 完了 → 品質評価（指令 0.9）。不合格なら報酬／PD 再チューン。Laikago は body の次

### 2026-11-24 04:31 UTC: Aliengo 合格・ANYmal-B 速度不合格・Spot PPO 起動

- **分かったこと:**
  - Aliengo `model_300`: 完走 **100%**、平均前進 ≈**0.494** m/s（指令 0.5・比 ≈**99%**）、RMSE ≈0.036 → **合格**（`results/aliengo_expert_quality.json`、動画あり）
  - ANYmal-B `model_300`: 完走 **100%** だが平均前進 ≈**0**（比 ≈0.01%）→ **不合格**。動画もほぼ立位のまま（学習ログの長さだけでは歩けると言えない）
  - #22 MorFiC 公式 play 完了で close
- **実行内容:** `eval_expert_quality.py`（anymalb / aliengo）。Spot PPO を密保存（`save_interval=20`）で起動（AD 待ちは起動しない）
- **次:** Spot 学習監視 → 品質評価。B1 / ANYmal-B 再チューニングは Spot 後または設計メモ並行

### 2026-11-24 04:15 UTC: AD 方針撤回（未知評価のみ）

- **分かったこと:** 拡充ロボットは未知ターゲット評価用。AD は不要
- **実行内容:** ANYmal-B AD（PID 1981084）と Spot-AD 連鎖待ちを停止。`TASK2_PLAN.md` を PPO＋品質評価に書き換え。GPU 解放後は MorFiC (#22) を優先可
- **次:** ANYmal 品質評価＋動画。Spot PPO。MorFiC 公式 play

### 2026-11-24 03:50 UTC: ANYmal-B PPO 完了・履歴収集開始

- **分かったこと:** 学習 301 iter 到達。最終平均長 ≈**917**・平均報酬 ≈**7.0**。品質ゲート PASS（ピーク長 ≈982@210）→ `model_300.pt` で Adaptation 履歴収集開始（PID **1981084**）。序盤は探索ノイズ高く短いエピソード（B1 と同様の序盤挙動）
- **実行内容:** 監視のみ（稼働中ジョブ未介入）。`EXPERT_QUALITY.md` の ANYmal-B 行を更新。並行で #22 Isaac Gym 導入（別 env）
- **次:** 履歴完了 → `eval_expert_quality.py`＋動画。B1 再チューニング準備

### 2026-11-24 03:12: B1 第1波完了・ANYmal-B PPO 起動

- **分かったこと:** B1 Adaptation 履歴が完了（合計 **2,005,870** transitions、ε=0、直近長 ≈911、直近報酬 ≈+3.7）。短評価は完走率 100%・平均報酬 ≈2.3（Aliengo ≈22.6 の約 1/10）。連鎖が ANYmal-B PPO を起動済み（PID **1981081**、03:12 時点 ≈52/301・平均長 ≈78・ETA ≈+35 min）。密保存で `model_0/20/40.pt` あり
- **実行内容:** `results/b1_wave1_summary.json` をチャンク名集計で再作成（初回サマリは flat key 想定で 0 と誤読）。`wait_after_b1_ad_start_anymalb.sh` の h5 集計を修正。品質比較・TASK2_PLAN・README を同期。稼働中 ANYmal-B 学習には未介入
- **次:** ANYmal-B 学習監視 → 履歴収集 → 短評価。#22 は合意で close 可

### 2026-11-24 01:58: B1 履歴≈80%・Aliengo比較と Spot 事前メモ

- **分かったこと:** 正しい進捗 ≈**80%**（合計 ≈159万 / 200万）、ε≈**0.022**、直近平均長 **≈813**、直近報酬 ≈**+3.25**。完走率は Aliengo と同じ 100% だが平均報酬は ≈2.3 vs ≈22（約10倍弱い追従）。収集は健全（ETA ≈**03:08 UTC**）
- **実行内容:** `results/b1_ad_collect_mid.json` 更新。`results/wave1_quality_compare.md` と `SPOT_PREFLIGHT.md` を追加。稼働中ジョブ（PID **1961754** / 連鎖 **1970393**）には未介入
- **次:** B1 履歴完了 → 第1波サマリ → ANYmal-B。#22 は合意で close 可

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


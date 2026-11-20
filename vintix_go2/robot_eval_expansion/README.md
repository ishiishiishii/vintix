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
6. **次**: GPU 空いたら `prepare_standing_render.py --run`（aliengo/b1）。他機体は Env default angles 後 → タスク2は本線後

## やらないこと（今）
- PPO 専門家の本学習（タスク2）
- 稼働中の #21 評価ジョブへの介入

## 実行ログ

### 2026-11-19 09:04（autopilot）: 立位レンダ前の関節命名 QA（CPU）

- **実行内容:** URDF を XML パースし、Go2 既定名との一致／別名スキームを `JOINT_NAMING_QA.md` に整理
- **結果:** aliengo/b1 は Unitree 名でそのまま可。ANYmal-B/C・Spot・Laikago は明示マップ必須
- **次:** #18 完了後に `render_robot_comparison.py` で立位 PNG

### 2026-11-20 03:23（autopilot）: 立位レンダ起動スクリプト（CPU dry-run）

- **実行内容:** `prepare_standing_render.py` で 6 機体の joint alias を URDF 突合しマニフェスト化
- **結果:** 全 OK。`--run` は Unitree 系のみ即可。他は default angles 未配線で defer
- **次:** #21 解放後に `--run`

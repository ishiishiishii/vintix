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
5. **次**: GPU 空いたら立位レンダ（Unitree 系は GO2 名で可。ANYmal/Spot/Laikago は Env の joint map 必須）→ タスク2は本線後

## やらないこと（今）
- PPO 専門家の本学習（タスク2）
- 稼働中の #18 / #21 待機ジョブへの介入

## 実行ログ

### 2026-11-19 09:04（autopilot）: 立位レンダ前の関節命名 QA（CPU）

- **実行内容:** URDF を XML パースし、Go2 既定名との一致／別名スキームを `JOINT_NAMING_QA.md` に整理
- **結果:** aliengo/b1 は Unitree 名でそのまま可。ANYmal-B/C・Spot・Laikago は明示マップ必須
- **次:** #18 完了後に `render_robot_comparison.py` で立位 PNG

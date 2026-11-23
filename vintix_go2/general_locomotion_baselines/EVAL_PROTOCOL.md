# 公式評価手順メモ（#22・非実験）

Vintix と比較するとき、各ベースラインの**公式再生と Genesis 配線の差分**を揃えるための手順書。
重い再生・学習は #18→#21 の GPU 解放後。ここではコマンドと指標の対応だけを固定する。

## 共通の本線指標（Genesis 側）

| 指標 | 定義 |
|------|------|
| Success rate / completion | 1000 step 転倒せず完走したエピソード割合 |
| tracking_success_rate | 速度指令追従（本リポ locomotion_metrics） |
| avg_forward_speed_mps | 平均前進速度 |

## P0 手法ごとの公式入口

### MorFiC
- 公式: Isaac Gym 系 play（リポ `third_party/morfic-rl`）
- 本リポ: `python wrappers/morfic/run_morfic_genesis_eval.py --dry_run`（GPU 不要）
- Genesis `--run`（2026-11-23 初回）: `python wrappers/morfic/run_morfic_genesis_eval.py --run --num_envs 4 --max_steps 500`
  - 結果 JSON: `results/morfic_genesis/morfic_go2_ne4_s500.json`（completion 0%、mean len ≈11.6）
  - 次: 公式 Isaac play と並べて「sim/obs 差」列を埋める
- 注意: 観測は 70×30 履歴。Genesis 45 からの再構成は近似

### GenLoco
- 公式シミュレータ: PyBullet `motion_imitation`（pace / spin / inverse の学習済み zip）
- **推奨（公式に近い再生・TensorFlow 不要）:**
  `python wrappers/genloco/run_genloco_pybullet_eval.py --robot go1 --policy pace --num_episodes 3`
  - 結果例（2026-11-23）: `results/genloco_pybullet/genloco_pace_go1_ep3_s300.json`
    — 3 エピソードとも 300/300 ステップ生存、平均 return ≈254
  - GenLoco のロボット一覧に Go2 は無い（Go1 / A1 など）
- 参考（Genesis・観測近似）: `python wrappers/genloco/run_genloco_genesis_eval.py --run --num_envs 4 --max_steps 500 --policy pace`
  - 結果: `results/genloco_genesis/genloco_pace_go2_ne4_s500.json`（完走率 0%、平均長 100）
- **公式は動きの模倣**なので、本線の速度追従とはタスクが違う → 比較表に「タスクの違い」列を残す
- 観測: 451（関節角×15 + IMU6×15 + 直前行動×15 + 位相）


### URMA
- 公式: RL-X（Flax/msgpack）の play
- 本リポ: 観測 380 の寸法スモークまで。actor forward は RL-X 依存で deferred
- GPU 空き後: RL-X 環境を隔離 venv で入れ、公式 play → 同指標で Genesis 配線

## 比較表の列（論文用に先に固定）

| 列 | 内容 |
|----|------|
| method | MorFiC / GenLoco / URMA / Vintix / Vintix+decoder FT |
| train_distribution | 学習した体・タスク |
| eval_robot | Go2（本線）＋ #16 機体は後段 |
| task | velocity tracking / motion imitation / … |
| success / tracking / speed | 上記共通指標（タスク非対応なら N/A） |
| notes | 観測再構成・sim 差 |

## いまやらないこと
- McARL（コード未収録）
- 本線 Vintix 学習設定の変更
- URMA actor forward（RL-X 隔離 venv が必要）

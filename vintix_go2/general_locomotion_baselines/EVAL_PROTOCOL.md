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
- GPU 空き後: 同スクリプト `--run`（Go2 Env）→ 公式 play と完走・速度を並べる
- 注意: 観測は 70×30 履歴。Genesis 45 からの再構成は近似

### GenLoco
- 公式: PyBullet `motion_imitation`（pace / spin / inverse zip）
- 本リポ: `python wrappers/genloco/run_genloco_genesis_eval.py --dry_run`
- GPU 空き後: `--run` で Go2 Env 接続。**公式はモーション模倣**なので本線の速度追従とはタスク定義が違う → 比較表に「タスク差」列を残す
- 観測: 451（Motor×15 + IMU6×15 + Act×15 + phase）

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
- `--run` 本体（#18 評価が GPU を占有）
- McARL（コード未収録）
- 本線 Vintix 学習設定の変更

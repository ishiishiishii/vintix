# 評価で保存する指標の対応表（Issue #15）

入力: `metrics_literature_survey.md` / `.csv`（#14）。実装: `Genesis/examples/locomotion/locomotion_metrics.py`、`vintix_go2/scripts/save_vintix.py`、`save_expert.py`。

## 算出可能 → 保存する

| カテゴリ | 指標 | 保存先 | 備考 |
|---|---|---|---|
| Success/Survival | 完走率（1000 step 未転倒） | `*_episodes.csv` の `episode_length` から集計、`*_metrics.json` の `completion_rate`、体スケール `results.csv` | 既存 |
| TTF / episode 長 | episode length (steps) | `*_episodes.csv`、`*_mean_reward.txt`、`*_metrics.json` | 既存 |
| return | episode / step return | `*_episodes.csv`、step CSV、`*_mean_reward.txt`、`*_metrics.json` | 既存 |
| 速度/距離 | avg forward speed, mean distance | `*_mean_reward.txt`（Locomotion）、`*_metrics.json` → `locomotion` | `LocomotionMetrics` |
| 指令追従 | lin/ang vel RMSE、tracking success ±15% | 同上 | 同上 |
| エネルギー | Cost of Transport | 同上 | トルク×角速度の近似 |
| 形態正規化 | Froude number（脚長があるとき） | 同上（任意） | `env_cfg.leg_length_m` が必要 |

## 現状では算出不能（勝手に近似しない）

| 指標 | 理由 | 追加で必要なログ |
|---|---|---|
| Ground impact / foot contact force | 接触力センサ／foot contact ログを評価ループで保存していない | 各 step の contact force / contact flag |
| Torque smoothness δτ の文献定義どおりの正規化 | 生トルク時系列をエピソード単位で保存していない | エピソードごとの τ 系列、または δτ のオンライン集計 |
| Heading deviation（方位誤差）単体 | yaw 指令追従は ang-vel RMSE に含まれるが、見出し角そのものの誤差は未保存 | base yaw vs 目標 heading |
| 実機凍結／歩容破綻判定 | sim のみ。実機プロトコル未定（#19） | 実機ログ定義 |
| CARO 式 Success（15s で 10m） | 本評価は 1000 step 転倒なし定義。距離閾値 Success は別定義 | 明示的な距離閾値フラグを追加する合意が必要 |

## 成果物の互換

- 既存の `*.csv` / `*_episodes.csv` / `*_mean_reward.txt` の列は壊さない。
- 追加: `*_metrics.json`（集計単位の機械可読）。
- エピソード単位の loco 列（速度・距離・CoT）は未実装 → 次の一手。

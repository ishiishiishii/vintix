# データ 10% デコーダ FT：平均累積報酬（10 env × 10 ep）

各列は、**そのロボット用 AD データの 10% のみ**でデコーダのみ FT したうえで、**同じロボット**上で `save_vintix.py` 評価した結果です（チェックポイント: 各 `checkpoints/<model_key>/p10/0001_epoch`）。

評価設定: 10 環境 × 10 エピソード（計 100 エピソード）、各エピソード最大 1000 step。

数値は `eval/<model_key>/p10/*_10envs_10episodes_episodes.csv` の `cumulative_reward` 列（100 件）から計算し、`results.csv` および `*_mean_reward.txt` の **Mean Reward per Episode** と一致することを確認済みです。

---

## 2 × 4 サマリ表

| | Go1 | Go2 | A1 | Minicheetah |
|--|:---:|:---:|:---:|:---:|
| **FT 元モデル（10% データ）** | `go1_without` | `go2_without` | `a1_without` | `minicheetah_without` |
| **平均累積報酬（全 100 エピソード）** | **19.273** | **21.232** | **20.703** | **19.984** |

---

## 参照値（同一評価から）

| ロボット | 平均 | 標準偏差（100 ep） | エピソード数 | ソース |
|----------|-----:|-------------------:|-------------:|--------|
| Go1 | 19.273 | 7.470 | 100 | `eval/go1_without/p10/go1_10envs_10episodes_mean_reward.txt` |
| Go2 | 21.232 | 0.069 | 100 | `eval/go2_without/p10/go2_10envs_10episodes_mean_reward.txt` |
| A1 | 20.703 | 0.121 | 100 | `eval/a1_without/p10/a1_10envs_10episodes_mean_reward.txt` |
| Minicheetah | 19.984 | 2.031 | 100 | `eval/minicheetah_without/p10/minicheetah_10envs_10episodes_mean_reward.txt` |

`results.csv` 上の `mean_cumulative_reward`（10% 行）も上表の平均と同一です。

---

## 効率性の主比較（10% FT vs PPO クロス FT・1 ロボット）

| 指標 | Vintix 10% | PPO クロス FT（300 iter） |
|------|------------|---------------------------|
| **オンライン環境ステップ** | **0** | **2.95×10⁷** |
| **オフラインデータステップ** | **≈ 10⁶**（(10⁶/軌跡×10 軌跡)×10%） | — |

AD 全プール: **10⁷** データステップ/ロボット（収集デフォルト）。学習実装では **~7,800 窓**（全プール ~78k 窓の 10%）。

補助: 2 epoch、1,950 更新、実効バッチ 64 → [`EXPERIMENT.md` §9–§10](./EXPERIMENT.md)

---

## 補足

- 本実験では **クロスロボット評価**（例: `go1_without` を 10% FT して Go2 で評価）は行っていません。表の各セルは **対角（同一ロボット）** のみです。
- Go1 の 10% 条件はエピソード長のばらつきがあり（平均長 ≈ 868 step）、標準偏差が他ロボットより大きいです。ep 4 以降は報酬 ~22 で安定（`eval/go1_without/p10/*_mean_reward.txt`）。

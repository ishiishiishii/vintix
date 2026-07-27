# Online Decoder PPO Fine-Tuning — 実験レポート

> 対象時点: **Go2 安定化スクラッチ完了**。MiniCheetah は即死で未成功。  
> 予定: seed 再試行付きで MiniCheetah → Go1 → A1。  
> レポート形式: [`REPORT_FORMAT.md`](../../REPORT_FORMAT.md)

---

## ゼミ進捗報告（約10分）

### 概要

未知ロボット向けに、AD データなしで **decoder のみを PPO でオンライン FT** できるかを Go2 で検証した。  
初期は不安定だったが、**探索ノイズ縮小＋報酬飽和停止**によりスクラッチからも学習が伸び、Eval でゼロショットを大きく上回った。  
一方 MiniCheetah は同一設定でも即死ループから抜け出せておらず、**即死検知＋シード再試行**を追加して再挑戦する。

### 目的

オフライン decoder FT（対象ロボットの AD が必要）の代替として、**シミュレーション上のオンライン更新だけ**で適応できるか。

### 方法

- 事前学習 Vintix（`*_without`）を actor に使用
- **encoder / transformer は freeze**、decoder（＋critic・action std）のみ PPO
- 正規化は事前学習 metadata（未知タスクは既知タスクの平均）

### 実験内容・設定（Go2）

| 項目 | 値 |
|------|-----|
| モデル | `go2_without` |
| envs | 64 |
| context_len | 2048 |
| seed | 1（主実験） |
| 初期ノイズ | 1.0 → **改善後 0.2** |
| 停止 | ハード上限＋**`--saturate`**（絶対 ε）／即死は ep_len で別検知 |

### 結果（Eval: 10×10、平均エピソード報酬）

| 条件 | Train 付近 | Eval |
|------|-----------:|-----:|
| ゼロショット | — | 0.53 |
| オフライン FT (10% AD) | — | **21.23** |
| オンライン（初期成功 1000） | ~7.7 | **15.48** |
| 失敗スクラッチ（std=1.0） | ~−0.02 | 崩壊 |
| **安定化スクラッチ（std=0.2）** | ~13.3 | **14.76** |

### うまくいかなかったこと → 修正

- 即死ループ（ep_len≈4）で学習が進まないことが多かった  
  → **`init_noise_std=0.2`**、**飽和停止**（大きめ上限で頭打ちまで連続学習）
- 相対％飽和は `|mean|≈0` の即死で発火しない  
  → **絶対 ε（max−min ≤ 0.25、WINDOW=20）**＋成功時は mean≥1.0
- MiniCheetah は Go2 と同設定でも **ep_len≈6 のまま数千 iter**（相対％では止まらず）  
  → **ep_len で即死検知**し、**`--seed_retries`（推奨 3）**で seed を変えて再試行

### 考察・いま言えること

オンライン decoder FT は Go2 では可能。大ノイズだと不安定。小ノイズ＋飽和停止でスクラッチ再現性が改善。オフラインには未到達だが 15 帯まで到達。  
他ロボットは単純移植では足りず、シード再試行などのブートストラップ対策が必要。

### 次の予定

同一安定化設定＋**`--seed_retries 3`** で **MiniCheetah → Go1 → A1**（各ロボットで学習→評価）。

### 参照グラフ（パス）

- [`result/go2_without_noise02_saturate/train_mean_reward.png`](result/go2_without_noise02_saturate/train_mean_reward.png)
- [`result/go2_without_noise02_saturate/eval_iter4006/`](result/go2_without_noise02_saturate/eval_iter4006/)
- [`result/go2_without/train_mean_reward.png`](result/go2_without/train_mean_reward.png)
- [`result/minicheetah_without_noise02_saturate/train_mean_reward.png`](result/minicheetah_without_noise02_saturate/train_mean_reward.png)

---

<details>
<summary><strong>詳細レポート（研究資料）</strong> — クリックで展開</summary>

## 1. 概要

本実験は Vintix leave-one-out 事前学習モデルに対し、**対象ロボットの AD 軌跡を使わず**、Genesis シミュレーション上で **action decoder のみを PPO で更新**するオンライン適応を検証する。Go2 で手法の成立と安定化を確認し、他3ロボットへ展開する。

## 2. 目的

> オフラインではなく、オンラインで decoder fine-tuning は可能か？

比較軸:

1. ゼロショット / in-context（勾配なし）
2. オフライン decoder FT（対象 AD の一部が必要）
3. 本手法: オンライン decoder-only PPO FT

## 3. 方法

### 3.1 モデル

| 部品 | 扱い |
|------|------|
| Encoder + Transformer | freeze |
| `acs_decoders[quadruped_locomotion]` | PPO 更新 |
| Critic | MLP `[512,256,128]` + ELU（ゼロから） |
| Action std | 学習可能 Gaussian（初期値は `init_noise_std`） |

実装: `scripts/vintix_actor_critic.py`, `scripts/train_online_decoder_ppo.py`

### 3.2 学習ループ

SAR 履歴（env ごと）→ backbone で最終特徴（no grad）→ decoder で mean → ノイズ付き行動 → 環境 step → PPO update。

## 4. 実験設定（共通）と設計理由

チャット・実装で議論した項目を、**設定 / 理由 / 考察**で残す。

### 4.1 正規化（観測・行動）

| | |
|--|--|
| **設定** | 未知タスク追加時、事前学習 ckpt の `metadata` にある **既知タスクの `obs_mean/std`, `acs_mean/std` を平均**して使用（`compute_averaged_stats`）。decoder 出力は `unnorm_acs` で物理行動へ戻す。 |
| **理由** | leave-one-out では対象ロボット（例: Go2）の専用 stats が無い。オフライン FT のように対象 AD から stats を取れないため、評価スクリプト（`save_vintix.py`）と同じ平均 stats 方針にした。 |
| **考察** | 平均 stats はゼロショット不安定の一因にもなり得る。それでもオンライン更新で大幅改善できた。オフライン FT（Go2 専用 stats）が更に強い理由の一つでもある。 |

### 4.2 Seed

| | |
|--|--|
| **設定** | 主実験は **`seed=1`**。失敗調査中に **`seed=2`** の短時間試行もあり（停電で中断、序盤は ep_len が伸びる兆し）。即死時は `--seed_retries` で `seed+1, …` を試す。 |
| **理由** | `get_train_cfg(..., seed=...)` 経由で numpy/torch/env 側の乱数を揃える意図。 |
| **考察** | **同じ seed=1 でも成功と失敗が両立**した。GPU 演算の非決定性＋大ノイズにより、「seed 固定＝再現」にはならない。論文では複数 seed の成功率報告が必要。seed を変えるのは早期打ち切り再試行の手段としては有効。 |

### 4.3 探索ノイズ（`init_noise_std`）

| | |
|--|--|
| **設定** | 初期 **1.0**（`train.py` 踏襲）→ 安定化後 **0.2**。ノイズは **非正規化後の行動空間**に乗る。 |
| **理由** | PPO は確率的方策が前提のため **完全ゼロは不可**（logprob / ratio が壊れる）。一方、事前学習済み mean があるので、ゼロから RL するほど大きな探索は不要、という議論。 |
| **考察** | std=1.0 は `acs_std`（関節あたり ~0.8–2.5）に対して過大で、即死を増やしやすい。0.2 は「転ばなくする魔法」ではなく、**寿命のばらつき＝学習信号を残す**ための局所探索。転んでいても PPO は学べるが、**全部が同じように即死だと学べない**。 |

### 4.4 context / sparsity / env 数

| 項目 | 設定 | 理由・考察 |
|------|------|------------|
| `context_len=2048` | ckpt と一致 | 事前学習の履歴長に合わせる。短いと in-context 能力を損なう |
| `trajectory_sparsity=128` | ckpt と一致 | オンラインでは連続 step を保持（評価と同様）。値自体は整合用 |
| `num_envs=64` | 主に使用 | 4096 より軽い。分散は小さめで離陸の当たり外れが出やすい |
| 1 iter | 64×24=1536 steps | `train.py` と同型 |

### 4.5 停止条件（iter 指定＋飽和＋即死）

| | |
|--|--|
| **設定** | `--max_iterations` はハード上限。`--saturate` で直近 **20** iter の **絶対レンジ（max−min）≤ ε=0.25** かつ mean≥1.0 なら成功飽和で早期終了（最低 1000 iter 後）。即死は直近 20 iter の **mean_ep_len < 30** で検知し、`--seed_retries`（推奨 3）で seed を変えて再試行。失敗試行は `failed_seed{S}/` へ退避。 |
| **理由** | 固定 1000 だと伸び切る前に終わりがち。再開すると履歴リセット等で **報酬が一時的に大きく落ち、曲線が不自然**。大きめ上限＋飽和停止なら **一続きの run で頭打ちまで**行ける。相対％は近ゼロ即死で誤作動するため廃止。 |
| **考察** | Go2 安定化ランは旧相対％で **iter 4006 で自動停止**。MiniCheetah は相対％では止まらず即死平坦が継続したため、絶対 ε＋即死／seed 再試行に変更。 |

### 4.6 評価プロトコル

- `save_vintix.py`: **10 envs × 10 episodes**（主評価）
- 主指標: Mean Reward per Episode（累積）
- 参照: zero-shot 0.53 / offline FT 21.23（既存評価）

### 4.7 その他（議論メモ）

- **Critic**: Vintix に value head を付けず独立 MLP（PPO との相性・実装単純さ）
- **報酬 ~20 で頭打ち**: エピソード上限 1000 step に既に近い。これ以上は追従品質。backbone 固定も制約
- **resume 成功 ≠ スクラッチ再現**: 歩ける ckpt から始めただけ、という点を混同しない

## 5. 実験ログ（時系列）

### 実験1: 初期スクラッチ（std=1.0）— 成功と失敗が混在

- **設定**: 64 env, seed=1, std=1.0, ctx=2048
- **結果A（canonical 1000）**: Train ~7.7, Eval **15.48**。序盤から ep_len が伸長
- **結果B（再実行 5000 / 10k 等）**: Train ~−0.02, ep_len≈4 の即死ループ。長時間回しても回復せず
- **問題**: 同一設定でも再現しない。失敗時は `v_loss→0` で学習信号消失
- **考察**: 手法不可ではなく、**ブートストラップの当たり外れ**。大ノイズが失敗モードを増幅

成果物: `models/go2_without_1000_canonical/`, 失敗例 `models/go2_without/`

### 実験2: 成功 ckpt からの延長（上限性能の確認）

- **変更点**: 実験1成功の `model_1000` から resume（+4000 iter → 絶対 5000）
- **理由**: スクラッチ不安定とは切り分け、オンライン FT の性能上限を見る
- **結果**: Train max ~17.5。最良付近 **model_4000（~6.1×10⁶ steps）** で Eval **19.08**（offline 21.23 に接近）
- **考察**: 条件が有利（既に歩行可能）。コード修正の効果ではない。ただし「オンラインで offline 近くまで行ける」証拠にはなる

### 実験3: 安定化スクラッチ（std=0.2 + saturate）← 本命の再現性改善

- **変更点（実験1失敗への対応）**:
  1. `init_noise_std: 1.0 → 0.2`
  2. `--saturate` + `max_iterations=10000`
  3. ディレクトリ名で設定が分かるように `go2_without_noise02_saturate`
- **結果**: スクラッチから ep_len が伸長。Train ~13.3、**iter 4006 で飽和停止**。Eval **14.76**
- **考察**: 即死ループを避け、実験1の「運任せ」を緩和。Eval は canonical@1000（15.48）と同程度、resume 最良（19.08）より低いが、**ゼロからの安定化**としては成功

![noise02 train](result/go2_without_noise02_saturate/train_mean_reward.png)

![noise02 eval](result/go2_without_noise02_saturate/eval_iter4006/go2_10envs_10episodes_episodes.png)

### 実験4: MiniCheetah（同一安定化設定）— 未成功

- **設定**: Go2 実験3と同じ（std=0.2, saturate, seed=1, 上限 10000）
- **結果**: ~5400 iter 時点でも Train reward≈−0.04、**ep_len≈6** の即死平坦。相対％飽和は発火せず
- **対応**: 学習を停止。曲線は `result/minicheetah_without_noise02_saturate/`。以降は **絶対 ε 飽和＋即死検知＋`--seed_retries 3`** で再試行予定

![minicheetah train](result/minicheetah_without_noise02_saturate/train_mean_reward.png)

### 実験5: 実装追加（即死時シード再試行）

- `DEATH_*`: mean_ep_len で即死早期停止
- `SATURATE_MIN_MEAN_REWARD`: 近ゼロ平坦を成功飽和にしない
- `--seed_retries`: 失敗を `failed_seed{S}/` へ退避し seed+1 で subprocess 再起動
- 詳細: `EXPERIMENT.md` §3.3–3.4

## 6. 結果まとめ

### 6.1 表（Eval）

| 条件 | Mean ep. reward | Mean ep. len |
|------|----------------:|-------------:|
| Zero-shot | 0.53 | 短 |
| Offline FT 10% | **21.23** | — |
| Online @1000（実験1成功） | **15.48** | 1000 |
| Online resume 最良 @4000（実験2） | **19.08** | 1000 |
| Online 安定化最終 @4006（実験3） | **14.76** | 990 |
| Online 失敗スクラッチ（実験1失敗） | ~0（train） | ~4 |
| MiniCheetah 安定化（実験4） | 未評価（即死） | ~6（train） |

### 6.2 学習曲線パス

| ラン | パス |
|------|------|
| 失敗 | [`result/go2_without/train_mean_reward.png`](result/go2_without/train_mean_reward.png) |
| canonical | [`result/go2_without_1000_canonical/train_mean_reward.png`](result/go2_without_1000_canonical/train_mean_reward.png) |
| resume | [`result/go2_without_resume5000/train_mean_reward.png`](result/go2_without_resume5000/train_mean_reward.png) |
| 安定化 | [`result/go2_without_noise02_saturate/train_mean_reward.png`](result/go2_without_noise02_saturate/train_mean_reward.png) |
| MiniCheetah | [`result/minicheetah_without_noise02_saturate/train_mean_reward.png`](result/minicheetah_without_noise02_saturate/train_mean_reward.png) |

## 7. 考察

1. **オンライン decoder-only PPO は Go2 で成立**（AD なしでゼロショットを大幅上回る）
2. 事前学習 FT では **探索ノイズのスケールが再現性の鍵**。PPO なのでゼロは不可、小さめが適切
3. 正規化を事前学習の平均 stats に頼るのは一貫した設計だが、offline 専用 stats とのギャップは残る
4. seed 固定だけでは再現保証にならない（GPU 非決定性）→ **即死時の seed 再試行**を実装
5. 飽和停止は曲線の連続性と計算効率の両方に寄与（相対％から絶対 ε へ修正）
6. MiniCheetah は Go2 安定化設定の単純移植では未成功。横断評価と seed 再試行が次の必須材料

## 8. 今後の予定

- [ ] MiniCheetah / Go1 / A1 を実験3と同一設定で **学習→評価**（順次、`--seed_retries 3`）
- [ ] 本レポートに横断表を追記
- [ ] 即死時シード再試行の成功率を記録（`seed_attempts.json`）

オーケストレーション: `run_remaining_noise02_saturate.py`（`SEED_RETRIES=3`）

## 9. 成果物パス

```
online_decoder_ft/
  models/
    go2_without_1000_canonical/
    go2_without/                      # 失敗
    go2_without_resume5000/
    go2_without_noise02_saturate/     # 安定化（本命・スクラッチ）
    minicheetah_without_noise02_saturate/  # 即死で停止済み（再試行予定）
    go1_without_noise02_saturate/
    a1_without_noise02_saturate/
  result/ …
  REPORT.md
  EXPERIMENT.md
  run_remaining_noise02_saturate.py
```

</details>

# Decoder-only ファインチューニング：データ量スイープ実験

**実験ディレクトリ:** `vintix_go2/experience_decoder_ft_data_fraction_20260517`  
**実行日:** 2026-05-17（再実行・`save_vintix` 評価版）  
**ステータス:** 全 44 条件完了（4 モデル × 11 データ割合）

---

## 1. 目的と背景

### 目的

マルチタスク事前学習時に**対象ロボットを含まなかった** Vintix モデル（`*_without`）に対し、**未知ロボットの Algorithm Distillation（AD）データのみ**でデコーダをファインチューニングしたとき、**学習に使うデータ量（10%〜100%）**が歩行性能にどう効くかを定量的に比較する。

### 背景

- Vintix はエンコーダ＋トランスフォーマを固定し、ロボットごとのデコーダで行動を出す構成である。
- 「未知ロボット」では事前学習 ckpt に当該タスクが無く、FT 時にタスク統計の追加とデコーダの学習が必要になる。
- データ効率（少量 AD でどこまで性能が出るか）は実用・論文の両方で重要な論点である。
- 本リポジトリでは過去に単一条件（例: 10% や 50%）のデコーダ FT を個別に行ってきたが、**同一プロトコルで 0%〜100% をスイープした系統的な比較**は本実験が初めてである。
- 初回実行の評価結果は無効だったため、**`save_vintix.py` で全条件を再実行**した（無効結果は `experience_decoder_ft_data_fraction_20260517_backup_broken_eval_20260517` に退避）。

---

## 2. 主な結果（Mean Reward per Episode）

評価指標は `save_vintix.py` が出力する **エピソードあたり平均累積報酬**（10 env × 10 ep、各 ep 最大 1000 step）。  
詳細は `results.csv` および `eval/<model_key>/pXX/` 内の `*_mean_reward.txt` を参照。

| モデル（FT・評価ロボット） | 0%（FT なし） | 10% | 50% | 100% | 傾向の要約 |
|---------------------------|--------------|-----|-----|------|------------|
| Go1 (`go1_without`) | 15.17 | 19.27 | 20.41 | 20.36 | 10% で大幅改善、以降は ~20 前後で横ばい |
| Go2 (`go2_without`) | 0.53 | 21.23 | 21.30 | 21.26 | 0% はほぼ歩行不能、10% 以降は ~21 で飽和 |
| A1 (`a1_without`) | 15.19 | 20.70 | 20.79 | 20.62 | 10% で ~20 到達、以降ほぼ平坦 |
| Minicheetah (`minicheetah_without`) | -0.17 | 19.98 | 20.43 | 20.38 | 0% は転倒級、10% 以降は ~20 で安定 |

**解釈の要点**

- **0%** は各 `*_without` の `0001_epoch` をそのまま評価したベースライン（未知ロボット用デコーダ未適応）。
- Go1 / A1 は 0% でも歩行はするが報酬は ~15（コンテキスト・正規化の影響あり）。Go2 / Minicheetah の 0% は特に低く、**少量 FT の効果が大きく見える**。
- **10% 以降**は Go2 / A1 / Minicheetah でおおよそ性能飽和。**データ量の主効果は 0%→10% 付近**に集中している。
- 全条件ともエピソード長 1000 step 完走（FT 後）が確認でき、評価パイプラインは正常に動作した。

### グラフ

`graphs/` 以下（縦軸 **-5〜28**、色は `ppo_leave_one_out_20260519` と同一。各点は **100 評価エピソードの平均のみ**（σ 帯なし。AD 評価では序盤エピソードのばらつきが大きく、帯が誤解を招くため））:

| ファイル | 用途 |
|----------|------|
| `all_models_data_fraction.png` | **論文用**。凡例 `Finetune Go1` など |
| `all_models_data_fraction_poster.png` | **ポスター用**。凡例なし |

---

## 3. 実験プロトコル（何をしたか）

### 3.1 モデルと未知ロボットの対応

いずれも事前学習チェックポイント **`models/<key>/<key>/0001_epoch`**（epoch 1）を起点とする。

| `model_key` | ベース ckpt | FT・評価ロボット | データセット設定 |
|-------------|-------------|------------------|------------------|
| `go1_without` | `models/go1_without/go1_without/0001_epoch` | go1 | `configs/go1_only_onegroup_config.yaml` |
| `go2_without` | `models/go2_without/go2_without/0001_epoch` | go2 | `configs/go2_only_onegroup_config.yaml` |
| `a1_without` | `models/a1_without/a1_without/0001_epoch` | a1 | `configs/a1_finetune_config.yaml` |
| `minicheetah_without` | `models/minicheetah_without/minicheetah_without/0001_epoch` | minicheetah | `configs/minicheetah_only_onegroup_config.yaml` |

各条件で **常に同じベース ckpt から** FT を開始する（データ割合ごとに独立した run）。

### 3.2 データ量

- 横軸: **0, 10, 20, …, 100（%）**
- **0%:** 追加学習なし。ベース `0001_epoch` を直接評価。
- **10%〜100%:** `train_vintix.py` の `--random_sample_frac`（0.1〜1.0）で AD 軌道全体からランダムサブサンプル（`seed=5` 固定）。

### 3.3 ファインチューニング

- **方式:** デコーダのみ（`--finetune_decoder_only true`）
- **エポック数:** **2**（`--epochs 2`, `--save_every 1`）
- **評価に使う重み:** 各 run の **`0001_epoch`**（2 epoch 終了時点）。`0000_epoch` は保存されるが本実験の集計には未使用。
- **出力先:** `checkpoints/<model_key>/p10/` … `p100/`（0% は `models/` 直下のベース ckpt のみ）

### 3.4 評価

- **スクリプト:** `scripts/save_vintix.py`
- **設定:** `--num_envs 10 --max_episodes 10`（並列 10 環境、各 10 エピソード）
- **正規化:** FT 時に追加したタスク統計、または事前学習済みタスクの `metadata.json`（FT サブセット統計は使わない）
- **結果コピー:** `eval/<model_key>/pXX/` に mean_reward.txt / episodes.csv 等を保存

---

## 4. なぜ追加訓練は 2 エポックか

本実験では **全データ割合で一律 2 epoch** とした。論文・レビュー向けの整理:

| 観点 | 2 epoch を採用する理由 |
|------|------------------------|
| **公平な比較** | 0%〜100% を「同一の学習予算」で比較するため。割合ごとに epoch を変えると、性能差がデータ量なのか訓練量なのか切り分けられない。 |
| **デコーダ適応の十分性** | デコーダのみ FT でも未知タスクの追加・統計計算があり、1 epoch では未収束の可能性がある。過去の単発 FT（1 epoch・`0000_epoch` 評価）では Minicheetah 等で不安定な例があった。 |
| **リポジトリ内の慣行** | 既存の go1 10% デコーダ FT 参照（`0001_epoch`）などと整合する。 |
| **計算コスト** | 学習可能パラメータは全体の約 16%（デコーダのみ）であり、2 epoch でも事前学習比では軽量。 |

**「なぜ 1 epoch では？」への答え方（例）**

- 本研究の主眼は **データ量の効果** であり、訓練 epoch は **固定のコントロール変数** として 2 に設定した。
- 1 epoch は「最小更新」として魅力的だが、**適応不足で低く出るリスク**があり、特に Minicheetah のような難条件でデータ量曲線が歪む可能性がある。
- 必要なら `0000_epoch` と `0001_epoch` の両方が残っているため、**1 epoch 相当の事後解析**は可能（本 README の主表は `0001_epoch`）。

---

## 5. 使用スクリプトとファイル

| 役割 | パス |
|------|------|
| オーケストレーション | `scripts/run_decoder_ft_data_fraction_experiment.py` |
| 学習 | `scripts/train_vintix.py` |
| 評価 | `scripts/save_vintix.py` |
| 実験設定の記録 | `experiment_config.json`（本 README と対応） |
| 数値結果 | `results.csv` |
| 進捗サマリ | `summary.json` |
| 実行ログ | `run.log` |

---

## 6. 実行方法（再現）

Docker コンテナ `genesis` 内で、`vintix_go2` をカレントに実行する。

```bash
cd /path/to/genesis_project
docker compose exec -T genesis bash -lc \
  'cd /workspace/vintix_go2 && PYTHONPATH=/workspace/vintix_go2 WANDB_MODE=disabled \
   python -u scripts/run_decoder_ft_data_fraction_experiment.py \
   --exp_root /workspace/vintix_go2/experience_decoder_ft_data_fraction_20260517'
```

**主な CLI オプション（オーケストレータ）**

- `--exp_root` — 出力ルート（省略時は `experience_decoder_ft_data_fraction_YYYYMMDD`）
- `--models` — 例: `go1_without,go2_without`（省略時は 4 モデルすべて）
- `--fractions` — 例: `0,10,20`（省略時は 0,10,…,100）
- `--dry-run` — コマンド表示のみ
- `--skip-train` / `--skip-eval` — 再評価・再プロット用

完了済み行は `results.csv` の `status=ok` を見てスキップするため、**中断後の再開が可能**。

**グラフのみ再生成**

```bash
cd /workspace/vintix_go2
python3 -c "
from pathlib import Path
from scripts.run_decoder_ft_data_fraction_experiment import plot_graphs, load_results_csv
exp = Path('experience_decoder_ft_data_fraction_20260517')
plot_graphs(exp, load_results_csv(exp / 'results.csv'))
"
```

---

## 7. 学習ハイパーパラメータ一覧

`experiment_config.json` の `train_hparams` と同一。

| 項目 | 値 |
|------|-----|
| `context_len` | 2048 |
| `trajectory_sparsity` | 128 |
| `lr` | 3e-4 |
| `betas` | (0.9, 0.99) |
| `weight_decay` | 0.1 |
| `precision` | bf16 |
| `grad_accum_steps` | 8 |
| `warmup_ratio` | 0.005 |
| `batch_size` | 8 |
| `seed` | 5 |
| `epochs` | **2** |
| `save_every` | 1 |
| `data_dir` | `data` |
| `stats_path` | `vintix/stats.json` |
| W&B | `WANDB_MODE=disabled` |

---

## 8. ディレクトリ構成

```
experience_decoder_ft_data_fraction_20260517/
├── EXPERIMENT.md          # 本ファイル
├── experiment_config.json
├── results.csv            # Git 追跡（全条件の数値）
├── summary.json
├── run.log                # 非追跡（大容量ログ）
├── checkpoints/           # 非追跡（FT 済み重み）
│   └── <model_key>/p10|p20|…|p100/
│       ├── 0000_epoch/
│       └── 0001_epoch/    # ← 評価に使用
├── eval/                  # *_mean_reward.txt のみ Git 追跡（CSV は非追跡）
│   └── <model_key>/p00|p10|…/
├── readable_comparisons/  # 0% vs 10% の readable 重ね合わせ PNG（§8.1）
└── graphs/                # Git 追跡（データ量 vs 報酬プロット）
```

### 8.1 0% vs 10% 比較グラフ（readable）

`scripts/readable_eval_graphs/comparisons/` にあった形式と同様に、**本実験の評価 CSV**（`eval/<model>/p00` = 追加学習なし、`p10` = 10% デコーダ FT）から 4 ロボット分の重ね合わせ PNG を生成する。

```bash
cd /workspace/vintix_go2
python3 scripts/generate_readable_finetune_comparison.py \
  --exp-root experience_decoder_ft_data_fraction_20260517
```

出力（縦軸 **-5〜28**、10% 曲線の色はロボット色と一致）:

| ファイル | 用途 |
|----------|------|
| `<model_key>/{robot}_without_vs_finetune_readable.png` | 0% vs 10%（凡例あり。エピソードごとの平均±σ は AD のウォームアップ説明用） |

旧実験の `go1_without_epoch1_0p1data_finetune` 等の ckpt は使わない。

---

## 9. 10% デコーダ FT の学習コスト（安定歩行に必要な量）

10% 条件で **同一ロボット上で安定歩行（報酬 ~20、多くは 1000 step 完走）** が得られたときの、オフライン学習コストを `run.log` の学習ログから整理する。数値の一次ソースは各 `checkpoints/<model_key>/p10/` の `train_vintix.py` 出力（`run.log` 内の `Training configuration:` ブロック）である。

### 9.1 用語（「ステップ」の区別）

| 用語 | 本実験での意味 |
|------|----------------|
| **学習サンプル（コンテキスト窓）** | HDF5 上の専門家軌跡から切り出した 1 件の訓練例。`train_vintix.py` がカウントする `Dataset size` はこの件数（**シミュレータを回す環境ステップではない**）。 |
| **1 サンプルが含む履歴** | 最大 `context_len + 1 = 2049` 遷移の連続区間（`trajectory_sparsity=128` は窓の**開始位置**を 128 遷移ごとに間引くインデックス用）。 |
| **エポック内ステップ数** | DataLoader の 1 epoch あたりイテレーション数 = `ceil(サンプル数 / batch_size)`。ログ上の `Steps per epoch`。 |
| **パラメータ更新回数** | 勾配蓄積後に `optimizer.step()` が走った回数。ログ上の `Total updates`（= `Steps per epoch × epochs`）。 |
| **FT 中のオンライン環境ステップ** | **0**（記録済み AD データのみ。PPO ベースラインとはここが最も異なる）。 |

### 9.2 ロボット別：10% で使ったデータ量

| ロボット | 全 AD プール（100%） | 10% サンプル数 | Steps/epoch | Epochs | **Total updates** |
|----------|---------------------:|---------------:|------------:|-------:|------------------:|
| Go1 | 78,003 | **7,800** | 975 | 2 | **1,950** |
| Go2 | 78,011 | **7,801** | 975 | 2 | **1,950** |
| A1 | 78,016 | **7,801** | 975 | 2 | **1,950** |
| Minicheetah | 78,025 | **7,802** | 975 | 2 | **1,950** |

- サブサンプリング: `--random_sample_frac 0.1`、`seed=5` 固定（再現性のため全ロボット同一シード）。
- **2 epoch 通算**で各サンプルは理論上 **2 回** DataLoader を通る → 窓の提示回数は **約 1.56×10⁴**（7.8×10³ × 2）/ ロボット。

### 9.3 更新・バッチ・学習可能パラメータ（共通）

| 項目 | 値 | 備考 |
|------|-----|------|
| `batch_size` | 8 | マイクロバッチ |
| `grad_accum_steps` | 8 | 8 マイクロバッチごとに 1 回 `optimizer.step()` |
| **実効バッチサイズ** | **64** サンプル / update | 8 × 8 |
| **マイクロバッチ forward 回数** | **15,600** / ロボット | 1,950 updates × 8 |
| `lr` | 3×10⁻⁴ | Adam β=(0.9, 0.99), weight_decay=0.1 |
| **学習可能パラメータ** | **200,460 / 1,231,706（16.27%）** | デコーダ `quadruped_locomotion` のみ |
| 評価 ckpt | `0001_epoch` | 2 epoch 終了時点 |

### 9.4 学習効率のための補助指標

| 指標 | 10% FT（1 ロボットあたり） |
|------|---------------------------|
| オフライン学習サンプル | ~7.8×10³ 窓 |
| 勾配更新回数 | 1,950 |
| オンライン環境ステップ | 0 |
| GPU 学習時間（参考） | 1 epoch ≈ **68 s** → 2 epoch ≈ **2.3 min**（`run.log` の tqdm、Go1 p10 代表） |
| 評価コスト | `save_vintix` 10 env × 10 ep × 1000 step cap（FT とは別計測） |

**データ効率（本実験内）:** 全プールの **約 10%** の窓だけで、0% ベースラインから **+4〜+21** ポイントのエピソード報酬改善（§2 表）。特に Go2 / Minicheetah は 0% が転倒級（平均エピソード長 **45 / 28 step**）→ 10% で **~1000 step** 完走に近づく。

---

## 10. PPO クロスロボット FT ベースラインとの比較

対照実験: [`ppo_cross_robot_ft_20260517/EXPERIMENT.md`](../ppo_cross_robot_ft_20260517/EXPERIMENT.md)（4 ソース × 3 ターゲット × 10 シード、各 run **300 PPO iter**）。

### 10.1 比較の前提（解釈上の注意）

| | Vintix 10% デコーダ FT（本実験） | PPO クロス FT（ベースライン） |
|--|----------------------------------|-------------------------------|
| 初期重み | `*_without` の `0001_epoch`（対象ロボット未学習） | **別ロボット**の PPO 専門家 `model_300.pt` |
| 学習データ | **対象ロボット**の AD 軌跡（10%） | ターゲット環境での **オンライン** ロールアウト |
| 評価 | `save_vintix.py`（10×10 ep） | 学習中 `Train/mean_reward`（完了エピソードの移動平均） |
| 指標の一致 | 厳密には異なるログだが、いずれも **~20 前後が「歩行できている」目安** として横並び参考にする |

本節は「**同じターゲットロボットに適応させる**」という目的での **データ効率・安定性のオーダー感** を示す。厳密な同一指標・同一初期条件の対戦ではない。

### 10.2 性能と安定性（ターゲットロボット別）

**Vintix:** 対角評価（10% FT → 同一ロボット）。**PPO:** 3 ソースのうち iter 300 時 `Train/mean_reward` が最大の組み合わせ（10 シード平均）。

| ターゲット | Vintix 0%（FT なし） | Vintix **10%** | 最良 PPO クロス @ iter 300 | 最良ソース |
|------------|---------------------:|---------------:|---------------------------:|------------|
| Go1 | 15.17（ep 長 1000、σ≈1.07） | **19.27**（σ≈7.43、平均 ep 長 ≈868） | **19.63**（σ≈0.14） | go2 → go1 |
| Go2 | 0.53（ep 長 **45**） | **21.23**（σ≈0.07、ep 長 **1000**） | 18.51（σ≈0.48） | go1 → go2 |
| A1 | 15.19 | **20.70**（σ≈0.12） | 19.63（σ≈0.12） | go1 → a1 |
| Minicheetah | -0.17（ep 長 **28**） | **19.98**（σ≈2.02、ep 長 ≈990） | 18.05（σ≈0.17） | go1 → minicheetah |

**安定性の読み取り（評価 100 ep ベース）:**

- **Go2 / A1 @ 10%:** 平均エピソード長 **1000 step**、報酬 σ **< 0.13** と非常に安定。
- **Minicheetah @ 10%:** 1 ep 目以外は σ **< 0.12**、ほぼ全 ep で ~20 報酬（ep 1 のみ転倒あり得る）。
- **Go1 @ 10%:** コンテキストウォームアップの影響で ep 1–2 はばらつき大。ep 4 以降は平均 **~22.3**、σ **< 0.2**（`eval/go1_without/p10/*_mean_reward.txt` の per-episode 表）。
- **PPO:** Go2→Minicheetah（1.58）、Minicheetah→Go1（≈0）など、**クロス FT でも学習不能に近い組**があり、ソース依存が強い。Vintix 10% は **対象ロボット AD のみ**で 4 体とも ~20 前後に揃う。

### 10.3 効率性の主比較：オンライン環境ステップ vs オフラインデータステップ

異なる学習方式を比べるときは、**パラメータ更新回数ではなく、相互作用した環境ステップ（オンライン）と、学習に用いた記録データのステップ数（オフライン）** を主指標にするのが妥当である（§9.1 の窓数・更新回数は補助）。

#### オフライン側の前提（AD データ収集）

専門家軌跡は `scripts/collect_ad_data_parallel.py` のデフォルトに従い、ロボットごとに次で収集する（`data/<robot>_trajectories/` 配下の HDF5）。

| 項目 | 値 |
|------|-----|
| 軌跡本数 | **10**（`--num_envs 10`） |
| 1 軌跡あたりの収集ステップ | **1,000,000**（`--target_steps_per_env`） |
| **全プール（100%）** | **10 × 10⁶ = 10⁷** データステップ / ロボット |

`train_vintix.py` はこの HDF5 から `trajectory_sparsity=128` で学習窓を切り出し、全プールで **約 7.8×10⁴ 窓**（Go1 で 78,003 窓）になる。窓数 × 128 ≈ 10⁷ となり、上記 1000 万 step と整合する。

#### 10% FT で実際に使った量

| 指標 | 値 | 意味 |
|------|-----|------|
| **オフラインデータステップ（主指標）** | **≈ 10⁶** / ロボット | 全プール **10⁷ step の 10%**。計算式: **(10⁶ step/軌跡 × 10 軌跡) × 10% = 10⁶** |
| 学習窓数（実装） | **~7,800** / ロボット | `--random_sample_frac 0.1` により **78,003 窓の 10%** をサブサンプル（重複窓があるため step 数と 1:1 ではないが、**プールの 10% と同オーダー**） |
| FT 中のオンライン環境ステップ | **0** | シミュレータは回さない |

**ご質問の式について:** 「100万ステップ × 10 軌跡 × 10%」は、**(100万/軌跡 × 10 軌跡) × 10% = 100万データステップ** という読み方なら **正しい**。全プール 1000 万 step の 10% が FT に使うオフライン量の目安である。

#### オンライン側（PPO クロス FT・1 run）

| 指標 | 値 |
|------|-----|
| **オンライン環境ステップ（主指標）** | **2.949×10⁷** |
| 内訳 | 4096 env × 24 step/iter × **300** iter |

#### 主比較表（適応フェーズ・1 ターゲットロボットあたり）

| | Vintix デコーダ FT **10%** | PPO クロス FT **300 iter** |
|--|---------------------------|----------------------------|
| **オンライン環境ステップ** | **0**（学習中） | **2.95×10⁷** |
| **オフラインデータステップ** | **≈ 10⁶**（10⁷ プールの 10%） | —（学習データとしては未使用） |
| 比率（オンライン / オフライン） | — | **約 30 倍**（2.95×10⁷ / 10⁶） |

- Vintix は **適応時にオンライン step を一切使わない**一方、PPO クロス FT は **約 3000 万 step** のターゲット環境ロールアウトが必要。
- オフライン **10⁶ step** は、**既に収集済みの AD プール**からの利用量。端到端で「データ収集から」数える場合は、別途 **収集コスト 10⁷ step/ロボット**（専門家ポリシーで 1 回収集）を加える（PPO 側もソース専門家学習は別コスト）。

#### 補助指標（参考のみ）

| | Vintix 10% | PPO クロス FT |
|--|------------|---------------|
| ポリシー更新回数 | 1,950 | 300 iter |
| GPU 学習時間 | ~2–3 min / ロボット | 1 run 数十分〜数時間級 |

**効率のまとめ:**

1. **主指標:** 同程度の歩行性能（§10.2）に対し、適応のオンラインコストは **PPO ≈ 3×10⁷ step**、Vintix 10% は **オフライン ≈ 10⁶ step・オンライン 0**。
2. **対 PPO クロス:** Go2 / A1 / Minicheetah では Vintix 10% が同等以上の報酬。Go1 は最良 PPO クロスと同程度で、評価 ep 序盤の分散は Vintix 側が大きい。
3. **安定性:** Vintix は対象ロボット AD に限定しソース依存の失敗がない。PPO は組み合わせにより iter 300 でも報酬 **< 2** のケースがある。

詳細な PPO 数値・学習曲線は [`ppo_cross_robot_ft_20260517/`](../ppo_cross_robot_ft_20260517/) の `results.csv` / `graphs/` を参照。

---

## 11. 注意事項・既知の比較

- **Go1 / A1 の 0% 報酬（~15）** は、FT 後（~20）より低い。エピソード間のコンテキスト再利用や正規化の影響があり、絶対値の解釈は同一モデル内の **0% vs FT 後** の比較を優先すること。
- **過去の単発実験**（例: `minicheetah_without_epoch0_0p5data_finetune` は `0000_epoch` 起点・1 epoch 相当の評価）とはプロトコルが異なるため、数値の直接比較は避けること。
- 無効だった初回結果は **`_backup_broken_eval_20260517`** にあり、本ディレクトリの `results.csv` が有効な結果である。

---

## 12. 変更履歴

| 日付 | 内容 |
|------|------|
| 2026-05-17 | `save_vintix` 評価で全条件再実行完了。グラフ凡例をロボット名のみに簡略化。本 README 作成。 |
| 2026-05-23 | 評価は `save_vintix.py` に統一。§9–§10 学習コスト・PPO 比較。データ割合グラフは平均のみ（論文/ポスター各1枚）。 |

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

`graphs/` 以下:

- `all_models_data_fraction.png` / `.pdf` — 4 モデル重ね合わせ
- `<model_key>_data_fraction.png` / `.pdf` — モデル別（凡例はロボット名のみ: Go1, Go2, A1, Minicheetah）

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
└── graphs/                # Git 追跡（データ量 vs 報酬プロット）
```

---

## 9. 注意事項・既知の比較

- **Go1 / A1 の 0% 報酬（~15）** は、FT 後（~20）より低い。エピソード間のコンテキスト再利用や正規化の影響があり、絶対値の解釈は同一モデル内の **0% vs FT 後** の比較を優先すること。
- **過去の単発実験**（例: `minicheetah_without_epoch0_0p5data_finetune` は `0000_epoch` 起点・1 epoch 相当の評価）とはプロトコルが異なるため、数値の直接比較は避けること。
- 無効だった初回結果は **`_backup_broken_eval_20260517`** にあり、本ディレクトリの `results.csv` が有効な結果である。

---

## 10. 変更履歴

| 日付 | 内容 |
|------|------|
| 2026-05-17 | `save_vintix` 評価で全条件再実行完了。グラフ凡例をロボット名のみに簡略化。本 README 作成。 |
| 2026-05-23 | 評価は `save_vintix.py` に統一（リポジトリの公式評価スクリプト）。 |

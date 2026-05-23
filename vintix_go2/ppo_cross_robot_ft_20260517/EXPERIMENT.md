# クロスロボット PPO ファインチューニング実験

**実験ディレクトリ:** `vintix_go2/ppo_cross_robot_ft_20260517`  
**実行日:** 2026-05-17〜2026-05-18（Docker コンテナ `genesis_exp` 内）  
**ステータス:** 全 120 run 完了（4 ソース × 3 ターゲット × 10 シード）

---

## 1. 目的と背景

### 目的

Vintix のアルゴリズム蒸留（AD）による適応と対照として、**PPO をそのまま別ロボット環境でファインチューニング**したときの学習曲線を定量的に可視化する。各四足ロボットで 300 イテレーションまで学習した専門家ポリシーを初期値とし、**他 3 ロボットへ 300 イテレーション追加学習**したときの `Train/mean_reward` の推移を、10 シード平均 ± 標準偏差で比較する。

### 背景

- 本リポジトリでは Go1 / Go2 / A1 / MiniCheetah について、Genesis 上で PPO + MLP の**歩行専門家**（`Genesis/logs/<robot>-walking/model_300.pt`）を既に学習済みである。
- Vintix 側ではクロスロボット適応を AD データとデコーダ FT で調査してきたが、**素の PPO FT がどれだけ機能するか**は別問題である。
- 同一 MLP 構造（観測 45 次元 → 行動 12 次元）を共有するため、チェックポイントの重みをそのまま載せ替えて FT できるが、ダイナミクス・報酬スケールの違いにより転移性能はロボットの組み合わせに強く依存する。
- 本実験は **12 通りのソース→ターゲット × 10 ランダムシード** を同一プロトコルで走らせ、再現性のある学習曲線を得ることを目的とする。

---

## 2. 主な結果

### 指標

- **報酬:** TensorBoard スカラー `Train/mean_reward`（学習ループ内の完了エピソード報酬の移動平均）
- **集計:** 各 (ソース, ターゲット) について 10 シードの平均・標準偏差
- **イテレーション 300 時点**（追加 FT 300 ステップ後）の代表値:

| Source → Target | Mean @ iter 300 | Std（シード間） | 備考 |
|-----------------|-----------------|-----------------|------|
| go1 → go2 | 18.51 | 0.48 | 良好 |
| go1 → a1 | 19.63 | 0.12 | 良好 |
| go1 → minicheetah | 18.05 | 0.17 | 良好 |
| go2 → go1 | 19.63 | 0.14 | 良好 |
| go2 → a1 | 19.16 | 1.03 | 良好 |
| go2 → minicheetah | **1.58** | 0.43 | **転移困難** |
| a1 → go1 | 12.44 | 0.96 | 中程度 |
| a1 → go2 | 13.20 | 1.05 | 中程度 |
| a1 → minicheetah | 14.66 | 0.35 | 中程度 |
| minicheetah → go1 | **≈ 0.00** | ≈ 0.00 | **ほぼ学習不能** |
| minicheetah → go2 | 17.13 | 0.78 | 良好 |
| minicheetah → a1 | 14.06 | 1.40 | 中程度 |

**イテレーション 0（FT 直後）** は全組み合わせでおおよそ **-0.25 〜 -0.04** と低く、ソース専門家を載せ替えた直後はターゲット環境ではまだ歩行報酬が出ていない。300 イテレ後に上表の差が開く。

### 解釈の要点

- **Go1 / Go2 をソース**にした FT は、多くのターゲットで iter 300 時に **18〜20 前後**まで到達し、PPO によるクロスロボット適応が比較的有效。
- **Go2 → MiniCheetah**、**MiniCheetah → Go1** は顕著に低く、単純な重み載せ替え + 300 iter PPO では不十分な組み合わせがある。
- Vintix AD 実験（`experience_decoder_ft_data_fraction_20260517/`）との対比用ベースラインとして、本結果の学習曲線（`graphs/`）を参照する。

### 提案手法（10% デコーダ FT）とのデータ量比較（参考）

| | PPO クロス FT（1 run） | Vintix デコーダ FT 10% |
|--|------------------------|-------------------------|
| 学習方式 | オンライン PPO（シミュレータ内） | オフライン（記録済み AD 軌道） |
| **オンライン環境ステップ**（主指標） | **約 2.95×10⁷**（4096×24×300） | FT 中 **0** |
| **オフラインデータステップ**（主指標） | — | **約 10⁶** / ロボット（AD 全プール 10⁷×10%、収集は 10 軌跡×10⁶ step/軌跡） |
| 学習更新回数（補助） | 300 PPO iter | **1,950**（975 step/epoch × 2 epoch） |

Vintix 10% の学習コスト・性能・安定性の詳細比較は [`experience_decoder_ft_data_fraction_20260517/EXPERIMENT.md` §9–§10](../experience_decoder_ft_data_fraction_20260517/EXPERIMENT.md) および [`TABLE_10PCT_MEAN_REWARD.md`](../experience_decoder_ft_data_fraction_20260517/TABLE_10PCT_MEAN_REWARD.md) を参照。

### グラフ

`graphs/` 以下（readable スタイル、`ylim = [-5, 28]`）:

| ファイル | 内容 |
|----------|------|
| `source_<robot>_mean_reward_vs_iteration.png` | 横軸: PPO イテレーション 0〜300 |
| `source_<robot>_mean_reward_vs_env_steps.png` | 横軸: 環境ステップ数（`iter × 4096 × 24`） |

各図に **3 本の曲線**（同一ソースからの 3 ターゲット FT）。実線 = 10 シード平均、塗り = ±1 標準偏差。

---

## 3. 実験プロトコル（何をしたか）

### 3.1 ロボットと組み合わせ

- **ロボット:** `go1`, `go2`, `a1`, `minicheetah`（CLI・ログ名はすべて `a1` に統一）
- **ソース（初期チェックポイント）:** 各 `Genesis/logs/<robot>-walking/model_300.pt`
- **ターゲット:** ソースと異なる残り 3 ロボット
- **Run 数:** 4 × 3 × 10 = **120**
- **Run ID 例:** `ft_go1_to_go2_seed01`

### 3.2 学習

- **スクリプト:** `Genesis/examples/locomotion/train.py`（`train_with_history.py` は使用しない。軌跡 HDF5 は保存しない）
- **方式:** `--pretrained_path` でソース専門家をロード → ターゲット環境で PPO 継続
- **イテレーション:** `max_iterations=301`（**0〜300** を 1 run。打ち切り 300）
- **外部 ckpt ロード時:** `train.py` で `current_learning_iteration` を **0 にリセット**（横軸を FT 0 起点にする。**重みは専門家のまま**、カウンタのみ 0）
- **並列環境数:** 4096
- **ドメインランダマイゼーション:** なし
- **シード:** 1〜10（run ごとに `--seed` を変更）
- **実行:** 120 run を **直列・1 プロセス**ずつ

### 3.3 ログと曲線抽出

- 各 run の TensorBoard / チェックポイント: `runs/ft_<src>_to_<tgt>_seed<NN>/`
- 完了後、`Train/mean_reward` を `curves/ft_<src>_to_<tgt>_seed<NN>.csv` に抽出
- CSV 列: `iteration`, `env_steps`, `train_mean_reward`

### 3.4 可視化

- オーケストレータが 120 run 完了後に `graphs/` を自動生成
- 体裁は `generate_readable_finetune_comparison.py` / `save_vintix.py` と同系（大フォント、題名なし PNG）

---

## 4. 使用スクリプトとファイル

| 役割 | パス |
|------|------|
| オーケストレーション | `scripts/run_ppo_cross_robot_ft_experiment.py` |
| PPO 学習 | `Genesis/examples/locomotion/train.py` |
| 実験設定 | `experiment_config.json` |
| 進捗・結果一覧 | `results.csv` |
| 完了サマリ | `summary.json` |
| 実行ログ（Docker） | `run_experiment_docker.log` |
| リソースチェック（初回ホスト実行時） | `resource_check.json` |

### ソース専門家チェックポイント

| ロボット | パス（コンテナ内） | パス（ホスト） |
|----------|-------------------|----------------|
| go1 | `/workspace/Genesis/logs/go1-walking/model_300.pt` | `Genesis/logs/go1-walking/model_300.pt` |
| go2 | `/workspace/Genesis/logs/go2-walking/model_300.pt` | `Genesis/logs/go2-walking/model_300.pt` |
| a1 | `/workspace/Genesis/logs/a1-walking/model_300.pt` | `Genesis/logs/a1-walking/model_300.pt` |
| minicheetah | `/workspace/Genesis/logs/minicheetah-walking/model_300.pt` | `Genesis/logs/minicheetah-walking/model_300.pt` |

---

## 5. 実行方法（再現）

**Genesis は Docker イメージ内にのみインストールされている。** ホストの `python3` では `import genesis` が失敗する。

### 一括実行（学習 + 曲線抽出 + グラフ）

```bash
cd /path/to/genesis_project
docker compose exec -d genesis bash -lc \
  'cd /workspace/vintix_go2 && \
   pip install -q tensorboard && \
   nohup python3 -u scripts/run_ppo_cross_robot_ft_experiment.py \
     --exp_root ppo_cross_robot_ft_20260517 \
     >> ppo_cross_robot_ft_20260517/run_experiment_docker.log 2>&1 &'
```

対話シェルでフォアグラウンド実行する場合:

```bash
docker compose exec genesis bash
cd /workspace/vintix_go2
pip install tensorboard   # 初回のみ
python -u scripts/run_ppo_cross_robot_ft_experiment.py \
  --exp_root ppo_cross_robot_ft_20260517
```

### 主な CLI オプション

| オプション | 説明 |
|------------|------|
| `--exp_root` | 出力ルート（省略時 `ppo_cross_robot_ft_YYYYMMDD`） |
| `--plot_only` | `curves/` から `graphs/` のみ再生成 |
| `--skip_train` | 学習をスキップ（曲線抽出・プロットのみ） |
| `--dry_run` | コマンド表示のみ |
| `--force_run` | GPU 使用率チェックを無視して学習開始 |

完了済み run は `results.csv` の `status=ok` と `curves/*.csv` の存在でスキップされるため、**中断後の再開が可能**。

### グラフのみ再生成

```bash
docker compose exec genesis bash -lc \
  'cd /workspace/vintix_go2 && python3 scripts/run_ppo_cross_robot_ft_experiment.py \
     --exp_root ppo_cross_robot_ft_20260517 --plot_only'
```

### 進捗確認

```bash
tail -f vintix_go2/ppo_cross_robot_ft_20260517/run_experiment_docker.log
wc -l vintix_go2/ppo_cross_robot_ft_20260517/results.csv   # 完了 run 数 + 1
```

---

## 6. PPO ハイパーパラメータ

`Genesis/examples/locomotion/train.py` の `get_train_cfg` 既定値（全 run 共通）。

| 項目 | 値 |
|------|-----|
| アルゴリズム | PPO |
| `num_envs` | 4096 |
| `num_steps_per_env` | 24 |
| `max_iterations` | 301（iter 0〜300） |
| `learning_rate` | 0.001 |
| `clip_param` | 0.2 |
| `entropy_coef` | 0.01 |
| `num_learning_epochs` | 5 |
| `num_mini_batches` | 4 |
| Actor / Critic MLP | [512, 256, 128], ELU |
| `save_interval` | 100 |
| ドメインランダマイゼーション | 無効 |
| シード | 1〜10（run ごと） |

---

## 7. ディレクトリ構成

```
ppo_cross_robot_ft_20260517/
├── EXPERIMENT.md              # 本ファイル
├── experiment_config.json
├── results.csv                # 120 run の status / パス
├── summary.json
├── run_experiment_docker.log  # 実行ログ（有効）
├── run_experiment.log         # ホスト初回試行（失敗分）
├── resource_check.json
├── runs/                      # 各 run の TensorBoard・model_*.pt
│   └── ft_<src>_to_<tgt>_seed<NN>/
├── curves/                    # 抽出済み学習曲線 CSV（120 本）
│   └── ft_<src>_to_<tgt>_seed<NN>.csv
└── graphs/                    # ソース別比較プロット（8 PNG）
    ├── source_go1_mean_reward_vs_iteration.png
    ├── source_go1_mean_reward_vs_env_steps.png
    └── …
```

---

## 8. 実装上の注意（再現時）

1. **イテレーション 0 リセット:** 2026-05-18 に `train.py` を修正。外部 `model_300.pt` ロード後、`current_learning_iteration` を 0 にしないと ckpt 内の iter=300 が引き継がれ **601 iter まで走る**不具合があった。重みは `runner.load()` で専門家のまま、カウンタのみ 0 に戻す。
2. **Docker 必須:** ホスト Python では Genesis 未インストール。必ず `genesis_exp`（または `docker compose exec genesis`）内で実行すること。
3. **tensorboard:** 曲線抽出に `pip install tensorboard` がコンテナ内で必要（初回のみ）。
4. **報酬の意味:** `Train/mean_reward` は学習中のオンライン平均であり、`save_vintix.py` の評価用固定ポリシー報酬とは定義が異なる。Vintix 実験との数値比較は指標を揃えて解釈すること。

---

## 9. 関連実験

| 実験 | ディレクトリ | 関係 |
|------|-------------|------|
| Vintix デコーダ FT データ量スイープ | `experience_decoder_ft_data_fraction_20260517/` | AD + デコーダ FT の対照 |
| PPO 履歴付きクロス FT | `train_with_history.py` `--run_all_cross_ft` | 軌跡記録あり・シード 1 の旧プロトコル |

---

## 10. 変更履歴

| 日付 | 内容 |
|------|------|
| 2026-05-17 | 実験スクリプト作成。ホスト実行は Genesis 未導入で失敗。 |
| 2026-05-18 | Docker 内で実行。`train.py` に外部 FT 時の iter リセットを追加。全 120 run 完了・グラフ生成。 |
| 2026-05-19 | 本 README 再作成（Undo 復旧）。 |

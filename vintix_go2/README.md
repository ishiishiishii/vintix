# Vintix for Genesis Go2

Genesis環境でVintixのAlgorithm Distillation手法を四足歩行ロボットGo2に適用した実装です。

## 概要

このプロジェクトは、[Vintix: Action Model via In-Context Reinforcement Learning](https://arxiv.org/abs/2501.19400)の手法を、Genesis物理シミュレーション環境の四足歩行ロボットGo2に適用したものです。Algorithm Distillationによって、ロボットが異なるドメインや条件下で適応的に歩行を学習できるようにします。

### 主な特徴

- **Algorithm Distillation**: トランスフォーマーベースのモデルで、観察・行動・報酬のシーケンスから次の行動を予測
- **In-Context Learning**: 新しいタスクやドメインに推論時に適応可能
- **Genesis環境統合**: 高速な物理シミュレーションとGPU並列化
- **Domain Randomization対応**: 質量変化などのドメインランダマイゼーションに対応
- **クロスドメイン転移**: 異なる物理パラメータ間での歩行制御の転移

## ディレクトリ構造

```
vintix_go2/
├── configs/                      # 設定ファイル
│   ├── train_config.json         # 標準モデル訓練設定
│   └── train_config_large.json   # 大規模モデル訓練設定
├── scripts/                      # スクリプト
│   ├── collect_trajectories.py   # データ収集
│   ├── train_vintix.py           # モデル訓練
│   └── eval_vintix_go2.py        # モデル評価
├── data/                         # データディレクトリ（生成される）
│   └── go2_trajectories/         # 収集したトラジェクトリ
├── models/                       # モデル保存ディレクトリ（生成される）
│   └── vintix_go2/               # 訓練済みモデル
└── README.md                     # このファイル
```

## セットアップ

### 必要なパッケージ

1. **Genesis**: 物理シミュレーション環境
   ```bash
   # Genesisは既にインストールされていると仮定
   ```

2. **Vintix**: Algorithm Distillationフレームワーク
   ```bash
   cd ../vintix
   pip install -e .
   ```

3. **その他の依存関係**:
   ```bash
   pip install rsl-rl-lib==2.2.4
   pip install h5py numpy torch
   ```

## 使い方

### 1. データ収集

まず、訓練済みのPPOポリシーを使ってGo2のトラジェクトリデータを収集します。

```bash
# 標準環境でのデータ収集
python scripts/collect_trajectories.py \
    --policy_path ../Genesis/logs/go2-walking/model_300.pt \
    --output_dir data/go2_trajectories/go2_walking_standard \
    --num_episodes 1000 \
    --num_envs 4096

# Domain Randomizationを有効にしたデータ収集
python scripts/collect_trajectories.py \
    --policy_path ../Genesis/logs/go2-walking/model_300.pt \
    --output_dir data/go2_trajectories/go2_walking_dr \
    --num_episodes 1000 \
    --num_envs 16 \
    --domain_randomization \
    --mass_range_min 0.8 \
    --mass_range_max 1.2
```

**引数説明:**
- `--policy_path`: 訓練済みPPOポリシーのパス
- `--output_dir`: トラジェクトリデータの保存先
- `--num_episodes`: 収集するエピソード数
- `--num_envs`: 並列環境数
- `--domain_randomization`: ドメインランダマイゼーションを有効化
- `--mass_range_min/max`: 質量スケールの範囲

### 2. Vintixモデルの訓練

収集したデータを使用してVintixモデルを訓練します。

```bash
# シングルGPUでの訓練
python scripts/train_vintix.py \
    --config configs/train_config.json \
    --data_dir data/go2_trajectories \
    --save_dir models/vintix_go2 \
    --name vintix_go2_standard \
    --epochs 100 \
    --batch_size 16

# マルチGPUでの訓練
export WORLD_SIZE=4
OMP_NUM_THREADS=1 torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$WORLD_SIZE \
    scripts/train_vintix.py \
    --config configs/train_config.json
```

**引数説明:**
- `--config`: 訓練設定ファイル
- `--data_dir`: トラジェクトリデータのディレクトリ
- `--save_dir`: モデル保存先
- `--name`: 実験名
- `--epochs`: エポック数
- `--batch_size`: バッチサイズ
- `--load_ckpt`: チェックポイントから再開する場合のパス
- `--use_wandb`: Weights & Biasesでログを記録

### 3. モデルの評価

訓練したVintixモデルでGo2を制御し、性能を評価します。

```bash
# 標準環境での評価
python scripts/eval_vintix_go2.py \
    --model_path models/vintix_go2/vintix_go2_standard/0100_epoch \
    --num_episodes 50 \
    --save_results \
    --results_path results/eval_standard.json

# ビューワーを表示して評価
python scripts/eval_vintix_go2.py \
    --model_path models/vintix_go2/vintix_go2_standard/0100_epoch \
    --num_episodes 10 \
    --show_viewer

# Domain Randomizationでの評価
python scripts/eval_vintix_go2.py \
    --model_path models/vintix_go2/vintix_go2_standard/0100_epoch \
    --num_episodes 50 \
    --domain_randomization \
    --mass_range_min 0.7 \
    --mass_range_max 1.3 \
    --save_results \
    --results_path results/eval_dr.json
```

**引数説明:**
- `--model_path`: 訓練済みVintixモデルのパス
- `--num_episodes`: 評価エピソード数
- `--show_viewer`: Genesis Viewerを表示
- `--domain_randomization`: ドメインランダマイゼーションを有効化
- `--use_cache`: KVキャッシュを使用（高速化）
- `--use_fp16`: FP16精度で推論
- `--save_results`: 結果をJSON形式で保存
- `--results_path`: 結果の保存先

## ワークフロー例

### 基本的なワークフロー

```bash
# 1. データ収集
python scripts/collect_trajectories.py \
    --policy_path ../Genesis/logs/go2-walking/model_300.pt \
    --output_dir data/go2_trajectories/go2_walking_standard \
    --num_episodes 1000 \
    --num_envs 16

# 2. モデル訓練
python scripts/train_vintix.py \
    --config configs/train_config.json \
    --data_dir data/go2_trajectories \
    --name vintix_go2_exp1 \
    --epochs 100

# 3. モデル評価
python scripts/eval_vintix_go2.py \
    --model_path models/vintix_go2/vintix_go2_exp1/0100_epoch \
    --num_episodes 50 \
    --save_results
```

### Domain Randomizationを使った汎化学習

```bash
# 1. 多様な条件でデータ収集
python scripts/collect_trajectories.py \
    --policy_path ../Genesis/logs/go2-walking/model_300.pt \
    --output_dir data/go2_trajectories/go2_walking_diverse \
    --num_episodes 2000 \
    --num_envs 16 \
    --domain_randomization \
    --mass_range_min 0.7 \
    --mass_range_max 1.3

# 2. 大規模モデルで訓練
python scripts/train_vintix.py \
    --config configs/train_config_large.json \
    --name vintix_go2_robust \
    --epochs 150

# 3. 未知の条件で評価
python scripts/eval_vintix_go2.py \
    --model_path models/vintix_go2_large/vintix_go2_robust/0150_epoch \
    --num_episodes 100 \
    --domain_randomization \
    --mass_range_min 0.6 \
    --mass_range_max 1.4 \
    --save_results
```

## 設定ファイル

### train_config.json (標準モデル)

- **Context Length**: 2048 (約40秒の履歴)
- **Transformer**: 12層、8ヘッド
- **Hidden Dim**: 514
- **Batch Size**: 16
- **適用場面**: 標準的な学習、単一ドメイン

### train_config_large.json (大規模モデル)

- **Context Length**: 4096 (約80秒の履歴)
- **Transformer**: 20層、16ヘッド
- **Hidden Dim**: 1028
- **Batch Size**: 8
- **適用場面**: 複数ドメイン、高い汎化性能が必要な場合

## トラブルシューティング

### データ収集が遅い

```bash
# 並列環境数を増やす
--num_envs 32
```

### メモリ不足

```bash
# バッチサイズを減らす
--batch_size 8

# コンテキスト長を短くする
--context_len 1024
```

### 学習が収束しない

- 学習率を下げる: `--lr 1e-4`
- Warmupステップを増やす: `warmup_ratio: 0.02`
- データ収集量を増やす: `--num_episodes 2000`

## Algorithm Distillationについて

Vintixは、従来の強化学習とは異なり、**エキスパートの学習プロセス全体**を蒸留します：

1. **データ収集**: 様々な学習段階のポリシーからトラジェクトリを収集
2. **シーケンスモデリング**: 観察・行動・報酬のシーケンスをトランスフォーマーで学習
3. **In-Context適応**: 新しい条件下でも、過去の履歴から適切な行動を生成

これにより、モデルは：
- 新しいドメインに迅速に適応
- 多様な歩行パターンを学習
- オンライン学習なしで性能向上

## 参考文献

- [Vintix: Action Model via In-Context Reinforcement Learning](https://arxiv.org/abs/2501.19400)
- [Genesis: A Platform for Robot Learning](https://github.com/Genesis-Embodied-AI/Genesis)
- [Algorithm Distillation](https://arxiv.org/abs/2210.14215)

## ライセンス

このプロジェクトは研究目的で作成されています。Vintixおよび Genesis のライセンスに従ってください。

## 作成者

このプログラムは、VintixのAlgorithm Distillation手法をGenesis環境のGo2ロボットに適用するために作成されました。既存のコードに一切変更を加えず、統合プログラムとして実装されています。


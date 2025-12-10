# Go2 + Minicheetah マルチタスク学習ガイド

## 1. Noleakモデルが使用したデータセット

**データセット名**: `data_1M`
**パス**: `data/go2_trajectories/data_1M`
**設定ファイル**: `configs/go2_dataset_config.yaml`

このデータセットは：
- Go2ロボット用のAlgorithm Distillationデータ
- データリーク修正済み（観測値から行動を除外、33次元）
- 10環境 × 100万ステップ/環境
- p=0.6, f=0.05, max_perf=1.0

## 2. マルチタスク学習の実装方法

### 2.1 データセットの準備

既に以下のデータセットが収集済みです：
- **Go2**: `data/go2_trajectories/data_1M`
- **Minicheetah**: `data/minicheetah_trajectories/minicheetah_ad`

### 2.2 設定ファイルの作成

マルチタスク用の設定ファイル `configs/multitask_go2_minicheetah_config.yaml` を作成しました。

各タスクの設定：
- `type: "default"`: デフォルトタイプのデータセット
- `path`: `data_dir`からの相対パス
- `group`: タスクグループ名（Vintixモデルでタスクを識別するために使用）
- `reward_scale`: 報酬のスケーリング係数
- `episode_sparsity`: エピソードの間引き率

### 2.3 訓練コマンド

```bash
docker exec genesis_tensorboard bash -lc "cd /workspace/vintix_go2 && python scripts/train_vintix.py \
  --data_dir data \
  --dataset_config_paths configs/multitask_go2_minicheetah_config.yaml \
  --context_len 2048 \
  --batch_size 8 \
  --trajectory_sparsity 128 \
  --epochs 100 \
  --save_every 5 \
  --name go2_minicheetah_multitask"
```

**重要なポイント**:
- `--data_dir` は `data` に設定（両方のデータセットの親ディレクトリ）
- `--dataset_config_paths` でマルチタスク設定ファイルを指定
- 設定ファイル内の`path`は`data_dir`からの相対パス

### 2.4 データセットの構造

```
data/
├── go2_trajectories/
│   └── data_1M/
│       ├── data_1M.json  # メタデータ（必須）
│       ├── trajectories_env_0000.h5
│       ├── trajectories_env_0000.json
│       └── ...
└── minicheetah_trajectories/
    └── minicheetah_ad/
        ├── minicheetah_ad.json  # メタデータ（必要に応じて作成）
        ├── trajectories_env_0000.h5
        ├── trajectories_env_0000.json
        └── ...
```

### 2.5 メタデータの確認と修正

**注意**: Minicheetahデータのメタデータ（`trajectories_env_0000.json`）の`task_name`が`go2_walking_ad`になっている可能性があります。

`MultiTaskMapDataset`は`task_name`と`group_name`の両方を使用しますが、`group_name`が正しく設定されていれば動作します。

もし問題が発生する場合は、`collect_ad_data_parallel.py`を修正して、Minicheetahの場合は適切な`task_name`（例: `minicheetah_walking_ad`）を設定する必要があります。

### 2.6 訓練の動作

1. **データローダー**: `MultiTaskMapDataset`が両方のデータセットを読み込み、ランダムにミックスしてバッチを生成
2. **タスク識別**: 各サンプルには`task_name`と`group_name`が含まれ、Vintixモデルがタスクを識別
3. **統計情報**: `compute_stats`が両方のデータセットの統計を計算し、正規化に使用

### 2.7 評価方法

訓練後、各ロボットで個別に評価：
- Go2: `save_vintix.py`で評価
- Minicheetah: `save_vintix.py`をMinicheetah環境用に修正して評価

## 3. トラブルシューティング

### 3.1 メタデータファイルが見つからない

`compute_stats`関数は各データセットのメタデータファイル（`<dataset_name>.json`）を探します。
- Go2: `data/go2_trajectories/data_1M/data_1M.json` ✓
- Minicheetah: `data/minicheetah_trajectories/minicheetah_ad/minicheetah_ad.json` が必要

Minicheetahのメタデータファイルが存在しない場合は、`trajectories_env_0000.json`をコピーして作成：
```bash
cp data/minicheetah_trajectories/minicheetah_ad/trajectories_env_0000.json \
   data/minicheetah_trajectories/minicheetah_ad/minicheetah_ad.json
```

### 3.2 観測値の次元が異なる

Go2とMinicheetahで観測値の次元が異なる場合、Vintixモデルの入力層を調整する必要があるかもしれません。
現在の実装では、両方とも33次元（行動除外後）のはずです。

### 3.3 タスクグループの設定

`group`名はVintixモデルでタスクを識別するために使用されます。
異なるタスクには異なる`group`名を設定してください。


# シングルタスクとマルチタスクの切り替え方法

## 切り替え方法

**設定ファイル（`dataset_config_paths`）で切り替えます。**

### シングルタスクの場合

```yaml
# configs/go2_dataset_config.yaml
go2_walking_ad:
  type: "default"
  path: "data_1M"
  group: "go2_locomotion"
  reward_scale: 1.0
  episode_sparsity: 1
```

**実行コマンド**:
```bash
python scripts/train_vintix.py \
  --dataset_config_paths configs/go2_dataset_config.yaml \
  ...
```

**結果**:
- `dataset_names`に1つのデータセット（`go2_walking_ad`）のみが含まれる
- `MultiTaskMapDataset`を使うが、実質的にシングルタスクとして動作

### マルチタスクの場合

```yaml
# configs/multitask_go2_minicheetah_config.yaml
go2_walking_ad:
  type: "default"
  path: "go2_trajectories/data_1M"
  group: "go2_locomotion"
  reward_scale: 1.0
  episode_sparsity: 1

minicheetah_walking_ad:
  type: "default"
  path: "minicheetah_trajectories/minicheetah_ad"
  group: "minicheetah_locomotion"
  reward_scale: 1.0
  episode_sparsity: 1
  episode_range: [0.0, 0.5]
```

**実行コマンド**:
```bash
python scripts/train_vintix.py \
  --dataset_config_paths '["configs/multitask_go2_minicheetah_config.yaml"]' \
  ...
```

**結果**:
- `dataset_names`に2つのデータセット（`go2_walking_ad`と`minicheetah_walking_ad`）が含まれる
- `MultiTaskMapDataset`が複数のデータセットを組み合わせてマルチタスク学習を行う

## 実装の詳細

### `train_vintix.py`の動作

```python
# 設定ファイルからデータセット情報を読み込む
self.dataset_config = {}
for dcp in self.dataset_config_paths:
    dc = OmegaConf.load(dcp)
    self.dataset_config = {**self.dataset_config, **dc}  # マージ

# データセット名の辞書を作成
self.dataset_names = {
    v.path: v.group
    for k, v in self.dataset_config.items() if v.type == "default"
}

# 常にMultiTaskMapDatasetを使用
dataset = MultiTaskMapDataset(
    data_dir=config.data_dir,
    datasets_info=config.dataset_names,  # 1つでも複数でもOK
    ...
)
```

### `MultiTaskMapDataset`の動作

```python
# 設定ファイルに定義されたデータセットの数だけ、FoundationMapDatasetを作成
for i, ds_path in enumerate(self.dataset_paths):
    new_dataset = FoundationMapDataset(...)
    self.datasets.append(new_dataset)
    
# 各サンプルにtask_nameを追加
sample['task_name'] = self.datasets[ds_num].metadata['task_name']
```

**重要なポイント**:
- `MultiTaskMapDataset`は、設定ファイルに定義されたデータセットの数に応じて動作する
- 1つのデータセット → シングルタスク（実質的）
- 複数のデータセット → マルチタスク

## 今回の訓練でマルチタスクが使えた理由

今回の訓練では、以下のコマンドで実行しました：

```bash
python scripts/train_vintix.py \
  --dataset_config_paths '["configs/multitask_go2_minicheetah_config.yaml"]' \
  ...
```

この設定ファイルには、**2つのデータセット**（`go2_walking_ad`と`minicheetah_walking_ad`）が定義されているため、`MultiTaskMapDataset`が2つのデータセットを組み合わせてマルチタスク学習を行いました。

## まとめ

- **切り替え方法**: 設定ファイル（`dataset_config_paths`）で定義するデータセットの数で決まる
- **実装**: `train_vintix.py`は常に`MultiTaskMapDataset`を使用（1つでも複数でも対応）
- **シングルタスク**: 設定ファイルに1つのデータセットのみ定義
- **マルチタスク**: 設定ファイルに複数のデータセットを定義


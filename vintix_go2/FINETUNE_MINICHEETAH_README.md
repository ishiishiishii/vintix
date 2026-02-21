# Minicheetah追加訓練ガイド

## 概要

`Minicheetah_without_separategroup`モデルを基に、Transformerパラメータを凍結してMinicheetahタスク用のエンコーダー/デコーダーのみを追加訓練する方法。

## 実装のポイント

### 1. 既存モデルの構造
- **Encoder**: `IndividualTaskEncoderNew` - グループごとにエンコーダーを持つ（`obs_encoders`, `acs_encoders`, `rews_encoders`）
- **Transformer**: `Transformer` - ポリシー改善部分（凍結対象）
- **Head**: `IndividualTaskHeadNew` - グループごとにデコーダーを持つ（`acs_decoders`）

### 2. 追加訓練の流れ

1. **事前訓練済みモデルのロード**
   - `Minicheetah_without_separategroup`の15エポックモデルをロード

2. **新しいグループのエンコーダー/デコーダーの追加**
   - `minicheetah_locomotion`グループ用のエンコーダー/デコーダーを`ModuleDict`に追加
   - メタデータを更新（`task2group`, `task_metadata`, `group_metadata`）

3. **Transformerパラメータの凍結**
   - `model.transformer.parameters()`の`requires_grad = False`を設定

4. **訓練可能パラメータのみを最適化器に追加**
   - 新しいグループのエンコーダー/デコーダーのパラメータのみを最適化

5. **追加訓練の実行**
   - Minicheetahデータのみで訓練（既存のGo1/Go2/A1データは使用しない）

## 使用方法

### 1. 設定ファイルの作成

`configs/minicheetah_finetune_config.yaml`を作成（既に作成済み）:

```yaml
minicheetah_walking_ad:
  type: "default"
  path: "minicheetah_trajectories"
  group: "minicheetah_locomotion"
  reward_scale: 1.0
  episode_sparsity: 1
```

### 2. 追加訓練の実行

```bash
cd /home/kawa37/genesis_project
docker exec -e PYTHONPATH=/workspace/vintix_go2 genesis_tensorboard bash -c "
cd /workspace/vintix_go2 && \
python3 scripts/finetune_minicheetah.py \
  --pretrained_path models/vintix_go2/Minicheetah_without_separategroup/Minicheetah_without_separategroup/0015_epoch \
  --dataset_config_paths '[\"configs/minicheetah_finetune_config.yaml\"]' \
  --name Minicheetah_without_separategroup_finetuned \
  --epochs 10 \
  --data_dir data \
  --freeze_transformer \
  --lr 0.0001
" 2>&1 | tee /tmp/finetune_minicheetah.log
```

### 3. パラメータ説明

- `--pretrained_path`: 事前訓練済みモデルのパス（15エポックモデル）
- `--dataset_config_paths`: Minicheetahデータセット設定ファイルのパス
- `--name`: 保存されるモデル名
- `--epochs`: 追加訓練のエポック数（通常は5-10エポック）
- `--freeze_transformer`: Transformerを凍結するフラグ
- `--lr`: 学習率（通常は0.0001程度、事前訓練より低め）

## 実装の詳細

### `add_new_group_to_model`関数

新しいグループのエンコーダー/デコーダーを既存モデルに追加:

```python
def add_new_group_to_model(model: Vintix, new_task_name: str, new_group_name: str,
                           group_metadata: dict, stats: dict) -> None:
    # エンコーダーの追加
    model.encoder.obs_encoders[new_group_name] = get_obs_encoder(...)
    model.encoder.acs_encoders[new_group_name] = get_acs_encoder(...)
    model.encoder.rews_encoders[new_group_name] = RewardEncoder(...)
    
    # デコーダーの追加
    model.head.acs_decoders[new_group_name] = AcsDecoder(...)
    
    # メタデータの更新
    model.encoder.task2group[new_task_name] = new_group_name
    model.head.task2group[new_task_name] = new_group_name
```

### `freeze_transformer_parameters`関数

Transformerパラメータを凍結:

```python
def freeze_transformer_parameters(model: Vintix) -> None:
    for param in model.transformer.parameters():
        param.requires_grad = False
```

### `get_trainable_parameters`関数

訓練可能なパラメータ（新しいグループのエンコーダー/デコーダーのみ）を取得:

```python
def get_trainable_parameters(model: Vintix, new_group_name: str) -> List[torch.nn.Parameter]:
    trainable_params = []
    trainable_params.extend(model.encoder.obs_encoders[new_group_name].parameters())
    trainable_params.extend(model.encoder.acs_encoders[new_group_name].parameters())
    trainable_params.extend(model.encoder.rews_encoders[new_group_name].parameters())
    trainable_params.extend(model.head.acs_decoders[new_group_name].parameters())
    return trainable_params
```

## 期待される効果

1. **ゼロショット評価**: 事前訓練済みモデルでのMinicheetah評価（Transformerが既存の3グループのデータで学習済み）
2. **追加訓練後**: Minicheetah専用のエンコーダー/デコーダーが追加訓練され、Minicheetahでの性能が向上

## 注意事項

1. **メタデータの整合性**: 新しいグループのメタデータ（観測/行動の次元など）が正しく設定されている必要があります
2. **統計情報**: `stats.json`にMinicheetahの統計情報が含まれている必要があります
3. **デバイス**: エンコーダー/デコーダーを追加後、`model.to(device)`でデバイスに移動する必要があります

## 評価

追加訓練後の評価:

```bash
python3 scripts/save_vintix.py \
  --vintix_path models/vintix_go2/Minicheetah_without_separategroup_finetuned/Minicheetah_without_separategroup_finetuned/0010_epoch \
  --robot_type minicheetah \
  --parallel \
  --num_envs 100 \
  --max_steps 1000
```

# エピソードスパーシティ（Episode Sparsity）の説明

## エピソードスパーシティとは？

**エピソードスパーシティ（`episode_sparsity`）**は、データセットから軌跡（trajectory）を構築する際に、どのエピソードを使用するかを制御するパラメータです。

### 動作

- **`episode_sparsity = 1`**: 全てのエピソードを使用（エピソードを間引かない）
- **`episode_sparsity = 2`**: 2番目ごとのエピソードのみを使用（エピソードを50%間引く）
- **`episode_sparsity = 3`**: 3番目ごとのエピソードのみを使用（エピソードを67%間引く）

### コード内の実装

```python
# vintix/data/torch_dataloaders.py の116行目付近
transes = transes[::self.ep_sparsity]  # エピソードを間引く
```

このコードは、エピソードのリストを`ep_sparsity`の間隔でスライスしています。

### 使用例

データセットに1000個のエピソードがある場合：
- `episode_sparsity = 1`: 1000個全てのエピソードを使用
- `episode_sparsity = 2`: 500個のエピソードを使用（1, 3, 5, 7, ...）
- `episode_sparsity = 4`: 250個のエピソードを使用（1, 5, 9, 13, ...）

### エピソードの順序

重要な点として、エピソードの順序は保持されます。つまり、時系列的な順序は維持されます。

## Noleakモデルとの設定比較

### Noleakモデル（go2_ad_1m_no_leak）の設定

```yaml
go2_walking_ad:
  type: "default"
  path: "data_1M"
  group: "go2_locomotion"
  reward_scale: 1.0
  episode_sparsity: 1  # 全てのエピソードを使用
```

### マルチタスクモデル（go2_minicheetah_multitask）の設定

```yaml
go2_walking_ad:
  type: "default"
  path: "go2_trajectories/data_1M"
  group: "go2_locomotion"
  reward_scale: 1.0
  episode_sparsity: 1  # 全てのエピソードを使用

minicheetah_walking_ad:
  type: "default"
  path: "minicheetah_trajectories/minicheetah_ad"
  group: "minicheetah_locomotion"
  reward_scale: 1.0
  episode_sparsity: 1  # 全てのエピソードを使用
```

## 結論

**はい、Noleakモデルと同じ設定になっています。**

両方とも`episode_sparsity: 1`を使用しているため、全てのエピソードが訓練に使用されます。これは、データを最大限に活用する設定です。

### その他の訓練パラメータの比較

| パラメータ | Noleakモデル | マルチタスクモデル |
|-----------|-------------|------------------|
| `context_len` | 2048 | 2048 ✓ |
| `trajectory_sparsity` | 128 | 128 ✓ |
| `batch_size` | 8 | 8 ✓ |
| `episode_sparsity` | 1 | 1 ✓ |
| `epochs` | 100 | 100 ✓ |
| `save_every` | 5 | 5 ✓ |
| `lr` | 0.0003 | 0.0003 ✓ |

**全ての訓練パラメータが同じ設定になっています。**


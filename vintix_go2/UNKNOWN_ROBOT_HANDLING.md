# 未知のロボットでの評価に関する問題と改善案

## 質問1: 未知のタスクを未知のタスクとして扱うことはできるか？

### 回答: 可能ですが、制約があります

**技術的には可能：**
- `Vintix`クラスには`add_task()`メソッドがあり、実行時に新しいタスクを追加できます
- ただし、以下が必要です：
  - `task_name`: 新しいタスク名
  - `group_name`: グループ名（既存の`group_name`を使用可能）
  - `stats`: 正規化統計（`obs_mean`, `obs_std`, `acs_mean`, `acs_std`）
  - `rew_scale`: 報酬スケール

**問題点：**
- 未知のロボットの正規化統計を取得するには、そのロボットのデータが必要
- 評価時にはデータがないため、既存のロボットの統計を使用する必要がある

## 質問2: 選ぶロボットによって性能が変わってしまうことはあるか？

### 回答: はい、性能に影響する可能性があります

**影響要因：**

1. **正規化統計の違い**
   - 各ロボットの観測値と行動の分布が異なる
   - 例：Go2の統計を使用してMinicheetahを評価すると、正規化が不適切になる可能性
   - 影響：モデルの入力分布が訓練時と異なり、性能が低下する可能性

2. **group_nameの違い**
   - `group_name`が異なる場合、異なるエンコーダ/デコーダが使用される
   - 現在の実装では統一された`group_name`（`quadruped_locomotion`）を使用しているため、この影響は小さい

3. **正規化統計の計算方法**
   - 観測値: `(obs - obs_mean) / (obs_std + 1e-6)`
   - 行動: `(acs - acs_mean) / (acs_std + 1e-6)` (エンコード時)
   - 行動: `action * acs_std + acs_mean` (デコード時)
   - 統計が不適切な場合、正規化後の値が訓練時の分布から大きく外れる可能性がある

## 改善案

### 案1: 実行時に正規化統計を計算して新しいタスクとして追加（推奨）

評価時に、未知のロボットのデータを収集して正規化統計を計算し、`add_task()`で追加する。

```python
def compute_stats_for_unknown_robot(env, num_samples=1000):
    """未知のロボットの正規化統計を計算"""
    obses = []
    acses = []
    
    obs, _ = env.reset()
    obs = obs[:, :-12]  # 行動を除外
    
    for _ in range(num_samples):
        # ランダムな行動を実行
        action = torch.randn(env.num_actions, device=gs.device)
        obs, _, _, _ = env.step(action)
        obs = obs[:, :-12]
        
        obses.append(obs.cpu().numpy())
        acses.append(action.cpu().numpy())
    
    obses = np.vstack(obses)
    acses = np.vstack(acses)
    
    stats = {
        "obs_mean": obses.mean(axis=0).tolist(),
        "obs_std": obses.std(axis=0, ddof=1).tolist(),
        "acs_mean": acses.mean(axis=0).tolist(),
        "acs_std": acses.std(axis=0, ddof=1).tolist(),
    }
    return stats
```

**メリット：**
- 未知のロボット専用の正規化統計を使用できる
- 性能への影響を最小化できる

**デメリット：**
- 評価前に統計計算が必要（時間がかかる）
- ランダムな行動で収集した統計は、実際の分布と異なる可能性がある

### 案2: ロボットの類似性を考慮した選択

物理的特性や観測/行動空間の類似性を考慮して、最も近い既存ロボットを選択する。

```python
def select_best_task_name(robot_type: str, model_metadata: dict) -> str:
    """ロボットの類似性を考慮して最適なtask_nameを選択"""
    # 類似性マトリックス（経験的に決定）
    similarity_map = {
        "go1": ["go2_walking_ad", "go1_walking_ad", "unitreea1_walking_ad"],
        "go2": ["go2_walking_ad", "go1_walking_ad"],
        "minicheetah": ["minicheetah_walking_ad", "go2_walking_ad"],
        "unitreea1": ["unitreea1_walking_ad", "go2_walking_ad", "go1_walking_ad"],
    }
    
    # 類似性順に試す
    for similar_task in similarity_map.get(robot_type, []):
        if similar_task in model_metadata:
            return similar_task
    
    # フォールバック
    return list(model_metadata.keys())[0]
```

### 案3: 正規化を無効にするオプション

未知のロボットでは正規化を無効にする（ただし、モデルが正規化を前提としている場合、性能が大幅に低下する可能性がある）

### 案4: 複数の既存タスクを試して最良のものを選択

評価時に複数の既存タスク名で評価し、最も性能が良いものを選択する。

## 現在の実装の影響

現在の実装では、フォールバックリストの順序で最初に見つかったタスク名を使用しています：
1. `go2_walking_ad`
2. `go1_walking_ad`
3. `unitreea1_walking_ad`
4. `minicheetah_walking_ad`
5. `laikago_walking_ad`

この順序は、四足歩行ロボットの一般的な類似性を考慮していますが、最適とは限りません。

## 推奨事項

1. **短期的な改善**: ロボットの類似性を考慮した選択ロジックを実装
2. **長期的な改善**: 実行時に正規化統計を計算して新しいタスクとして追加する機能を実装
3. **実験**: 異なる既存タスク名を使用した場合の性能を比較し、最適な選択方法を決定


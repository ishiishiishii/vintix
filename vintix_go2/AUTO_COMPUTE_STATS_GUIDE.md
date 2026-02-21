# 実行時に正規化統計を計算して新しいタスクとして追加する機能

## 概要

未知のロボットを評価する際、モデルのメタデータにそのロボットの正規化統計が存在しない場合、既存のロボットの統計を使用することになります。しかし、これにより性能が低下する可能性があります。

この機能は、評価時に自動的に未知のロボットの正規化統計を計算し、モデルに新しいタスクとして追加することで、最適な性能を実現します。

## 使用方法

### 基本的な使用方法

```bash
python scripts/save_vintix.py \
    --vintix_path models/vintix_go2/model_name/0015_epoch \
    --robot_type anymalc \
    --auto_compute_stats \
    --stats_num_samples 1000 \
    --parallel \
    --num_envs 100 \
    --max_steps 1000
```

### 引数の説明

- `--auto_compute_stats`: 未知のロボットの場合、自動的に統計を計算してタスクを追加するフラグ
- `--stats_num_samples`: 統計計算に使用するサンプル数（デフォルト: 1000）
  - より多くのサンプルを使用すると、より正確な統計が得られますが、計算時間が長くなります
  - 推奨値: 1000-5000

## 動作の流れ

1. **モデルのロード**: Vintixモデルをロードし、メタデータを確認
2. **タスク名の確認**: 指定されたロボットタイプに対応するタスク名がメタデータに存在するか確認
3. **統計計算（必要な場合）**: 
   - `--auto_compute_stats`が有効で、タスク名が存在しない場合
   - 単一環境を作成し、ランダムな行動でデータを収集
   - 観測値と行動の平均・標準偏差を計算
4. **タスクの追加**: 計算した統計を使用して`add_task()`メソッドでモデルに新しいタスクを追加
5. **評価の実行**: 追加されたタスク名を使用して評価を実行

## 実装の詳細

### `_compute_stats_for_robot()`関数

```python
def _compute_stats_for_robot(env, num_samples: int = 1000) -> dict:
    """未知のロボットの正規化統計を計算"""
    # 1. 環境をリセット
    # 2. ランダムな行動でnum_samples回データを収集
    # 3. 観測値と行動の平均・標準偏差を計算
    # 4. 統計を返す
```

### `_add_unknown_task_to_model()`関数

```python
def _add_unknown_task_to_model(vintix_model, robot_type: str, env, 
                                group_name: str = "quadruped_locomotion", 
                                num_samples: int = 1000) -> str:
    """未知のロボットのタスクをモデルに追加"""
    # 1. task_nameを生成
    # 2. 既に存在する場合は追加しない
    # 3. 正規化統計を計算
    # 4. モデルにタスクを追加
    # 5. task_nameを返す
```

## 注意事項

1. **計算時間**: 統計計算には時間がかかります（1000サンプルで約30秒-1分程度）
2. **メモリ**: 統計計算用に単一環境を作成するため、追加のメモリが必要です
3. **統計の精度**: ランダムな行動で収集した統計は、実際のポリシーの分布と異なる可能性があります
   - より正確な統計を得るには、既存のポリシーを使用してデータを収集することを検討してください
4. **既存タスク**: 既にメタデータに存在するタスク名の場合は、統計計算はスキップされます

## 今後の改善案

1. **既存ポリシーを使用した統計計算**: ランダムな行動ではなく、既存のポリシー（例：Go2のポリシー）を使用してデータを収集
2. **統計のキャッシュ**: 計算した統計をファイルに保存し、次回以降は再利用
3. **複数環境での統計計算**: 単一環境ではなく、複数環境で並列にデータを収集して統計を計算

## 使用例

### 例1: Anymal-Cを評価（モデルにanymalc_walking_adが存在しない場合）

```bash
python scripts/save_vintix.py \
    --vintix_path models/vintix_go2/go2_go1_a1_minicheetah/go2_go1_a1_minicheetah/0010_epoch \
    --robot_type anymalc \
    --auto_compute_stats \
    --stats_num_samples 2000 \
    --parallel \
    --num_envs 100 \
    --max_steps 1000
```

このコマンドは：
1. `anymalc_walking_ad`がメタデータに存在しないことを検出
2. 統計計算用の単一環境を作成
3. 2000サンプルを収集して統計を計算
4. `anymalc_walking_ad`をモデルに追加
5. 追加されたタスク名を使用して評価を実行

### 例2: 既存タスクが存在する場合

```bash
python scripts/save_vintix.py \
    --vintix_path models/vintix_go2/go2_go1_a1_minicheetah/go2_go1_a1_minicheetah/0010_epoch \
    --robot_type go2 \
    --auto_compute_stats \
    --parallel \
    --num_envs 100 \
    --max_steps 1000
```

この場合、`go2_walking_ad`が既にメタデータに存在するため、統計計算はスキップされ、既存のタスク名が使用されます。


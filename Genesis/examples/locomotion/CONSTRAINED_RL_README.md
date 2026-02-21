# Constrained Reinforcement Learning Implementation

この実装は、"Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion"論文の手法をベースに、制約付き強化学習（Constrained RL）による歩行ポリシーの訓練を可能にします。

## 概要

従来の強化学習では、報酬（reward）のみを最大化することでポリシーを学習しますが、この手法では：

1. **報酬（Reward）**: 最大化したい目標（例: 速度追従）
2. **コスト（Cost）**: 守るべき制約（例: 高さ維持、接触回避）

を明確に分離し、制約を満たしながら報酬を最大化するポリシーを学習します。

## 主な特徴

### 1. 報酬とコストの分離
- **報酬**: ポリシーの性能を評価する指標（最大化）
- **コスト**: 安全性や制約違反を評価する指標（最小化・制約内に維持）

### 2. コストの種類

#### Probabilistic Constraints（確率的制約）
- 二値インジケータ（0: 満足、1: 違反）
- 例: ベース高さが範囲外、胴体が地面に接触

#### Average Constraints（平均制約）
- 連続値で平均的に低く抑えたいもの
- 例: 足の滑り速度、トルク使用量

### 3. 制約付きPPOアルゴリズム

論文のIPO（Interior-Point Policy Optimization）の考え方を実装：

- **Cost Critic**: 各コストの将来累積値を予測するValue Function
- **Log Barrier Penalty**: 制約違反を防ぐための対数バリア関数によるペナルティ
- **Adaptive Thresholding**: 学習初期の制約違反を防ぐ適応的閾値調整

## ファイル構成

```
Genesis/examples/locomotion/
├── constrained_env_base.py      # 制約付き環境の基底クラス
├── constrained_ppo.py           # 制約付きPPOアルゴリズム実装
├── train_constrained.py         # 制約付き訓練スクリプト
└── CONSTRAINED_RL_README.md     # このファイル
```

## 使用方法

### 基本的な使い方

```bash
python train_constrained.py \
    -e go2-constrained-walking \
    -r go2 \
    -B 4096 \
    --max_iterations 301 \
    --seed 1
```

### パラメータ説明

- `-e, --exp_name`: 実験名（ログディレクトリ名）
- `-r, --robot_type`: ロボットタイプ（現在は`go2`のみ）
- `-B, --num_envs`: 並列環境数
- `--max_iterations`: 最大訓練イテレーション数
- `--seed`: 乱数シード

## 実装の詳細

### 1. 制約付き環境（ConstrainedEnvMixin）

既存の環境クラスに`ConstrainedEnvMixin`を継承させることで、コスト計算機能を追加：

```python
class ConstrainedGo2Env(ConstrainedEnvMixin, Go2Env):
    def __init__(self, *args, cost_cfg=None, **kwargs):
        super().__init__(*args, cost_cfg=cost_cfg, **kwargs)
    
    def _cost_base_height_violation(self):
        """Probabilistic constraint: ベース高さが範囲外"""
        height_min = 0.25
        height_max = 0.50
        violation = (self.base_pos[:, 2] < height_min) | (self.base_pos[:, 2] > height_max)
        return violation.float()
```

### 2. コスト設定（cost_cfg）

```python
cost_cfg = {
    "cost_functions": {
        "base_height_violation": True,
        "body_contact": True,
        "action_smoothness": True,
    },
    "cost_types": {
        "base_height_violation": "probabilistic",  # 二値制約
        "body_contact": "probabilistic",
        "action_smoothness": "average",  # 平均制約
    },
    "cost_thresholds": {
        "base_height_violation": 0.05,  # 最大5%違反率
        "body_contact": 0.01,  # 最大1%接触率
        "action_smoothness": 0.1,  # 最大平均アクション率
    },
}
```

### 3. 制約付きPPOアルゴリズム

`ConstrainedPPO`クラスは以下の機能を提供：

- **MultiHeadCostCritic**: 複数のコストを同時に予測するCost Critic
- **GAE for Costs**: コスト用のGeneralized Advantage Estimation
- **Log Barrier Penalty**: 制約ペナルティ項の計算

損失関数：
```
L_total = L_PPO - Σ_k [λ_k * log(d_k - J_c_k)] + L_value + L_cost_value
```

ここで：
- `L_PPO`: 標準PPOクリップ済みサロゲート損失
- `λ_k`: k番目の制約のペナルティ係数
- `d_k`: k番目の制約の閾値
- `J_c_k`: k番目のコストの期待値
- `L_value`: 報酬のValue損失
- `L_cost_value`: コストのValue損失

## カスタマイズ

### 新しいコスト関数の追加

環境クラスに新しいコスト関数を追加：

```python
def _cost_custom_constraint(self):
    """カスタム制約の実装"""
    # 制約違反を計算
    violation = ...
    return violation.float()  # Probabilistic
    # または
    return continuous_value  # Average
```

`cost_cfg`に追加：

```python
cost_cfg["cost_functions"]["custom_constraint"] = True
cost_cfg["cost_types"]["custom_constraint"] = "probabilistic"  # or "average"
cost_cfg["cost_thresholds"]["custom_constraint"] = 0.1
```

### 制約閾値の調整

制約の閾値（`cost_thresholds`）を調整することで、制約の厳しさを変更できます：

- **厳しい制約**: より小さい閾値（例: 0.01）
- **緩い制約**: より大きい閾値（例: 0.1）

## 注意事項

### 現在の実装状況

現在の`train_constrained.py`は**デモンストレーションフレームワーク**です。

完全な実装のためには：

1. **rsl-rl-libとの統合**: `OnPolicyRunner`を拡張して制約付きPPOを統合
2. **Cost Criticの最適化器**: コストCritic用の最適化器を追加
3. **ロールアウト収集**: コスト情報を含むロールアウト収集の実装
4. **ログ機能**: TensorBoardへのコスト違反率などのログ出力

### 本番環境での使用

本番環境で使用する場合：

1. `constrained_ppo.py`の`ConstrainedPPO`クラスをrsl-rl-libの`OnPolicyRunner`に統合
2. または、OmniSafeライブラリを使用（論文で比較対象として挙げられている）

## 参考文献

- "Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion"
- OmniSafe: https://github.com/PKU-MARL/OmniSafe
- rsl-rl-lib: https://github.com/leggedrobotics/rsl_rl

## トラブルシューティング

### 制約違反が起きない

- 制約閾値を確認（大きすぎる可能性）
- コスト関数の実装を確認
- `cost_cfg`の設定を確認

### 学習が不安定

- `penalty_coef`を調整（デフォルト: 1.0）
- `adaptive_threshold`を有効にする（デフォルト: True）
- 学習率を下げる

### メモリエラー

- `num_envs`を減らす
- `num_steps_per_env`を減らす
- `num_mini_batches`を増やす



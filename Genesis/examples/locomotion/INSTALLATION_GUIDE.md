# 制約付き強化学習 インストール・使用方法ガイド

## 概要

この実装は「Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion」論文の手法をベースに、制約付きPPOで歩行ポリシーを訓練します。

## 必要なライブラリ

既にインストールされているもの：
- `rsl-rl-lib==2.2.4` ✅
- `torch` ✅
- `genesis` ✅

追加で必要なもの：
- なし（既存のrsl-rl-libを使用）

## 使用方法

### 基本的な訓練コマンド

```bash
# Dockerコンテナ内で実行
docker exec genesis_tensorboard bash -c "cd /workspace/Genesis/examples/locomotion && python train_constrained.py -e go2-constrained-walking -r go2 -B 4096 --max_iterations 301"
```

### パラメータ説明

- `-e, --exp_name`: 実験名（ログディレクトリ名）
- `-r, --robot_type`: ロボットタイプ（`go2` または `laikago`）
- `-B, --num_envs`: 並列環境数（デフォルト: 4096）
- `--max_iterations`: 最大訓練イテレーション数（デフォルト: 301）
- `--seed`: 乱数シード（デフォルト: 1）
- `--penalty_coef`: 制約ペナルティ係数（デフォルト: 1.0）
- `--cost_lr`: Cost Criticの学習率（デフォルト: 0.001）

### 制約のカスタマイズ

制約をカスタマイズするには、`train_constrained.py`の`get_constrained_train_cfg()`や`main()`関数内の`cost_cfg`を編集してください。

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

## 実装状況

### ✅ 完成した部分

1. 制約付き環境（ConstrainedEnvMixin）
2. Cost Criticネットワーク（MultiHeadCostCritic）
3. 制約付きPPOアルゴリズム（ConstrainedPPOAlgorithm）
4. 制約付きランナー（ConstrainedOnPolicyRunner）
5. 訓練スクリプト（train_constrained.py）

### ⚠️ 注意事項

現在の実装は、rsl-rl-libのOnPolicyRunnerを拡張していますが、完全な統合には以下の作業が必要です：

1. **Cost Criticの更新**: ロールアウト収集時にコストも収集し、Cost Criticを更新する処理を完全に実装
2. **制約ペナルティの適用**: PPOの更新時に制約ペナルティを適用する処理を完全に実装
3. **ログ機能**: TensorBoardへのコスト違反率などのログ出力

現在の実装では、Cost Criticは初期化されますが、完全な更新ループは実装中です。

## トラブルシューティング

### インポートエラー

```bash
# Dockerコンテナ内で実行
docker exec genesis_tensorboard bash -c "cd /workspace/Genesis/examples/locomotion && python -c 'from train_constrained import ConstrainedGo2Env; print(\"OK\")'"
```

### メモリエラー

環境数を減らしてください：
```bash
python train_constrained.py -e test -r go2 -B 2048  # 4096から2048に減らす
```

## 今後の拡張

完全な制約付きPPO実装のためには：

1. OnPolicyRunner.learn()メソッドを完全にオーバーライドして、コスト収集と更新を実装
2. PPO.update()メソッドを拡張して、制約ペナルティを適用
3. TensorBoardへのコスト関連メトリクスのログ出力

または、OmniSafeライブラリを使用することを検討してください（ただし、Genesis環境との統合は別途必要）。



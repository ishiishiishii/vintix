# OmniSafe統合ガイド

このドキュメントは、OmniSafeライブラリを使用した制約付き強化学習の実装方法を説明します。

## 現状

OmniSafeは`omnisafe==0.5.0`がインストールされていますが、完全な動作には以下の依存関係が必要です：

- ✅ `gymnasium==1.2.3` - インストール済み
- ❌ `safety-gymnasium` - インストールに問題あり（依存関係の競合）

## 解決策

### 方法1: OmniSafeアルゴリズムを直接使用（推奨）

OmniSafeのアルゴリズム実装（IPO、P3Oなど）を参考に、既存の`constrained_ppo.py`を拡張します。これにより、OmniSafeの完全なインストールなしで動作します。

### 方法2: 環境ラッパーの完成

`omnisafe_env_wrapper.py`を作成済みです。これを完成させて、Genesis環境をOmniSafe互換にします。

### 方法3: OmniSafeの依存関係を修正

`safety-gymnasium`の依存関係の問題を解決して、完全なOmniSafeをインストールします。

## 実装済みのファイル

1. **`omnisafe_env_wrapper.py`** - Genesis環境をOmniSafe互換にラップ
2. **`train_omnisafe.py`** - OmniSafeを使用した訓練スクリプト（フォールバック付き）
3. **`constrained_ppo.py`** - カスタム制約付きPPO実装（IPO/P3Oスタイル）

## 使用方法

### カスタム実装を使用（現在の推奨方法）

```bash
# カスタム制約付きPPOを使用
docker exec genesis_tensorboard bash -c "cd /workspace/Genesis/examples/locomotion && python train_constrained.py -e go2-constrained -r go2 -B 4096 --max_iterations 301"
```

### OmniSafe統合を試す（safety-gymnasiumの問題解決後）

```bash
# OmniSafeのアルゴリズムを使用（要：safety-gymnasiumのインストール）
docker exec genesis_tensorboard bash -c "cd /workspace/Genesis/examples/locomotion && python train_omnisafe.py -e go2-omnisafe -r go2 --algorithm IPO --max_iterations 301"
```

## OmniSafeアルゴリズムの実装参考

論文「Not Only Rewards But Also Constraints」で使用されているアルゴリズム：

1. **IPO (Interior-Point Policy Optimization)** ✅ 実装済み（`constrained_ppo.py`）
2. **P3O (Penalized Proximal Policy Optimization)** - 実装可能
3. **CPO (Constrained Policy Optimization)** - 実装可能

## 今後の作業

### 優先度: 高

1. ✅ OmniSafeのアルゴリズム実装を参考にしたカスタム実装（完了）
2. ⚠️ OmniSafe環境ラッパーの完成（部分的完了）
3. ⏳ safety-gymnasiumの依存関係問題の解決

### 優先度: 中

1. P3Oアルゴリズムの実装追加
2. CPOアルゴリズムの実装追加
3. OmniSafeとの完全な統合

## 参考リソース

- OmniSafe GitHub: https://github.com/OmniSafe/OmniSafe
- OmniSafe Documentation: https://omnisafe.readthedocs.io/
- 論文: "Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion"

## トラブルシューティング

### safety-gymnasiumのインストールエラー

```bash
# 代替インストール方法を試す
pip install safety-gymnasium --no-build-isolation
# または
pip install safety-gymnasium==1.0.0 --no-deps
pip install mujoco gymnasium
```

### OmniSafeのインポートエラー

現在の実装では、OmniSafeの完全なインポートができない場合、自動的にカスタム実装にフォールバックします。

```python
# train_omnisafe.py内で自動的に処理
if not OMNISAFE_AVAILABLE:
    print("Falling back to custom constrained PPO implementation")
    # カスタム実装を使用
```

## 結論

現在の実装では、**カスタム制約付きPPO実装（`train_constrained.py`）**が最も実用的です。OmniSafeのアルゴリズムの考え方を取り入れた実装になっており、実際に訓練を行うことができます。

OmniSafeの完全な統合は、依存関係の問題が解決された後に進めることをお勧めします。



# 自己改善型ファインチューニングガイド

## 概要

未知のロボット（Minicheetah、Go1、Go2、A1など）に適応するための、自己改善型のファインチューニングスクリプト。

既存のVintixモデルを基に、Transformerパラメータを凍結して、エンコーダー/デコーダーのみを自己改善的なデータ収集とファインチューニングで学習します。

## アルゴリズム

1. **初期ポリシーの準備**
   - 新しいグループのエンコーダー/デコーダーを追加
   - 参考ロボット（最も似ているロボット）のエンコーダー/デコーダーをコピー
   - Transformerを凍結

2. **自己改善ループ（複数回繰り返し）**
   - **データ収集フェーズ**: 現在のモデルで環境を実行し、Nエピソード分のデータを収集
   - **ファインチューニングフェーズ**: 収集したデータでエンコーダー/デコーダーのみをMステップ訓練
   - ループを繰り返す

3. **収束**
   - ループを数回繰り返すことで、ロボット専用のエンコーダー/デコーダーが適応

## 使用方法

### 基本的な使用例

```bash
cd /home/kawa37/genesis_project
docker exec -e PYTHONPATH=/workspace/vintix_go2 genesis_tensorboard bash -c "
cd /workspace/vintix_go2 && \
python3 scripts/self_improving_finetune.py \
  --pretrained_path models/vintix_go2/Minicheetah_without_separategroup/Minicheetah_without_separategroup/0015_epoch \
  --robot_type minicheetah \
  --reference_robot go2 \
  --num_improvement_loops 5 \
  --episodes_per_loop 50 \
  --finetune_steps_per_loop 1000 \
  --output_dir data/self_improving_finetune/minicheetah \
  --lr 0.0001
"
```

### パラメータ説明

- `--pretrained_path`: 事前訓練済みモデルのパス（必須）
- `--robot_type`: ターゲットロボットタイプ（`go1`, `go2`, `minicheetah`, `unitreea1`, `a1`）
- `--reference_robot`: 参考ロボットタイプ（初期ポリシーとしてコピーするロボット、デフォルト: `go2`）
- `--num_improvement_loops`: 自己改善ループの回数（デフォルト: 5）
- `--episodes_per_loop`: ループあたりのデータ収集エピソード数（デフォルト: 50）
- `--finetune_steps_per_loop`: ループあたりのファインチューニングステップ数（デフォルト: 1000）
- `--output_dir`: 出力ディレクトリ（データとモデルの保存先、必須）
- `--max_steps_per_episode`: エピソードあたりの最大ステップ数（デフォルト: 1000）
- `--context_len`: コンテキスト長（デフォルト: 2048）
- `--batch_size`: ファインチューニングのバッチサイズ（デフォルト: 4）
- `--lr`: 学習率（デフォルト: 0.0001）
- `--seed`: 乱数シード（デフォルト: 5）

## 出力構造

```
output_dir/
├── models/
│   ├── initial/          # ループ前の初期モデル
│   ├── loop_01/          # 1回目のループ後のモデル
│   ├── loop_02/          # 2回目のループ後のモデル
│   └── ...
└── loop_00/              # 1回目のデータ収集結果
    ├── trajectories.h5   # HDF5形式のトラジェクトリデータ
    └── loop_00.json      # メタデータ
└── loop_01/              # 2回目のデータ収集結果
    └── ...
```

## 実装の詳細

### 1. 初期ポリシーの準備

`add_new_group_to_model`関数で新しいグループのエンコーダー/デコーダーを追加し、`copy_encoder_decoder_from_reference`関数で参考ロボットのエンコーダー/デコーダーをコピーします。

### 2. データ収集

`collect_data_with_model`関数で、現在のモデルを使用して環境を実行し、トラジェクトリデータを収集します。データはHDF5形式で保存され、MultiTaskMapDatasetで読み込める形式になっています。

### 3. ファインチューニング

`finetune_on_collected_data`関数で、収集したデータを使用してエンコーダー/デコーダーのみをファインチューニングします。Transformerは凍結されているため、訓練されるパラメータは新しいグループのエンコーダー/デコーダーのみです。

### 4. 自己改善ループ

メイン関数で、データ収集→ファインチューニングのループを指定回数繰り返します。

## 注意事項

1. **メモリ**: データ収集とファインチューニングの両方でGPUメモリを使用します。必要に応じて`batch_size`や`episodes_per_loop`を調整してください。

2. **時間**: ループあたりの時間は、エピソード数とファインチューニングステップ数に依存します。`episodes_per_loop`と`finetune_steps_per_loop`を調整して、時間と性能のバランスを取ってください。

3. **データ収集**: 初期のループでは、モデルがまだ適応していないため、エピソードが短くなる可能性があります。これは正常な動作です。

4. **参考ロボット**: 参考ロボットは、ターゲットロボットに最も似ているロボットを選択してください（例: Minicheetahの場合、Go2やGo1を選択）。

## 評価

ファインチューニング後のモデルを評価するには：

```bash
python3 scripts/save_vintix.py \
  --vintix_path data/self_improving_finetune/minicheetah/models/loop_05 \
  --robot_type minicheetah \
  --parallel \
  --num_envs 100 \
  --max_steps 1000
```

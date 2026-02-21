#!/bin/bash
# Algorithm Distillation データ収集 → Vintix訓練 の自動パイプライン

set -e  # エラーで停止

WORKSPACE="/workspace/vintix_go2"
LOG_DIR="$WORKSPACE/pipeline_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ログディレクトリ作成
mkdir -p "$LOG_DIR"

echo "=============================================================================="
echo "AD Data Collection + Vintix Training Pipeline"
echo "=============================================================================="
echo "Start time: $(date)"
echo "Workspace: $WORKSPACE"
echo ""

cd "$WORKSPACE"
export PYTHONPATH="$WORKSPACE:$PYTHONPATH"
# WANDB: 環境変数 WANDB_API_KEY を参照（公開リポジトリにキーを載せないこと）
if [ -n "${WANDB_API_KEY:-}" ]; then export WANDB_API_KEY; fi

# ============================================================================
# Step 1: データ収集
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Algorithm Distillation Data Collection (Parallel)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

COLLECTION_LOG="$LOG_DIR/data_collection_${TIMESTAMP}.log"

OUTPUT_DIR="data/go2_trajectories/go2_ad_p06_f005"

python scripts/collect_ad_data_parallel.py \
    --model_path /workspace/Genesis/logs/go2-walking/model_300.pt \
    --output_dir "$OUTPUT_DIR" \
    --num_envs 10 \
    --target_steps_per_env 1000000 \
    --max_perf 1.0 \
    --noise_free_fraction 0.05 \
    --decay_power 0.6 \
    --max_steps 1000 \
    2>&1 | tee "$COLLECTION_LOG"

# データ収集の成功を確認（各環境ごとのファイルを確認）
if [ ! -f "$OUTPUT_DIR/trajectories_env_0000.h5" ]; then
    echo "ERROR: Data collection failed - trajectories file not found"
    exit 1
fi

echo ""
echo "✓ Data collection completed successfully"
echo "  Log: $COLLECTION_LOG"
echo ""

# データの簡易チェック
python3 << PYEOF
import h5py
import os
output_dir = "$OUTPUT_DIR"
total_transitions = 0
num_files = 0
for i in range(10):
    data_file = f"{output_dir}/trajectories_env_{i:04d}.h5"
    if os.path.exists(data_file):
        num_files += 1
        with h5py.File(data_file, 'r') as f:
            num_groups = len(f.keys())
            for group_name in f.keys():
                total_transitions += len(f[group_name]['proprio_observation'][:])
print(f"  Files found: {num_files}/10")
print(f"  Total transitions: {total_transitions:,}")
PYEOF

echo ""
sleep 5

# ============================================================================
# Step 2: Vintix訓練
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Vintix Model Training"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

TRAINING_LOG="$LOG_DIR/vintix_training_${TIMESTAMP}.log"
MODEL_NAME="go2_ad_p06_f005_${TIMESTAMP}"

echo "Updating dataset config..."
cat > configs/go2_dataset_config.yaml << YAML_EOF
# Go2 Dataset Configuration for Vintix

# Use the newly collected AD dataset (p=0.6, f=0.05) at data/go2_trajectories/go2_ad_p06_f005

go2_ad_p06_f005:
  type: "default"
  path: "go2_ad_p06_f005"
  group: "go2_locomotion"
  reward_scale: 1.0
  episode_sparsity: 1
YAML_EOF

python scripts/train_vintix.py \
    --data_dir data/go2_trajectories \
    --epochs 300 \
    --batch_size 4 \
    --context_len 1024 \
    --save_every 10 \
    --project "vintix_go2_ad" \
    --name "${MODEL_NAME}" \
    2>&1 | tee "$TRAINING_LOG"

# 訓練の成功を確認（モデルディレクトリを確認）
if [ ! -d "models/vintix_go2/${MODEL_NAME}" ]; then
    echo "ERROR: Training failed - model directory not found: models/vintix_go2/${MODEL_NAME}"
    exit 1
fi

echo ""
echo "✓ Vintix training completed successfully"
echo "  Log: $TRAINING_LOG"
echo ""

# ============================================================================
# 完了
# ============================================================================
echo ""
echo "=============================================================================="
echo "Pipeline Completed Successfully!"
echo "=============================================================================="
echo "End time: $(date)"
echo ""
echo "Outputs:"
echo "  - Data: $OUTPUT_DIR/"
echo "  - Model: models/vintix_go2/${MODEL_NAME}/"
echo "  - Logs: $LOG_DIR/"
echo ""
echo "Next step: Evaluate the trained model with scripts/eval_vintix.py"
echo "=============================================================================="


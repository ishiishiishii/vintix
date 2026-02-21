#!/bin/bash
#
# 単一環境でのAlgorithm Distillationデータ収集 + Vintix訓練
# 自動パイプライン
#

set -e  # エラーで停止

echo "================================================================================"
echo "Algorithm Distillation Auto Pipeline (Single Environment)"
echo "================================================================================"
echo ""
echo "Phase 1: Data Collection (Single Environment)"
echo "  - Episodes: 5120"
echo "  - Estimated time: ~58 hours (2.4 days)"
echo ""
echo "Phase 2: Vintix Training"
echo "  - Epochs: 100"
echo "  - Estimated time: ~1-2 days"
echo ""
echo "Total estimated time: ~3.5-4.5 days"
echo "================================================================================"
echo ""

# ================================================================================
# Phase 1: データ収集
# ================================================================================

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 1: Data Collection..."
echo ""

OUTPUT_DIR="data/go2_trajectories/go2_ad_single_env"

python scripts/collect_ad_data.py \
    --model_path /workspace/Genesis/logs/go2-walking/model_300.pt \
    --output_dir ${OUTPUT_DIR} \
    --num_episodes 5120 \
    --max_perf 1.0

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 1 completed!"
echo ""

# データ収集の検証
echo "Verifying collected data..."
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: Output directory not found: ${OUTPUT_DIR}"
    exit 1
fi

DATA_FILE="${OUTPUT_DIR}/trajectories_0000.h5"
if [ ! -f "${DATA_FILE}" ]; then
    echo "ERROR: Data file not found: ${DATA_FILE}"
    exit 1
fi

echo "✓ Data file exists: ${DATA_FILE}"

# Pythonでデータの基本的な検証
python << EOF
import h5py
import sys

try:
    with h5py.File("${DATA_FILE}", 'r') as f:
        num_groups = len(f.keys())
        print(f"✓ HDF5 groups: {num_groups}")
        
        if num_groups == 0:
            print("ERROR: No data groups found")
            sys.exit(1)
        
        # 最初のグループをチェック
        first_group = list(f.keys())[0]
        group = f[first_group]
        
        required_keys = ['proprio_observation', 'action', 'reward', 'step_num']
        for key in required_keys:
            if key not in group:
                print(f"ERROR: Missing key '{key}' in HDF5 group")
                sys.exit(1)
        
        print(f"✓ All required keys present")
        print(f"✓ Data validation passed!")
        
except Exception as e:
    print(f"ERROR: Data validation failed: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "Data validation failed. Exiting."
    exit 1
fi

echo ""
echo "================================================================================"
echo ""

# ================================================================================
# Phase 2: Vintix訓練
# ================================================================================

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 2: Vintix Training..."
echo ""

# データセット設定を更新
echo "Updating dataset config..."
cat > configs/go2_dataset_config.yaml << YAML_EOF
go2_ad_single_env:
  obs_dim: 45
  action_dim: 12
  path: go2_ad_single_env
  domain_name: go2_locomotion
YAML_EOF

echo "✓ Updated configs/go2_dataset_config.yaml"
echo ""

# Vintix訓練を実行
python scripts/train_vintix.py \
    --data_dir data/go2_trajectories \
    --epochs 100 \
    --batch_size 4 \
    --context_len 1024 \
    --save_every 10 \
    --wandb_project "vintix_go2_ad" \
    --wandb_run_name "go2_ad_single_env_$(date +%Y%m%d_%H%M%S)" \
    --disable_wandb

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2 completed!"
echo ""

# ================================================================================
# 完了
# ================================================================================

echo "================================================================================"
echo "Auto Pipeline Completed Successfully!"
echo "================================================================================"
echo ""
echo "Output Data:"
echo "  - Collected data: ${OUTPUT_DIR}/"
echo "  - Trained models: models/vintix_go2/"
echo ""
echo "Next steps:"
echo "  1. Evaluate the trained Vintix model:"
echo "     python scripts/eval_vintix.py --vintix_path models/vintix_go2/model_final.pt"
echo ""
echo "  2. Visualize training data:"
echo "     python scripts/visualize_multi_trajectory.py --input_dir ${OUTPUT_DIR}"
echo ""
echo "================================================================================"


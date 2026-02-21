#!/bin/bash
# p=0.1から1.0まで（0.1刻み）データ収集と可視化を一度に実行

set -e

BASE_DIR="data/go2_trajectories"
NUM_ENVS=10
TARGET_STEPS_PER_ENV=10000
MAX_PERF=1.0
MAX_STEPS=1000

echo "=========================================="
echo "Collecting and visualizing data for p=0.1 to 1.0 (step 0.1)"
echo "=========================================="
echo ""

# p=0.1から1.0まで（0.1刻み）ループ
for p in $(seq 0.1 0.1 1.0); do
    p_str=$(printf "%.1f" $p)
    p_str_no_dot=$(echo $p_str | tr -d '.')
    
    output_dir="${BASE_DIR}/test_parallel_p${p_str_no_dot}"
    
    echo "=========================================="
    echo "Processing p=${p_str}..."
    echo "=========================================="
    
    # データ収集
    echo "[Phase 1] Collecting data with p=${p_str}..."
    docker exec genesis_tensorboard bash -lc "cd /workspace/vintix_go2 && python scripts/collect_ad_data_parallel.py \
        --model_path /workspace/Genesis/logs/go2-walking/model_300.pt \
        --output_dir ${output_dir} \
        --target_steps_per_env ${TARGET_STEPS_PER_ENV} \
        --num_envs ${NUM_ENVS} \
        --max_perf ${MAX_PERF} \
        --decay_power ${p_str} \
        --max_steps ${MAX_STEPS}"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Data collection failed for p=${p_str}"
        exit 1
    fi
    
    # 可視化
    echo "[Phase 2] Visualizing data for p=${p_str}..."
    docker exec genesis_tensorboard bash -lc "cd /workspace/vintix_go2 && python scripts/visualize_trajectories.py \
        ${output_dir} \
        --output ${output_dir}/test_parallel_p${p_str_no_dot}_analysis.png \
        --target_steps_per_env ${TARGET_STEPS_PER_ENV} \
        --max_perf ${MAX_PERF} \
        --p ${p_str}"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Visualization failed for p=${p_str}"
        exit 1
    fi
    
    echo "✓ Completed p=${p_str}"
    echo ""
done

echo "=========================================="
echo "All done! Collected and visualized data for p=0.1 to 1.0"
echo "=========================================="


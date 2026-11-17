#!/usr/bin/env bash
# Train same-body-only Vintix for Go1 / A1 / Mini Cheetah (#18).
# Mirrors models/go2_data1M_5epochs (epochs=5, context=2048, sparsity=128, seed=5).
# Resume-safe: skips a robot if its final epoch folder already exists.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
EXP="$ROOT/same_body_scale_zeroshot"
LOGDIR="$EXP/logs"
mkdir -p "$LOGDIR" "$EXP/models"

train_one() {
  local robot="$1"
  local cfg="$2"
  local group="$3"
  local name="${robot}_data1M_5epochs"
  local out="$EXP/models/${name}"
  local final="$out/${name}/0004_epoch"
  if [[ -d "$final" ]]; then
    echo "[skip] $robot already has $final"
    return 0
  fi
  echo "[train] $robot -> $out"
  python -u scripts/train_vintix.py \
    --data_dir data \
    --dataset_config_paths "[\"same_body_scale_zeroshot/configs/${cfg}\"]" \
    --context_len 2048 \
    --trajectory_sparsity 128 \
    --lr 0.0003 \
    --weight_decay 0.1 \
    --precision bf16 \
    --grad_accum_steps 8 \
    --warmup_ratio 0.005 \
    --batch_size 8 \
    --seed 5 \
    --epochs 5 \
    --save_every 1 \
    --stats_path vintix/stats.json \
    --project Vintix_Go2 \
    --group "$group" \
    --name "$name" \
    --save_dir "$out" \
    2>&1 | tee -a "$LOGDIR/train_${robot}.log"
}

train_one go1 go1_only_dataset.yaml go1_walking
train_one a1 a1_only_dataset.yaml a1_walking
train_one minicheetah minicheetah_only_dataset.yaml minicheetah_walking
echo "[done] same-body model training finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"

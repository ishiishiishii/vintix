#!/usr/bin/env bash
# Train Vintix on successful narrow-DR PPO histories (#21).
# Hyperparams mirror models/go2_data1M_5epochs / same_body train_missing_models.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
EXP="$ROOT/dr_narrow_vintix_bodyscale"
LOGDIR="$EXP/logs"
NAME="dr_narrow_success_5epochs"
OUT="$EXP/models/${NAME}"
FINAL="$OUT/${NAME}/0004_epoch"
mkdir -p "$LOGDIR" "$EXP/models" "$EXP/configs"

python -u "$EXP/build_dataset.py"

if [[ -d "$FINAL" ]]; then
  echo "[skip] already trained: $FINAL"
  exit 0
fi

echo "[train] $NAME -> $OUT"
python -u scripts/train_vintix.py \
  --data_dir data \
  --dataset_config_paths "[\"dr_narrow_vintix_bodyscale/configs/dr_narrow_success_dataset.yaml\"]" \
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
  --stats_path "$EXP/stats.json" \
  --project Vintix_Go2 \
  --group dr_narrow_vintix \
  --name "$NAME" \
  --save_dir "$OUT" \
  2>&1 | tee -a "$LOGDIR/train.log"

echo "[done] train finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"

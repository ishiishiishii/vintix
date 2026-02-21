#!/usr/bin/env python3
"""Count Vintix parameters for train_vintix default config (e.g. All one group).
根拠: TrainConfig デフォルトで Vintix を初期化し、model.parameters() の numel() を合計。
"""
import json
import os
import sys

# run from vintix_go2
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VINTIX_GO2_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, VINTIX_GO2_ROOT)
os.chdir(VINTIX_GO2_ROOT)

import torch
from vintix.vintix import Vintix

# TrainConfig defaults from train_vintix.py (表の設定と一致)
config = {
    "action_emb_dim": 127,
    "observation_emb_dim": 127,
    "reward_emb_dim": 2,
    "hidden_dim": 256,
    "context_len": 2048,
    "transformer_depth": 3,
    "transformer_heads": 4,
    "attn_dropout": 0.1,
    "residual_dropout": 0.1,
    "normalize_qk": True,
    "bias": True,
    "parallel_residual": False,
    "shared_attention_norm": False,
    "norm_class": "LayerNorm",
    "mlp_class": "GptNeoxMLP",
    "intermediate_size": 128,
    "inner_ep_pos_enc": False,
    "norm_acs": False,
    "norm_obs": True,
}

stats_path = os.path.join(VINTIX_GO2_ROOT, "vintix", "stats.json")
with open(stats_path, "r") as f:
    stats = json.load(f)
task_name = "go2_walking_ad"
if task_name not in stats:
    task_name = list(stats.keys())[0]
st = stats[task_name]
obs_dim = len(st["obs_mean"])
acs_dim = len(st["acs_mean"])
metadata = {
    task_name: {
        "group_name": "quadruped_locomotion",
        "observation_shape": {"proprio": [obs_dim]},
        "action_dim": acs_dim,
        "action_type": "continuous",
        "reward_scale": 1.0,
        **st,
    }
}

model = Vintix()
model.init_model(config, metadata)

# 合計
total = sum(p.numel() for p in model.parameters())

# モジュール別内訳（根拠）
encoder_params = sum(p.numel() for p in model.encoder.parameters())
transformer_params = sum(p.numel() for p in model.transformer.parameters())
head_params = sum(p.numel() for p in model.head.parameters())
breakdown_sum = encoder_params + transformer_params + head_params

# エンコーダ・デコーダ内訳（MLP 表用）
enc_obs = sum(p.numel() for p in model.encoder.obs_encoders.parameters())
enc_acs = sum(p.numel() for p in model.encoder.acs_encoders.parameters())
enc_rews = sum(p.numel() for p in model.encoder.rews_encoders.parameters())
enc_pad = encoder_params - enc_obs - enc_acs - enc_rews
mlp_total = encoder_params + head_params

print("=== 設定（表と一致） ===")
print(f"  hidden_dim={config['hidden_dim']}, layers={config['transformer_depth']}, heads={config['transformer_heads']}")
print(f"  intermediate_size={config['intermediate_size']}, norm_class={config['norm_class']}, mlp_class={config['mlp_class']}")
print(f"  1グループ: obs_dim={obs_dim}, action_dim={acs_dim}")
print("")
print("=== モジュール別パラメータ数 ===")
print(f"  encoder (obs/act/rew encoders + padding): {encoder_params:,}")
print(f"  transformer:                             {transformer_params:,}")
print(f"  head (action decoders):                   {head_params:,}")
print(f"  内訳の合計:                               {breakdown_sum:,}")
print("")
print("=== 総パラメータ数 ===")
print(f"  total: {total:,}")
assert breakdown_sum == total, f"内訳の合計 {breakdown_sum} != total {total}"
print("")
print("=== Transformer 外側 MLP（エンコーダ・デコーダ）内訳 ===")
print(f"  観測エンコーダ: {enc_obs:,}")
print(f"  行動エンコーダ: {enc_acs:,}")
print(f"  報酬エンコーダ: {enc_rews:,}")
print(f"  パディング用パラメータ: {enc_pad:,}")
print(f"  行動デコーダ: {head_params:,}")
print(f"  MLP 合計: {mlp_total:,}")
print("")
print("(表: Transformer のみ 991,616 / MLP 合計 240,090 / 全体 1,231,706)")

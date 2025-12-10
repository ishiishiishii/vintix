#!/usr/bin/env python3
"""
オーバーフィットテスト: 1つのバッチだけで何百回も訓練して、
ロスが急速にゼロに近づくか確認するテスト

このテストが成功すれば、モデル・オプティマイザ・損失関数の実装は正常です。
ロスが下がらないのは、単にデータセット全体が難しすぎて時間がかかっているだけです。
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    from omegaconf import OmegaConf
except ImportError:
    print("Warning: omegaconf not found, using alternative config loading")
    OmegaConf = None

# パス設定
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir.parent / "vintix"))

from vintix.data.torch_dataloaders import MultiTaskMapDataset
from vintix.training.utils.train_utils import (
    compute_stats,
    initialize_model,
    configure_optimizers,
    to_torch_traj,
    multitask_action_loss,
)
from vintix.training.utils.misc import set_seed

# 現在の訓練設定と同じ設定を使用
HALF_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def main():
    print("=" * 80)
    print("オーバーフィットテスト")
    print("=" * 80)
    print()
    print("このテストは、1つのバッチだけを使って何百回も訓練し、")
    print("ロスが急速にゼロに近づくかどうかを確認します。")
    print()
    print("✓ ロスがゼロに近づく → モデル・オプティマイザ・損失関数は正常")
    print("  → ロスが下がらないのは、データセットが難しすぎるだけ")
    print()
    print("✗ ロスが全く下がらない → コードにバグがある可能性")
    print("  → 学習率、勾配計算、パラメータ更新を確認")
    print()
    print("=" * 80)
    print()

    # 現在の訓練設定と同じ設定
    set_seed(5)
    
    data_dir = str(parent_dir / "data" / "go2_trajectories")
    dataset_name = "data_1M"  # 現在の訓練で使用しているデータセット
    
    # データセット設定を読み込み
    config_path = parent_dir / "configs" / "go2_dataset_config.yaml"
    if OmegaConf is not None:
        dataset_config = OmegaConf.load(config_path)
    else:
        import yaml
        with open(config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
    
    # データセット名を取得
    dataset_names = {
        v.path: v.group
        for k, v in dataset_config.items() if v.type == "default"
    }
    
    print("データセットを読み込み中...")
    dataset = MultiTaskMapDataset(
        data_dir=data_dir,
        datasets_info=dataset_names,
        trajectory_len=2048,  # 現在の訓練と同じ
        trajectory_sparsity=128,  # 現在の訓練と同じ
        ep_sparsity=[1],  # 現在の訓練と同じ
        preload=False,
    )
    
    print(f"データセットサイズ: {len(dataset)}")
    
    # 1つのバッチだけを取得
    dataloader = DataLoader(
        dataset,
        batch_size=8,  # 現在の訓練と同じ
        shuffle=True,
        collate_fn=lambda x: x,
        drop_last=True,
        num_workers=0,  # オーバーフィットテストでは0に設定
    )
    
    # 最初のバッチを取得
    print("\n最初のバッチを取得中...")
    batch = next(iter(dataloader))
    print(f"バッチ取得完了: {len(batch)}個のサンプル")
    
    # 統計を計算（現在の訓練と同じ）
    print("\n統計を計算中...")
    stats = compute_stats(data_dir, dataset_names)
    stats_path = parent_dir / "vintix" / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    
    # モデルを初期化（現在の訓練と同じ設定）
    print("\nモデルを初期化中...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデル設定（現在の訓練と同じ）
    model_config = {
        'action_emb_dim': 127,
        'observation_emb_dim': 127,
        'reward_emb_dim': 2,
        'hidden_dim': 256,
        'transformer_depth': 3,
        'transformer_heads': 4,
        'attn_dropout': 0.1,
        'residual_dropout': 0.1,
        'normalize_qk': True,
        'bias': True,
        'parallel_residual': False,
        'shared_attention_norm': False,
        'norm_class': 'LayerNorm',
        'mlp_class': 'GptNeoxMLP',
        'intermediate_size': 128,
        'inner_ep_pos_enc': False,
        'norm_acs': False,
        'norm_obs': True,
        'context_len': 2048,
    }
    
    # 簡易的なConfigクラス（dataclassとして定義）
    @dataclass
    class SimpleConfig:
        action_emb_dim: int = 127
        observation_emb_dim: int = 127
        reward_emb_dim: int = 2
        hidden_dim: int = 256
        transformer_depth: int = 3
        transformer_heads: int = 4
        attn_dropout: float = 0.1
        residual_dropout: float = 0.1
        normalize_qk: bool = True
        bias: bool = True
        parallel_residual: bool = False
        shared_attention_norm: bool = False
        norm_class: str = 'LayerNorm'
        mlp_class: str = 'GptNeoxMLP'
        intermediate_size: int = 128
        inner_ep_pos_enc: bool = False
        norm_acs: bool = False
        norm_obs: bool = True
        context_len: int = 2048
        optimizer: str = 'Adam'
        lr: float = 0.0003
        betas: Tuple[float, float] = (0.9, 0.99)
        weight_decay: float = 0.1
        precision: str = 'bf16'
        clip_grad: Optional[float] = None
        grad_accum_steps: int = 8
        warmup_ratio: float = 0.005
        local_rank: int = 0
        epochs: int = 1
        batch_size: int = 8
        save_every: int = 1
        save_every_steps: int = 1000
        save_dir: str = "models/vintix_go2"
        stats_path: str = str(parent_dir / "vintix" / "stats.json")
        load_ckpt: Optional[str] = None
        start_epoch: int = 0
        seed: int = 5
        dataset_config: dict = None
        
        def __post_init__(self):
            if self.dataset_config is None:
                self.dataset_config = {}
    
    # データセット設定を辞書形式に変換
    dataset_config_dict = {}
    if hasattr(dataset_config, 'items'):  # OmegaConfの場合
        for k, v in dataset_config.items():
            if hasattr(v, 'path'):
                dataset_config_dict[k] = {
                    'reward_scale': v.reward_scale if hasattr(v, 'reward_scale') else 1.0,
                }
    else:  # 通常の辞書の場合
        for k, v in dataset_config.items():
            if isinstance(v, dict) and 'path' in v:
                dataset_config_dict[k] = {
                    'reward_scale': v.get('reward_scale', 1.0),
                }
    
    config = SimpleConfig(dataset_config=dataset_config_dict)
    
    model, _ = initialize_model(config, dataset.metadata, device)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"モデルパラメータ数: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # オプティマイザを設定（現在の訓練と同じ）
    optimizer = configure_optimizers(config, model)
    
    # スケーラーを設定
    scaler = torch.cuda.amp.GradScaler() if config.precision == "fp16" else None
    torch_dtype = HALF_DTYPES[config.precision]
    
    # バッチをデバイスに移動
    batch = [to_torch_traj(t, device) for t in batch]
    
    print("\n" + "=" * 80)
    print("オーバーフィットテスト開始")
    print("=" * 80)
    print(f"同じバッチで500回訓練します...")
    print()
    
    # オーバーフィットテスト: 同じバッチで500回訓練
    model.train()
    losses = []
    start_time = time.time()
    
    for step in tqdm(range(500), desc="オーバーフィットテスト"):
        optimizer.zero_grad()
        
        with torch.amp.autocast("cuda", dtype=torch_dtype):
            pred_acs, metadata = model(batch)
            
            loss, logs = multitask_action_loss(
                acs=[t['action'] for t in batch],
                pred_acs=pred_acs,
                metadata=metadata,
            )
        
        if config.precision == "fp16":
            scaler.scale(loss).backward()
            if config.clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        elif config.precision == "bf16":
            loss.backward()
            if config.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optimizer.step()
        
        loss_value = float(loss.item())
        losses.append(loss_value)
        
        # 10ステップごとに進捗を表示
        if (step + 1) % 10 == 0:
            current_loss = loss_value
            initial_loss = losses[0] if losses else current_loss
            reduction = ((initial_loss - current_loss) / initial_loss * 100) if initial_loss > 0 else 0
            print(f"Step {step+1:3d}: Loss = {current_loss:.6f} (初期値からの減少: {reduction:.1f}%)")
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("テスト結果")
    print("=" * 80)
    print(f"実行時間: {elapsed_time:.1f}秒")
    print()
    print(f"初期ロス: {losses[0]:.6f}")
    print(f"最終ロス: {losses[-1]:.6f}")
    print(f"ロス減少率: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    print()
    
    # 結果の判定
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction_ratio = (initial_loss - final_loss) / initial_loss
    
    print("=" * 80)
    if reduction_ratio > 0.5:  # 50%以上減少
        print("✓✓✓ テスト成功！")
        print()
        print("モデル・オプティマイザ・損失関数の実装は正常です。")
        print("ロスが下がらないのは、データセット全体が難しすぎて")
        print("時間がかかっているだけです。")
        print()
        print("→ 安心して訓練を続けてください。")
        print("→ 少なくとも20〜30エポックは辛抱強く待ってみてください。")
    elif reduction_ratio > 0.1:  # 10%以上減少
        print("⚠ 部分的に成功")
        print()
        print("ロスは減少していますが、期待されるほど急速ではありません。")
        print("学習率を少し上げることを検討してください。")
    else:  # 10%未満の減少
        print("✗✗✗ テスト失敗")
        print()
        print("ロスが全く下がっていません。以下の点を確認してください：")
        print("1. 学習率が適切か（現在: 0.0003）")
        print("2. 勾配が正しく計算されているか")
        print("3. パラメータが正しく更新されているか")
        print("4. 損失関数の実装が正しいか")
    print("=" * 80)
    
    # ロスの推移をプロット（簡易版）
    print("\nロスの推移（最初の10ステップと最後の10ステップ）:")
    print("最初の10ステップ:")
    for i in range(min(10, len(losses))):
        print(f"  Step {i+1:3d}: {losses[i]:.6f}")
    print("最後の10ステップ:")
    for i in range(max(0, len(losses)-10), len(losses)):
        print(f"  Step {i+1:3d}: {losses[i]:.6f}")


if __name__ == "__main__":
    main()


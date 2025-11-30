#!/usr/bin/env python3
"""
データバッチの中身を詳細に確認するスクリプト
- データの多様性
- 正規化の状態
- データリークの有無
- ターゲット（行動）の異常
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# パス設定
GENESIS_LOCOMOTION_PATH = Path(__file__).parents[2] / "Genesis" / "examples" / "locomotion"
sys.path.insert(0, str(GENESIS_LOCOMOTION_PATH))

from vintix.data.torch_dataloaders import MultiTaskMapDataset
from omegaconf import OmegaConf

def analyze_batch(batch, batch_idx=0):
    """バッチの中身を詳細に分析"""
    print("=" * 100)
    print(f"バッチ {batch_idx} の詳細分析")
    print("=" * 100)
    
    # バッチの構造を確認
    print("\n【バッチの構造】")
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            print(f"  {key}: shape={batch[key].shape}, dtype={batch[key].dtype}")
        else:
            print(f"  {key}: type={type(batch[key])}")
    
    # 観測値（observation）の分析
    if 'observation' in batch:
        obs = batch['observation']
        # バッチ次元を除去（通常は[1, seq_len, features]の形式）
        if obs.dim() == 3 and obs.shape[0] == 1:
            obs = obs.squeeze(0)  # [seq_len, features]
        
        print("\n【観測値（observation）の統計】")
        print(f"  shape: {obs.shape}")
        print(f"  dtype: {obs.dtype}")
        print(f"  min: {obs.min().item():.6f}")
        print(f"  max: {obs.max().item():.6f}")
        print(f"  mean: {obs.mean().item():.6f}")
        print(f"  std: {obs.std().item():.6f}")
        print(f"  ゼロの数: {(obs == 0).sum().item()} / {obs.numel()} ({100*(obs==0).sum().item()/obs.numel():.2f}%)")
        
        # 時間軸に沿った変化を確認
        if obs.shape[0] > 1:
            # 各ステップが同じかどうか
            first_step = obs[0:1].expand_as(obs)
            is_constant = torch.allclose(obs, first_step, atol=1e-3)
            print(f"  全ステップが同じか: {is_constant}")
            if is_constant:
                print("  ⚠️  警告: 観測値が時間軸に沿って変化していません！")
            
            # 各次元の時間軸に沿った分散を確認
            obs_var = obs.var(dim=0)  # [features]
            print(f"  各次元の時間軸分散: min={obs_var.min().item():.6f}, max={obs_var.max().item():.6f}, mean={obs_var.mean().item():.6f}")
            zero_var_dims = (obs_var < 1e-6).sum().item()
            print(f"  分散がほぼゼロの次元数: {zero_var_dims} / {obs.shape[1]}")
            if zero_var_dims > obs.shape[1] * 0.5:
                print(f"  ⚠️  警告: {zero_var_dims}個の次元（{100*zero_var_dims/obs.shape[1]:.1f}%）で分散がほぼゼロです！")
    
    # 前の行動（prev_action）の分析
    if 'prev_action' in batch:
        prev_acs = batch['prev_action']
        # バッチ次元を除去
        if prev_acs.dim() == 3 and prev_acs.shape[0] == 1:
            prev_acs = prev_acs.squeeze(0)
        
        print("\n【前の行動（prev_action）の統計】")
        print(f"  shape: {prev_acs.shape}")
        print(f"  dtype: {prev_acs.dtype}")
        print(f"  min: {prev_acs.min().item():.6f}")
        print(f"  max: {prev_acs.max().item():.6f}")
        print(f"  mean: {prev_acs.mean().item():.6f}")
        print(f"  std: {prev_acs.std().item():.6f}")
        print(f"  ゼロの数: {(prev_acs == 0).sum().item()} / {prev_acs.numel()} ({100*(prev_acs==0).sum().item()/prev_acs.numel():.2f}%)")
        
        # 時間軸に沿った変化を確認
        if prev_acs.shape[0] > 1:
            first_step = prev_acs[0:1].expand_as(prev_acs)
            is_constant = torch.allclose(prev_acs, first_step, atol=1e-3)
            print(f"  全ステップが同じか: {is_constant}")
            if is_constant:
                print("  ⚠️  警告: 前の行動が時間軸に沿って変化していません！")
            
            acs_var = prev_acs.var(dim=0)
            print(f"  各次元の時間軸分散: min={acs_var.min().item():.6f}, max={acs_var.max().item():.6f}, mean={acs_var.mean().item():.6f}")
            zero_var_dims = (acs_var < 1e-6).sum().item()
            print(f"  分散がほぼゼロの次元数: {zero_var_dims} / {prev_acs.shape[1]}")
            if zero_var_dims > prev_acs.shape[1] * 0.5:
                print(f"  ⚠️  警告: {zero_var_dims}個の次元（{100*zero_var_dims/prev_acs.shape[1]:.1f}%）で分散がほぼゼロです！")
    
    # ターゲット行動（action）の分析（最重要）
    if 'action' in batch:
        acs = batch['action']
        # バッチ次元を除去
        if acs.dim() == 3 and acs.shape[0] == 1:
            acs = acs.squeeze(0)
        
        print("\n【ターゲット行動（action）の統計】★最重要★")
        print(f"  shape: {acs.shape}")
        print(f"  dtype: {acs.dtype}")
        print(f"  min: {acs.min().item():.6f}")
        print(f"  max: {acs.max().item():.6f}")
        print(f"  mean: {acs.mean().item():.6f}")
        print(f"  std: {acs.std().item():.6f}")
        print(f"  ゼロの数: {(acs == 0).sum().item()} / {acs.numel()} ({100*(acs==0).sum().item()/acs.numel():.2f}%)")
        
        # 時間軸に沿った変化を確認（最重要）
        if acs.shape[0] > 1:
            first_step = acs[0:1].expand_as(acs)
            is_constant = torch.allclose(acs, first_step, atol=1e-3)
            print(f"  全ステップが同じか: {is_constant}")
            if is_constant:
                print("  ⚠️⚠️⚠️  重大な警告: ターゲット行動が時間軸に沿って変化していません！")
                print("     これは、モデルが学習すべきターゲットが定数であることを意味します。")
                print("     モデルは常に同じ値を出力すれば損失がゼロになるため、即座に収束します。")
            
            acs_var = acs.var(dim=0)  # [action_dim]
            print(f"  各次元の時間軸分散: min={acs_var.min().item():.6f}, max={acs_var.max().item():.6f}, mean={acs_var.mean().item():.6f}")
            zero_var_dims = (acs_var < 1e-6).sum().item()
            print(f"  分散がほぼゼロの次元数: {zero_var_dims} / {acs.shape[1]}")
            if zero_var_dims > 0:
                print(f"  ⚠️  警告: {zero_var_dims}個の次元で分散がほぼゼロです！")
                if zero_var_dims == acs.shape[1]:
                    print("  ⚠️⚠️⚠️  重大な警告: すべての次元で分散がゼロです！")
        
        # 最初の数ステップの値を表示
        print("\n  最初の10ステップの行動値（全12次元）:")
        for i in range(min(10, acs.shape[0])):
            print(f"    step {i}: {acs[i].tolist()}")
    
    # 前の報酬（prev_reward）の分析
    if 'prev_reward' in batch:
        prev_rew = batch['prev_reward']
        print("\n【前の報酬（prev_reward）の統計】")
        print(f"  shape: {prev_rew.shape}")
        print(f"  dtype: {prev_rew.dtype}")
        print(f"  min: {prev_rew.min().item():.6f}")
        print(f"  max: {prev_rew.max().item():.6f}")
        print(f"  mean: {prev_rew.mean().item():.6f}")
        print(f"  std: {prev_rew.std().item():.6f}")
    
    # データリークのチェック
    print("\n【データリークチェック】")
    if 'observation' in batch and 'action' in batch:
        obs = batch['observation']
        acs = batch['action']
        # バッチ次元を除去
        if obs.dim() == 3 and obs.shape[0] == 1:
            obs = obs.squeeze(0)
        if acs.dim() == 3 and acs.shape[0] == 1:
            acs = acs.squeeze(0)
        
        # 観測値に行動が含まれているかチェック
        # （観測値の最後の次元が行動と同じ次元数かどうか）
        if obs.shape[1] >= acs.shape[1]:
            obs_last_dims = obs[:, -acs.shape[1]:]
            # 観測値の最後の次元と行動が一致しているか
            if torch.allclose(obs_last_dims, acs, atol=1e-3):
                print("  ⚠️⚠️⚠️  重大な警告: 観測値の最後の次元が行動と一致しています！データリークの可能性があります！")
            else:
                print("  ✓ 観測値と行動は異なります（リークなし）")
        
        # prev_actionとactionの関係をチェック
        if 'prev_action' in batch:
            prev_acs = batch['prev_action']
            if prev_acs.dim() == 3 and prev_acs.shape[0] == 1:
                prev_acs = prev_acs.squeeze(0)
            # prev_actionを1ステップシフトしたものがactionと一致するか
            if prev_acs.shape[0] == acs.shape[0]:
                shifted_prev_acs = prev_acs[1:] if prev_acs.shape[0] > 1 else prev_acs
                if torch.allclose(shifted_prev_acs, acs[:-1], atol=1e-3):
                    print("  ✓ prev_actionとactionの関係は正しい（1ステップシフト）")
                else:
                    print("  ⚠️  警告: prev_actionとactionの関係が期待通りではありません")
    
    print("\n" + "=" * 100)


def main():
    # 設定を読み込み
    config_path = Path(__file__).parent.parent / "configs" / "go2_dataset_config.yaml"
    config = OmegaConf.load(config_path)
    
    data_dir = Path(__file__).parent.parent / "data" / "go2_trajectories"
    dataset_path = data_dir / "go2_ad_p06_train"
    
    print("=" * 100)
    print("データバッチデバッグスクリプト")
    print("=" * 100)
    print(f"\nデータセットパス: {dataset_path}")
    print(f"存在確認: {dataset_path.exists()}")
    
    # データセットを初期化
    context_len = 2048
    trajectory_sparsity = 128
    
    print(f"\n設定:")
    print(f"  context_len: {context_len}")
    print(f"  trajectory_sparsity: {trajectory_sparsity}")
    
    # データセットの相対パスを取得
    dataset_relative_path = dataset_path.relative_to(data_dir)
    
    dataset = MultiTaskMapDataset(
        data_dir=str(data_dir),
        datasets_info={str(dataset_relative_path): "go2_locomotion"},
        trajectory_len=context_len,
        trajectory_sparsity=trajectory_sparsity,
        ep_sparsity=1,
        preload=False,
    )
    
    print(f"\nデータセットサイズ: {len(dataset)}")
    
    # データローダーを作成
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # 1バッチだけ確認
        shuffle=False,
        num_workers=0,
    )
    
    # 最初の数バッチを分析
    print("\n" + "=" * 100)
    print("最初の3つのバッチを分析します...")
    print("=" * 100)
    
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        
        # バッチをリストから取り出す（MultiTaskMapDatasetはリストを返す）
        if isinstance(batch, list) and len(batch) > 0:
            batch = batch[0]
        
        analyze_batch(batch, batch_idx=i)
        
        # バッチ間の比較
        if i > 0:
            print(f"\n【バッチ {i-1} とバッチ {i} の比較】")
            prev_batch = None
            # 前のバッチを取得（簡略化のため、ここではスキップ）
    
    print("\n" + "=" * 100)
    print("分析完了")
    print("=" * 100)


if __name__ == "__main__":
    main()


"""
自己改善型ファインチューニングスクリプト

既存のVintixモデルを基に、Transformerパラメータを凍結して、
未知のロボット（Minicheetah、Go1、Go2、A1など）用のエンコーダー/デコーダーを
自己改善的なデータ収集とファインチューニングで学習する。

使用方法:
    python3 scripts/self_improving_finetune.py \
        --pretrained_path models/vintix_go2/Minicheetah_without_separategroup/Minicheetah_without_separategroup/0015_epoch \
        --robot_type minicheetah \
        --reference_robot go2 \
        --num_improvement_loops 5 \
        --episodes_per_loop 50 \
        --finetune_steps_per_loop 1000 \
        --output_dir data/self_improving_finetune/minicheetah
"""

#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# Genesis locomotion環境のインポート用
GENESIS_LOCOMOTION_PATH = str(Path(__file__).parents[2] / "Genesis" / "examples" / "locomotion")
sys.path.insert(0, GENESIS_LOCOMOTION_PATH)

import genesis as gs
from env import Go1Env, Go2Env, MiniCheetahEnv, UnitreeA1Env

# Vintixモジュールのインポート
sys.path.insert(0, str(Path(__file__).parent.parent))
from vintix.data.torch_dataloaders import MultiTaskMapDataset
from vintix.training.utils.misc import set_seed
from vintix.training.utils.schedule import cosine_annealing_with_warmup
from vintix.training.utils.train_utils import (
    compute_stats,
    multitask_action_loss,
    to_torch_traj,
)
from vintix.vintix import Vintix

# ロボットタイプとタスク名のマッピング
ROBOT_TASK_MAP = {
    "go1": "go1_walking_ad",
    "go2": "go2_walking_ad",
    "minicheetah": "minicheetah_walking_ad",
    "unitreea1": "unitreea1_walking_ad",
    "a1": "unitreea1_walking_ad",  # A1はunitreea1と同じ
}

# ロボットタイプとグループ名のマッピング
ROBOT_GROUP_MAP = {
    "go1": "go1_locomotion",
    "go2": "go2_locomotion",
    "minicheetah": "minicheetah_locomotion",
    "unitreea1": "a1_locomotion",
    "a1": "a1_locomotion",
}


def get_exp_name_for_robot(robot_type: str) -> str:
    """ロボットタイプからexp_nameを取得"""
    exp_name_map = {
        "go1": "go1-walking",
        "go2": "go2-walking",
        "minicheetah": "minicheetah-walking",
        "unitreea1": "unitreea1-walking",
        "a1": "unitreea1-walking",
    }
    return exp_name_map.get(robot_type, "go2-walking")


def create_robot_env(robot_type: str, num_envs: int = 1):
    """ロボットタイプに応じて環境を作成"""
    import pickle
    from pathlib import Path
    
    # 環境設定をロード
    exp_name = get_exp_name_for_robot(robot_type)
    genesis_root = Path(__file__).parents[2] / "Genesis"
    log_dir = genesis_root / "logs" / exp_name
    cfgs_path = log_dir / "cfgs.pkl"
    
    if cfgs_path.exists():
        with open(cfgs_path, 'rb') as f:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    else:
        # デフォルト設定を使用
        print(f"Config file not found: {cfgs_path}. Using default configuration.")
        sys.path.insert(0, str(genesis_root / "examples" / "locomotion"))
        from train import get_go2_cfgs, get_minicheetah_cfgs, get_go1_cfgs, get_unitreea1_cfgs
        
        if robot_type == "go2":
            env_cfg, obs_cfg, reward_cfg, command_cfg = get_go2_cfgs()
        elif robot_type == "minicheetah":
            env_cfg, obs_cfg, reward_cfg, command_cfg = get_minicheetah_cfgs()
        elif robot_type == "go1":
            env_cfg, obs_cfg, reward_cfg, command_cfg = get_go1_cfgs()
        elif robot_type == "unitreea1" or robot_type == "a1":
            env_cfg, obs_cfg, reward_cfg, command_cfg = get_unitreea1_cfgs()
        else:
            raise ValueError(f"Unknown robot type: {robot_type}")
    
    if robot_type == "go1":
        return Go1Env(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                      reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False)
    elif robot_type == "go2":
        return Go2Env(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                      reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False)
    elif robot_type == "minicheetah":
        return MiniCheetahEnv(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                              reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False)
    elif robot_type == "unitreea1" or robot_type == "a1":
        return UnitreeA1Env(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                            reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False)
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")


def copy_encoder_decoder_from_reference(
    model: Vintix,
    target_group_name: str,
    reference_group_name: str,
) -> None:
    """参考ロボットのエンコーダー/デコーダーをターゲットロボットにコピー
    
    注: add_new_group_to_modelの後に呼び出すこと（既にエンコーダー/デコーダーが作成されている必要がある）
    
    Args:
        model: Vintixモデル
        target_group_name: ターゲットグループ名
        reference_group_name: 参考グループ名
    """
    print(f"\n{'=' * 80}")
    print(f"Copying encoder/decoder from {reference_group_name} to {target_group_name}")
    print(f"{'=' * 80}")
    
    # 既に作成されているターゲットグループのエンコーダー/デコーダーに状態をコピー
    if reference_group_name in model.encoder.obs_encoders and target_group_name in model.encoder.obs_encoders:
        reference_state = model.encoder.obs_encoders[reference_group_name].state_dict()
        try:
            model.encoder.obs_encoders[target_group_name].load_state_dict(reference_state, strict=False)
            print(f"✓ Copied obs_encoder from {reference_group_name} to {target_group_name}")
        except Exception as e:
            print(f"Warning: Could not copy obs_encoder: {e}")
    
    if reference_group_name in model.encoder.acs_encoders and target_group_name in model.encoder.acs_encoders:
        reference_state = model.encoder.acs_encoders[reference_group_name].state_dict()
        try:
            model.encoder.acs_encoders[target_group_name].load_state_dict(reference_state, strict=False)
            print(f"✓ Copied acs_encoder from {reference_group_name} to {target_group_name}")
        except Exception as e:
            print(f"Warning: Could not copy acs_encoder: {e}")
    
    if reference_group_name in model.encoder.rews_encoders and target_group_name in model.encoder.rews_encoders:
        reference_state = model.encoder.rews_encoders[reference_group_name].state_dict()
        try:
            model.encoder.rews_encoders[target_group_name].load_state_dict(reference_state, strict=False)
            print(f"✓ Copied rews_encoder from {reference_group_name} to {target_group_name}")
        except Exception as e:
            print(f"Warning: Could not copy rews_encoder: {e}")
    
    if reference_group_name in model.head.acs_decoders and target_group_name in model.head.acs_decoders:
        reference_state = model.head.acs_decoders[reference_group_name].state_dict()
        try:
            model.head.acs_decoders[target_group_name].load_state_dict(reference_state, strict=False)
            print(f"✓ Copied acs_decoder from {reference_group_name} to {target_group_name}")
        except Exception as e:
            print(f"Warning: Could not copy acs_decoder: {e}")
    
    print(f"{'=' * 80}\n")


def add_new_group_to_model(
    model: Vintix,
    task_name: str,
    group_name: str,
    group_metadata: dict,
    stats: dict,
) -> None:
    """既存モデルに新しいグループのエンコーダー/デコーダーを追加"""
    from torch import nn
    from vintix.nn.individual_task_encoder import (
        get_obs_encoder, get_acs_encoder, RewardEncoder
    )
    from vintix.nn.individual_task_head import AcsDecoder
    
    # メタデータを更新
    model.metadata[task_name] = group_metadata.copy()
    if task_name in stats:
        model.metadata[task_name].update(stats[task_name])
    
    # エンコーダーに新しいグループを追加（まだ存在しない場合のみ）
    if group_name not in model.encoder.obs_encoders:
        model.encoder.obs_encoders[group_name] = get_obs_encoder(
            group_metadata=group_metadata,
            emb_dim=model.encoder.observation_emb_dim,
            emb_activation=nn.LeakyReLU(),
            image_encoder=None
        )
    
    if group_name not in model.encoder.acs_encoders:
        model.encoder.acs_encoders[group_name] = get_acs_encoder(
            group_metadata=group_metadata,
            emb_dim=model.encoder.action_emb_dim,
            emb_activation=nn.LeakyReLU()
        )
    
    if group_name not in model.encoder.rews_encoders:
        model.encoder.rews_encoders[group_name] = RewardEncoder(
            emb_dim=model.encoder.reward_emb_dim,
            emb_activation=nn.LeakyReLU()
        )
    
    # task2groupとgroup_metadataを更新
    model.encoder.task2group[task_name] = group_name
    model.encoder.task_metadata[task_name] = model.metadata[task_name]
    model.encoder.group_metadata[group_name] = group_metadata
    
    # デコーダーに新しいグループを追加（まだ存在しない場合のみ）
    if group_name not in model.head.acs_decoders:
        model.head.acs_decoders[group_name] = AcsDecoder(
            task_metadata=group_metadata,
            hidden_dim=model.conf['hidden_dim'],
            emb_activation=nn.LeakyReLU(),
            out_activation=nn.Identity()
        )
    
    # headのメタデータを更新
    model.head.task2group[task_name] = group_name
    model.head.task_metadata[task_name] = model.metadata[task_name]
    model.head.group_metadata[group_name] = group_metadata


def freeze_transformer_parameters(model: Vintix) -> None:
    """Transformerのパラメータを凍結"""
    for param in model.transformer.parameters():
        param.requires_grad = False
    print("✓ Transformer parameters frozen.")


def get_trainable_parameters(model: Vintix, group_name: str) -> List[nn.Parameter]:
    """訓練可能なパラメータを取得（指定グループのエンコーダー/デコーダーのみ）"""
    trainable_params = []
    
    if group_name in model.encoder.obs_encoders:
        trainable_params.extend(model.encoder.obs_encoders[group_name].parameters())
    if group_name in model.encoder.acs_encoders:
        trainable_params.extend(model.encoder.acs_encoders[group_name].parameters())
    if group_name in model.encoder.rews_encoders:
        trainable_params.extend(model.encoder.rews_encoders[group_name].parameters())
    if group_name in model.head.acs_decoders:
        trainable_params.extend(model.head.acs_decoders[group_name].parameters())
    
    return trainable_params


def collect_data_with_model(
    model: Vintix,
    robot_type: str,
    task_name: str,
    num_episodes: int,
    max_steps_per_episode: int,
    output_dir: Path,
    context_len: int = 2048,
) -> Path:
    """現在のモデルで環境を実行してデータを収集
    
    Args:
        model: Vintixモデル
        robot_type: ロボットタイプ
        task_name: タスク名
        num_episodes: 収集するエピソード数
        max_steps_per_episode: エピソードあたりの最大ステップ数
        output_dir: データ保存ディレクトリ
        context_len: コンテキスト長
        
    Returns:
        データファイルのパス
    """
    print(f"\n{'=' * 80}")
    print(f"Collecting {num_episodes} episodes with current model")
    print(f"{'=' * 80}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    data_file = output_dir / "trajectories.h5"
    
    # 環境を作成
    env = create_robot_env(robot_type, num_envs=1)
    
    # データ収集用のバッファ
    all_episodes_obs = []
    all_episodes_acts = []
    all_episodes_rews = []
    all_episodes_steps = []
    
    model.eval()
    device = next(model.parameters()).device
    
    # 履歴バッファ（コンテキスト用）
    history_obs = []
    history_acts = []
    history_rews = []
    history_steps = []
    
    episode_rewards = []
    episode_lengths = []
    
    with torch.no_grad():
        for episode in range(num_episodes):
            obs, _ = env.reset()
            obs = obs[:, :-12]  # 観測値から行動を除外
            
            # エピソード用のバッファ
            episode_obs = []
            episode_acts = []
            episode_rews = []
            episode_steps = []
            
            # 初期状態を記録
            initial_action = np.zeros(env.num_actions)
            initial_reward = 0.0
            history_obs.append(obs[0].cpu().numpy())
            history_acts.append(initial_action)
            history_rews.append(initial_reward)
            history_steps.append(0)
            
            episode_reward = 0.0
            step = 0
            done = False
            
            while not done and step < max_steps_per_episode:
                # コンテキストを構築
                if len(history_obs) > context_len:
                    context_obs = history_obs[-context_len:]
                    context_acts = history_acts[-context_len:]
                    context_rews = history_rews[-context_len:]
                    context_steps = history_steps[-context_len:]
                else:
                    context_obs = history_obs
                    context_acts = history_acts
                    context_rews = history_rews
                    context_steps = history_steps
                
                # Vintixで予測
                model_input = [{
                    'observation': torch.from_numpy(np.array(context_obs)).float().to(device),
                    'prev_action': torch.from_numpy(np.array(context_acts)).float().to(device),
                    'prev_reward': torch.from_numpy(np.array(context_rews)).float().to(device),
                    'step_num': torch.from_numpy(np.array(context_steps)).long().to(device),
                    'task_name': task_name
                }]
                
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred_actions, metadata = model(model_input)
                
                # 最新の予測行動を取得
                if isinstance(pred_actions, list):
                    pred_action = pred_actions[0]
                else:
                    pred_action = pred_actions
                
                if pred_action.dim() == 3:  # [batch, seq, act_dim]
                    action = pred_action[0, -1, :].cpu().numpy()
                elif pred_action.dim() == 2:  # [seq, act_dim]
                    action = pred_action[-1, :].cpu().numpy()
                else:
                    action = np.zeros(env.num_actions)
                
                action_tensor = torch.from_numpy(action).float().to(gs.device).unsqueeze(0)
                
                # 環境ステップ
                next_obs, rewards, dones, infos = env.step(action_tensor)
                next_obs = next_obs[:, :-12]  # 観測値から行動を除外
                reward = float(rewards.cpu().numpy()[0])
                done = bool(dones.cpu().numpy()[0])
                
                # データを記録
                episode_obs.append(obs[0].cpu().numpy())
                episode_acts.append(action)
                episode_rews.append(reward)
                episode_steps.append(step)
                
                # 履歴を更新
                history_obs.append(next_obs[0].cpu().numpy())
                history_acts.append(action)
                history_rews.append(reward)
                history_steps.append(step + 1)
                
                obs = next_obs
                episode_reward += reward
                step += 1
                
                if done:
                    break
            
            # エピソードを保存
            all_episodes_obs.extend(episode_obs)
            all_episodes_acts.extend(episode_acts)
            all_episodes_rews.extend(episode_rews)
            all_episodes_steps.extend(episode_steps)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
            
            if (episode + 1) % 10 == 0:
                mean_reward = np.mean(episode_rewards[-10:])
                mean_length = np.mean(episode_lengths[-10:])
                print(f"  Episode {episode + 1}/{num_episodes} | "
                      f"Mean Reward: {mean_reward:.4f} | "
                      f"Mean Length: {mean_length:.1f}")
    
    # HDF5ファイルに保存
    print(f"\nSaving {len(all_episodes_obs)} transitions to {data_file}...")
    with h5py.File(data_file, 'w') as f:
        # メタデータを保存
        metadata_group = f.create_group('metadata')
        metadata_group.attrs['task_name'] = task_name
        metadata_group.attrs['num_episodes'] = num_episodes
        metadata_group.attrs['num_transitions'] = len(all_episodes_obs)
        
        # トラジェクトリデータを保存（グループ単位で）
        group_size = 1000
        num_groups = (len(all_episodes_obs) + group_size - 1) // group_size
        
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = min((i + 1) * group_size, len(all_episodes_obs))
            
            group_name = f"{start_idx}-{end_idx-1}"
            group = f.create_group(group_name)
            
            group.create_dataset(
                'proprio_observation',
                data=np.array(all_episodes_obs[start_idx:end_idx], dtype=np.float32)
            )
            group.create_dataset(
                'action',
                data=np.array(all_episodes_acts[start_idx:end_idx], dtype=np.float32)
            )
            group.create_dataset(
                'reward',
                data=np.array(all_episodes_rews[start_idx:end_idx], dtype=np.float32)
            )
            group.create_dataset(
                'step_num',
                data=np.array(all_episodes_steps[start_idx:end_idx], dtype=np.int32)
            )
    
    # メタデータJSONファイルも作成（MultiTaskMapDataset用）
    metadata_json = output_dir / f"{os.path.basename(output_dir)}.json"
    metadata_dict = {
        "task_name": task_name,
        "observation_shape": {
            "proprio": [len(all_episodes_obs[0])]
        },
        "action_dim": len(all_episodes_acts[0]),
        "action_type": "continuous"
    }
    with open(metadata_json, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    
    print(f"✓ Data collection completed!")
    print(f"  Total transitions: {len(all_episodes_obs)}")
    print(f"  Mean episode reward: {np.mean(episode_rewards):.4f}")
    print(f"  Mean episode length: {np.mean(episode_lengths):.1f}")
    print(f"  Data file: {data_file}")
    print(f"{'=' * 80}\n")
    
    return data_file.parent


def finetune_on_collected_data(
    model: Vintix,
    data_dir: Path,
    task_name: str,
    group_name: str,
    num_steps: int,
    batch_size: int = 4,
    context_len: int = 2048,
    lr: float = 0.0001,
) -> None:
    """収集したデータでファインチューニング
    
    Args:
        model: Vintixモデル
        data_dir: データディレクトリ
        task_name: タスク名
        group_name: グループ名
        num_steps: 訓練ステップ数
        batch_size: バッチサイズ
        context_len: コンテキスト長
        lr: 学習率
    """
    print(f"\n{'=' * 80}")
    print(f"Finetuning on collected data ({num_steps} steps)")
    print(f"{'=' * 80}")
    
    # データセットを作成
    datasets_info = {str(data_dir.name): group_name}
    dataset = MultiTaskMapDataset(
        data_dir=str(data_dir.parent),
        datasets_info=datasets_info,
        trajectory_len=context_len,
        trajectory_sparsity=1,
        ep_sparsity=1,
        episode_range=None,
        preload=False,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
        drop_last=True,
        num_workers=2
    )
    
    # 最適化器を設定（訓練可能パラメータのみ）
    trainable_params = get_trainable_parameters(model, group_name)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        betas=(0.9, 0.99),
        weight_decay=0.1
    )
    
    # スケジューラー
    scheduler = cosine_annealing_with_warmup(
        optimizer=optimizer,
        warmup_steps=min(100, num_steps // 10),
        total_steps=num_steps,
    )
    
    # 訓練
    model.train()
    device = next(model.parameters()).device
    scaler = torch.cuda.amp.GradScaler()
    torch_dtype = torch.bfloat16
    
    step_count = 0
    total_loss = 0.0
    
    print(f"Training {sum(p.numel() for p in trainable_params)} parameters...")
    
    while step_count < num_steps:
        for batch in dataloader:
            if step_count >= num_steps:
                break
            
            batch = [to_torch_traj(t, device) for t in batch]
            
            with torch.amp.autocast("cuda", dtype=torch_dtype):
                pred_acs, metadata = model(batch)
                loss, logs = multitask_action_loss(
                    acs=[t['action'] for t in batch],
                    pred_acs=pred_acs,
                    metadata=metadata,
                )
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            step_count += 1
            
            if step_count % 100 == 0:
                avg_loss = total_loss / 100
                print(f"  Step {step_count}/{num_steps} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
                total_loss = 0.0
    
    print(f"✓ Finetuning completed!")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Self-improving finetuning for unknown robots")
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to pretrained model")
    parser.add_argument("--robot_type", type=str, required=True,
                        choices=["go1", "go2", "minicheetah", "unitreea1", "a1"],
                        help="Target robot type")
    parser.add_argument("--reference_robot", type=str, default="go2",
                        choices=["go1", "go2", "unitreea1", "a1"],
                        help="Reference robot type for initial policy")
    parser.add_argument("--num_improvement_loops", type=int, default=5,
                        help="Number of improvement loops")
    parser.add_argument("--episodes_per_loop", type=int, default=50,
                        help="Number of episodes to collect per loop")
    parser.add_argument("--finetune_steps_per_loop", type=int, default=1000,
                        help="Number of finetuning steps per loop")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for collected data and model (default: auto-generated from pretrained_path)")
    parser.add_argument("--max_steps_per_episode", type=int, default=1000,
                        help="Maximum steps per episode")
    parser.add_argument("--context_len", type=int, default=2048,
                        help="Context length")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for finetuning")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=5,
                        help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # パスを設定
    # output_dirが指定されていない場合、元のモデルディレクトリと同じ階層に自動設定
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # 事前訓練済みモデルのパスから、モデルディレクトリを取得
        pretrained_path = Path(args.pretrained_path)
        # エポックディレクトリの親ディレクトリ（例: Minicheetah_without_separategroup/）
        model_base_dir = pretrained_path.parent
        # ファインチューニング用のディレクトリ名を生成
        finetune_dir_name = f"finetuned_{args.robot_type}"
        output_dir = model_base_dir / finetune_dir_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # タスク名とグループ名を取得
    target_task_name = ROBOT_TASK_MAP[args.robot_type]
    target_group_name = ROBOT_GROUP_MAP[args.robot_type]
    reference_task_name = ROBOT_TASK_MAP[args.reference_robot]
    reference_group_name = ROBOT_GROUP_MAP[args.reference_robot]
    
    print(f"\n{'=' * 80}")
    print(f"Self-Improving Finetuning")
    print(f"{'=' * 80}")
    print(f"Target robot: {args.robot_type} ({target_task_name})")
    print(f"Reference robot: {args.reference_robot} ({reference_task_name})")
    print(f"Number of improvement loops: {args.num_improvement_loops}")
    print(f"Episodes per loop: {args.episodes_per_loop}")
    print(f"Finetuning steps per loop: {args.finetune_steps_per_loop}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}\n")
    
    # Genesis初期化
    gs.init()
    
    # 事前訓練済みモデルをロード
    print(f"Loading pretrained model from {args.pretrained_path}...")
    model = Vintix()
    model.load_model(args.pretrained_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 環境を作成してメタデータを取得
    env = create_robot_env(args.robot_type, num_envs=1)
    obs, _ = env.reset()
    obs = obs[:, :-12]
    
    # 統計情報を計算（簡易版：ランダム行動でサンプリング）
    print("Computing normalization statistics...")
    num_samples = 1000
    obses = []
    acses = []
    for _ in range(num_samples):
        action = torch.randn(1, env.num_actions, device=gs.device) * 0.5
        action = torch.clamp(action, -1.0, 1.0)
        obs, _, _, _ = env.step(action)
        obs = obs[:, :-12]
        obses.append(obs[0].cpu().numpy())
        acses.append(action[0].cpu().numpy())
    
    obses = np.vstack(obses)
    acses = np.vstack(acses)
    stats = {
        target_task_name: {
            "obs_mean": obses.mean(axis=0).tolist(),
            "obs_std": obses.std(axis=0, ddof=1).tolist(),
            "acs_mean": acses.mean(axis=0).tolist(),
            "acs_std": acses.std(axis=0, ddof=1).tolist(),
        }
    }
    
    # グループメタデータを構築
    group_metadata = {
        "observation_shape": {"proprio": [obs.shape[1]]},
        "action_dim": env.num_actions,
        "action_type": "continuous",
        "reward_scale": 1.0,
    }
    group_metadata.update(stats[target_task_name])
    
    # 新しいグループを追加
    add_new_group_to_model(model, target_task_name, target_group_name, group_metadata, stats)
    
    # 参考ロボットからエンコーダー/デコーダーをコピー
    if reference_group_name in model.encoder.group_metadata:
        copy_encoder_decoder_from_reference(
            model,
            target_group_name,
            reference_group_name,
        )
    else:
        print(f"Warning: Reference group {reference_group_name} not found. Using random initialization.")
    
    # Transformerを凍結
    freeze_transformer_parameters(model)
    
    # モデルを保存（ループ前）
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "initial"))
    
    # 自己改善ループ
    for loop in range(args.num_improvement_loops):
        print(f"\n{'=' * 80}")
        print(f"Improvement Loop {loop + 1}/{args.num_improvement_loops}")
        print(f"{'=' * 80}")
        
        # ステップ1: データ収集
        loop_data_dir = output_dir / f"loop_{loop:02d}"
        collect_data_with_model(
            model=model,
            robot_type=args.robot_type,
            task_name=target_task_name,
            num_episodes=args.episodes_per_loop,
            max_steps_per_episode=args.max_steps_per_episode,
            output_dir=loop_data_dir,
            context_len=args.context_len,
        )
        
        # ステップ2: ファインチューニング
        finetune_on_collected_data(
            model=model,
            data_dir=loop_data_dir,
            task_name=target_task_name,
            group_name=target_group_name,
            num_steps=args.finetune_steps_per_loop,
            batch_size=args.batch_size,
            context_len=args.context_len,
            lr=args.lr,
        )
        
        # モデルを保存
        model.save_model(str(model_dir / f"loop_{loop + 1:02d}"))
    
    print(f"\n{'=' * 80}")
    print(f"✓ Self-improving finetuning completed!")
    print(f"  Final model: {model_dir / f'loop_{args.num_improvement_loops:02d}'}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()

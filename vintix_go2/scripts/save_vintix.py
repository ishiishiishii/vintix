#!/usr/bin/env python3
"""
Vintix モデルの評価と録画を行うスクリプト
"""
import argparse
import os
import pickle
import sys
import json
from pathlib import Path
from collections import deque
from importlib import metadata

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

GENESIS_LOCOMOTION_PATH = str(Path(__file__).parents[2] / "Genesis" / "examples" / "locomotion")
sys.path.insert(0, GENESIS_LOCOMOTION_PATH)
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

import genesis as gs
from env import Go2Env
from env import MiniCheetahEnv
from env import LaikagoEnv
from env import Go1Env
from env import UnitreeA1Env
from env import ANYmalCEnv

sys.path.insert(0, str(Path(__file__).parent.parent))
from vintix.vintix import Vintix


def generate_output_filename(vintix_path: str, robot_type: str, num_envs: int, max_episodes: int, output_dir: str = None, base_mass: float = None):
    """出力ファイル名を自動生成: robotname_envsnumber_episodenumber"""
    filename = f"{robot_type}_{num_envs}envs_{max_episodes}episodes"
    
    if base_mass is not None:
        filename = f"{filename}_mass_{base_mass:.3f}kg"
    
    if output_dir is None:
        model_dir = Path(vintix_path).parent
        result_dir = model_dir / "Result" / robot_type
        result_dir.mkdir(parents=True, exist_ok=True)
        output_path = result_dir / filename
    else:
        output_path = Path(output_dir) / filename
    return str(output_path)


def _create_env(robot_type: str, num_envs: int, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer: bool = False):
    """環境を作成するヘルパー関数"""
    env_classes = {
        "go2": Go2Env,
        "minicheetah": MiniCheetahEnv,
        "laikago": LaikagoEnv,
        "go1": Go1Env,
        "unitreea1": UnitreeA1Env,
        "anymalc": ANYmalCEnv,
    }
    if robot_type not in env_classes:
        raise ValueError(f"Unknown robot type: {robot_type}")
    
    return env_classes[robot_type](
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
    )


def _load_stats_from_trajectory_json(json_path: str, task_name: str) -> dict:
    """Trajectory JSONファイルから正規化統計を読み込む"""
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    if task_name not in metadata.get("task_name", ""):
        print(f"Warning: task_name mismatch. Expected '{task_name}', found '{metadata.get('task_name', 'N/A')}'")
    
    stats = {
        task_name: {
            "obs_mean": metadata["obs_mean"],
            "obs_std": metadata["obs_std"],
            "acs_mean": metadata["acs_mean"],
            "acs_std": metadata["acs_std"],
        }
    }
    return stats


def _compute_averaged_stats_from_known_tasks(model_metadata: dict, task_name: str) -> dict:
    """未知タスク用に、学習済みタスク（既知タスク）の正規化情報を平均して1つの正規化情報を生成する。

    ゼロショット評価ではこの方法をメインで使う。対象ロボットのデータを一切使わず、
    モデルが学習時に用いたタスクの obs_mean/obs_std, acs_mean/acs_std を
    要素ごとに平均したものを未知タスクの正規化として利用する。
    """
    if not model_metadata:
        raise ValueError("Cannot compute averaged stats: model has no known tasks")
    obs_means = []
    obs_stds = []
    acs_means = []
    acs_stds = []
    for tn, meta in model_metadata.items():
        if "obs_mean" in meta and "obs_std" in meta and "acs_mean" in meta and "acs_std" in meta:
            obs_means.append(np.array(meta["obs_mean"]))
            obs_stds.append(np.array(meta["obs_std"]))
            acs_means.append(np.array(meta["acs_mean"]))
            acs_stds.append(np.array(meta["acs_std"]))
    if not obs_means:
        raise ValueError(
            "No task in model_metadata has obs_mean/obs_std/acs_mean/acs_std. "
            "Cannot compute averaged normalization for unknown task."
        )
    obs_mean_avg = np.mean(obs_means, axis=0)
    obs_std_avg = np.mean(obs_stds, axis=0)
    acs_mean_avg = np.mean(acs_means, axis=0)
    acs_std_avg = np.mean(acs_stds, axis=0)
    return {
        task_name: {
            "obs_mean": obs_mean_avg.tolist(),
            "obs_std": obs_std_avg.tolist(),
            "acs_mean": acs_mean_avg.tolist(),
            "acs_std": acs_std_avg.tolist(),
        }
    }


def _get_action_limits_for_robot(robot_type: str, env_cfg, device):
    """ロボットタイプに基づいてaction_limitsを計算"""
    if robot_type == "go2":
        default_joint_angles = torch.tensor([
            0.0, 0.8, -1.5, 0.0, 0.8, -1.5,
            0.0, 1.0, -1.5, 0.0, 1.0, -1.5,
        ], device=device)
        joint_limits = torch.tensor([
            [-1.0472, 1.0472], [-1.5708, 3.4907], [-2.7227, -0.83776],
            [-1.0472, 1.0472], [-1.5708, 3.4907], [-2.7227, -0.83776],
            [-1.0472, 1.0472], [-0.5236, 4.5379], [-2.7227, -0.83776],
            [-1.0472, 1.0472], [-0.5236, 4.5379], [-2.7227, -0.83776],
        ], device=device)
    elif robot_type == "unitreea1":
        default_joint_angles = torch.tensor([
            0.0, 0.8, -1.5, 0.0, 0.8, -1.5,
            0.0, 1.0, -1.5, 0.0, 1.0, -1.5,
        ], device=device)
        joint_limits = torch.tensor([
            [-0.80, 0.80], [-1.05, 4.19], [-2.70, -0.92],
            [-0.80, 0.80], [-1.05, 4.19], [-2.70, -0.92],
            [-0.80, 0.80], [-1.05, 4.19], [-2.70, -0.92],
            [-0.80, 0.80], [-1.05, 4.19], [-2.70, -0.92],
        ], device=device)
    elif robot_type == "go1":
        default_joint_angles = torch.tensor([
            0.0, 0.8, -1.6, 0.0, 0.8, -1.6,
            0.0, 1.0, -1.6, 0.0, 1.0, -1.6,
        ], device=device)
        joint_limits = torch.tensor([
            [-0.863, 0.863], [-0.686, 4.501], [-2.818, -0.888],
            [-0.863, 0.863], [-0.686, 4.501], [-2.818, -0.888],
            [-0.863, 0.863], [-0.686, 4.501], [-2.818, -0.888],
            [-0.863, 0.863], [-0.686, 4.501], [-2.818, -0.888],
        ], device=device)
    elif robot_type == "minicheetah":
        default_joint_angles = torch.tensor([
            0.0, 0.8, -1.5, 0.0, 0.8, -1.5,
            0.0, 1.0, -1.5, 0.0, 1.0, -1.5,
        ], device=device)
        joint_limits = torch.tensor([
            [-0.802, 0.802], [-1.047, 4.189], [-2.697, -0.916],
            [-0.802, 0.802], [-1.047, 4.189], [-2.697, -0.916],
            [-0.802, 0.802], [-1.047, 4.189], [-2.697, -0.916],
            [-0.802, 0.802], [-1.047, 4.189], [-2.697, -0.916],
        ], device=device)
    else:
        return None
    
    action_scale = env_cfg.get("action_scale", 0.25)
    action_limits = (joint_limits - default_joint_angles.unsqueeze(1)) / action_scale
    return action_limits


def _get_task_name_for_robot(robot_type: str, model_metadata: dict):
    """ロボットタイプに基づいてtask_nameを取得
    
    Args:
        robot_type: ロボットタイプ（go2, go1, minicheetah, unitreea1等）
        model_metadata: モデルのメタデータ辞書
        
    Returns:
        (task_name, is_unknown): タスク名と未知タスクかどうかのフラグ
    """
    task_name_map = {
        "go2": "go2_walking_ad",
        "minicheetah": "minicheetah_walking_ad",
        "laikago": "laikago_walking_ad",
        "go1": "go1_walking_ad",
        "unitreea1": "unitreea1_walking_ad",
        "anymalc": "anymalc_walking_ad",
    }
    
    task_name = task_name_map.get(robot_type)
    if task_name is None:
        raise ValueError(f"Unknown robot_type: {robot_type}")
    
    if task_name in model_metadata:
        return task_name, False
    
    return task_name, True


def _run_parallel_evaluation(args, env_cfg, obs_cfg, reward_cfg, command_cfg):
    """並列評価を実行"""
    NUM_ENVS = args.num_envs
    MAX_EPISODE_STEPS = 1000
    MAX_EPISODES = args.max_episodes
    
    print(f"Loading Vintix model from {args.vintix_path}...")
    vintix_model = Vintix()
    vintix_model.load_model(args.vintix_path)
    vintix_model = vintix_model.to(gs.device)
    vintix_model.eval()
    
    print(f"Creating {NUM_ENVS} parallel environments...")
    env = _create_env(args.robot_type, NUM_ENVS, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False)
    print(f"✓ Created {NUM_ENVS} parallel {args.robot_type} environments")
    
    if args.base_mass is not None:
        try:
            base_link = env.robot.get_link("base")
            original_mass = base_link.get_mass()
            base_link.set_mass(args.base_mass)
            print(f"✓ Changed base mass from {original_mass:.3f} kg to {args.base_mass:.3f} kg")
        except Exception as e:
            print(f"Warning: Could not change base mass: {e}")
    
    print("✓ Vintix model loaded")
    
    model_metadata = vintix_model.metadata if hasattr(vintix_model, 'metadata') else {}
    print(f"Available task names in model: {list(model_metadata.keys())}")
    
    task_name, is_unknown = _get_task_name_for_robot(args.robot_type, model_metadata)
    
    if is_unknown:
        print(f"\n{'=' * 80}")
        print(f"Unknown task detected: {task_name}")
        # 既知のデータセットから読み取る設定: --trajectory_stats_path を指定した場合はそちらを優先
        if hasattr(args, 'trajectory_stats_path') and args.trajectory_stats_path is not None:
            stats_path = Path(args.trajectory_stats_path)
            if stats_path.exists():
                print(f"Loading normalization statistics from trajectory JSON: {args.trajectory_stats_path}")
                print(f"{'=' * 80}")
                stats = _load_stats_from_trajectory_json(args.trajectory_stats_path, task_name)
                print(f"✓ Loaded statistics from trajectory data")
            else:
                # メイン: 未知タスクの正規化は学習タスクの正規化情報を平均したものを使用する
                print(f"Trajectory stats file not found. Using averaged normalization from known tasks (main).")
                print(f"{'=' * 80}")
                stats = _compute_averaged_stats_from_known_tasks(model_metadata, task_name)
                print(f"✓ Using averaged obs/acs mean/std from {len([k for k, v in model_metadata.items() if 'obs_mean' in v])} known task(s)")
        else:
            # メイン: 未知タスクの正規化は学習タスクの正規化情報を平均したものを使用する（ゼロショット評価のデフォルト）
            print(f"Using averaged normalization from known tasks (main for zero-shot evaluation).")
            print(f"{'=' * 80}")
            stats = _compute_averaged_stats_from_known_tasks(model_metadata, task_name)
            print(f"✓ Using averaged obs/acs mean/std from {len([k for k, v in model_metadata.items() if 'obs_mean' in v])} known task(s)")
        
        group_name = "quadruped_locomotion"
        print(f"\nAdding task '{task_name}' to model with group '{group_name}'...")
        vintix_model.add_task(task_name, group_name, stats, rew_scale=1.0)
        print(f"✓ Task '{task_name}' added successfully")
        print(f"{'=' * 80}\n")
    
    print(f"Using task_name: {task_name} for robot_type: {args.robot_type}")
    
    CONTEXT_LEN = 2048
    history_buffers = [VintixHistoryBuffer(max_len=CONTEXT_LEN, task_name=task_name) for _ in range(NUM_ENVS)]
    
    obs, _ = env.reset()
    
    from genesis.utils.geom import transform_quat_by_quat as transform_quat
    env_indices = torch.arange(NUM_ENVS, device=gs.device, dtype=torch.long)
    
    pos_offset = (torch.rand(NUM_ENVS, 3, device=gs.device) - 0.5) * 0.2
    pos_offset[:, 2] = 0.0
    env.base_pos[env_indices] = env.base_init_pos + pos_offset
    env.robot.set_pos(env.base_pos[env_indices], zero_velocity=False, envs_idx=env_indices)
    
    roll = (torch.rand(NUM_ENVS, device=gs.device) - 0.5) * 10.0 * np.pi / 180.0
    pitch = (torch.rand(NUM_ENVS, device=gs.device) - 0.5) * 10.0 * np.pi / 180.0
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    quat_noise = torch.stack([cr * cp, cr * sp, sr * cp, -sr * sp], dim=1)
    base_init_quat_expanded = env.base_init_quat.reshape(1, -1).expand(NUM_ENVS, -1)
    env.base_quat[env_indices] = transform_quat(base_init_quat_expanded, quat_noise)
    env.robot.set_quat(env.base_quat[env_indices], zero_velocity=False, envs_idx=env_indices)
    
    dof_noise = (torch.rand(NUM_ENVS, env.num_actions, device=gs.device) - 0.5) * 0.2
    env.dof_pos[env_indices] = env.default_dof_pos + dof_noise
    env.robot.set_dofs_position(
        position=env.dof_pos[env_indices],
        dofs_idx_local=env.motors_dof_idx,
        zero_velocity=True,
        envs_idx=env_indices,
    )
    
    zero_actions = torch.zeros(NUM_ENVS, env.num_actions, device=gs.device)
    obs, _, _, _ = env.step(zero_actions)
    obs = obs[:, :-12]
    
    initial_action = np.zeros(env.num_actions)
    initial_reward = 0.0
    for env_idx in range(NUM_ENVS):
        history_buffers[env_idx].add(obs[env_idx].cpu().numpy(), initial_action, initial_reward)
    
    output_path = Path(args.output)
    graph_path = output_path.parent / f"{output_path.stem}.png"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRunning parallel evaluation...")
    
    all_rewards = []
    env_episode_steps = [0 for _ in range(NUM_ENVS)]
    env_episode_rewards = [[] for _ in range(NUM_ENVS)]
    env_episode_lengths = [[] for _ in range(NUM_ENVS)]
    # Per-environment cumulative step counter (actual executed steps per env).
    # We record it at episode boundaries for episodes.csv.
    env_total_steps = [0 for _ in range(NUM_ENVS)]
    env_episode_cumulative_steps = [[] for _ in range(NUM_ENVS)]
    env_current_episode_rewards = [0.0 for _ in range(NUM_ENVS)]
    env_just_reset = [False for _ in range(NUM_ENVS)]
    env_episode_counts = [0 for _ in range(NUM_ENVS)]
    env_completed = [False for _ in range(NUM_ENVS)]
    
    step_count = 0
    with torch.no_grad():
        while True:
            if all(env_completed):
                break
            
            actions = torch.zeros(NUM_ENVS, env.num_actions, device=gs.device)
            for env_idx in range(NUM_ENVS):
                if env_completed[env_idx]:
                    continue
                
                context = history_buffers[env_idx].get_context(CONTEXT_LEN)
                if context is not None:
                    for key in context[0]:
                        if isinstance(context[0][key], torch.Tensor):
                            context[0][key] = context[0][key].to(gs.device)
                    
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        pred_actions_list, metadata = vintix_model(context)
                    
                    pred_actions = pred_actions_list[0]
                    if isinstance(pred_actions, list):
                        pred_actions = pred_actions[0]
                    
                    if pred_actions.dim() == 3:
                        action = pred_actions[0, -1, :].float()
                    elif pred_actions.dim() == 2:
                        action = pred_actions[-1, :].float()
                    else:
                        raise ValueError(f"Unexpected pred_actions shape: {pred_actions.shape}")
                    
                    actions[env_idx] = action
            
            obs, rewards, dones, infos = env.step(actions)
            obs = obs[:, :-12]
            
            step_rewards = []
            rewards_cpu = rewards.cpu().numpy()
            obs_cpu = obs.cpu().numpy()
            actions_cpu = actions.cpu().numpy()
            
            for env_idx in range(NUM_ENVS):
                if env_completed[env_idx]:
                    continue
                
                reward_value = float(rewards_cpu[env_idx])
                
                if env_just_reset[env_idx]:
                    if env_episode_counts[env_idx] >= MAX_EPISODES:
                        env_completed[env_idx] = True
                        continue
                    
                    step_rewards.append(reward_value)
                    history_buffers[env_idx].add(obs_cpu[env_idx], actions_cpu[env_idx], reward_value)
                    env_episode_steps[env_idx] += 1
                    env_current_episode_rewards[env_idx] += reward_value
                    env_total_steps[env_idx] += 1
                    env_just_reset[env_idx] = False
                    continue
                
                step_rewards.append(reward_value)
                history_buffers[env_idx].add(obs_cpu[env_idx], actions_cpu[env_idx], reward_value)
                env_episode_steps[env_idx] += 1
                env_current_episode_rewards[env_idx] += reward_value
                env_total_steps[env_idx] += 1
                
                if dones[env_idx] or (env_episode_steps[env_idx] >= MAX_EPISODE_STEPS):
                    env_episode_rewards[env_idx].append(env_current_episode_rewards[env_idx])
                    env_episode_lengths[env_idx].append(env_episode_steps[env_idx])
                    env_episode_cumulative_steps[env_idx].append(env_total_steps[env_idx])
                    
                    env_episode_counts[env_idx] += 1
                    if env_episode_counts[env_idx] >= MAX_EPISODES:
                        env_completed[env_idx] = True
                    
                    if not env_completed[env_idx]:
                        reset_indices = torch.tensor([env_idx], device=gs.device, dtype=torch.long)
                        env.reset_idx(reset_indices)
                        obs[env_idx] = env.obs_buf[env_idx, :-12]
                        env_episode_steps[env_idx] = 0
                        env_current_episode_rewards[env_idx] = 0.0
                        env_just_reset[env_idx] = True
            
            all_rewards.append(step_rewards)
            step_count += 1
            
            if step_count % 100 == 0:
                if step_rewards:
                    mean_reward = np.mean(step_rewards)
                    std_reward = np.std(step_rewards) if len(step_rewards) > 1 else 0.0
                else:
                    mean_reward = 0.0
                    std_reward = 0.0
                
                completed_info = [f"{count}/{MAX_EPISODES}{'✓' if env_completed[i] else ''}" 
                                 for i, count in enumerate(env_episode_counts)]
                episode_info = f"Episodes: {completed_info}"
                print(f"Step {step_count:5d} | {episode_info} | Mean Reward: {mean_reward:7.5f} | Std: {std_reward:7.5f}")
    
    print(f"\nCreating performance graphs...")
    
    steps = np.arange(1, len(all_rewards) + 1)
    mean_rewards = [np.mean(rewards) for rewards in all_rewards]
    std_rewards = [np.std(rewards) for rewards in all_rewards]
    
    exclude_last_n_steps = 10
    if len(all_rewards) > exclude_last_n_steps:
        steps = steps[:-exclude_last_n_steps]
        mean_rewards = mean_rewards[:-exclude_last_n_steps]
        std_rewards = std_rewards[:-exclude_last_n_steps]
        all_rewards = all_rewards[:-exclude_last_n_steps]
    
    all_episodes_data = []
    all_episodes_by_episode_num = []
    episode_counter = 0
    for env_idx in range(NUM_ENVS):
        for cum_steps, cum_reward, ep_length in zip(
            env_episode_cumulative_steps[env_idx], 
            env_episode_rewards[env_idx],
            env_episode_lengths[env_idx]
        ):
            all_episodes_data.append((cum_steps, cum_reward, ep_length))
            all_episodes_by_episode_num.append((episode_counter, cum_reward))
            episode_counter += 1
    
    all_episodes_data.sort(key=lambda x: x[0])
    all_episodes_by_episode_num.sort(key=lambda x: x[0])
    
    episode_num_to_rewards = {}
    for env_idx in range(NUM_ENVS):
        for ep_idx, cum_reward in enumerate(env_episode_rewards[env_idx]):
            episode_num_to_rewards.setdefault(ep_idx, []).append(cum_reward)
    
    episode_nums = []
    episode_means = []
    episode_stds = []
    for ep_num in sorted(episode_num_to_rewards.keys()):
        rewards = episode_num_to_rewards[ep_num]
        if rewards:
            episode_nums.append(ep_num + 1)
            episode_means.append(np.mean(rewards))
            episode_stds.append(np.std(rewards) if len(rewards) > 1 else 0.0)
    
    if all_episodes_data:
        max_cum_steps = max(x[0] for x in all_episodes_data)
        num_bins = 100
        step_bins = np.linspace(0, max_cum_steps, num_bins + 1)
        
        bin_cumulative_rewards = []
        for i in range(num_bins):
            rewards_in_bin = [x[1] for x in all_episodes_data if step_bins[i] <= x[0] < step_bins[i + 1]]
            bin_cumulative_rewards.append(rewards_in_bin)
        
        mean_cum_rewards = [np.mean(rews) if rews else np.nan for rews in bin_cumulative_rewards]
        std_cum_rewards = [np.std(rews) if rews and len(rews) > 1 else (0.0 if rews else np.nan) 
                          for rews in bin_cumulative_rewards]
        bin_centers = [(step_bins[i] + step_bins[i + 1]) / 2 for i in range(num_bins)]
        
        valid_mask = ~np.isnan(mean_cum_rewards)
        valid_bin_centers = np.array(bin_centers)[valid_mask]
        valid_mean_rewards = np.array(mean_cum_rewards)[valid_mask]
        valid_std_rewards = np.array(std_cum_rewards)[valid_mask]
    else:
        valid_bin_centers = np.array([])
        valid_mean_rewards = np.array([])
        valid_std_rewards = np.array([])
    
    episode_numbers = [ep[0] for ep in all_episodes_by_episode_num]
    episode_rewards_list = [ep[1] for ep in all_episodes_by_episode_num]
    
    valid_ep_bin_centers = np.array(episode_nums) if episode_nums else np.array([])
    valid_ep_mean_rewards = np.array(episode_means) if episode_means else np.array([])
    valid_ep_std_rewards = np.array(episode_stds) if episode_stds else np.array([])
    
    # グラフスタイル: generate_readable_eval_graphs と同じ文字サイズ、題名なし、凡例は右上
    FONT_SIZE_LABEL = 34
    FONT_SIZE_TICK = 28
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 10))
    ax1.plot(steps, mean_rewards, linewidth=2, label='Mean Reward', color='blue')
    ax1.fill_between(steps,
                     np.array(mean_rewards) - np.array(std_rewards),
                     np.array(mean_rewards) + np.array(std_rewards),
                     alpha=0.3, color='blue', label='±1 Std')
    ax1.set_xlabel('Step', fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel('Reward', fontsize=FONT_SIZE_LABEL)
    ax1.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    ax1.set_ylim(-0.03, 0.03)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=0.5)
    ax1.legend(fontsize=FONT_SIZE_TICK, loc='upper right')
    plt.tight_layout()
    plt.savefig(str(graph_path), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"✓ Graph saved: {graph_path}")
    
    episode_graph_path = graph_path.parent / f"{graph_path.stem}_episodes.png"
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
    if len(valid_ep_bin_centers):
        ax2.plot(valid_ep_bin_centers, valid_ep_mean_rewards, linewidth=2, label='Mean Cumulative Reward per Episode', color='green')
        ax2.fill_between(valid_ep_bin_centers,
                           valid_ep_mean_rewards - valid_ep_std_rewards,
                           valid_ep_mean_rewards + valid_ep_std_rewards,
                           alpha=0.3, color='green', label='±1 Std')
    else:
        if episode_rewards_list:
            mean_ep_reward = np.mean(episode_rewards_list)
            std_ep_reward = np.std(episode_rewards_list)
            ax2.axhline(y=mean_ep_reward, linewidth=2, color='green', linestyle='-', label='Mean Cumulative Reward per Episode')
            ax2.axhspan(mean_ep_reward - std_ep_reward, mean_ep_reward + std_ep_reward,
                        alpha=0.3, color='green', label='±1 Std')
            ax2.plot(episode_numbers, episode_rewards_list, linewidth=1, alpha=0.3, color='green', label='Individual Episodes')
    ax2.set_xlabel('Episode Number', fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel('Cumulative Reward per Episode', fontsize=FONT_SIZE_LABEL)
    ax2.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    ax2.set_ylim(-5, 28)
    ax2.set_xlim(0, 11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=FONT_SIZE_TICK, loc='upper right')
    plt.tight_layout()
    plt.savefig(str(episode_graph_path), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ Episode graph saved: {episode_graph_path}")
    
    csv_path = graph_path.with_suffix('.csv')
    with open(csv_path, 'w') as f:
        f.write("step,mean_reward,std_reward\n")
        for i, (mean_r, std_r) in enumerate(zip(mean_rewards, std_rewards), 1):
            f.write(f"{i},{mean_r:.6f},{std_r:.6f}\n")
    print(f"✓ CSV saved: {csv_path}")
    
    episode_csv_path = graph_path.parent / f"{graph_path.stem}_episodes.csv"
    with open(episode_csv_path, 'w') as f:
        f.write("episode_number,env_index,cumulative_steps,cumulative_reward,episode_length\n")
        for env_idx in range(NUM_ENVS):
            for ep_idx, (cum_steps, cum_reward, ep_length) in enumerate(zip(
                env_episode_cumulative_steps[env_idx],
                env_episode_rewards[env_idx],
                env_episode_lengths[env_idx]
            )):
                f.write(f"{ep_idx},{env_idx},{cum_steps},{cum_reward:.6f},{ep_length}\n")
    print(f"✓ Episode CSV saved: {episode_csv_path}")
    
    final_mean_reward = np.mean(mean_rewards)
    final_std_reward = np.mean(std_rewards)

    # Actual executed steps (sum over environments). Note: step_count is the number of parallel loop iterations.
    total_steps_executed = int(sum(env_total_steps))
    
    all_episode_rewards_flat = [reward for env_rewards in env_episode_rewards for reward in env_rewards]
    all_episode_lengths_flat = [length for env_lengths in env_episode_lengths for length in env_lengths]
    mean_episode_reward = np.mean(all_episode_rewards_flat) if all_episode_rewards_flat else 0.0
    std_episode_reward = np.std(all_episode_rewards_flat) if len(all_episode_rewards_flat) > 1 else 0.0
    mean_episode_length = np.mean(all_episode_lengths_flat) if all_episode_lengths_flat else 0.0
    std_episode_length = np.std(all_episode_lengths_flat) if len(all_episode_lengths_flat) > 1 else 0.0
    total_episodes = len(all_episode_rewards_flat)
    
    actual_episodes_per_env = [len(env_rewards) for env_rewards in env_episode_rewards]
    min_episodes_per_env = min(actual_episodes_per_env) if actual_episodes_per_env else 0
    max_episodes_per_env = max(actual_episodes_per_env) if actual_episodes_per_env else 0
    mean_episodes_per_env = np.mean(actual_episodes_per_env) if actual_episodes_per_env else 0.0
    
    episode_num_to_rewards = {}
    for env_idx in range(NUM_ENVS):
        for ep_idx, cum_reward in enumerate(env_episode_rewards[env_idx]):
            episode_num_to_rewards.setdefault(ep_idx, []).append(cum_reward)
    
    episode_stats = []
    for ep_num in sorted(episode_num_to_rewards.keys()):
        rewards = episode_num_to_rewards[ep_num]
        if rewards:
            episode_stats.append({
                'episode_num': ep_num + 1,
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards) if len(rewards) > 1 else 0.0,
                'num_envs': len(rewards)
            })
    
    mean_reward_path = graph_path.parent / f"{graph_path.stem}_mean_reward.txt"
    with open(mean_reward_path, 'w') as f:
        f.write(f"Vintix Model Evaluation Summary\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Robot Type: {args.robot_type}\n")
        f.write(f"Number of Environments: {NUM_ENVS}\n")
        f.write(f"Evaluation Mode: Episode-based\n")
        f.write(f"\n")
        f.write(f"=== Actual Results (from collected data) ===\n")
        f.write(f"Total Steps Executed: {total_steps_executed}\n")
        f.write(f"Total Episodes Collected: {total_episodes}\n")
        f.write(f"\n")
        f.write(f"Episodes per Environment:\n")
        f.write(f"  Min: {min_episodes_per_env}\n")
        f.write(f"  Max: {max_episodes_per_env}\n")
        f.write(f"  Mean: {mean_episodes_per_env:.2f}\n")
        f.write(f"  Per Environment: {actual_episodes_per_env}\n")
        f.write(f"\n")
        f.write(f"=== Step-based Statistics ===\n")
        f.write(f"Mean Reward per Step: {final_mean_reward:.6f}\n")
        f.write(f"Std Reward per Step: {final_std_reward:.6f}\n")
        f.write(f"\n")
        f.write(f"=== Episode-based Statistics ===\n")
        f.write(f"Mean Reward per Episode: {mean_episode_reward:.6f}\n")
        f.write(f"Std Reward per Episode: {std_episode_reward:.6f}\n")
        if all_episode_rewards_flat:
            f.write(f"Min Reward per Episode: {min(all_episode_rewards_flat):.6f}\n")
            f.write(f"Max Reward per Episode: {max(all_episode_rewards_flat):.6f}\n")
        else:
            f.write(f"Min Reward per Episode: 0.000000\n")
            f.write(f"Max Reward per Episode: 0.000000\n")
        f.write(f"\n")
        f.write(f"Mean Episode Length (steps): {mean_episode_length:.2f}\n")
        f.write(f"Std Episode Length (steps): {std_episode_length:.2f}\n")
        if all_episode_lengths_flat:
            f.write(f"Min Episode Length (steps): {min(all_episode_lengths_flat)}\n")
            f.write(f"Max Episode Length (steps): {max(all_episode_lengths_flat)}\n")
        else:
            f.write(f"Min Episode Length (steps): 0\n")
            f.write(f"Max Episode Length (steps): 0\n")
        f.write(f"\n")
        if episode_stats:
            f.write(f"=== Reward per Episode Number (across all environments) ===\n")
            for stat in episode_stats[:20]:
                f.write(f"Episode {stat['episode_num']:3d}: Mean={stat['mean_reward']:8.6f}, Std={stat['std_reward']:8.6f}, N={stat['num_envs']}\n")
            if len(episode_stats) > 20:
                f.write(f"... (showing first 20 episodes, total {len(episode_stats)} episodes)\n")
        f.write(f"{'=' * 60}\n")
    print(f"✓ Mean reward summary saved: {mean_reward_path}")
    
    print(f"\n{'=' * 80}")
    print(f"✓ Parallel evaluation completed!")
    print(f"  Output graph: {graph_path.absolute()}")
    print(f"  Total steps: {total_steps_executed}")
    print(f"  Number of environments: {NUM_ENVS}")
    print(f"  Mean reward per step: {final_mean_reward:.6f}")
    print(f"  Std reward per step: {final_std_reward:.6f}")
    print(f"{'=' * 80}")
    
    # グラフパスを返す（単一録画で使用）
    return graph_path


def _run_video_recording(args, env_cfg, obs_cfg, reward_cfg, command_cfg):
    """単一環境で1000ステップの動画録画を実行"""
    MAX_RECORDING_STEPS = 1000
    
    print(f"Loading Vintix model from {args.vintix_path}...")
    vintix_model = Vintix()
    vintix_model.load_model(args.vintix_path)
    vintix_model = vintix_model.to(gs.device)
    vintix_model.eval()
    
    model_metadata = vintix_model.metadata if hasattr(vintix_model, 'metadata') else {}
    task_name, is_unknown = _get_task_name_for_robot(args.robot_type, model_metadata)
    
    if is_unknown:
        print(f"\n{'=' * 80}")
        print(f"Unknown task detected: {task_name}")
        # 既知のデータセットから読み取る設定: --trajectory_stats_path を指定した場合はそちらを優先
        if hasattr(args, 'trajectory_stats_path') and args.trajectory_stats_path is not None:
            stats_path = Path(args.trajectory_stats_path)
            if stats_path.exists():
                print(f"Loading normalization statistics from trajectory JSON: {args.trajectory_stats_path}")
                print(f"{'=' * 80}")
                stats = _load_stats_from_trajectory_json(args.trajectory_stats_path, task_name)
                print(f"✓ Loaded statistics from trajectory data")
            else:
                # メイン: 未知タスクの正規化は学習タスクの正規化情報を平均したものを使用する
                print(f"Trajectory stats file not found. Using averaged normalization from known tasks (main).")
                print(f"{'=' * 80}")
                stats = _compute_averaged_stats_from_known_tasks(model_metadata, task_name)
                print(f"✓ Using averaged obs/acs mean/std from {len([k for k, v in model_metadata.items() if 'obs_mean' in v])} known task(s)")
        else:
            # メイン: 未知タスクの正規化は学習タスクの正規化情報を平均したものを使用する（ゼロショット評価のデフォルト）
            print(f"Using averaged normalization from known tasks (main for zero-shot evaluation).")
            print(f"{'=' * 80}")
            stats = _compute_averaged_stats_from_known_tasks(model_metadata, task_name)
            print(f"✓ Using averaged obs/acs mean/std from {len([k for k, v in model_metadata.items() if 'obs_mean' in v])} known task(s)")
        
        group_name = "quadruped_locomotion"
        print(f"\nAdding task '{task_name}' to model with group '{group_name}'...")
        vintix_model.add_task(task_name, group_name, stats, rew_scale=1.0)
        print(f"✓ Task '{task_name}' added successfully")
        print(f"{'=' * 80}\n")
    
    print(f"Using task_name: {task_name} for robot_type: {args.robot_type}")
    
    print(f"\n{'=' * 80}")
    print(f"Starting video recording...")
    print(f"{'=' * 80}")
    
    env = _create_env(args.robot_type, 1, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False)
    print(f"✓ Created {args.robot_type} environment")
    
    if args.base_mass is not None:
        try:
            base_link = env.robot.get_link("base")
            original_mass = base_link.get_mass()
            base_link.set_mass(args.base_mass)
            print(f"✓ Changed base mass from {original_mass:.3f} kg to {args.base_mass:.3f} kg")
        except Exception as e:
            print(f"Warning: Could not change base mass: {e}")
    
    CONTEXT_LEN = 2048
    history_buffer = VintixHistoryBuffer(max_len=CONTEXT_LEN, task_name=task_name)
    
    obs, _ = env.reset()
    obs = obs[:, :-12]
    
    from genesis.utils.geom import transform_quat_by_quat as transform_quat
    env_idx = torch.tensor([0], device=gs.device, dtype=torch.long)
    
    pos_offset = (torch.rand(1, 3, device=gs.device) - 0.5) * 0.2
    pos_offset[:, 2] = 0.0
    env.base_pos[env_idx] = env.base_init_pos + pos_offset
    env.robot.set_pos(env.base_pos[env_idx], zero_velocity=False, envs_idx=env_idx)
    
    roll = (torch.rand(1, device=gs.device) - 0.5) * 10.0 * np.pi / 180.0
    pitch = (torch.rand(1, device=gs.device) - 0.5) * 10.0 * np.pi / 180.0
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    quat_noise = torch.stack([cr * cp, cr * sp, sr * cp, -sr * sp], dim=1)
    base_init_quat_expanded = env.base_init_quat.reshape(1, -1).expand(1, -1)
    env.base_quat[env_idx] = transform_quat(base_init_quat_expanded, quat_noise)
    env.robot.set_quat(env.base_quat[env_idx], zero_velocity=False, envs_idx=env_idx)
    
    dof_noise = (torch.rand(1, env.num_actions, device=gs.device) - 0.5) * 0.2
    env.dof_pos[env_idx] = env.default_dof_pos + dof_noise
    env.robot.set_dofs_position(
        position=env.dof_pos[env_idx],
        dofs_idx_local=env.motors_dof_idx,
        zero_velocity=True,
        envs_idx=env_idx,
    )
    
    zero_actions = torch.zeros(1, env.num_actions, device=gs.device)
    obs, _, _, _ = env.step(zero_actions)
    obs = obs[:, :-12]
    
    initial_action = np.zeros(env.num_actions)
    initial_reward = 0.0
    history_buffer.add(obs[0].cpu().numpy(), initial_action, initial_reward)
    
    output_path = Path(args.output)
    if output_path.suffix == '':
        output_path = output_path.with_suffix('.mp4')
        args.output = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    FPS = 30
    print(f"Recording {MAX_RECORDING_STEPS} steps at {FPS} FPS...")
    env.cam.start_recording()
    
    step_count = 0
    with torch.no_grad():
        while step_count < MAX_RECORDING_STEPS:
            context = history_buffer.get_context(CONTEXT_LEN)
            
            if context is not None:
                for key in context[0]:
                    if isinstance(context[0][key], torch.Tensor):
                        context[0][key] = context[0][key].to(gs.device)
                
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred_actions, metadata = vintix_model(context)
                
                if isinstance(pred_actions, list):
                    pred_actions = pred_actions[0]
                
                if pred_actions.dim() == 3:
                    action = pred_actions[0, -1, :].unsqueeze(0).float()
                elif pred_actions.dim() == 2:
                    action = pred_actions[-1, :].unsqueeze(0).float()
                else:
                    raise ValueError(f"Unexpected pred_actions shape: {pred_actions.shape}")
            else:
                action = torch.zeros(1, env.num_actions, device=gs.device)
            
            obs, rewards, dones, infos = env.step(action)
            obs = obs[:, :-12]
            env.cam.render()
            
            reward_value = float(rewards.cpu().numpy()[0])
            history_buffer.add(obs[0].cpu().numpy(), action[0].cpu().numpy(), reward_value)
            
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"Recording step {step_count:5d} / {MAX_RECORDING_STEPS}...")
            
            if dones[0]:
                reset_indices = torch.tensor([0], device=gs.device, dtype=torch.long)
                env.reset_idx(reset_indices)
                obs[0] = env.obs_buf[0, :-12]
    
    print(f"\nStopping recording and saving to {args.output}...")
    env.cam.stop_recording(save_to_filename=str(args.output), fps=FPS)
    
    print(f"\n{'=' * 80}")
    print(f"✓ Video recording completed!")
    print(f"  Output video: {output_path.absolute()}")
    if output_path.exists():
        print(f"  Video file size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'=' * 80}")




class VintixHistoryBuffer:
    """Vintix用の履歴バッファ（環境リセット後も保持）"""
    
    def __init__(self, max_len=1024, task_name='go2_walking_ad'):
        self.max_len = max_len
        self.task_name = task_name
        self.observations = deque(maxlen=max_len)
        self.actions = deque(maxlen=max_len)
        self.rewards = deque(maxlen=max_len)
        self.step_nums = deque(maxlen=max_len)
        self.current_step = 0
    
    def add(self, obs, action, reward):
        """履歴に追加"""
        self.observations.append(obs.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.step_nums.append(self.current_step)
        self.current_step += 1
    
    def get_context(self, context_len=1024):
        """Vintix用のコンテキストを取得"""
        if not self.observations:
            return None
        
        obs_list = list(self.observations)[-context_len:]
        act_list = list(self.actions)[-context_len:]
        rew_list = list(self.rewards)[-context_len:]
        step_list = list(self.step_nums)[-context_len:]
        
        batch = [{
            'observation': torch.tensor(np.array(obs_list), dtype=torch.float32),
            'prev_action': torch.tensor(np.array(act_list), dtype=torch.float32),
            'prev_reward': torch.tensor(np.array(rew_list), dtype=torch.float32).unsqueeze(1),
            'step_num': torch.tensor(step_list, dtype=torch.int32),
            'task_name': self.task_name,
        }]
        
        return batch


def main():
    parser = argparse.ArgumentParser(description="Save Vintix model behavior as video")
    parser.add_argument("-r", "--robot_type", type=str, choices=["go2", "minicheetah", "laikago", "go1", "unitreea1", "anymalc"], 
                        default=None, action="append", help="Robot type (can be specified multiple times for multiple robots). Default: go2")
    parser.add_argument("--vintix_path", type=str, required=True,
                        help="Path to Vintix model directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (if not specified, auto-generated as robotname_envsnumber_episodenumber)")
    parser.add_argument("--max_episodes", type=int, default=10,
                        help="Maximum number of episodes to evaluate (default: 10)")
    parser.add_argument("--num_envs", type=int, default=10,
                        help="Number of parallel environments (default: 10)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory name (relative to Result directory). If not specified, uses robot_type as directory name.")
    parser.add_argument("--base_mass", type=float, default=None,
                        help="Override base mass of the robot (in kg). If not specified, uses URDF default mass.")
    parser.add_argument("--record", action="store_true",
                        help="Record video of the evaluation")
    parser.add_argument("--trajectory_stats_path", type=str, default=None,
                        help="Path to trajectory JSON file containing normalization statistics. If specified, skips random data collection and uses statistics from this file.")
    args = parser.parse_args()
    
    if args.robot_type is None:
        args.robot_type = ["go2"]
    elif not isinstance(args.robot_type, list):
        args.robot_type = [args.robot_type]
    
    seen = set()
    unique_robot_types = []
    for robot_type in args.robot_type:
        if robot_type not in seen:
            seen.add(robot_type)
            unique_robot_types.append(robot_type)
    
    robot_types = unique_robot_types
    
    if not robot_types:
        raise ValueError("No robot types specified. Please specify at least one robot type with -r/--robot_type.")
    
    def get_exp_name_for_robot(robot_type):
        if robot_type == "go2":
            return "go2-walking"
        elif robot_type == "minicheetah":
            return "minicheetah-walking2"
        elif robot_type == "laikago":
            return "laikago-walking"
        elif robot_type == "go1":
            return "go1-walking"
        elif robot_type == "unitreea1":
            return "unitreea1-walking"
        elif robot_type == "anymalc":
            return "anymalc-walking"
        else:
            return "go2-walking"
    
    print("=" * 80)
    print("Vintix Parallel Evaluation (Multiple Robots)")
    print("=" * 80)
    print(f"Robot types: {', '.join(robot_types)}")
    print(f"Vintix model: {args.vintix_path}")
    print(f"Max episodes: {args.max_episodes}")
    print(f"Mode: Parallel ({args.num_envs} envs, {args.max_episodes} episodes each)")
    if args.base_mass is not None:
        print(f"Base mass override: {args.base_mass:.3f} kg")
    print("=" * 80)
    print()
    
    gs.init(performance_mode=True)
    for robot_idx, robot_type in enumerate(robot_types, 1):
        print(f"\n{'=' * 80}")
        print(f"Evaluating Robot {robot_idx}/{len(robot_types)}: {robot_type}")
        print(f"{'=' * 80}")

        robot_args = argparse.Namespace(**vars(args))
        robot_args.robot_type = robot_type
        
        robot_exp_name = get_exp_name_for_robot(robot_type)
        robot_args.exp_name = robot_exp_name
        
        if args.output_dir is not None:
            model_dir = Path(args.vintix_path).parent
            custom_result_dir = model_dir / "Result" / args.output_dir
            custom_result_dir.mkdir(parents=True, exist_ok=True)
            output_dir_str = str(custom_result_dir)
        else:
            output_dir_str = None
        
        if args.output is None:
            robot_args.output = generate_output_filename(args.vintix_path, robot_type, args.num_envs, args.max_episodes, output_dir=output_dir_str, base_mass=args.base_mass)
        else:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            robot_args.output = str(output_path)
        
        print(f"Robot type: {robot_type}")
        print(f"Experiment name: {robot_args.exp_name}")
        print(f"Output: {robot_args.output}")
        print()
        
        genesis_root = Path(__file__).parents[2] / "Genesis"
        log_dir = genesis_root / "logs" / robot_args.exp_name
        cfgs_path = log_dir / "cfgs.pkl"
        
        if cfgs_path.exists():
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfgs_path, "rb"))
            print(f"Loaded config from: {cfgs_path}")
        else:
            print(f"Config file not found: {cfgs_path}. Using default configuration.")
            from train import get_go2_cfgs, get_minicheetah_cfgs, get_laikago_cfgs, get_go1_cfgs, get_unitreea1_cfgs, get_anymalc_cfgs
            
            if robot_type == "go2":
                env_cfg, obs_cfg, reward_cfg, command_cfg = get_go2_cfgs()
            elif robot_type == "minicheetah":
                env_cfg, obs_cfg, reward_cfg, command_cfg = get_minicheetah_cfgs()
            elif robot_type == "laikago":
                env_cfg, obs_cfg, reward_cfg, command_cfg = get_laikago_cfgs()
            elif robot_type == "go1":
                env_cfg, obs_cfg, reward_cfg, command_cfg = get_go1_cfgs()
            elif robot_type == "unitreea1":
                env_cfg, obs_cfg, reward_cfg, command_cfg = get_unitreea1_cfgs()
            elif robot_type == "anymalc":
                env_cfg, obs_cfg, reward_cfg, command_cfg = get_anymalc_cfgs()
            else:
                raise ValueError(f"Unknown robot type: {robot_type}")
            train_cfg = None
        
        graph_path = _run_parallel_evaluation(robot_args, env_cfg, obs_cfg, reward_cfg, command_cfg)
        
        if args.record:
            print(f"\n{'=' * 80}")
            print(f"Parallel evaluation completed for {robot_type}. Starting video recording...")
            print(f"{'=' * 80}")
            
            video_output_path = graph_path.parent / f"{graph_path.stem}.mp4"
            robot_args.output = str(video_output_path)
            _run_video_recording(robot_args, env_cfg, obs_cfg, reward_cfg, command_cfg)
    print(f"\n{'=' * 80}")
    print(f"✓ All evaluations completed!")
    print(f"  Evaluated {len(robot_types)} robot(s): {', '.join(robot_types)}")
    print(f"{'=' * 80}")
    return
if __name__ == "__main__":
    main()

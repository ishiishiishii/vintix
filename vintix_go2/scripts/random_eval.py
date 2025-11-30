#!/usr/bin/env python3
"""Evaluate PPO policy with controllable epsilon-random action mixing."""

import argparse
import os
import pickle
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

# Genesis locomotion環境のインポート用
GENESIS_LOCOMOTION_PATH = str(Path(__file__).parents[2] / "Genesis" / "examples" / "locomotion")
sys.path.insert(0, GENESIS_LOCOMOTION_PATH)

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as exc:
    raise ImportError(
        "Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'."
    ) from exc

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from env import Go2Env
from env import MiniCheetahEnv
from env import LaikagoEnv


def build_env(robot_type, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True):
    if robot_type == "go2":
        return Go2Env(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
        )
    if robot_type == "minicheetah":
        return MiniCheetahEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
        )
    if robot_type == "laikago":
        return LaikagoEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
        )
    raise ValueError(f"Unknown robot type: {robot_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO expert with epsilon-random action mixing",
    )
    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        default="go2-walking",
        help="Experiment log directory under logs/",
    )
    parser.add_argument(
        "-r",
        "--robot_type",
        type=str,
        choices=["go2", "minicheetah", "laikago"],
        default="go2",
        help="Robot type to evaluate",
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default=300,
        help="Checkpoint index to load (model_<ckpt>.pt)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Fixed interpolation factor between random actions (1.0) and expert actions (0.0). If not specified, epsilon schedule will be used instead.",
    )
    parser.add_argument(
        "--decay_power",
        type=float,
        default=0.5,
        help="Power parameter p for epsilon decay schedule (used when --epsilon is not specified). Lower values -> quicker drop from 1.0 to 0.0",
    )
    parser.add_argument(
        "--noise_free_fraction",
        type=float,
        default=0.05,
        help="Fraction f of trajectory where epsilon=0 (final f*max_steps steps are expert-only, default=0.05). Used with epsilon schedule.",
    )
    parser.add_argument(
        "--random_std",
        type=float,
        default=0.5,
        help="Standard deviation of Gaussian noise for random actions (deprecated: now using uniform distribution)",
    )
    parser.add_argument(
        "--show_viewer",
        action="store_true",
        help="Display the Genesis viewer",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save a video and statistics (similar to save_vintix.py).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="movie/random_eval.mp4",
        help="Output video path when --save_video is enabled.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum number of steps to run when saving (≈10 seconds at 50 Hz).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video FPS when saving.",
    )
    args = parser.parse_args()

    # イプシロン値の設定を決定
    use_epsilon_schedule = args.epsilon is None
    
    if use_epsilon_schedule:
        # イプシロンスケジュールを使用（collect_ad_data_parallel.pyと同じ方法）
        p = args.decay_power
        f = args.noise_free_fraction
        print(f"Using epsilon schedule with p={p:.3f}, f={f:.2f}")
        print(f"  ε(n_s) = (1 - (n_s / ((1-f)*N_s))^p)^{1/p}  if n_s <= (1-f)*N_s")
        print(f"  ε(n_s) = 0                                  if n_s > (1-f)*N_s")
        print(f"  where N_s = max_steps = {args.max_steps}")
    else:
        # 固定のイプシロン値を使用（後方互換性のため）
        if ',' in str(args.epsilon):
            epsilon_values = [float(x.strip()) for x in str(args.epsilon).split(',')]
        else:
            epsilon_values = [float(args.epsilon)]
        
        # 各イプシロン値が有効範囲内かチェック
        for eps in epsilon_values:
            if not (0.0 <= eps <= 1.0):
                raise ValueError(f"epsilon must be within [0.0, 1.0], got {eps}")
        print(f"Using fixed epsilon values: {epsilon_values}")

    gs.init()

    # cfgs.pklのパスを探す（collect_ad_data_parallel.pyと同じ方法）
    log_dir = Path(f"logs/{args.exp_name}")
    cfg_path = log_dir / "cfgs.pkl"
    
    if not cfg_path.exists():
        # Genesisディレクトリ内も探す
        genesis_logs = Path(__file__).parents[2] / "Genesis" / "logs" / args.exp_name
        cfg_path = genesis_logs / "cfgs.pkl"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Could not find cfgs.pkl at {log_dir / 'cfgs.pkl'} or {genesis_logs / 'cfgs.pkl'}")

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(str(cfg_path), "rb"))
    reward_cfg["reward_scales"] = {}

    env = build_env(
        args.robot_type,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        show_viewer=args.show_viewer,
    )

    # モデルパスの解決（collect_ad_data_parallel.pyと同じ方法）
    model_dir = Path(f"logs/{args.exp_name}")
    resume_path = model_dir / f"model_{args.ckpt}.pt"
    if not resume_path.exists():
        genesis_logs = Path(__file__).parents[2] / "Genesis" / "logs" / args.exp_name
        resume_path = genesis_logs / f"model_{args.ckpt}.pt"
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_dir / f'model_{args.ckpt}.pt'} or {genesis_logs / f'model_{args.ckpt}.pt'}")
        model_dir = genesis_logs
    
    runner = OnPolicyRunner(env, train_cfg, str(model_dir), device=gs.device)
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()

    # ランダムアクション生成用のaction_limitsを計算（collect_ad_data_parallel.pyと同じ方法）
    # Go2の各関節の可動域（URDFから）をアクション空間にマッピング
    if args.robot_type == "go2":
        default_joint_angles = torch.tensor([
            0.0,   # FR_hip
            0.8,   # FR_thigh
            -1.5,  # FR_calf
            0.0,   # FL_hip
            0.8,   # FL_thigh
            -1.5,  # FL_calf
            0.0,   # RR_hip
            1.0,   # RR_thigh
            -1.5,  # RR_calf
            0.0,   # RL_hip
            1.0,   # RL_thigh
            -1.5,  # RL_calf
        ], device=gs.device)
        
        joint_limits = torch.tensor([
            [-1.0472,  1.0472],   # FR_hip
            [-1.5708,  3.4907],   # FR_thigh
            [-2.7227, -0.83776],  # FR_calf
            [-1.0472,  1.0472],   # FL_hip
            [-1.5708,  3.4907],   # FL_thigh
            [-2.7227, -0.83776],  # FL_calf
            [-1.0472,  1.0472],   # RR_hip
            [-0.5236,  4.5379],   # RR_thigh
            [-2.7227, -0.83776],  # RR_calf
            [-1.0472,  1.0472],   # RL_hip
            [-0.5236,  4.5379],   # RL_thigh
            [-2.7227, -0.83776],  # RL_calf
        ], device=gs.device)  # shape: [12, 2] where [:, 0]=lower, [:, 1]=upper
        
        # アクション空間での範囲を計算
        # target_dof_pos = action * action_scale + default_dof_pos
        # より: action = (target_dof_pos - default_dof_pos) / action_scale
        action_scale = env_cfg["action_scale"]
        action_limits = (joint_limits - default_joint_angles.unsqueeze(1)) / action_scale  # shape: [12, 2]
    else:
        # 他のロボットタイプの場合は、適切な範囲を設定（必要に応じて拡張）
        # とりあえず、大きな範囲を設定してクリッピングに任せる
        action_limits = None

    print("============================================================")
    print("Randomized PPO Evaluation")
    print(f" Experiment   : {args.exp_name}")
    print(f" Robot        : {args.robot_type}")
    print(f" Checkpoint   : model_{args.ckpt}.pt")
    if use_epsilon_schedule:
        print(f" Epsilon      : Schedule (p={args.decay_power:.3f}, f={args.noise_free_fraction:.2f})")
    else:
        print(f" Epsilon      : Fixed values {epsilon_values}")
    print(f" Random method: Uniform distribution (same as collect_ad_data_parallel.py)")
    print(f" Action mixing: action = epsilon * random + (1 - epsilon) * expert")
    print("============================================================")

    video_enabled = args.save_video
    if video_enabled:
        print("Video capture enabled")
        print(f" Max steps    : {args.max_steps}")
        print(f" FPS          : {args.fps}")

    # イプシロンスケジュールの設定（collect_ad_data_parallel.pyと同じ方法）
    if use_epsilon_schedule:
        p = args.decay_power
        f = args.noise_free_fraction
        target_steps = args.max_steps
        threshold_steps = (1.0 - f) * target_steps
        threshold_steps_tensor = torch.tensor(float(threshold_steps), device=gs.device) if threshold_steps > 0 else None
        threshold_valid = threshold_steps > 0
        
        if not threshold_valid:
            print("Warning: threshold_steps <= 0, epsilon will be 0 from the start")
    else:
        # 固定値モード用の変数（後方互換性）
        threshold_steps_tensor = None
        threshold_valid = False

    # 環境をリセット
    obs, _ = env.reset()
    
    # ビデオ保存の設定
    if video_enabled:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f" Output video : {output_path}")
        env.cam.start_recording()
        episode_rewards = []
        episode_lengths = []
        episode_avg_rewards = []
        total_reward = 0.0
        step_count = 0
        episode_reward = 0.0
        episode_step_count = 0
        episode_count = 0
        epsilon_history = []  # イプシロンの履歴を記録

    # 累積ステップ数（イプシロンスケジュール用）
    total_step_count = 0

    with torch.no_grad():
        step_idx = 0
        while True:
            # イプシロンの計算
            if use_epsilon_schedule:
                if not threshold_valid:
                    eps = 0.0
                else:
                    # ε(n_s) = (1 - (n_s / ((1-f)*N_s))^p)^{1/p}  if n_s <= (1-f)*N_s
                    # ε(n_s) = 0                                  if n_s > (1-f)*N_s
                    ratio = torch.clamp(torch.tensor(total_step_count / threshold_steps, device=gs.device), 0.0, 1.0)
                    ratio_term = torch.pow(ratio, p)
                    eps = float(torch.pow(torch.clamp(1.0 - ratio_term, 0.0), 1.0 / p).item())
            else:
                # 固定値モード（後方互換性のため、最初の値を使用）
                eps = epsilon_values[0] if epsilon_values else 0.0
            
            expert_actions = policy(obs)
            
            # 一様分布からランダムアクションを生成（collect_ad_data_parallel.pyと同じ方法）
            if action_limits is not None:
                # 各関節について、可動域の範囲で一様分布からサンプリング
                # torch.rand() は [0, 1) の一様分布なので、線形変換で [amin, amax] にスケール
                # 式: random_action = amin + rand() * (amax - amin)
                random_actions = action_limits[:, 0] + torch.rand(
                    expert_actions.shape,
                    device=gs.device
                ) * (action_limits[:, 1] - action_limits[:, 0])
            else:
                # フォールバック: 正規分布を使用（他のロボットタイプの場合）
                random_actions = torch.randn_like(expert_actions) * args.random_std
            
            # 線形補間: action = epsilon * random + (1 - epsilon) * expert
            actions = eps * random_actions + (1.0 - eps) * expert_actions
            
            # 関節の可動域範囲でクリッピング（線形補間により範囲外になる可能性があるため）
            if action_limits is not None:
                actions = torch.clamp(actions, action_limits[:, 0], action_limits[:, 1])
            
            obs, rews, dones, infos = env.step(actions)
            step_idx += 1
            total_step_count += 1
            
            if video_enabled:
                env.cam.render()
                reward_value = float(rews[0].item())
                total_reward += reward_value
                episode_reward += reward_value
                step_count += 1
                episode_step_count += 1
                epsilon_history.append(eps)

                if step_count % 100 == 0:
                    avg_reward = total_reward / step_count
                    print(
                        f"Step {step_count:5d} / {args.max_steps} | ε={eps:.4f} | Episode {episode_count + 1} | "
                        f"Ep Step: {episode_step_count:4d} | Ep Reward: {episode_reward:7.3f} | "
                        f"Avg Reward: {avg_reward:7.5f}"
                    )

                if dones[0]:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_step_count)
                    episode_avg_rewards.append(
                        episode_reward / episode_step_count if episode_step_count > 0 else 0.0
                    )
                    print(
                        f"Episode {episode_count + 1} completed | ε={eps:.4f} | Reward: {episode_reward:.3f} | "
                        f"Steps: {episode_step_count}"
                    )
                    episode_count += 1
                    episode_reward = 0.0
                    episode_step_count = 0
                    obs, _ = env.reset()
                if step_count >= args.max_steps:
                    break
            else:
                if dones[0]:
                    obs, _ = env.reset()
                # ビデオ保存なしの場合は、一定ステップ数で終了
                if step_idx >= args.max_steps:
                    break

        if video_enabled:
            print(f"\nStopping recording and saving to {output_path}...")
            env.cam.stop_recording(save_to_filename=str(output_path), fps=args.fps)
            avg_reward_per_step = total_reward / step_count if step_count > 0 else 0.0

            if episode_count > 0:
                graph_path = output_path.with_suffix(".png")
                csv_path = output_path.with_suffix(".csv")

                if use_epsilon_schedule:
                    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                    fig.suptitle(
                        f'Random Eval Performance - ε Schedule (p={p:.3f}, f={f:.2f}) - {output_path.stem}',
                        fontsize=16,
                        fontweight="bold",
                    )
                else:
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    fig.suptitle(
                        f'Random Eval Performance - ε={eps:.2f} - {output_path.stem}',
                        fontsize=16,
                        fontweight="bold",
                    )

                ax1 = axes[0, 0]
                ax1.plot(
                    range(1, len(episode_rewards) + 1),
                    episode_rewards,
                    marker="o",
                    linewidth=2,
                    markersize=4,
                )
                ax1.set_xlabel("Episode", fontsize=11)
                ax1.set_ylabel("Cumulative Reward", fontsize=11)
                ax1.set_title("Episode Cumulative Reward", fontsize=12, fontweight="bold")
                ax1.grid(True, alpha=0.3)
                ax1.axhline(
                    y=np.mean(episode_rewards),
                    color="r",
                    linestyle="--",
                    label=f"Mean: {np.mean(episode_rewards):.3f}",
                )
                ax1.legend()

                ax2 = axes[0, 1]
                ax2.plot(
                    range(1, len(episode_lengths) + 1),
                    episode_lengths,
                    marker="s",
                    color="green",
                    linewidth=2,
                    markersize=4,
                )
                ax2.set_xlabel("Episode", fontsize=11)
                ax2.set_ylabel("Episode Length (steps)", fontsize=11)
                ax2.set_title("Episode Length", fontsize=12, fontweight="bold")
                ax2.grid(True, alpha=0.3)
                ax2.axhline(
                    y=np.mean(episode_lengths),
                    color="r",
                    linestyle="--",
                    label=f"Mean: {np.mean(episode_lengths):.1f}",
                )
                ax2.legend()

                ax3 = axes[1, 0]
                ax3.plot(
                    range(1, len(episode_avg_rewards) + 1),
                    episode_avg_rewards,
                    marker="^",
                    color="orange",
                    linewidth=2,
                    markersize=4,
                )
                ax3.set_xlabel("Episode", fontsize=11)
                ax3.set_ylabel("Average Reward per Step", fontsize=11)
                ax3.set_title(
                    "Episode Average Reward (Reward / Steps)",
                    fontsize=12,
                    fontweight="bold",
                )
                ax3.grid(True, alpha=0.3)
                ax3.axhline(
                    y=np.mean(episode_avg_rewards),
                    color="r",
                    linestyle="--",
                    label=f"Mean: {np.mean(episode_avg_rewards):.5f}",
                )
                ax3.legend()

                if use_epsilon_schedule:
                    # イプシロンスケジュールのグラフを追加
                    ax4 = axes[0, 2]
                    ax4.plot(
                        range(len(epsilon_history)),
                        epsilon_history,
                        color="purple",
                        linewidth=2,
                    )
                    ax4.set_xlabel("Step", fontsize=11)
                    ax4.set_ylabel("Epsilon (ε)", fontsize=11)
                    ax4.set_title("Epsilon Schedule", fontsize=12, fontweight="bold")
                    ax4.grid(True, alpha=0.3)
                    ax4.set_ylim([0, 1.1])
                    
                    # 報酬とイプシロンの関係をプロット
                    ax5 = axes[1, 2]
                    if len(epsilon_history) == len(episode_rewards):
                        # エピソードごとの平均イプシロンと報酬の関係
                        episode_eps = []
                        eps_idx = 0
                        for ep_len in episode_lengths:
                            ep_eps = np.mean(epsilon_history[eps_idx:eps_idx+ep_len]) if ep_len > 0 else 0.0
                            episode_eps.append(ep_eps)
                            eps_idx += ep_len
                        ax5.scatter(episode_eps, episode_rewards, alpha=0.6, s=50)
                        ax5.set_xlabel("Average Epsilon per Episode", fontsize=11)
                        ax5.set_ylabel("Episode Reward", fontsize=11)
                        ax5.set_title("Reward vs Epsilon", fontsize=12, fontweight="bold")
                        ax5.grid(True, alpha=0.3)
                    else:
                        ax5.axis("off")
                        ax5.text(0.5, 0.5, "Epsilon history\nlength mismatch", 
                                ha="center", va="center", fontsize=12)
                    
                    ax6 = axes[1, 1]
                else:
                    ax4 = axes[1, 1]
                
                ax4.axis("off")
                if use_epsilon_schedule:
                    summary_text = f"""
Performance Summary

Total Episodes: {len(episode_rewards)}
Total Steps: {step_count}

Epsilon Schedule:
  p (decay power): {p:.3f}
  f (noise-free): {f:.2f}
  Initial ε: {(epsilon_history[0] if len(epsilon_history) > 0 else 0.0):.4f}
  Final ε: {(epsilon_history[-1] if len(epsilon_history) > 0 else 0.0):.4f}

Cumulative Reward:
  Mean: {np.mean(episode_rewards):.3f}
  Std: {np.std(episode_rewards):.3f}
  Min: {np.min(episode_rewards):.3f}
  Max: {np.max(episode_rewards):.3f}

Episode Length:
  Mean: {np.mean(episode_lengths):.1f}
  Std: {np.std(episode_lengths):.1f}
  Min: {np.min(episode_lengths):.0f}
  Max: {np.max(episode_lengths):.0f}

Avg Reward per Step:
  Mean: {np.mean(episode_avg_rewards):.5f}
  Std: {np.std(episode_avg_rewards):.5f}
  Overall: {avg_reward_per_step:.6f}

Policy: PPO model_{args.ckpt}.pt
"""
                else:
                    summary_text = f"""
Performance Summary

Total Episodes: {len(episode_rewards)}
Total Steps: {step_count}

Cumulative Reward:
  Mean: {np.mean(episode_rewards):.3f}
  Std: {np.std(episode_rewards):.3f}
  Min: {np.min(episode_rewards):.3f}
  Max: {np.max(episode_rewards):.3f}

Episode Length:
  Mean: {np.mean(episode_lengths):.1f}
  Std: {np.std(episode_lengths):.1f}
  Min: {np.min(episode_lengths):.0f}
  Max: {np.max(episode_lengths):.0f}

Avg Reward per Step:
  Mean: {np.mean(episode_avg_rewards):.5f}
  Std: {np.std(episode_avg_rewards):.5f}
  Overall: {avg_reward_per_step:.6f}

Policy: PPO model_{args.ckpt}.pt
Epsilon: {eps:.2f}
Action mixing: {eps:.2f} * random + {1.0 - eps:.2f} * expert
"""
                ax4.text(
                    0.1,
                    0.5,
                    summary_text,
                    fontsize=11,
                    verticalalignment="center",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
                )

                plt.tight_layout()
                plt.savefig(str(graph_path), dpi=150, bbox_inches="tight")
                plt.close(fig)

                with open(csv_path, "w") as f:
                    if use_epsilon_schedule:
                        # エピソードごとの平均イプシロンを計算
                        episode_eps = []
                        eps_idx = 0
                        for ep_len in episode_lengths:
                            ep_eps = np.mean(epsilon_history[eps_idx:eps_idx+ep_len]) if ep_len > 0 and eps_idx < len(epsilon_history) else 0.0
                            episode_eps.append(ep_eps)
                            eps_idx += ep_len
                        f.write(
                            "episode,cumulative_reward,episode_length,avg_reward_per_step,avg_epsilon\n"
                        )
                        for i, (rew, length, avg_rew, ep_eps) in enumerate(
                            zip(episode_rewards, episode_lengths, episode_avg_rewards, episode_eps), 1
                        ):
                            f.write(f"{i},{rew:.6f},{length},{avg_rew:.6f},{ep_eps:.6f}\n")
                    else:
                        f.write(
                            "episode,cumulative_reward,episode_length,avg_reward_per_step\n"
                        )
                        for i, (rew, length, avg_rew) in enumerate(
                            zip(episode_rewards, episode_lengths, episode_avg_rewards), 1
                        ):
                            f.write(f"{i},{rew:.6f},{length},{avg_rew:.6f}\n")

                print(f"✓ Graph saved: {graph_path}")
                print(f"✓ CSV saved  : {csv_path}")

            print("=" * 70)
            if use_epsilon_schedule:
                initial_eps = epsilon_history[0] if len(epsilon_history) > 0 else 0.0
                final_eps = epsilon_history[-1] if len(epsilon_history) > 0 else 0.0
                print(f"✓ Video saved successfully (ε schedule: {initial_eps:.4f} → {final_eps:.4f})")
            else:
                print(f"✓ Video saved successfully (ε={eps:.2f})")
            print(f"  Output : {output_path.resolve()}")
            print(f"  Steps  : {step_count}")
            print(f"  Episodes: {episode_count}")
            print(f"  Avg reward / step: {avg_reward_per_step:.6f}")
            if output_path.exists():
                print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
            print("=" * 70)


if __name__ == "__main__":
    main()

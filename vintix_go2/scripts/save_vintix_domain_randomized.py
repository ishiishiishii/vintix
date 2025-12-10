#!/usr/bin/env python3
"""
Vintix モデルの動作をドメインランダマイゼーション環境で評価し、成長過程を観測するスクリプト

save_vintix.pyをベースに、ドメインランダマイゼーション（質量変化など）を追加

Usage:
    python scripts/save_vintix_domain_randomized.py --vintix_path models/vintix_go2/vintix_go2_ad/0095_epoch --output movie/vintix_domain_randomized.mp4
"""
import argparse
import os
import pickle
import sys
import random
from pathlib import Path
from collections import deque
from importlib import metadata

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Genesis locomotion環境のインポート用
GENESIS_LOCOMOTION_PATH = str(Path(__file__).parents[2] / "Genesis" / "examples" / "locomotion")
sys.path.insert(0, GENESIS_LOCOMOTION_PATH)

# rsl_rl バージョンチェック
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

# Vintixモジュールのインポート
sys.path.insert(0, str(Path(__file__).parent.parent))
from vintix.vintix import Vintix

# PPOモデル用のインポート
from rsl_rl.runners import OnPolicyRunner


class VintixHistoryBuffer:
    """Vintix用の履歴バッファ（環境リセット後も保持）"""
    
    def __init__(self, max_len=1024):
        self.max_len = max_len
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
        """Vintix用のコンテキストを取得（eval_vintix.pyと同じ）"""
        if len(self.observations) == 0:
            return None
        
        # 最新のcontext_len分を取得
        obs_list = list(self.observations)[-context_len:]
        act_list = list(self.actions)[-context_len:]
        rew_list = list(self.rewards)[-context_len:]
        step_list = list(self.step_nums)[-context_len:]
        
        # Vintixの入力形式：eval_vintix.pyと同じ
        batch = [{
            'observation': torch.tensor(np.array(obs_list), dtype=torch.float32),
            'prev_action': torch.tensor(np.array(act_list), dtype=torch.float32),
            'prev_reward': torch.tensor(np.array(rew_list), dtype=torch.float32).unsqueeze(1),
            'step_num': torch.tensor(step_list, dtype=torch.int32),
            'task_name': 'go2_walking_ad',
        }]
        
        return batch


def set_gravity_for_envs(env, gravity_scales, env_indices=None, verbose=False):
    """指定された環境に重力を設定"""
    try:
        # 標準重力は (0.0, 0.0, -9.81)
        default_gravity = np.array([0.0, 0.0, -9.81])
        
        if env_indices is None:
            env_indices = list(range(len(gravity_scales)))
        
        for i, env_idx in enumerate(env_indices):
            gravity_scale = gravity_scales[i] if isinstance(gravity_scales, (list, np.ndarray)) else gravity_scales
            new_gravity = default_gravity * gravity_scale
            # numpy配列として渡す（tensorではなく）
            env.scene.sim.set_gravity(
                new_gravity.astype(np.float32),
                envs_idx=torch.tensor([env_idx], dtype=torch.long, device=gs.device)
            )
            if verbose:
                print(f"  Set gravity for env {env_idx}: {gravity_scale:.3f}x (gravity vector: {new_gravity})")
    except Exception as e:
        print(f"Warning: Could not set gravity: {e}")
        import traceback
        traceback.print_exc()


def randomize_domain_for_all_envs(env, num_envs, mass_range=(0.5, 2.0), gravity_range=(0.9, 1.1)):
    """全環境のドメイン（重力のみ）をランダマイズ（質量は固定）"""
    try:
        # 質量は固定（1.0スケール）
        mass_scales = [1.0] * num_envs
        
        # 各環境で異なる重力スケールを生成（論文では±10%）
        gravity_scales = np.random.uniform(gravity_range[0], gravity_range[1], num_envs)
        
        # 重力を設定
        set_gravity_for_envs(env, gravity_scales, list(range(num_envs)))
        
        return mass_scales, gravity_scales.tolist()
    except Exception as e:
        print(f"Warning: Could not randomize domain: {e}")
        import traceback
        traceback.print_exc()
        return [1.0] * num_envs, [1.0] * num_envs


def _run_single_evaluation(args, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, mass_range, gravity_range=(0.9, 1.1)):
    """単一環境評価を実行（動画保存用）"""
    MAX_STEPS = args.max_steps
    MAX_EPISODE_STEPS = 1000
    
    # 環境作成
    print(f"\nCreating single environment with domain randomization...")
    if args.robot_type == "go2":
        env = Go2Env(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,  # 単一環境の場合はビジュアライズを有効化
            domain_randomization=False,
        )
    elif args.robot_type == "minicheetah":
        env = MiniCheetahEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,
            domain_randomization=False,
        )
    elif args.robot_type == "laikago":
        env = LaikagoEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,
            domain_randomization=False,
        )
    else:
        raise ValueError(f"Unknown robot type: {args.robot_type}")
    
    print(f"✓ Created single {args.robot_type} environment")
    
    # 重力をランダマイズ（質量は固定）
    print(f"\nRandomizing domain parameters (gravity only, mass fixed)...")
    mass_scales, gravity_scales = randomize_domain_for_all_envs(env, 1, mass_range, gravity_range)
    mass_scale = mass_scales[0]
    gravity_scale = gravity_scales[0]
    print(f"  Mass scale: {mass_scale:.3f} (fixed)")
    print(f"  Gravity scale: {gravity_scale:.3f} ({gravity_scale * 100:.1f}% of standard)")
    print(f"✓ Domain randomization completed")
    
    # 重力設定を確認（デバッグ用）
    print(f"\n[DEBUG] Setting gravity to {gravity_scale:.3f}x standard...")
    set_gravity_for_envs(env, [gravity_scale], [0], verbose=True)
    print(f"[DEBUG] Gravity setting completed")
    
    # モデルのロード（VintixまたはPPO）
    use_expert = args.expert_path is not None
    if use_expert:
        # PPO専門家モデルのロード
        print(f"\nLoading PPO expert model from {args.expert_path}...")
        genesis_path = Path(__file__).parents[2] / "Genesis"
        model_dir = genesis_path / "logs" / args.exp_name
        runner = OnPolicyRunner(env, train_cfg, str(model_dir), device=gs.device)
        # パスを絶対パスに変換
        expert_path = Path(args.expert_path)
        if not expert_path.is_absolute():
            # Genesis/で始まる場合は除去
            expert_path_str = args.expert_path
            if expert_path_str.startswith("Genesis/"):
                expert_path_str = expert_path_str[len("Genesis/"):]
            expert_path = genesis_path / expert_path_str
        runner.load(str(expert_path))
        expert_policy = runner.get_inference_policy(device=gs.device)
        print("✓ PPO expert model loaded")
        vintix_model = None
        history_buffer = None
    else:
        # Vintixモデルのロード
        print(f"\nLoading Vintix model from {args.vintix_path}...")
        vintix_model = Vintix()
        vintix_model.load_model(args.vintix_path)
        vintix_model = vintix_model.to(gs.device)
        vintix_model.eval()
        print("✓ Vintix model loaded")
        expert_policy = None
        # 履歴バッファの初期化
        history_buffer = VintixHistoryBuffer(max_len=args.context_len)
    
    # 環境リセット
    obs, _ = env.reset()
    if not use_expert:
        obs = obs[:, :-12]  # Vintixの場合は観測値から行動を除外
    else:
        # PPOの場合は観測値に行動を含める（PPOは45次元の観測値を使用）
        pass
    
    # 重力を再設定（reset()でリセットされる可能性があるため）
    set_gravity_for_envs(env, [gravity_scale], [0], verbose=False)
    
    # 初期状態をランダム化
    randomize_initial_state(env, [0])
    
    # 観測値を更新
    zero_actions = torch.zeros(1, env.num_actions, device=gs.device)
    obs, _, _, _ = env.step(zero_actions)
    if not use_expert:
        obs = obs[:, :-12]
    
    # 初期履歴の追加（Vintixの場合のみ）
    if not use_expert:
        initial_action = np.zeros(env.num_actions)
        initial_reward = 0.0
        history_buffer.add(obs[0].cpu().numpy(), initial_action, initial_reward)
    
    # 出力ディレクトリの作成
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting video recording...")
    print(f"Recording {MAX_STEPS} steps at {args.fps} FPS...")
    env.cam.start_recording()
    
    step_count = 0
    episode_count = 0
    episode_reward = 0.0
    episode_step_count = 0
    total_reward = 0.0
    
    step_rewards = []
    cumulative_rewards = []
    episode_starts = []
    episode_rewards_list = []
    episode_lengths = []
    
    episode_starts.append(0)
    
    with torch.no_grad():
        while step_count < MAX_STEPS:
            # 行動予測（VintixまたはPPO）
            if use_expert:
                # PPO専門家から行動予測
                action = expert_policy(obs)
            else:
                # Vintixから行動予測
                context = history_buffer.get_context(args.context_len)
                
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
            
            # 環境ステップ
            obs, rewards, dones, infos = env.step(action)
            if not use_expert:
                obs = obs[:, :-12]  # Vintixの場合は観測値から行動を除外
            env.cam.render()
            
            reward_value = float(rewards.cpu().numpy()[0])
            if not use_expert:
                # Vintixの場合は履歴バッファに追加
                history_buffer.add(
                    obs[0].cpu().numpy(),
                    action[0].cpu().numpy(),
                    reward_value
                )
            
            step_rewards.append(reward_value)
            total_reward += reward_value
            cumulative_rewards.append(total_reward)
            episode_reward += reward_value
            episode_step_count += 1
            step_count += 1
            
            # エピソード終了判定
            episode_done = dones[0] or (episode_step_count >= MAX_EPISODE_STEPS)
            if episode_done:
                episode_rewards_list.append(episode_reward)
                episode_lengths.append(episode_step_count)
                episode_count += 1
                episode_starts.append(step_count)
                
                # 環境リセット
                obs, _ = env.reset()
                if not use_expert:
                    obs = obs[:, :-12]  # Vintixの場合は観測値から行動を除外
                
                # 重力を再設定（reset()でリセットされる可能性があるため）
                if episode_count == 1:  # 最初のエピソードリセット時のみデバッグ出力
                    print(f"[DEBUG] After reset: Re-setting gravity to {gravity_scale:.3f}x standard...")
                set_gravity_for_envs(env, [gravity_scale], [0], verbose=(episode_count == 1))
                
                randomize_initial_state(env, [0])
                
                zero_actions = torch.zeros(1, env.num_actions, device=gs.device)
                obs, _, _, _ = env.step(zero_actions)
                if not use_expert:
                    obs = obs[:, :-12]
                
                # 履歴バッファに追加（Vintixの場合のみ）
                if not use_expert:
                    history_buffer.add(obs[0].cpu().numpy(), initial_action, initial_reward)
                episode_reward = 0.0
                episode_step_count = 0
    
    # 録画停止と保存
    print(f"\nStopping recording and saving to {args.output}...")
    env.cam.stop_recording(save_to_filename=str(args.output), fps=args.fps)
    
    # グラフの作成
    if len(step_rewards) > 0:
        print(f"\nCreating performance graphs...")
        graph_path = output_path.with_suffix('.png')
        if '_domain_randomized' not in graph_path.stem:
            graph_path = graph_path.parent / f"{graph_path.stem}_domain_randomized.png"
        
        steps = np.arange(1, len(step_rewards) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Vintix Model Performance (Gravity Randomized: {gravity_scale:.3f}x) - {output_path.stem}', fontsize=16, fontweight='bold')
        
        # 1. 累積報酬
        ax1 = axes[0]
        ax1.plot(steps, cumulative_rewards, linewidth=2, label='Cumulative Reward', color='blue')
        for ep_start in episode_starts:
            if ep_start < len(step_rewards):
                ax1.axvline(x=ep_start, color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax1.set_xlabel('Step', fontsize=11)
        ax1.set_ylabel('Cumulative Reward', fontsize=11)
        ax1.set_title('Cumulative Reward vs Steps', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. ステップごとの報酬
        ax2 = axes[1]
        ax2.plot(steps, step_rewards, linewidth=1, alpha=0.6, label='Reward per Step', color='blue')
        for ep_start in episode_starts:
            if ep_start < len(step_rewards):
                ax2.axvline(x=ep_start, color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax2.set_xlabel('Step', fontsize=11)
        ax2.set_ylabel('Reward', fontsize=11)
        ax2.set_title('Reward per Step', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=0.5)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(str(graph_path), dpi=150, bbox_inches='tight')
        print(f"✓ Graph saved: {graph_path}")
        
        # CSVファイルにも保存
        csv_path = graph_path.with_suffix('.csv')
        with open(csv_path, 'w') as f:
            f.write("step,reward,cumulative_reward\n")
            for i, (reward, cum_reward) in enumerate(zip(step_rewards, cumulative_rewards), 1):
                f.write(f"{i},{reward:.6f},{cum_reward:.6f}\n")
        print(f"✓ CSV saved: {csv_path}")
    
    # 最終統計
    print(f"\n{'=' * 80}")
    print(f"✓ Single environment evaluation completed!")
    print(f"  Output video: {output_path.absolute()}")
    if len(step_rewards) > 0:
        print(f"  Output graph: {graph_path.absolute()}")
    print(f"  Total steps: {step_count}")
    print(f"  Number of episodes: {episode_count}")
    print(f"  Gravity scale: {gravity_scale:.3f}x standard ({gravity_scale * 100:.1f}%)")
    if len(step_rewards) > 0:
        print(f"  Mean reward per step: {np.mean(step_rewards):.6f}")
        print(f"  Total cumulative reward: {cumulative_rewards[-1]:.6f}")
    if output_path.exists():
        print(f"  Video file size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'=' * 80}")


def randomize_initial_state(env, env_indices):
    """各環境の初期状態をランダム化"""
    if len(env_indices) == 0:
        return
    
    env_indices_tensor = torch.tensor(env_indices, device=gs.device, dtype=torch.long)
    
    # 初期位置にランダムなオフセット（±0.1m）
    pos_offset = (torch.rand(len(env_indices), 3, device=gs.device) - 0.5) * 0.2
    pos_offset[:, 2] = 0.0  # Z軸（高さ）は変更しない
    env.base_pos[env_indices_tensor] = env.base_init_pos + pos_offset
    env.robot.set_pos(env.base_pos[env_indices_tensor], zero_velocity=False, envs_idx=env_indices_tensor)
    
    # 初期姿勢（ロール・ピッチ）にランダムな角度（±5度）
    from genesis.utils.geom import transform_quat_by_quat as transform_quat
    roll = (torch.rand(len(env_indices), device=gs.device) - 0.5) * 10.0 * np.pi / 180.0
    pitch = (torch.rand(len(env_indices), device=gs.device) - 0.5) * 10.0 * np.pi / 180.0
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    quat_noise = torch.stack([cr * cp, cr * sp, sr * cp, -sr * sp], dim=1)
    base_init_quat_expanded = env.base_init_quat.reshape(1, -1).expand(len(env_indices), -1)
    env.base_quat[env_indices_tensor] = transform_quat(base_init_quat_expanded, quat_noise)
    env.robot.set_quat(env.base_quat[env_indices_tensor], zero_velocity=False, envs_idx=env_indices_tensor)
    
    # 関節角度にランダムなオフセット（±0.1ラジアン）
    dof_noise = (torch.rand(len(env_indices), env.num_actions, device=gs.device) - 0.5) * 0.2
    env.dof_pos[env_indices_tensor] = env.default_dof_pos + dof_noise
    env.robot.set_dofs_position(
        position=env.dof_pos[env_indices_tensor],
        dofs_idx_local=env.motors_dof_idx,
        zero_velocity=True,
        envs_idx=env_indices_tensor,
    )


def main():
    parser = argparse.ArgumentParser(description="Save Vintix model behavior with domain randomization")
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking",
                        help="Experiment name (for loading env config)")
    parser.add_argument("-r", "--robot_type", type=str, choices=["go2", "minicheetah", "laikago"], 
                        default="go2", help="Robot type")
    parser.add_argument("--vintix_path", type=str, default=None,
                        help="Path to Vintix model directory (required if --expert_path is not specified)")
    parser.add_argument("--expert_path", type=str, default=None,
                        help="Path to PPO expert model (e.g., Genesis/logs/go2-walking/model_300.pt). If specified, uses PPO instead of Vintix.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output video file path (e.g., movie/vintix_domain_randomized.mp4)")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum number of steps to record")
    parser.add_argument("--context_len", type=int, default=2048,
                        help="Context length for Vintix model")
    parser.add_argument("--fps", type=int, default=30,
                        help="FPS for video recording")
    parser.add_argument("--num_envs", type=int, default=100,
                        help="Number of parallel environments")
    parser.add_argument("--mass_range_min", type=float, default=0.5,
                        help="Minimum mass scale (e.g., 0.5 = 50% of original)")
    parser.add_argument("--mass_range_max", type=float, default=2.0,
                        help="Maximum mass scale (e.g., 2.0 = 200% of original)")
    parser.add_argument("--gravity_range_min", type=float, default=0.9,
                        help="Minimum gravity scale (e.g., 0.9 = 90% of standard, -10%)")
    parser.add_argument("--gravity_range_max", type=float, default=1.1,
                        help="Maximum gravity scale (e.g., 1.1 = 110% of standard, +10%)")
    
    args = parser.parse_args()
    
    # 引数の検証
    if args.vintix_path is None and args.expert_path is None:
        raise ValueError("Either --vintix_path or --expert_path must be specified")
    if args.vintix_path is not None and args.expert_path is not None:
        raise ValueError("Cannot specify both --vintix_path and --expert_path. Please specify only one.")
    
    # 設定ファイルの読み込み
    genesis_path = Path(__file__).parents[2] / "Genesis"
    log_dir = genesis_path / "logs" / args.exp_name
    cfgs_path = log_dir / "cfgs.pkl"
    
    if not cfgs_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfgs_path}")
    
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfgs_path, "rb"))
    
    # Genesis初期化（環境数1の場合はperformance_modeをFalseにしてビジュアライズを有効化）
    NUM_ENVS = args.num_envs
    gs.init(performance_mode=(NUM_ENVS > 1))
    
    MAX_STEPS = args.max_steps
    MAX_EPISODE_STEPS = 1000
    MASS_RANGE = (args.mass_range_min, args.mass_range_max)
    GRAVITY_RANGE = (args.gravity_range_min, args.gravity_range_max)
    
    print("=" * 80)
    if args.expert_path:
        print("PPO Expert Domain Randomized Evaluation")
        print(f"PPO expert model: {args.expert_path}")
    else:
        print("Vintix Go2 Domain Randomized Evaluation")
        print(f"Vintix model: {args.vintix_path}")
    print("=" * 80)
    print(f"Output video: {args.output}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"FPS: {args.fps}")
    print(f"Number of environments: {NUM_ENVS}")
    print(f"Gravity range: {GRAVITY_RANGE[0]:.2f} - {GRAVITY_RANGE[1]:.2f}x standard (±{((GRAVITY_RANGE[1]-1.0)*100):.0f}%)")
    print(f"Mass: Fixed (1.0x original)")
    print("=" * 80)
    
    # 環境数1の場合は単一環境評価に切り替え
    if NUM_ENVS == 1:
        _run_single_evaluation(args, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, MASS_RANGE, GRAVITY_RANGE)
        return
    
    # 環境作成（並列評価）
    print(f"\nCreating {NUM_ENVS} parallel environments with domain randomization...")
    if args.robot_type == "go2":
        env = Go2Env(
            num_envs=NUM_ENVS,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
            domain_randomization=False,  # 手動でランダマイズするためFalse
        )
    elif args.robot_type == "minicheetah":
        env = MiniCheetahEnv(
            num_envs=NUM_ENVS,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
            domain_randomization=False,
        )
    elif args.robot_type == "laikago":
        env = LaikagoEnv(
            num_envs=NUM_ENVS,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
            domain_randomization=False,
        )
    else:
        raise ValueError(f"Unknown robot type: {args.robot_type}")
    
    print(f"✓ Created {NUM_ENVS} parallel {args.robot_type} environments")
    
    # 各環境の重力をランダマイズ（質量は固定）
    print(f"\nRandomizing domain parameters for each environment (gravity only, mass fixed)...")
    mass_scales, gravity_scales = randomize_domain_for_all_envs(env, NUM_ENVS, MASS_RANGE, GRAVITY_RANGE)
    print(f"  Mass: Fixed (1.0x original)")
    print(f"  Gravity scales: min={min(gravity_scales):.3f}, max={max(gravity_scales):.3f}, mean={np.mean(gravity_scales):.3f}")
    
    # 実際の重力値を確認（rigid solverから読み取る）
    try:
        # rigid solverの重力値を取得
        rigid_solver = env.scene.sim.rigid_solver
        if rigid_solver is not None and rigid_solver.gravity is not None:
            actual_gravity_np = rigid_solver.gravity
            if isinstance(actual_gravity_np, torch.Tensor):
                actual_gravity_np = actual_gravity_np.cpu().numpy()
            print(f"\n  Verifying gravity settings:")
            for env_idx in range(min(3, NUM_ENVS)):  # 最初の3環境のみ表示
                expected_gravity_z = -9.81 * gravity_scales[env_idx]
                if actual_gravity_np.ndim == 2:
                    actual_gravity_z = actual_gravity_np[env_idx, 2]
                else:
                    actual_gravity_z = actual_gravity_np[2]
                print(f"    Env {env_idx}: Expected Z={expected_gravity_z:.3f}, Actual Z={actual_gravity_z:.3f}, Scale={gravity_scales[env_idx]:.3f}x")
                if abs(actual_gravity_z - expected_gravity_z) > 0.1:
                    print(f"      ⚠️ WARNING: Gravity mismatch! Expected {expected_gravity_z:.3f} but got {actual_gravity_z:.3f}")
    except Exception as e:
        print(f"  Warning: Could not verify gravity: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"✓ Domain randomization completed")
    
    # モデルのロード（VintixまたはPPO）
    use_expert = args.expert_path is not None
    if use_expert:
        # PPO専門家モデルのロード
        print(f"\nLoading PPO expert model from {args.expert_path}...")
        genesis_path = Path(__file__).parents[2] / "Genesis"
        model_dir = genesis_path / "logs" / args.exp_name
        runner = OnPolicyRunner(env, train_cfg, str(model_dir), device=gs.device)
        # パスを絶対パスに変換
        expert_path = Path(args.expert_path)
        if not expert_path.is_absolute():
            # Genesis/で始まる場合は除去
            expert_path_str = args.expert_path
            if expert_path_str.startswith("Genesis/"):
                expert_path_str = expert_path_str[len("Genesis/"):]
            expert_path = genesis_path / expert_path_str
        runner.load(str(expert_path))
        expert_policy = runner.get_inference_policy(device=gs.device)
        print("✓ PPO expert model loaded")
        vintix_model = None
        history_buffers = None
    else:
        # Vintixモデルのロード
        print(f"\nLoading Vintix model from {args.vintix_path}...")
        vintix_model = Vintix()
        vintix_model.load_model(args.vintix_path)
        vintix_model = vintix_model.to(gs.device)
        vintix_model.eval()
        print("✓ Vintix model loaded")
        expert_policy = None
        # 各環境に独立した履歴バッファを作成
        history_buffers = [VintixHistoryBuffer(max_len=args.context_len) for _ in range(NUM_ENVS)]
    
    # 環境リセット
    obs, _ = env.reset()
    if not use_expert:
        obs = obs[:, :-12]  # Vintixの場合は観測値から行動を除外
    
    # 重力を再設定（reset()でリセットされる可能性があるため）
    set_gravity_for_envs(env, gravity_scales, list(range(NUM_ENVS)), verbose=False)
    
    # 各環境の初期状態をランダム化
    randomize_initial_state(env, list(range(NUM_ENVS)))
    
    # 観測値を更新（ランダム化後の状態を反映）
    zero_actions = torch.zeros(NUM_ENVS, env.num_actions, device=gs.device)
    obs, _, _, _ = env.step(zero_actions)
    if not use_expert:
        obs = obs[:, :-12]  # Vintixの場合は観測値から行動を除外
    
    # 初期履歴の追加（Vintixの場合のみ）
    if not use_expert:
        initial_action = np.zeros(env.num_actions)
        initial_reward = 0.0
        for env_idx in range(NUM_ENVS):
            history_buffers[env_idx].add(obs[env_idx].cpu().numpy(), initial_action, initial_reward)
    
    # 出力ディレクトリの作成
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ビジュアライズは環境0のみ
    # 録画開始（環境0のみ）
    print(f"\nStarting video recording (env 0 only)...")
    print(f"Recording {MAX_STEPS} steps at {args.fps} FPS...")
    env.cam.start_recording()
    
    # 各環境の報酬を記録（ステップごと）
    all_rewards = []  # 各ステップでの全環境の報酬
    all_mass_scales = mass_scales.copy()  # 各環境の質量スケール
    all_gravity_scales = gravity_scales.copy()  # 各環境の重力スケール
    
    # 各環境のエピソードステップ数を記録
    env_episode_steps = [0 for _ in range(NUM_ENVS)]
    all_episode_steps = []  # 各ステップでの各環境のエピソード長を記録
    
    step_count = 0
    with torch.no_grad():
        while step_count < MAX_STEPS:
            # 行動予測（VintixまたはPPO）
            if use_expert:
                # PPO専門家から行動予測（全環境を一度に処理）
                actions = expert_policy(obs)
            else:
                # 各環境のコンテキストを個別に処理（履歴長が異なるため）
                actions = torch.zeros(NUM_ENVS, env.num_actions, device=gs.device)
                for env_idx in range(NUM_ENVS):
                    context = history_buffers[env_idx].get_context(args.context_len)
                    if context is not None:
                        # デバイスに転送
                        for key in context[0]:
                            if isinstance(context[0][key], torch.Tensor):
                                context[0][key] = context[0][key].to(gs.device)
                        
                        # Vintixで予測（個別処理）
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            pred_actions_list, metadata = vintix_model(context)
                        
                        # 予測行動を取得
                        pred_actions = pred_actions_list[0]
                        if isinstance(pred_actions, list):
                            pred_actions = pred_actions[0]
                        
                        if pred_actions.dim() == 3:  # [batch, seq, act_dim]
                            action = pred_actions[0, -1, :].float()
                        elif pred_actions.dim() == 2:  # [seq, act_dim]
                            action = pred_actions[-1, :].float()
                        else:
                            raise ValueError(f"Unexpected pred_actions shape: {pred_actions.shape}")
                        
                        actions[env_idx] = action
            
            # 環境ステップ
            obs, rewards, dones, infos = env.step(actions)
            if not use_expert:
                obs = obs[:, :-12]  # Vintixの場合は観測値から行動を除外
            
            # ビジュアライズ（環境0のみ）
            if step_count % 5 == 0:  # レンダリング頻度を下げる
                env.cam.render()
            
            # 各環境の報酬と履歴を更新
            step_rewards = []
            rewards_cpu = rewards.cpu().numpy()
            obs_cpu = obs.cpu().numpy()
            actions_cpu = actions.cpu().numpy()
            
            for env_idx in range(NUM_ENVS):
                reward_value = float(rewards_cpu[env_idx])
                step_rewards.append(reward_value)
                
                if not use_expert:
                    # Vintixの場合は履歴バッファに追加
                    history_buffers[env_idx].add(
                        obs_cpu[env_idx],
                        actions_cpu[env_idx],
                        reward_value
                    )
                
                env_episode_steps[env_idx] += 1
                
                # エピソードリセット判定
                episode_done = dones[env_idx] or (env_episode_steps[env_idx] >= MAX_EPISODE_STEPS)
                if episode_done:
                    # 環境リセット
                    reset_indices = torch.tensor([env_idx], device=gs.device, dtype=torch.long)
                    env.reset_idx(reset_indices)
                    
                    # 重力を再設定（reset_idx()でリセットされる可能性があるため）
                    set_gravity_for_envs(env, [gravity_scales[env_idx]], [env_idx], verbose=False)
                    
                    # 初期状態をランダム化
                    randomize_initial_state(env, [env_idx])
                    
                    # 観測値を更新
                    zero_actions = torch.zeros(NUM_ENVS, env.num_actions, device=gs.device)
                    temp_obs, _, _, _ = env.step(zero_actions)
                    if not use_expert:
                        obs[env_idx] = temp_obs[env_idx, :-12]  # Vintixの場合は行動を除外
                    else:
                        obs[env_idx] = temp_obs[env_idx]  # PPOの場合は観測値に行動を含める
                    
                    # 履歴バッファに初期状態を追加（Vintixの場合のみ）
                    if not use_expert:
                        history_buffers[env_idx].add(
                            obs[env_idx].cpu().numpy(),
                            initial_action,
                            initial_reward
                        )
                    env_episode_steps[env_idx] = 0
            
            all_rewards.append(step_rewards)
            # 各ステップでの各環境のエピソード長を記録
            all_episode_steps.append(env_episode_steps.copy())
            step_count += 1
            
            # 進捗表示（100ステップごと）
            if step_count % 100 == 0:
                mean_reward = np.mean(step_rewards)
                std_reward = np.std(step_rewards)
                print(f"Step {step_count:5d} / {MAX_STEPS} | Mean Reward: {mean_reward:7.5f} | Std: {std_reward:7.5f}")
    
    # 録画停止と保存
    print(f"\nStopping recording and saving to {args.output}...")
    env.cam.stop_recording(save_to_filename=str(args.output), fps=args.fps)
    
    # グラフの作成（平均と標準偏差、質量別の分析）
    print(f"\nCreating performance graphs...")
    graph_path = output_path.with_suffix('.png')
    if '_domain_randomized' not in graph_path.stem:
        graph_path = graph_path.parent / f"{graph_path.stem}_domain_randomized.png"
    
    steps = np.arange(1, len(all_rewards) + 1)
    mean_rewards = [np.mean(rewards) for rewards in all_rewards]
    std_rewards = [np.std(rewards) for rewards in all_rewards]
    
    # エピソード長の平均と標準偏差を計算
    mean_episode_steps = [np.mean(ep_steps) for ep_steps in all_episode_steps]
    std_episode_steps = [np.std(ep_steps) for ep_steps in all_episode_steps]
    
    # 重力スケール別に報酬を分析
    gravity_bins = np.linspace(GRAVITY_RANGE[0], GRAVITY_RANGE[1], 5)
    gravity_groups = {}
    for i, gravity_scale in enumerate(all_gravity_scales):
        bin_idx = np.digitize(gravity_scale, gravity_bins) - 1
        bin_idx = max(0, min(bin_idx, len(gravity_bins) - 2))
        if bin_idx not in gravity_groups:
            gravity_groups[bin_idx] = []
        gravity_groups[bin_idx].append(i)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Vintix Model Performance (Domain Randomized) - {output_path.stem}', fontsize=16, fontweight='bold')
    
    # 1. エピソード長の平均と標準偏差
    ax1 = axes[0, 0]
    ax1.plot(steps, mean_episode_steps, linewidth=2, label='Mean Episode Length', color='green')
    ax1.fill_between(steps,
                     np.array(mean_episode_steps) - np.array(std_episode_steps),
                     np.array(mean_episode_steps) + np.array(std_episode_steps),
                     alpha=0.3, color='green', label='±1 Std')
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Episode Length', fontsize=11)
    ax1.set_title('Episode Length vs Steps (Mean ± Std)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. ステップごとの報酬の平均と標準偏差
    ax2 = axes[0, 1]
    ax2.plot(steps, mean_rewards, linewidth=2, label='Mean Reward', color='blue')
    ax2.fill_between(steps,
                     np.array(mean_rewards) - np.array(std_rewards),
                     np.array(mean_rewards) + np.array(std_rewards),
                     alpha=0.3, color='blue', label='±1 Std')
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Reward', fontsize=11)
    ax2.set_title('Reward per Step (Mean ± Std)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=0.5)
    ax2.legend()
    
    # 3. 重力スケール別のエピソード長
    ax3 = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(gravity_groups)))
    sorted_bin_indices = sorted(gravity_groups.keys())
    for color_idx, bin_idx in enumerate(sorted_bin_indices):
        env_indices = gravity_groups[bin_idx]
        if len(env_indices) == 0:
            continue
        gravity_range_str = f"{gravity_bins[bin_idx]:.2f}-{gravity_bins[bin_idx+1]:.2f}"
        env_episode_lengths = [[all_episode_steps[s][i] for i in env_indices] for s in range(len(all_episode_steps))]
        env_mean_episode_lengths = [np.mean(ep_lengths) for ep_lengths in env_episode_lengths]
        ax3.plot(steps, env_mean_episode_lengths, linewidth=1.5, alpha=0.7, 
                label=f'Gravity {gravity_range_str}x', color=colors[color_idx])
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('Episode Length', fontsize=11)
    ax3.set_title('Episode Length by Gravity Scale', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    
    # 4. 重力スケールと最終報酬の関係
    ax4 = axes[1, 1]
    final_rewards = [np.mean([all_rewards[s][i] for s in range(len(all_rewards))]) for i in range(NUM_ENVS)]
    scatter = ax4.scatter(all_gravity_scales, final_rewards, alpha=0.6, s=30, c=final_rewards, cmap='viridis')
    ax4.set_xlabel('Gravity Scale', fontsize=11)
    ax4.set_ylabel('Average Reward', fontsize=11)
    ax4.set_title('Gravity Scale vs Average Reward', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Average Reward')
    
    plt.tight_layout()
    plt.savefig(str(graph_path), dpi=150, bbox_inches='tight')
    print(f"✓ Graph saved: {graph_path}")
    
    # CSVファイルにも保存
    csv_path = graph_path.with_suffix('.csv')
    with open(csv_path, 'w') as f:
        f.write("step,mean_reward,std_reward,mean_episode_length,std_episode_length\n")
        for i, (mean_r, std_r, mean_ep, std_ep) in enumerate(zip(mean_rewards, std_rewards, mean_episode_steps, std_episode_steps), 1):
            f.write(f"{i},{mean_r:.6f},{std_r:.6f},{mean_ep:.2f},{std_ep:.2f}\n")
    print(f"✓ CSV saved: {csv_path}")
    
    # 重力別の統計も保存
    gravity_csv_path = graph_path.parent / f"{graph_path.stem}_gravity_analysis.csv"
    with open(gravity_csv_path, 'w') as f:
        f.write("env_idx,gravity_scale,average_reward\n")
        for i, (gravity_scale, avg_reward) in enumerate(zip(all_gravity_scales, final_rewards)):
            f.write(f"{i},{gravity_scale:.6f},{avg_reward:.6f}\n")
    print(f"✓ Gravity analysis CSV saved: {gravity_csv_path}")
    
    # 最終統計
    final_mean_reward = np.mean(mean_rewards)
    final_std_reward = np.mean(std_rewards)
    print(f"\n{'=' * 80}")
    print(f"✓ Domain randomized evaluation completed!")
    print(f"  Output video: {output_path.absolute()}")
    print(f"  Output graph: {graph_path.absolute()}")
    print(f"  Total steps: {MAX_STEPS}")
    print(f"  Number of environments: {NUM_ENVS}")
    print(f"  Gravity range: {GRAVITY_RANGE[0]:.2f} - {GRAVITY_RANGE[1]:.2f}x standard")
    print(f"  Mean reward per step: {final_mean_reward:.6f}")
    print(f"  Std reward per step: {final_std_reward:.6f}")
    if output_path.exists():
        print(f"  Video file size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()


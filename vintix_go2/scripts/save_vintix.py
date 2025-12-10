#!/usr/bin/env python3
"""
Vintix モデルの動作を動画として保存するスクリプト

eval_vintix.pyをベースに録画機能を追加

Usage:
    python scripts/save_vintix.py --vintix_path models/vintix_go2/vintix_go2_ad/0095_epoch --output movie/vintix_expert_only.mp4
"""
import argparse
import os
import pickle
import sys
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


def _run_parallel_evaluation(args, env_cfg, obs_cfg, reward_cfg, command_cfg):
    """並列評価を実行"""
    NUM_ENVS = args.num_envs
    MAX_STEPS = args.max_steps
    MAX_EPISODE_STEPS = 1000
    
    # 環境作成
    print(f"Creating {NUM_ENVS} parallel environments...")
    if args.robot_type == "go2":
        env = Go2Env(
            num_envs=NUM_ENVS,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
        )
    elif args.robot_type == "minicheetah":
        env = MiniCheetahEnv(
            num_envs=NUM_ENVS,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
        )
    elif args.robot_type == "laikago":
        env = LaikagoEnv(
            num_envs=NUM_ENVS,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
        )
    else:
        raise ValueError(f"Unknown robot type: {args.robot_type}")
    
    print(f"✓ Created {NUM_ENVS} parallel {args.robot_type} environments")
    
    # Vintixモデルのロード
    print(f"Loading Vintix model from {args.vintix_path}...")
    vintix_model = Vintix()
    vintix_model.load_model(args.vintix_path)
    vintix_model = vintix_model.to(gs.device)
    vintix_model.eval()
    print("✓ Vintix model loaded")
    
    # 各環境に独立した履歴バッファを作成
    history_buffers = [VintixHistoryBuffer(max_len=args.context_len) for _ in range(NUM_ENVS)]
    
    # 環境リセット
    obs, _ = env.reset()
    
    # 各環境の初期状態をランダム化（標準偏差が0になるのを防ぐため）
    from genesis.utils.geom import transform_quat_by_quat as transform_quat
    env_indices = torch.arange(NUM_ENVS, device=gs.device, dtype=torch.long)
    
    # 初期位置にランダムなオフセット（±0.1m）
    pos_offset = (torch.rand(NUM_ENVS, 3, device=gs.device) - 0.5) * 0.2
    pos_offset[:, 2] = 0.0  # Z軸（高さ）は変更しない
    env.base_pos[env_indices] = env.base_init_pos + pos_offset
    env.robot.set_pos(env.base_pos[env_indices], zero_velocity=False, envs_idx=env_indices)
    
    # 初期姿勢（ロール・ピッチ）にランダムな角度（±5度）
    roll = (torch.rand(NUM_ENVS, device=gs.device) - 0.5) * 10.0 * np.pi / 180.0
    pitch = (torch.rand(NUM_ENVS, device=gs.device) - 0.5) * 10.0 * np.pi / 180.0
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    quat_noise = torch.stack([cr * cp, cr * sp, sr * cp, -sr * sp], dim=1)
    # base_init_quatは[4]の形状なので、各環境に対して拡張する
    base_init_quat_expanded = env.base_init_quat.reshape(1, -1).expand(NUM_ENVS, -1)
    env.base_quat[env_indices] = transform_quat(base_init_quat_expanded, quat_noise)
    env.robot.set_quat(env.base_quat[env_indices], zero_velocity=False, envs_idx=env_indices)
    
    # 関節角度にランダムなオフセット（±0.1ラジアン）
    dof_noise = (torch.rand(NUM_ENVS, env.num_actions, device=gs.device) - 0.5) * 0.2
    env.dof_pos[env_indices] = env.default_dof_pos + dof_noise
    env.robot.set_dofs_position(
        position=env.dof_pos[env_indices],
        dofs_idx_local=env.motors_dof_idx,
        zero_velocity=True,
        envs_idx=env_indices,
    )
    
    # 観測値を更新（ランダム化後の状態を反映）
    # env.step()の最初でobs_bufが更新されるので、ゼロアクションで1ステップ進める
    zero_actions = torch.zeros(NUM_ENVS, env.num_actions, device=gs.device)
    obs, _, _, _ = env.step(zero_actions)
    obs = obs[:, :-12]  # 観測値から行動を除外
    
    # 初期履歴の追加
    initial_action = np.zeros(env.num_actions)
    initial_reward = 0.0
    for env_idx in range(NUM_ENVS):
        history_buffers[env_idx].add(obs[env_idx].cpu().numpy(), initial_action, initial_reward)
    
    # 出力ディレクトリの作成
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ビジュアライズは環境0のみ（rendered_envs_idx=[0]）
    # 録画開始（環境0のみ）
    print(f"\nStarting video recording (env 0 only)...")
    print(f"Recording {MAX_STEPS} steps at {args.fps} FPS...")
    env.cam.start_recording()
    
    # 各環境の報酬を記録（ステップごと）
    # all_rewards[step][env_idx] = reward
    all_rewards = []  # 各ステップでの全環境の報酬
    
    # 各環境のエピソードステップ数を記録
    env_episode_steps = [0 for _ in range(NUM_ENVS)]
    
    # 各環境のエピソードごとの累積報酬を記録
    # env_episode_rewards[env_idx] = [ep1_reward, ep2_reward, ...]
    env_episode_rewards = [[] for _ in range(NUM_ENVS)]
    # 各環境のエピソード開始時の累積ステップ数を記録
    # env_episode_cumulative_steps[env_idx] = [ep1_start_step, ep2_start_step, ...]
    env_episode_cumulative_steps = [[0] for _ in range(NUM_ENVS)]
    # 各環境の現在のエピソード累積報酬
    env_current_episode_rewards = [0.0 for _ in range(NUM_ENVS)]
    # 各環境がリセット直後かどうかを記録（リセット直後の報酬を除外するため）
    env_just_reset = [False for _ in range(NUM_ENVS)]
    
    step_count = 0
    with torch.no_grad():
        while step_count < MAX_STEPS:
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
            obs = obs[:, :-12]  # 観測値から行動を除外
            
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
                
                # リセット直後のステップの報酬は除外（前のステップの報酬を維持）
                if env_just_reset[env_idx]:
                    # リセット直後の報酬は記録しない（前のステップの報酬を維持）
                    # ただし、履歴バッファには追加する（Vintixの学習には必要）
                    history_buffers[env_idx].add(
                        obs_cpu[env_idx],
                        actions_cpu[env_idx],
                        reward_value
                    )
                    # 報酬は前のステップの平均報酬を使用（グラフの連続性のため）
                    if len(all_rewards) > 0:
                        prev_reward = all_rewards[-1][env_idx]
                    else:
                        prev_reward = initial_reward
                    step_rewards.append(prev_reward)
                    env_just_reset[env_idx] = False  # リセットフラグをクリア
                else:
                    step_rewards.append(reward_value)
                    
                    history_buffers[env_idx].add(
                        obs_cpu[env_idx],
                        actions_cpu[env_idx],
                        reward_value
                    )
                
                env_episode_steps[env_idx] += 1
                env_current_episode_rewards[env_idx] += reward_value
                
                # エピソードリセット判定
                episode_done = dones[env_idx] or (env_episode_steps[env_idx] >= MAX_EPISODE_STEPS)
                if episode_done:
                    # エピソードの累積報酬を記録
                    env_episode_rewards[env_idx].append(env_current_episode_rewards[env_idx])
                    env_episode_cumulative_steps[env_idx].append(step_count)
                    env_current_episode_rewards[env_idx] = 0.0
                    
                    # 環境リセット
                    reset_indices = torch.tensor([env_idx], device=gs.device, dtype=torch.long)
                    env.reset_idx(reset_indices)
                    obs[env_idx] = env.obs_buf[env_idx, :-12]  # 行動を除外
                    
                    # 履歴はクリアしない（Vintixは履歴を保持する）
                    
                    # 履歴バッファに初期状態を追加
                    history_buffers[env_idx].add(
                        obs[env_idx].cpu().numpy(),
                        initial_action,
                        initial_reward
                    )
                    env_episode_steps[env_idx] = 0
                    env_just_reset[env_idx] = True  # リセットフラグを設定
            
            all_rewards.append(step_rewards)
            step_count += 1
            
            # 進捗表示（100ステップごと）
            if step_count % 100 == 0:
                mean_reward = np.mean(step_rewards)
                std_reward = np.std(step_rewards)
                print(f"Step {step_count:5d} / {MAX_STEPS} | Mean Reward: {mean_reward:7.5f} | Std: {std_reward:7.5f}")
    
    # 録画停止と保存
    print(f"\nStopping recording and saving to {args.output}...")
    env.cam.stop_recording(save_to_filename=str(args.output), fps=args.fps)
    
    # グラフの作成（平均と標準偏差）
    print(f"\nCreating performance graphs...")
    graph_path = output_path.with_suffix('.png')
    # 元のファイル名を使用（_parallelサフィックスを追加しない）
    
    steps = np.arange(1, len(all_rewards) + 1)
    mean_rewards = [np.mean(rewards) for rewards in all_rewards]
    std_rewards = [np.std(rewards) for rewards in all_rewards]
    
    # 最後の10ステップを除外（エピソードリセットによる異常な低報酬を除去）
    exclude_last_n_steps = 10
    if len(all_rewards) > exclude_last_n_steps:
        steps = steps[:-exclude_last_n_steps]
        mean_rewards = mean_rewards[:-exclude_last_n_steps]
        std_rewards = std_rewards[:-exclude_last_n_steps]
        all_rewards = all_rewards[:-exclude_last_n_steps]
    
    # エピソードごとの累積報酬を集計（visualize_trajectories_by_steps.pyと同じ形式）
    # 全環境のエピソードを累積ステップ数でソートして集計
    all_episodes_data = []  # [(cumulative_steps, cumulative_reward), ...]
    for env_idx in range(NUM_ENVS):
        for ep_idx, (cum_steps, cum_reward) in enumerate(zip(env_episode_cumulative_steps[env_idx], 
                                                               env_episode_rewards[env_idx])):
            all_episodes_data.append((cum_steps, cum_reward))
    
    # 累積ステップ数でソート
    all_episodes_data.sort(key=lambda x: x[0])
    
    # ビン分割（100分割）して平均・標準偏差を計算
    if len(all_episodes_data) > 0:
        max_cum_steps = max([x[0] for x in all_episodes_data])
        num_bins = 100
        step_bins = np.linspace(0, max_cum_steps, num_bins + 1)
        
        bin_cumulative_rewards = []
        for i in range(num_bins):
            step_min = step_bins[i]
            step_max = step_bins[i + 1]
            rewards_in_bin = [x[1] for x in all_episodes_data if step_min <= x[0] < step_max]
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
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f'Vintix Model Performance (Parallel) - {output_path.stem}', fontsize=16, fontweight='bold')
    
    # ステップごとの報酬の平均と標準偏差
    ax.plot(steps, mean_rewards, linewidth=2, label='Mean Reward', color='blue')
    ax.fill_between(steps,
                     np.array(mean_rewards) - np.array(std_rewards),
                     np.array(mean_rewards) + np.array(std_rewards),
                     alpha=0.3, color='blue', label='±1 Std')
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Reward', fontsize=11)
    ax.set_title('Reward per Step (Mean ± Std)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=0.5)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(str(graph_path), dpi=150, bbox_inches='tight')
    print(f"✓ Graph saved: {graph_path}")
    
    # CSVファイルにも保存（ステップごとのデータ）
    csv_path = graph_path.with_suffix('.csv')
    with open(csv_path, 'w') as f:
        f.write("step,mean_reward,std_reward\n")
        for i, (mean_r, std_r) in enumerate(zip(mean_rewards, std_rewards), 1):
            f.write(f"{i},{mean_r:.6f},{std_r:.6f}\n")
    print(f"✓ CSV saved: {csv_path}")
    
    # エピソードごとの累積報酬のCSVも保存
    episode_csv_path = graph_path.parent / f"{graph_path.stem}_episodes.csv"
    with open(episode_csv_path, 'w') as f:
        f.write("cumulative_steps,cumulative_reward\n")
        for cum_steps, cum_reward in all_episodes_data:
            f.write(f"{cum_steps},{cum_reward:.6f}\n")
    print(f"✓ Episode CSV saved: {episode_csv_path}")
    
    # 最終統計
    final_mean_reward = np.mean(mean_rewards)
    final_std_reward = np.mean(std_rewards)
    print(f"\n{'=' * 80}")
    print(f"✓ Parallel evaluation completed!")
    print(f"  Output video: {output_path.absolute()}")
    print(f"  Output graph: {graph_path.absolute()}")
    print(f"  Total steps: {step_count}")
    print(f"  Number of environments: {NUM_ENVS}")
    print(f"  Mean reward per step: {final_mean_reward:.6f}")
    print(f"  Std reward per step: {final_std_reward:.6f}")
    if output_path.exists():
        print(f"  Video file size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'=' * 80}")


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


def main():
    parser = argparse.ArgumentParser(description="Save Vintix model behavior as video")
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking",
                        help="Experiment name (for loading env config)")
    parser.add_argument("-r", "--robot_type", type=str, choices=["go2", "minicheetah", "laikago"], 
                        default="go2", help="Robot type")
    parser.add_argument("--vintix_path", type=str, required=True,
                        help="Path to Vintix model directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output video file path (e.g., movie/vintix_expert.mp4)")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum steps to record (default: 500 = 10 seconds at 50Hz)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video FPS")
    parser.add_argument("--context_len", type=int, default=1024,
                        help="Context length for Vintix")
    parser.add_argument("--parallel", action="store_true",
                        help="Use parallel evaluation")
    parser.add_argument("--num_envs", type=int, default=100,
                        help="Number of parallel environments (default: 100)")
    args = parser.parse_args()

    print("=" * 80)
    if args.parallel:
        print("Vintix Go2 Parallel Evaluation")
    else:
        print("Vintix Go2 Video Recording")
    print("=" * 80)
    print(f"Vintix model: {args.vintix_path}")
    print(f"Output video: {args.output}")
    print(f"Max steps: {args.max_steps}")
    print(f"FPS: {args.fps}")
    if args.parallel:
        print(f"Mode: Parallel ({args.num_envs} envs, {args.max_steps} steps each)")
    print("=" * 80)
    print()

    # Genesis初期化
    if args.parallel:
        gs.init(performance_mode=True)  # 並列評価時はパフォーマンスモードを有効化
    else:
        gs.init()

    # 環境設定の読み込み（eval_vintix.pyと同じ）
    genesis_root = Path(__file__).parents[2] / "Genesis"
    log_dir = genesis_root / "logs" / args.exp_name
    cfgs_path = log_dir / "cfgs.pkl"
    
    if not cfgs_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfgs_path}")
    
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfgs_path, "rb"))
    
    # 並列評価の場合は別処理
    if args.parallel:
        _run_parallel_evaluation(args, env_cfg, obs_cfg, reward_cfg, command_cfg)
        return
    
    # 環境作成（eval_vintix.pyと同じ）
    print("Creating environment...")
    if args.robot_type == "go2":
        env = Go2Env(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,  # 録画時はビューアー非表示
        )
    elif args.robot_type == "minicheetah":
        env = MiniCheetahEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
        )
    elif args.robot_type == "laikago":
        env = LaikagoEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
        )
    else:
        raise ValueError(f"Unknown robot type: {args.robot_type}")
    
    print(f"✓ Created {args.robot_type} environment")

    # Vintixモデルのロード（eval_vintix.pyと同じ）
    print(f"Loading Vintix model from {args.vintix_path}...")
    vintix_model = Vintix()
    vintix_model.load_model(args.vintix_path)
    vintix_model = vintix_model.to(gs.device)
    vintix_model.eval()
    print("✓ Vintix model loaded")

    # 履歴バッファの初期化
    history_buffer = VintixHistoryBuffer(max_len=args.context_len)
    
    # エピソードの最大ステップ数（1000ステップ）
    MAX_EPISODE_STEPS = 1000

    # 環境リセット
    obs, _ = env.reset()
    # 観測値から行動を除外（訓練時と同じ33次元にする）
    obs = obs[:, :-12]
    
    # 初期履歴の追加（ゼロアクション、ゼロ報酬）
    initial_action = np.zeros(env.num_actions)
    initial_reward = 0.0
    history_buffer.add(obs[0].cpu().numpy(), initial_action, initial_reward)

    # 出力ディレクトリの作成
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting video recording...")
    print(f"Recording {args.max_steps} steps at {args.fps} FPS...")
    
    # 録画開始
    env.cam.start_recording()
    
    step_count = 0
    episode_count = 0
    episode_reward = 0.0
    episode_step_count = 0
    total_reward = 0.0
    
    # ステップごとの統計を記録（グラフの横軸をステップ数にするため）
    step_rewards = []  # 各ステップの報酬
    episode_starts = []  # エピソード開始位置（グラフで区切りを表示するため）
    
    # エピソード統計の記録（サマリー表示）
    episode_rewards = []  # 各エピソードの累積報酬
    episode_lengths = []
    episode_avg_rewards = []
    episode_cumulative_steps = []  # 各エピソード開始時の累積ステップ数
    
    # 最初のエピソード開始位置
    episode_starts.append(0)
    episode_cumulative_steps.append(0)
    
    with torch.no_grad():
        while step_count < args.max_steps:
            # Vintixから行動予測（eval_vintix.pyと同じロジック）
            context = history_buffer.get_context(args.context_len)
            
            if context is not None:
                # デバイスに転送
                for key in context[0]:
                    if isinstance(context[0][key], torch.Tensor):
                        context[0][key] = context[0][key].to(gs.device)
                
                # Vintixで予測（eval_vintix.pyと同じ）
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred_actions, metadata = vintix_model(context)
                
                # 最新の予測行動を取得（fp32に変換）
                # pred_actionsはリストの場合があるので、最初の要素を取得
                if isinstance(pred_actions, list):
                    pred_actions = pred_actions[0]
                
                # pred_actionsの次元を確認
                if pred_actions.dim() == 3:  # [batch, seq, act_dim]
                    action = pred_actions[0, -1, :].unsqueeze(0).float()
                elif pred_actions.dim() == 2:  # [seq, act_dim]
                    action = pred_actions[-1, :].unsqueeze(0).float()
                else:
                    raise ValueError(f"Unexpected pred_actions shape: {pred_actions.shape}")
            else:
                # 履歴がない場合はゼロアクション
                action = torch.zeros(1, env.num_actions, device=gs.device)
            
            # 環境ステップ
            obs, rewards, dones, infos = env.step(action)
            # 観測値から行動を除外（訓練時と同じ33次元にする）
            obs = obs[:, :-12]
            env.cam.render()
            
            # 報酬とアクション履歴に追加（行動を除外した観測値）
            reward_value = float(rewards.cpu().numpy()[0])
            history_buffer.add(
                obs[0].cpu().numpy(),
                action[0].cpu().numpy(),
                reward_value
            )
            
            # ステップごとのデータを記録
            step_rewards.append(reward_value)
            total_reward += reward_value
            
            episode_reward += reward_value
            step_count += 1
            episode_step_count += 1
            
            # 進捗表示（100ステップごと）
            if step_count % 100 == 0:
                avg_reward = total_reward / step_count
                print(f"Step {step_count:5d} / {args.max_steps} | Episode {episode_count + 1} | "
                      f"Ep Step: {episode_step_count:4d} | Ep Reward: {episode_reward:7.3f} | "
                      f"Avg Reward: {avg_reward:7.5f}")
            
            # 環境リセット判定（環境のdoneまたは1000ステップ到達）
            episode_done = dones[0] or (episode_step_count >= MAX_EPISODE_STEPS)
            
            if episode_done:
                if episode_step_count >= MAX_EPISODE_STEPS:
                    print(f"Episode {episode_count + 1} reached max steps ({MAX_EPISODE_STEPS}) | Reward: {episode_reward:.3f}")
                else:
                    print(f"Episode {episode_count + 1} completed | Reward: {episode_reward:.3f} | Steps: {episode_step_count}")
                
                # エピソード統計を記録
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_step_count)
                episode_avg_rewards.append(episode_reward / episode_step_count if episode_step_count > 0 else 0.0)
                
                # 次のエピソード開始位置を記録
                episode_starts.append(step_count)
                episode_cumulative_steps.append(step_count)
                
                episode_count += 1
                episode_reward = 0.0
                episode_step_count = 0
                obs, _ = env.reset()
                # 観測値から行動を除外（訓練時と同じ33次元にする）
                obs = obs[:, :-12]
                
                # 履歴はクリアしない（Vintixは履歴を保持する）
                # 新しい初期状態を履歴に追加
                history_buffer.add(obs[0].cpu().numpy(), initial_action, initial_reward)
    
    # 最後のエピソードが完了していない場合でも、その報酬を記録
    if episode_step_count > 0:
        print(f"Final episode (incomplete) | Reward: {episode_reward:.3f} | Steps: {episode_step_count}")
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step_count)
        episode_avg_rewards.append(episode_reward / episode_step_count if episode_step_count > 0 else 0.0)
        episode_starts.append(step_count)
        episode_cumulative_steps.append(step_count)
    
    # 録画停止と保存
    print(f"\nStopping recording and saving to {args.output}...")
    env.cam.stop_recording(save_to_filename=str(args.output), fps=args.fps)
    
    # 最終統計
    avg_reward_per_step = total_reward / step_count if step_count > 0 else 0.0
    
    # グラフの作成
    if len(step_rewards) > 0:
        print(f"\nCreating performance graphs...")
        
        # グラフのファイル名（動画と同じディレクトリに保存）
        graph_path = output_path.with_suffix('.png')
        
        # ステップ数配列
        steps = np.arange(1, len(step_rewards) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Vintix Model Performance - {output_path.stem}', fontsize=16, fontweight='bold')
        
        # 1. エピソードごとの累積報酬（visualize_trajectories_by_steps.pyと同じ形式）
        ax1 = axes[0]
        if len(episode_rewards) > 0:
            # 各エピソードの累積ステップ数と累積報酬をプロット
            ax1.plot(episode_cumulative_steps[:len(episode_rewards)], episode_rewards, 
                    marker='o', linewidth=2, markersize=4, label='Episode Cumulative Reward', color='blue')
            ax1.set_xlabel('Cumulative Steps', fontsize=11)
            ax1.set_ylabel('Cumulative Reward per Episode', fontsize=11)
            ax1.set_title('Episode Cumulative Reward vs Cumulative Steps', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No episodes completed', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_xlabel('Cumulative Steps', fontsize=11)
            ax1.set_ylabel('Cumulative Reward per Episode', fontsize=11)
            ax1.set_title('Episode Cumulative Reward vs Cumulative Steps', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # 2. ステップごとの報酬（ステップ数ベース）
        ax2 = axes[1]
        ax2.plot(steps, step_rewards, linewidth=1, alpha=0.6, label='Reward per Step', color='blue')
        ax2.set_xlabel('Step', fontsize=11)
        ax2.set_ylabel('Reward', fontsize=11)
        ax2.set_title('Reward per Step', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        # エピソードの区切りを縦線で表示
        for ep_start in episode_starts[1:]:  # 最初の0は除く
            if ep_start < len(step_rewards):
                ax2.axvline(x=ep_start, color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=0.5)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(str(graph_path), dpi=150, bbox_inches='tight')
        print(f"✓ Graph saved: {graph_path}")
        
        # CSVファイルにも保存（ステップごとのデータとエピソードごとの累積報酬）
        csv_path = output_path.with_suffix('.csv')
        with open(csv_path, 'w') as f:
            f.write("step,reward,episode\n")
            current_episode = 0
            for i, reward in enumerate(step_rewards, 1):
                # 現在のエピソード番号を判定
                if current_episode + 1 < len(episode_starts) and i >= episode_starts[current_episode + 1]:
                    current_episode += 1
                f.write(f"{i},{reward:.6f},{current_episode + 1}\n")
        print(f"✓ CSV saved: {csv_path}")
        
        # エピソードごとの累積報酬のCSVも保存
        episode_csv_path = output_path.with_name(output_path.stem + '_episodes.csv')
        with open(episode_csv_path, 'w') as f:
            f.write("episode,cumulative_steps,cumulative_reward,episode_length\n")
            for i, (cum_steps, cum_reward, ep_length) in enumerate(zip(episode_cumulative_steps[:len(episode_rewards)], 
                                                                        episode_rewards, episode_lengths), 1):
                f.write(f"{i},{cum_steps},{cum_reward:.6f},{ep_length}\n")
        print(f"✓ Episode CSV saved: {episode_csv_path}")
    
    print(f"\n{'=' * 80}")
    print(f"✓ Video saved successfully!")
    print(f"  Output: {output_path.absolute()}")
    print(f"  Total steps: {step_count}")
    print(f"  Total episodes: {episode_count}")
    print(f"  Average reward per step: {avg_reward_per_step:.6f}")
    if output_path.exists():
        print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

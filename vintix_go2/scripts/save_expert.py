#!/usr/bin/env python3
"""
PPO専門家ポリシーの動作を動画として保存するスクリプト

save_vintix.pyをベースに、PPO専門家ポリシー評価用に変更

Usage:
    python scripts/save_expert.py -r go2 -r go1 -r minicheetah -r unitreea1 --parallel --num_envs 10 --max_steps 1000
"""
import argparse
import copy
import os
import pickle
import sys
from pathlib import Path
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
from env import Go1Env
from env import UnitreeA1Env

# PPOモデル用のインポート
from rsl_rl.runners import OnPolicyRunner


def get_default_expert_model_path(robot_type: str):
    """各ロボットタイプのデフォルト専門家モデルパスを取得"""
    genesis_root = Path(__file__).parents[2] / "Genesis"
    
    if robot_type == "go2":
        return genesis_root / "logs" / "go2-walking" / "model_300.pt"
    elif robot_type == "minicheetah":
        return genesis_root / "logs" / "minicheetah-walking2" / "model_2990.pt"
    elif robot_type == "go1":
        return genesis_root / "logs" / "go1-walking" / "model_2000.pt"
    elif robot_type == "unitreea1":
        return genesis_root / "logs" / "unitreea1-walking" / "model_350.pt"
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")


def generate_output_filename(robot_type: str, output_dir: str = None):
    """ロボットタイプから出力ファイル名を自動生成"""
    # ファイル名を生成: expert_{robot_type}
    filename = f"expert_{robot_type}"
    
    # 出力ディレクトリが指定されていない場合、vintix_go2/Expert_result/{robot_type}/ディレクトリを使用
    if output_dir is None:
        vintix_root = Path(__file__).parent.parent
        result_dir = vintix_root / "Expert_result" / robot_type
        result_dir.mkdir(parents=True, exist_ok=True)
        output_path = result_dir / filename
    else:
        output_path = Path(output_dir) / filename
    return str(output_path)


def _run_parallel_evaluation(args, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg):
    """並列評価を実行（PPO専門家ポリシー使用）"""
    NUM_ENVS = args.num_envs
    MAX_EPISODE_STEPS = 1000
    
    # ステップ数ベースかエピソード数ベースかを判定
    use_episode_limit = (args.max_steps is None)
    if args.max_steps is not None:
        # 後方互換性のため、max_stepsが指定されている場合はそれを使用
        MAX_STEPS = args.max_steps
        MAX_EPISODES = None
    else:
        # エピソード数ベースで制御
        MAX_STEPS = None
        MAX_EPISODES = args.max_episodes
    
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
    elif args.robot_type == "go1":
        env = Go1Env(
            num_envs=NUM_ENVS,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
        )
    elif args.robot_type == "unitreea1":
        env = UnitreeA1Env(
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
    
    # PPO専門家ポリシーのロード
    # train_cfgをコピーして使用（OnPolicyRunnerがtrain_cfgを変更する可能性があるため）
    train_cfg_copy = copy.deepcopy(train_cfg)
    
    # 専門家モデルのパスを決定
    if hasattr(args, 'expert_model_path') and args.expert_model_path is not None:
        expert_path = Path(args.expert_model_path)
    elif hasattr(args, 'expert_model_type') and args.expert_model_type is not None:
        expert_path = get_default_expert_model_path(args.expert_model_type)
    else:
        expert_path = get_default_expert_model_path(args.robot_type)
    
    # 専門家モデルのexp_nameを決定（train_cfgの読み込み用）
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
        else:
            return "go2-walking"  # デフォルト
    
    if hasattr(args, 'expert_model_type') and args.expert_model_type is not None:
        expert_exp_name = get_exp_name_for_robot(args.expert_model_type)
    else:
        expert_exp_name = args.exp_name
    
    print(f"Loading PPO expert model from {expert_path}...")
    if hasattr(args, 'expert_model_type') and args.expert_model_type is not None:
        print(f"Expert model type: {args.expert_model_type} (zero-shot evaluation on {args.robot_type})")
    genesis_path = Path(__file__).parents[2] / "Genesis"
    model_dir = genesis_path / "logs" / expert_exp_name
    runner = OnPolicyRunner(env, train_cfg_copy, str(model_dir), device=gs.device)
    runner.load(str(expert_path))
    expert_policy = runner.get_inference_policy(device=gs.device)
    print("✓ PPO expert model loaded")
    
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
    # PPOポリシーは観測値から行動を除外しない（45次元のまま）
    
    # 出力ディレクトリの作成（グラフ用、動画は後で単一環境で録画）
    output_path = Path(args.output)
    # グラフのファイルパスを設定（.png拡張子）
    graph_path = output_path.parent / f"{output_path.stem}.png"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 並列評価では録画を行わない（グラフのみ作成）
    print(f"\nRunning parallel evaluation (no video recording)...")
    
    # 各環境の報酬を記録（ステップごと）
    # all_rewards[step][env_idx] = reward
    all_rewards = []  # 各ステップでの全環境の報酬
    
    # 各環境のエピソードステップ数を記録
    env_episode_steps = [0 for _ in range(NUM_ENVS)]
    
    # 各環境のエピソードごとの累積報酬を記録
    # env_episode_rewards[env_idx] = [ep1_reward, ep2_reward, ...]
    env_episode_rewards = [[] for _ in range(NUM_ENVS)]
    # 各環境のエピソードごとの長さを記録
    # env_episode_lengths[env_idx] = [ep1_length, ep2_length, ...]
    env_episode_lengths = [[] for _ in range(NUM_ENVS)]
    # 各環境のエピソード開始時の累積ステップ数を記録
    # env_episode_cumulative_steps[env_idx] = [ep1_start_step, ep2_start_step, ...]
    env_episode_cumulative_steps = [[0] for _ in range(NUM_ENVS)]
    # 各環境の現在のエピソード累積報酬
    env_current_episode_rewards = [0.0 for _ in range(NUM_ENVS)]
    # 各環境がリセット直後かどうかを記録（リセット直後の報酬を除外するため）
    env_just_reset = [False for _ in range(NUM_ENVS)]
    
    # エピソード数のカウント（各環境ごと）
    # 各環境でMAX_EPISODES回のエピソードを実行する
    env_episode_counts = [0 for _ in range(NUM_ENVS)]  # 各環境のエピソード完了数
    
    step_count = 0
    with torch.no_grad():
        while True:
            # 終了条件のチェック
            if use_episode_limit:
                # エピソード数ベース：全環境がそれぞれMAX_EPISODES回のエピソードを完了したら終了
                if all(count >= MAX_EPISODES for count in env_episode_counts):
                    break
            else:
                # ステップ数ベース：指定ステップ数に達したら終了
                if step_count >= MAX_STEPS:
                    break
            # PPO専門家ポリシーから行動予測（全環境を一度に処理）
            actions = expert_policy(obs)
            
            # 環境ステップ
            obs, rewards, dones, infos = env.step(actions)
            # PPOポリシーは観測値から行動を除外しない（45次元のまま）
            
            # 各環境の報酬を更新
            step_rewards = []
            rewards_cpu = rewards.cpu().numpy()
            
            for env_idx in range(NUM_ENVS):
                reward_value = float(rewards_cpu[env_idx])
                
                # リセット直後のステップの報酬は除外（前のステップの報酬を維持）
                if env_just_reset[env_idx]:
                    # 報酬は前のステップの平均報酬を使用（グラフの連続性のため）
                    if len(all_rewards) > 0:
                        prev_reward = all_rewards[-1][env_idx]
                    else:
                        prev_reward = 0.0
                    step_rewards.append(prev_reward)
                    env_just_reset[env_idx] = False  # リセットフラグをクリア
                else:
                    step_rewards.append(reward_value)
                
                env_episode_steps[env_idx] += 1
                env_current_episode_rewards[env_idx] += reward_value
                
                # エピソードリセット判定
                episode_done = dones[env_idx] or (env_episode_steps[env_idx] >= MAX_EPISODE_STEPS)
                if episode_done:
                    # エピソードの累積報酬を記録
                    env_episode_rewards[env_idx].append(env_current_episode_rewards[env_idx])
                    # エピソードの長さを記録
                    env_episode_lengths[env_idx].append(env_episode_steps[env_idx])
                    env_episode_cumulative_steps[env_idx].append(step_count)
                    env_current_episode_rewards[env_idx] = 0.0
                    
                    # エピソード数をカウント（各環境ごと）
                    if use_episode_limit:
                        env_episode_counts[env_idx] += 1
                        # 全環境がそれぞれMAX_EPISODES回のエピソードを完了したら終了
                        if all(count >= MAX_EPISODES for count in env_episode_counts):
                            break
                    
                    # 環境リセット
                    reset_indices = torch.tensor([env_idx], device=gs.device, dtype=torch.long)
                    env.reset_idx(reset_indices)
                    obs[env_idx] = env.obs_buf[env_idx]  # PPOは観測値から行動を除外しない
                    
                    env_episode_steps[env_idx] = 0
                    env_just_reset[env_idx] = True  # リセットフラグを設定
            
            all_rewards.append(step_rewards)
            step_count += 1
            
            # 進捗表示（100ステップごと）
            if step_count % 100 == 0:
                mean_reward = np.mean(step_rewards)
                std_reward = np.std(step_rewards)
                if use_episode_limit:
                    total_completed = sum(env_episode_counts)
                    total_target = MAX_EPISODES * NUM_ENVS
                    print(f"Step {step_count:5d} | Episodes: {total_completed}/{total_target} | Mean Reward: {mean_reward:7.5f} | Std: {std_reward:7.5f}")
                else:
                    print(f"Step {step_count:5d} / {MAX_STEPS} | Mean Reward: {mean_reward:7.5f} | Std: {std_reward:7.5f}")
    
    # グラフの作成（平均と標準偏差）
    print(f"\nCreating performance graphs...")
    
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
    all_episodes_data = []  # [(cumulative_steps, cumulative_reward, episode_length), ...]
    for env_idx in range(NUM_ENVS):
        for ep_idx, (cum_steps, cum_reward, ep_length) in enumerate(zip(
            env_episode_cumulative_steps[env_idx], 
            env_episode_rewards[env_idx],
            env_episode_lengths[env_idx]
        )):
            all_episodes_data.append((cum_steps, cum_reward, ep_length))
    
    # 累積ステップ数でソート
    all_episodes_data.sort(key=lambda x: x[0])
    
    # エピソード番号ごとに10環境の平均と標準偏差を計算
    # env_episode_rewards[env_idx][episode_idx] = 環境env_idxのepisode_idx番目のエピソードの累積報酬
    # 各エピソード番号（0, 1, 2, ...）に対して、全環境のそのエピソード番号の累積報酬を集計
    episode_num_to_rewards = {}  # {episode_num: [reward1, reward2, ...], ...}
    max_episode_num = 0
    
    # 各環境ごとに、その環境のエピソード番号（ep_idx）と累積報酬を対応付ける
    for env_idx in range(NUM_ENVS):
        # env_episode_rewards[env_idx]のインデックスがその環境のエピソード番号（0始まり）
        for ep_idx, cum_reward in enumerate(env_episode_rewards[env_idx]):
            # ep_idxがその環境でのエピソード番号（0始まり）
            # 同じep_idxを持つ全環境の累積報酬をグループ化
            if ep_idx not in episode_num_to_rewards:
                episode_num_to_rewards[ep_idx] = []
            episode_num_to_rewards[ep_idx].append(cum_reward)
            max_episode_num = max(max_episode_num, ep_idx)
    
    # エピソード番号ごとの平均と標準偏差を計算
    episode_nums_sorted = sorted(episode_num_to_rewards.keys())
    episode_means = []
    episode_stds = []
    episode_nums = []
    for ep_num in episode_nums_sorted:
        rewards = episode_num_to_rewards[ep_num]
        if len(rewards) > 0:
            # エピソード番号は1から始まる（表示用）
            episode_nums.append(ep_num + 1)
            episode_means.append(np.mean(rewards))
            episode_stds.append(np.std(rewards) if len(rewards) > 1 else 0.0)
    
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
    
    # ステップごとのグラフを作成
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    fig1.suptitle(f'Expert Policy Performance (Parallel) - {output_path.stem}', fontsize=16, fontweight='bold')
    
    # ステップごとの報酬の平均と標準偏差
    ax1.plot(steps, mean_rewards, linewidth=2, label='Mean Reward', color='blue')
    ax1.fill_between(steps,
                     np.array(mean_rewards) - np.array(std_rewards),
                     np.array(mean_rewards) + np.array(std_rewards),
                     alpha=0.3, color='blue', label='±1 Std')
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Reward', fontsize=11)
    ax1.set_title('Reward per Step (Mean ± Std)', fontsize=12, fontweight='bold')
    ax1.set_ylim(-0.03, 0.03)  # 縦軸を固定
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=0.5)
    ax1.legend()
    
    plt.tight_layout()
    plt.savefig(str(graph_path), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"✓ Graph saved: {graph_path}")
    
    # エピソードごとのグラフを別ファイルとして作成
    episode_graph_path = graph_path.parent / f"{graph_path.stem}_episodes.png"
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    fig2.suptitle(f'Expert Policy Performance - Episodes (Parallel) - {output_path.stem}', fontsize=16, fontweight='bold')
    
    # エピソード番号ごとの平均と標準偏差をプロット
    if len(episode_nums) > 0:
        ax2.plot(episode_nums, episode_means, linewidth=2, label='Mean Cumulative Reward per Episode', color='green')
        ax2.fill_between(episode_nums,
                         np.array(episode_means) - np.array(episode_stds),
                         np.array(episode_means) + np.array(episode_stds),
                         alpha=0.3, color='green', label='±1 Std')
    ax2.set_xlabel('Episode Number', fontsize=11)
    ax2.set_ylabel('Cumulative Reward per Episode', fontsize=11)
    ax2.set_title('Cumulative Reward per Episode (Mean ± Std)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 27)  # 縦軸を固定
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(str(episode_graph_path), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ Episode graph saved: {episode_graph_path}")
    
    # CSVファイルにも保存（ステップごとのデータ）
    csv_path = graph_path.with_suffix('.csv')
    with open(csv_path, 'w') as f:
        f.write("step,mean_reward,std_reward\n")
        for i, (mean_r, std_r) in enumerate(zip(mean_rewards, std_rewards), 1):
            f.write(f"{i},{mean_r:.6f},{std_r:.6f}\n")
    print(f"✓ CSV saved: {csv_path}")
    
    # エピソードごとの累積報酬のCSVも保存
    # 環境ごとのエピソードデータを保存（エピソード番号と環境インデックスを含める）
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
    
    # 最終統計
    final_mean_reward = np.mean(mean_rewards)
    final_std_reward = np.mean(std_rewards)
    
    # エピソード統計を計算
    all_episode_rewards_flat = [reward for env_rewards in env_episode_rewards for reward in env_rewards]
    all_episode_lengths_flat = [length for env_lengths in env_episode_lengths for length in env_lengths]
    mean_episode_reward = np.mean(all_episode_rewards_flat) if len(all_episode_rewards_flat) > 0 else 0.0
    std_episode_reward = np.std(all_episode_rewards_flat) if len(all_episode_rewards_flat) > 0 else 0.0
    mean_episode_length = np.mean(all_episode_lengths_flat) if len(all_episode_lengths_flat) > 0 else 0.0
    std_episode_length = np.std(all_episode_lengths_flat) if len(all_episode_lengths_flat) > 0 else 0.0
    total_episodes = len(all_episode_rewards_flat)
    
    # 平均報酬のテキストファイルを保存
    mean_reward_path = graph_path.parent / f"{graph_path.stem}_mean_reward.txt"
    with open(mean_reward_path, 'w') as f:
        f.write(f"Expert Policy Evaluation Summary\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Robot Type: {args.robot_type}\n")
        f.write(f"Total Steps: {step_count}\n")
        f.write(f"Number of Environments: {NUM_ENVS}\n")
        f.write(f"Total Episodes: {total_episodes}\n")
        f.write(f"\n")
        f.write(f"Mean Reward per Step: {final_mean_reward:.6f}\n")
        f.write(f"Std Reward per Step: {final_std_reward:.6f}\n")
        f.write(f"\n")
        f.write(f"Mean Reward per Episode: {mean_episode_reward:.6f}\n")
        f.write(f"Std Reward per Episode: {std_episode_reward:.6f}\n")
        f.write(f"Mean Episode Length: {mean_episode_length:.2f}\n")
        f.write(f"Std Episode Length: {std_episode_length:.2f}\n")
        f.write(f"{'=' * 60}\n")
    print(f"✓ Mean reward summary saved: {mean_reward_path}")
    
    print(f"\n{'=' * 80}")
    print(f"✓ Parallel evaluation completed!")
    print(f"  Output graph: {graph_path.absolute()}")
    print(f"  Total steps: {step_count}")
    print(f"  Number of environments: {NUM_ENVS}")
    print(f"  Mean reward per step: {final_mean_reward:.6f}")
    print(f"  Std reward per step: {final_std_reward:.6f}")
    print(f"{'=' * 80}")
    
    # グラフパスを返す（単一録画で使用）
    return graph_path


def _run_single_video_recording(args, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg):
    """単一環境で動画録画を実行（並列評価後に呼び出される、PPO専門家ポリシー使用）"""
    MAX_STEPS = args.max_steps
    MAX_EPISODE_STEPS = 1000
    
    # 環境作成（単一環境、ビジュアライズ有効）
    print(f"\n{'=' * 80}")
    print(f"Starting single environment video recording...")
    print(f"{'=' * 80}")
    print(f"Creating single {args.robot_type} environment for video recording...")
    
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
    elif args.robot_type == "go1":
        env = Go1Env(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
        )
    elif args.robot_type == "unitreea1":
        env = UnitreeA1Env(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
        )
    else:
        raise ValueError(f"Unknown robot type: {args.robot_type}")
    
    print(f"✓ Created single {args.robot_type} environment")
    
    # PPO専門家ポリシーのロード
    # train_cfgをコピーして使用（OnPolicyRunnerがtrain_cfgを変更する可能性があるため）
    train_cfg_copy = copy.deepcopy(train_cfg)
    
    # 専門家モデルのパスを決定
    if hasattr(args, 'expert_model_path') and args.expert_model_path is not None:
        expert_path = Path(args.expert_model_path)
    elif hasattr(args, 'expert_model_type') and args.expert_model_type is not None:
        expert_path = get_default_expert_model_path(args.expert_model_type)
    else:
        expert_path = get_default_expert_model_path(args.robot_type)
    
    # 専門家モデルのexp_nameを決定（train_cfgの読み込み用）
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
        else:
            return "go2-walking"  # デフォルト
    
    if hasattr(args, 'expert_model_type') and args.expert_model_type is not None:
        expert_exp_name = get_exp_name_for_robot(args.expert_model_type)
    else:
        expert_exp_name = args.exp_name
    
    print(f"Loading PPO expert model from {expert_path}...")
    if hasattr(args, 'expert_model_type') and args.expert_model_type is not None:
        print(f"Expert model type: {args.expert_model_type} (zero-shot evaluation on {args.robot_type})")
    genesis_path = Path(__file__).parents[2] / "Genesis"
    model_dir = genesis_path / "logs" / expert_exp_name
    runner = OnPolicyRunner(env, train_cfg_copy, str(model_dir), device=gs.device)
    runner.load(str(expert_path))
    expert_policy = runner.get_inference_policy(device=gs.device)
    print("✓ PPO expert model loaded")
    
    # 環境リセット
    obs, _ = env.reset()
    # PPOポリシーは観測値から行動を除外しない（45次元のまま）
    
    # 初期状態をランダム化（並列評価と同じ）
    from genesis.utils.geom import transform_quat_by_quat as transform_quat
    env_idx = torch.tensor([0], device=gs.device, dtype=torch.long)
    
    # 初期位置にランダムなオフセット（±0.1m）
    pos_offset = (torch.rand(1, 3, device=gs.device) - 0.5) * 0.2
    pos_offset[:, 2] = 0.0  # Z軸（高さ）は変更しない
    env.base_pos[env_idx] = env.base_init_pos + pos_offset
    env.robot.set_pos(env.base_pos[env_idx], zero_velocity=False, envs_idx=env_idx)
    
    # 初期姿勢（ロール・ピッチ）にランダムな角度（±5度）
    roll = (torch.rand(1, device=gs.device) - 0.5) * 10.0 * np.pi / 180.0
    pitch = (torch.rand(1, device=gs.device) - 0.5) * 10.0 * np.pi / 180.0
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    quat_noise = torch.stack([cr * cp, cr * sp, sr * cp, -sr * sp], dim=1)
    base_init_quat_expanded = env.base_init_quat.reshape(1, -1).expand(1, -1)
    env.base_quat[env_idx] = transform_quat(base_init_quat_expanded, quat_noise)
    env.robot.set_quat(env.base_quat[env_idx], zero_velocity=False, envs_idx=env_idx)
    
    # 関節角度にランダムなオフセット（±0.1ラジアン）
    dof_noise = (torch.rand(1, env.num_actions, device=gs.device) - 0.5) * 0.2
    env.dof_pos[env_idx] = env.default_dof_pos + dof_noise
    env.robot.set_dofs_position(
        position=env.dof_pos[env_idx],
        dofs_idx_local=env.motors_dof_idx,
        zero_velocity=True,
        envs_idx=env_idx,
    )
    
    # 観測値を更新（ランダム化後の状態を反映）
    zero_actions = torch.zeros(1, env.num_actions, device=gs.device)
    obs, _, _, _ = env.step(zero_actions)
    # PPOポリシーは観測値から行動を除外しない（45次元のまま）
    
    # 出力ディレクトリの作成
    output_path = Path(args.output)
    if output_path.suffix == '':
        output_path = output_path.with_suffix('.mp4')
        args.output = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 録画開始
    print(f"\nStarting video recording...")
    print(f"Recording {MAX_STEPS} steps at {args.fps} FPS...")
    env.cam.start_recording()
    
    step_count = 0
    with torch.no_grad():
        while step_count < MAX_STEPS:
            # PPO専門家ポリシーから行動予測
            action = expert_policy(obs)
            
            # 環境ステップ
            obs, rewards, dones, infos = env.step(action)
            # PPOポリシーは観測値から行動を除外しない（45次元のまま）
            
            # レンダリング（動画録画用）
            env.cam.render()
            
            step_count += 1
            
            # 進捗表示（100ステップごと）
            if step_count % 100 == 0:
                print(f"Recording step {step_count:5d} / {MAX_STEPS}...")
            
            # エピソードリセット判定
            episode_done = dones[0] or (step_count % MAX_EPISODE_STEPS == 0 and step_count > 0)
            if episode_done:
                reset_indices = torch.tensor([0], device=gs.device, dtype=torch.long)
                env.reset_idx(reset_indices)
                obs[0] = env.obs_buf[0]  # PPOは観測値から行動を除外しない
    
    # 録画停止と保存
    print(f"\nStopping recording and saving to {args.output}...")
    env.cam.stop_recording(save_to_filename=str(args.output), fps=args.fps)
    
    print(f"\n{'=' * 80}")
    print(f"✓ Video recording completed!")
    print(f"  Output video: {output_path.absolute()}")
    if output_path.exists():
        print(f"  Video file size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'=' * 80}")


def _run_single_environment_recording(args, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg):
    """単一環境での録画を実行（非並列モード用、PPO専門家ポリシー使用）"""
    MAX_EPISODE_STEPS = 1000
    
    # 環境作成
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
    elif args.robot_type == "go1":
        env = Go1Env(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
        )
    elif args.robot_type == "unitreea1":
        env = UnitreeA1Env(
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

    # PPO専門家ポリシーのロード
    # 専門家モデルのパスを決定
    if hasattr(args, 'expert_model_path') and args.expert_model_path is not None:
        expert_path = Path(args.expert_model_path)
    elif hasattr(args, 'expert_model_type') and args.expert_model_type is not None:
        expert_path = get_default_expert_model_path(args.expert_model_type)
    else:
        expert_path = get_default_expert_model_path(args.robot_type)

    # 専門家モデルのexp_nameを決定（train_cfgの読み込み用）
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
        else:
            return "go2-walking"  # デフォルト
    
    if hasattr(args, 'expert_model_type') and args.expert_model_type is not None:
        expert_exp_name = get_exp_name_for_robot(args.expert_model_type)
    else:
        expert_exp_name = args.exp_name
    print(f"Loading PPO expert model from {expert_path}...")
    if hasattr(args, 'expert_model_type') and args.expert_model_type is not None:
        print(f"Expert model type: {args.expert_model_type} (zero-shot evaluation on {args.robot_type})")
    genesis_path = Path(__file__).parents[2] / "Genesis"
    model_dir = genesis_path / "logs" / expert_exp_name
    runner = OnPolicyRunner(env, train_cfg, str(model_dir), device=gs.device)
    runner.load(str(expert_path))
    expert_policy = runner.get_inference_policy(device=gs.device)
    print("✓ PPO expert model loaded")

    # 環境リセット
    obs, _ = env.reset()
    # PPOポリシーは観測値から行動を除外しない（45次元のまま）

    # 出力ディレクトリの作成
    output_path = Path(args.output)
    # 拡張子がない場合は.mp4を追加（FFMPEGがフォーマットを判断するため必要）
    if output_path.suffix == '':
        output_path = output_path.with_suffix('.mp4')
        args.output = str(output_path)  # args.outputも更新
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
            # PPO専門家ポリシーから行動予測
            action = expert_policy(obs)
            
            # 環境ステップ
            obs, rewards, dones, infos = env.step(action)
            # PPOポリシーは観測値から行動を除外しない（45次元のまま）
            env.cam.render()
            
            # 報酬を記録
            reward_value = float(rewards.cpu().numpy()[0])
            
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
                    print(f"Episode {episode_count + 1} terminated | Reward: {episode_reward:.3f}")
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_step_count)
                episode_avg_rewards.append(episode_reward / episode_step_count)
                episode_cumulative_steps.append(step_count)
                
                episode_reward = 0.0
                episode_step_count = 0
                episode_count += 1
                episode_starts.append(step_count)
                
                # 環境リセット
                obs, _ = env.reset()  # PPOは観測値から行動を除外しない（45次元のまま）
    
    # 録画停止と保存
    print(f"\nStopping recording and saving to {args.output}...")
    env.cam.stop_recording(save_to_filename=str(args.output), fps=args.fps)
    
    # グラフの作成
    if len(step_rewards) > 0:
        graph_path = output_path.with_suffix('.png')
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Vintix Model Performance - {output_path.stem}', fontsize=16, fontweight='bold')
        
        steps = np.arange(1, len(step_rewards) + 1)
        
        # ステップごとの報酬
        axes[0].plot(steps, step_rewards, linewidth=1.5, alpha=0.7, color='blue', label='Reward per Step')
        axes[0].set_xlabel('Step', fontsize=11)
        axes[0].set_ylabel('Reward', fontsize=11)
        axes[0].set_title('Reward per Step', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=0.5)
        axes[0].legend()
        
        # エピソードごとの累積報酬
        if len(episode_rewards) > 0:
            episode_steps_for_plot = episode_cumulative_steps[1:] if len(episode_cumulative_steps) > 1 else [step_count]
            axes[1].plot(episode_steps_for_plot, episode_rewards, marker='o', linewidth=2, markersize=6, color='green', label='Cumulative Reward per Episode')
            axes[1].set_xlabel('Step', fontsize=11)
            axes[1].set_ylabel('Cumulative Reward', fontsize=11)
            axes[1].set_title('Cumulative Reward per Episode', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(str(graph_path), dpi=150, bbox_inches='tight')
        print(f"✓ Graph saved: {graph_path}")
        
        # CSVファイルにも保存（ステップごとのデータ）
        csv_path = graph_path.with_suffix('.csv')
        with open(csv_path, 'w') as f:
            f.write("step,reward\n")
            for i, reward in enumerate(step_rewards, 1):
                f.write(f"{i},{reward:.6f}\n")
        print(f"✓ CSV saved: {csv_path}")
        
        # エピソードごとの累積報酬のCSVも保存
        episode_csv_path = graph_path.parent / f"{graph_path.stem}_episodes.csv"
        with open(episode_csv_path, 'w') as f:
            f.write("episode,cumulative_steps,cumulative_reward,episode_length,avg_reward\n")
            for ep_idx, (cum_steps, cum_reward, ep_len, avg_reward) in enumerate(
                zip(episode_cumulative_steps[1:], episode_rewards, episode_lengths, episode_avg_rewards), 1
            ):
                f.write(f"{ep_idx},{cum_steps},{cum_reward:.6f},{ep_len},{avg_reward:.6f}\n")
        print(f"✓ Episode CSV saved: {episode_csv_path}")
    
    print(f"\n✓ Video saved successfully!")
    print(f"  Output: {output_path.absolute()}")
    
    if output_path.exists():
        print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Save PPO expert policy behavior as video")
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking",
                        help="Experiment name (for loading env config)")
    parser.add_argument("-r", "--robot_type", type=str, choices=["go2", "minicheetah", "laikago", "go1", "unitreea1"], 
                        default=None, action="append", help="Robot type (can be specified multiple times for multiple robots)")
    parser.add_argument("--expert_model_type", type=str, choices=["go2", "minicheetah", "laikago", "go1", "unitreea1"], default=None,
                        help="Expert model robot type (for zero-shot evaluation, e.g., use go2 model for go1 evaluation)")
    parser.add_argument("--expert_model_path", type=str, default=None,
                        help="Path to expert model file (if not specified, uses default based on expert_model_type or robot_type)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video file path (if not specified, auto-generated from robot type)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum steps to record (deprecated, use --max_episodes instead)")
    parser.add_argument("--max_episodes", type=int, default=1,
                        help="Maximum number of episodes to evaluate (default: 1)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video FPS")
    parser.add_argument("--parallel", action="store_true",
                        help="Use parallel evaluation")
    parser.add_argument("--num_envs", type=int, default=100,
                        help="Number of parallel environments (default: 100)")
    args = parser.parse_args()
    
    # robot_typeが指定されていない場合、デフォルト値を設定
    if args.robot_type is None:
        args.robot_type = ["go2"]
    
    # robot_typeがリストでない場合（後方互換性のため）、リストに変換
    if not isinstance(args.robot_type, list):
        args.robot_type = [args.robot_type]
    
    # 重複を除去し、順序を保持
    seen = set()
    unique_robot_types = []
    for robot_type in args.robot_type:
        if robot_type not in seen:
            seen.add(robot_type)
            unique_robot_types.append(robot_type)
    robot_types = unique_robot_types
    
    # robot_typeからexp_nameへのマッピング関数
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
        else:
            return "go2-walking"  # デフォルト
    
    # 全体のヘッダー表示
    print("=" * 80)
    if args.parallel:
        print("Expert Policy Parallel Evaluation (Multiple Robots)")
    else:
        print("Expert Policy Video Recording (Multiple Robots)")
    print("=" * 80)
    print(f"Robot types: {', '.join(robot_types)}")
    if args.max_steps is not None:
        print(f"Max steps: {args.max_steps} (deprecated, use --max_episodes instead)")
    else:
        print(f"Max episodes: {args.max_episodes}")
        print(f"Max steps (estimated): {args.max_episodes * 1000}")
    print(f"FPS: {args.fps}")
    if args.parallel:
        if args.max_steps is not None:
            print(f"Mode: Parallel ({args.num_envs} envs, {args.max_steps} steps each)")
        else:
            print(f"Mode: Parallel ({args.num_envs} envs, {args.max_episodes} episodes each)")
    print("=" * 80)
    print()
    
    # Genesis初期化（全ロボット共通）
    if args.parallel:
        gs.init(performance_mode=True)  # 並列評価時はパフォーマンスモードを有効化
    else:
        gs.init()
    
    # 各ロボットタイプごとに評価を実行
    for robot_idx, robot_type in enumerate(robot_types, 1):
        print(f"\n{'=' * 80}")
        print(f"Evaluating Robot {robot_idx}/{len(robot_types)}: {robot_type}")
        print(f"{'=' * 80}")
        
        # 現在のロボットタイプ用のargsを作成（コピー）
        robot_args = argparse.Namespace(**vars(args))
        robot_args.robot_type = robot_type
        
        # exp_nameを自動設定
        robot_exp_name = get_exp_name_for_robot(robot_type)
        if args.exp_name == "go2-walking":  # デフォルト値の場合のみ自動設定
            robot_args.exp_name = robot_exp_name
        
        # 出力ファイル名を自動生成（各ロボットごとに別々）
        # ゼロショット評価の場合は、カスタム出力ディレクトリを使用
        if args.output is None:
            # ゼロショット評価の場合、カスタムディレクトリ名を生成
            if hasattr(args, 'expert_model_type') and args.expert_model_type is not None and args.expert_model_type != robot_type:
                vintix_root = Path(__file__).parent.parent
                zero_shot_dir_name = f"{args.expert_model_type.capitalize()}to{robot_type.capitalize()}"
                result_dir = vintix_root / "Expert_result" / zero_shot_dir_name
                result_dir.mkdir(parents=True, exist_ok=True)
                filename = f"expert_{args.expert_model_type}_on_{robot_type}"
                robot_args.output = str(result_dir / filename)
            else:
                robot_args.output = generate_output_filename(robot_type)
        else:
            # 出力パスが指定されている場合はそのまま使用
            robot_args.output = args.output
        
        print(f"Robot type: {robot_type}")
        print(f"Experiment name: {robot_args.exp_name}")
        print(f"Output: {robot_args.output}")
        print()
        
        # 環境設定の読み込み（各ロボットごとに）
        genesis_root = Path(__file__).parents[2] / "Genesis"
        log_dir = genesis_root / "logs" / robot_args.exp_name
        cfgs_path = log_dir / "cfgs.pkl"
        
        if cfgs_path.exists():
            env_cfg, obs_cfg, reward_cfg, command_cfg, _ = pickle.load(open(cfgs_path, "rb"))
            print(f"Loaded env config from: {cfgs_path}")
        else:
            # デフォルト設定を使用
            print(f"Config file not found: {cfgs_path}. Using default configuration.")
            from train import get_go2_cfgs, get_minicheetah_cfgs, get_laikago_cfgs, get_go1_cfgs, get_unitreea1_cfgs
            
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
            else:
                raise ValueError(f"Unknown robot type: {robot_type}")
        
        # train_cfgの読み込み（専門家モデルのtrain_cfgを使用）
        # ゼロショット評価の場合は専門家モデルのtrain_cfgを読み込む
        if hasattr(args, 'expert_model_type') and args.expert_model_type is not None:
            expert_exp_name = get_exp_name_for_robot(args.expert_model_type)
        else:
            expert_exp_name = robot_args.exp_name
        
        expert_log_dir = genesis_root / "logs" / expert_exp_name
        expert_cfgs_path = expert_log_dir / "cfgs.pkl"
        
        if expert_cfgs_path.exists():
            _, _, _, _, train_cfg = pickle.load(open(expert_cfgs_path, "rb"))
            print(f"Loaded train_cfg from expert model: {expert_cfgs_path}")
        else:
            raise FileNotFoundError(f"Expert model config file not found: {expert_cfgs_path}. train_cfg is required for PPO policy loading. Please ensure the PPO model was trained and cfgs.pkl exists.")
        
        # 並列評価の場合は別処理
        if args.parallel:
            # 並列評価を実行（グラフのみ作成、録画なし）
            graph_path = _run_parallel_evaluation(robot_args, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)
            
            # 並列評価完了後、単一環境で動画録画を実行
            print(f"\n{'=' * 80}")
            print(f"Parallel evaluation completed for {robot_type}. Starting video recording...")
            print(f"{'=' * 80}")
            
            # 出力ファイル名を設定（グラフと同じディレクトリに動画を保存）
            video_output_path = graph_path.parent / f"{graph_path.stem}.mp4"
            robot_args.output = str(video_output_path)
            
            # 単一環境で録画を実行
            _run_single_video_recording(robot_args, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)
        else:
            # 単一環境での録画（既存のロジックを呼び出す）
            _run_single_environment_recording(robot_args, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)
    
    # 全ロボットの評価完了
    print(f"\n{'=' * 80}")
    print(f"✓ All evaluations completed!")
    print(f"  Evaluated {len(robot_types)} robot(s): {', '.join(robot_types)}")
    print(f"{'=' * 80}")
    return
if __name__ == "__main__":
    main()

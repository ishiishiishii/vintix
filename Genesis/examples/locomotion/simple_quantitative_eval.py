import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from importlib import metadata
import random

import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env


class SimpleQuantitativeEvaluator:
    def __init__(self, env, policy, num_episodes=10, episode_length=1000, num_seeds=5, robot_name="Go2"):
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.num_seeds = num_seeds
        self.robot_name = robot_name
        
        # 6つの重要指標のみ
        self.key_metrics = [
            'avg_forward_velocity',  # 平均前進速度
            'success_rate',          # 成功率
            'avg_pitch',             # 平均ピッチ角
            'avg_roll',              # 平均ロール角
            'total_reward',          # 総報酬
            'avg_base_height',       # 平均ベース高さ
            'avg_episode_length'     # 平均エピソード長
        ]
        
    def set_seed(self, seed):
        """シードを設定"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def collect_episode_data(self, episode_idx, seed):
        """1エピソードのデータを収集"""
        self.set_seed(seed)
        
        obs, _ = self.env.reset()
        episode_data = {
            'base_pos': [],
            'base_vel': [],
            'base_euler': [],
            'rewards': [],
            'episode_length': 0,
            'total_reward': 0.0,
            'success': False,
            'seed': seed
        }
        
        with torch.no_grad():
            for step in range(self.episode_length):
                actions = self.policy(obs)
                obs, rews, dones, infos = self.env.step(actions)
                
                # 重要データのみ収集
                episode_data['base_pos'].append(self.env.base_pos[0].cpu().numpy())
                episode_data['base_vel'].append(self.env.base_lin_vel[0].cpu().numpy())
                episode_data['base_euler'].append(self.env.base_euler[0].cpu().numpy())
                episode_data['rewards'].append(rews[0].cpu().numpy())
                
                episode_data['total_reward'] += float(rews[0].cpu().numpy())
                episode_data['episode_length'] = step + 1
                
                # 転倒判定（ピッチ・ロール角度が大きすぎる場合）
                if (abs(self.env.base_euler[0, 1]) > 45 or  # ピッチ
                    abs(self.env.base_euler[0, 0]) > 45):   # ロール
                    episode_data['success'] = False
                    break
                
                if dones[0]:
                    episode_data['success'] = False  # 通常の終了は失敗として扱う
                    break
                
                # エピソードの最後まで到達した場合は成功
                if step == self.episode_length - 1:
                    episode_data['success'] = True
        
        return episode_data
    
    def calculate_key_metrics(self, episode_data):
        """6つの重要指標のみを計算"""
        base_pos = np.array(episode_data['base_pos'])
        base_vel = np.array(episode_data['base_vel'])
        base_euler = np.array(episode_data['base_euler'])
        
        metrics = {}
        
        # 1. 歩行性能
        metrics['avg_forward_velocity'] = np.mean(base_vel[:, 0])  # X軸方向の平均速度
        metrics['success_rate'] = 1.0 if episode_data['success'] else 0.0
        
        # 2. 安定性
        metrics['avg_pitch'] = np.mean(np.abs(base_euler[:, 1]))  # ピッチ角の平均
        metrics['avg_roll'] = np.mean(np.abs(base_euler[:, 0]))   # ロール角の平均
        
        # 3. エネルギー効率
        metrics['total_reward'] = episode_data['total_reward']
        
        # 4. 動作品質
        metrics['avg_base_height'] = np.mean(base_pos[:, 2])  # Z軸方向の平均高さ
        
        # 5. 継続性
        metrics['avg_episode_length'] = episode_data['episode_length']
        
        # シード情報
        metrics['seed'] = episode_data['seed']
        
        return metrics
    
    def evaluate_single_seed(self, seed):
        """単一シードでの評価を実行"""
        print(f"Evaluating {self.robot_name} with seed {seed}...")
        
        seed_metrics = []
        
        for episode in range(self.num_episodes):
            print(f"  Episode {episode + 1}/{self.num_episodes}")
            episode_data = self.collect_episode_data(episode, seed)
            metrics = self.calculate_key_metrics(episode_data)
            seed_metrics.append(metrics)
            
            print(f"    Success: {metrics['success_rate']:.0%}, "
                  f"Velocity: {metrics['avg_forward_velocity']:.3f} m/s, "
                  f"Reward: {metrics['total_reward']:.1f}")
        
        return seed_metrics
    
    def evaluate(self):
        """複数シードで評価を実行"""
        print(f"Starting evaluation for {self.robot_name} with {self.num_seeds} seeds, {self.num_episodes} episodes each...")
        
        all_seed_results = []
        
        for seed_idx in range(self.num_seeds):
            seed = 42 + seed_idx
            seed_metrics = self.evaluate_single_seed(seed)
            all_seed_results.append(seed_metrics)
            
            # シードごとの平均を計算・表示
            seed_df = pd.DataFrame(seed_metrics)
            print(f"\n{self.robot_name} Seed {seed} Summary:")
            print(f"  Success Rate: {seed_df['success_rate'].mean():.1%}")
            print(f"  Avg Velocity: {seed_df['avg_forward_velocity'].mean():.3f} ± {seed_df['avg_forward_velocity'].std():.3f} m/s")
            print(f"  Avg Reward: {seed_df['total_reward'].mean():.1f} ± {seed_df['total_reward'].std():.1f}")
            print(f"  Avg Episode Length: {seed_df['avg_episode_length'].mean():.0f} steps")
            print(f"  Avg Pitch: {seed_df['avg_pitch'].mean():.1f}°")
            print(f"  Avg Roll: {seed_df['avg_roll'].mean():.1f}°")
        
        return all_seed_results
    
    def aggregate_results(self, all_seed_results):
        """全シードの結果を集約"""
        # 全エピソードのデータを結合
        all_episodes = []
        for seed_idx, seed_metrics in enumerate(all_seed_results):
            for episode_metrics in seed_metrics:
                all_episodes.append(episode_metrics)
        
        all_df = pd.DataFrame(all_episodes)
        
        # シードごとの平均
        seed_averages = []
        for seed_idx, seed_metrics in enumerate(all_seed_results):
            seed_df = pd.DataFrame(seed_metrics)
            seed_avg = seed_df.mean(numeric_only=True).to_dict()
            seed_avg['seed'] = 42 + seed_idx
            seed_averages.append(seed_avg)
        
        seed_avg_df = pd.DataFrame(seed_averages)
        
        # 全体の平均・標準偏差
        overall_avg = all_df.mean(numeric_only=True).to_dict()
        overall_std = all_df.std(numeric_only=True).to_dict()
        
        return all_df, seed_avg_df, overall_avg, overall_std
    
    def save_results(self, all_seed_results, save_dir):
        """結果を保存"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 結果を集約
        all_df, seed_avg_df, overall_avg, overall_std = self.aggregate_results(all_seed_results)
        
        # CSVファイルとして保存
        all_df.to_csv(os.path.join(save_dir, f'{self.robot_name.lower()}_all_episodes.csv'), index=False)
        seed_avg_df.to_csv(os.path.join(save_dir, f'{self.robot_name.lower()}_seed_averages.csv'), index=False)
        
        # 統計情報を計算
        summary_stats = all_df.describe()
        summary_stats.to_csv(os.path.join(save_dir, f'{self.robot_name.lower()}_summary.csv'))
        
        # グラフを作成
        self.create_simple_plots(all_df, seed_avg_df, save_dir)
        
        # 結果を表示
        print(f"\n=== {self.robot_name} EVALUATION SUMMARY ===")
        print(f"Results saved to: {save_dir}")
        print(f"Number of seeds: {self.num_seeds}")
        print(f"Episodes per seed: {self.num_episodes}")
        print(f"Total episodes: {len(all_df)}")
        print(f"Overall Success Rate: {overall_avg['success_rate']:.1%}")
        print(f"Overall Avg Velocity: {overall_avg['avg_forward_velocity']:.3f} ± {overall_std['avg_forward_velocity']:.3f} m/s")
        print(f"Overall Avg Reward: {overall_avg['total_reward']:.1f} ± {overall_std['total_reward']:.1f}")
        print(f"Overall Avg Episode Length: {overall_avg['avg_episode_length']:.0f} steps")
        print(f"Overall Avg Pitch: {overall_avg['avg_pitch']:.1f}°")
        print(f"Overall Avg Roll: {overall_avg['avg_roll']:.1f}°")
        
        return all_df, seed_avg_df, summary_stats
    
    def create_simple_plots(self, all_df, seed_avg_df, save_dir):
        """再現性・安定性を強調したグラフを作成"""
        plt.style.use('seaborn-v0_8')
        
        # メインの評価グラフ（2x3レイアウト）
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.robot_name} Quantitative Evaluation Results\n(Reproducible Performance Across Multiple Seeds)', 
                    fontsize=16, fontweight='bold')
        
        # 1. 成功率（箱ひげ図 + エラーバー）
        success_data = [all_df[all_df['seed'] == seed]['success_rate'].values for seed in seed_avg_df['seed']]
        bp1 = axes[0, 0].boxplot(success_data, labels=[f'Seed {s}' for s in seed_avg_df['seed']], 
                                patch_artist=True, showmeans=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightgreen')
            patch.set_alpha(0.7)
        axes[0, 0].set_title('Success Rate Distribution\n(Consistency Across Seeds)', fontweight='bold')
        axes[0, 0].set_xlabel('Seed')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_ylim(0, 1.1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 平均速度（箱ひげ図 + エラーバー）
        velocity_data = [all_df[all_df['seed'] == seed]['avg_forward_velocity'].values for seed in seed_avg_df['seed']]
        bp2 = axes[0, 1].boxplot(velocity_data, labels=[f'Seed {s}' for s in seed_avg_df['seed']], 
                                patch_artist=True, showmeans=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        axes[0, 1].set_title('Forward Velocity Distribution\n(Stable Performance)', fontweight='bold')
        axes[0, 1].set_xlabel('Seed')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 総報酬（箱ひげ図 + エラーバー）
        reward_data = [all_df[all_df['seed'] == seed]['total_reward'].values for seed in seed_avg_df['seed']]
        bp3 = axes[0, 2].boxplot(reward_data, labels=[f'Seed {s}' for s in seed_avg_df['seed']], 
                                patch_artist=True, showmeans=True)
        for patch in bp3['boxes']:
            patch.set_facecolor('orange')
            patch.set_alpha(0.7)
        axes[0, 2].set_title('Total Reward Distribution\n(Consistent Learning)', fontweight='bold')
        axes[0, 2].set_xlabel('Seed')
        axes[0, 2].set_ylabel('Total Reward')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 安定性（ピッチ・ロール）- エラーバー付き棒グラフ
        x_pos = np.arange(len(seed_avg_df))
        width = 0.35
        
        # エピソードごとのデータから標準偏差を計算
        pitch_std = [all_df[all_df['seed'] == seed]['avg_pitch'].std() for seed in seed_avg_df['seed']]
        roll_std = [all_df[all_df['seed'] == seed]['avg_roll'].std() for seed in seed_avg_df['seed']]
        
        axes[1, 0].bar(x_pos - width/2, seed_avg_df['avg_pitch'], width, 
                      yerr=pitch_std, label='Pitch', color='lightcoral', alpha=0.8, capsize=5)
        axes[1, 0].bar(x_pos + width/2, seed_avg_df['avg_roll'], width, 
                      yerr=roll_std, label='Roll', color='lightblue', alpha=0.8, capsize=5)
        axes[1, 0].set_title('Stability: Pitch vs Roll\n(Error bars show variability)', fontweight='bold')
        axes[1, 0].set_xlabel('Seed')
        axes[1, 0].set_ylabel('Average Angle (degrees)')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([f'Seed {s}' for s in seed_avg_df['seed']])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. エピソード長（箱ひげ図）
        length_data = [all_df[all_df['seed'] == seed]['avg_episode_length'].values for seed in seed_avg_df['seed']]
        bp5 = axes[1, 1].boxplot(length_data, labels=[f'Seed {s}' for s in seed_avg_df['seed']], 
                                patch_artist=True, showmeans=True)
        for patch in bp5['boxes']:
            patch.set_facecolor('purple')
            patch.set_alpha(0.7)
        axes[1, 1].set_title('Episode Length Distribution\n(Consistent Completion)', fontweight='bold')
        axes[1, 1].set_xlabel('Seed')
        axes[1, 1].set_ylabel('Episode Length (steps)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. ベース高さ（箱ひげ図）
        height_data = [all_df[all_df['seed'] == seed]['avg_base_height'].values for seed in seed_avg_df['seed']]
        bp6 = axes[1, 2].boxplot(height_data, labels=[f'Seed {s}' for s in seed_avg_df['seed']], 
                                patch_artist=True, showmeans=True)
        for patch in bp6['boxes']:
            patch.set_facecolor('teal')
            patch.set_alpha(0.7)
        axes[1, 2].set_title('Base Height Distribution\n(Stable Posture)', fontweight='bold')
        axes[1, 2].set_xlabel('Seed')
        axes[1, 2].set_ylabel('Base Height (m)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{self.robot_name.lower()}_evaluation_plots.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # サマリーテーブルを作成
        self.create_summary_table(seed_avg_df, save_dir)
    
    def create_summary_table(self, seed_avg_df, save_dir):
        """サマリーテーブルを作成"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # テーブルデータを準備
        table_data = []
        for _, row in seed_avg_df.iterrows():
            table_data.append([
                f"Seed {int(row['seed'])}",
                f"{row['success_rate']:.1%}",
                f"{row['avg_forward_velocity']:.3f}",
                f"{row['total_reward']:.1f}",
                f"{row['avg_episode_length']:.0f}",
                f"{row['avg_pitch']:.1f}°",
                f"{row['avg_roll']:.1f}°",
                f"{row['avg_base_height']:.3f}"
            ])
        
        # 全体平均を追加
        overall_avg = seed_avg_df.mean(numeric_only=True)
        table_data.append([
            "OVERALL",
            f"{overall_avg['success_rate']:.1%}",
            f"{overall_avg['avg_forward_velocity']:.3f}",
            f"{overall_avg['total_reward']:.1f}",
            f"{overall_avg['avg_episode_length']:.0f}",
            f"{overall_avg['avg_pitch']:.1f}°",
            f"{overall_avg['avg_roll']:.1f}°",
            f"{overall_avg['avg_base_height']:.3f}"
        ])
        
        # テーブルを作成
        table = ax.table(cellText=table_data,
                        colLabels=['Seed', 'Success Rate', 'Velocity (m/s)', 'Reward', 
                                 'Episode Length', 'Pitch', 'Roll', 'Base Height (m)'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # ヘッダーのスタイル
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 全体平均行のスタイル
        for i in range(len(table_data[0])):
            table[(len(table_data), i)].set_facecolor('#FFC107')
            table[(len(table_data), i)].set_text_props(weight='bold')
        
        plt.title(f'{self.robot_name} Evaluation Summary Table', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(os.path.join(save_dir, f'{self.robot_name.lower()}_summary_table.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Simple quantitative evaluation for Go2 robot")
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking",
                       help="Experiment name")
    parser.add_argument("--ckpt", type=int, default=100,
                       help="Checkpoint number")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes per seed")
    parser.add_argument("--episode_length", type=int, default=1000, help="Length of each episode")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of different seeds to test")
    parser.add_argument("--no_viewer", action="store_true", help="Disable viewer for faster evaluation")
    parser.add_argument("--robot_name", type=str, default="Go2", help="Robot name for display")
    
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    # 報酬スケールを保持（学習時と同じ設定を使用）
    # reward_cfg["reward_scales"] = {}  # この行をコメントアウト

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.no_viewer,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    # 簡易定量的評価を実行
    evaluator = SimpleQuantitativeEvaluator(
        env=env,
        policy=policy,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        num_seeds=args.num_seeds,
        robot_name=args.robot_name
    )
    
    all_seed_results = evaluator.evaluate()
    
    # 結果を保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"evaluation_results/simple_eval_{args.robot_name.lower()}_{args.exp_name}_ckpt{args.ckpt}_seeds{args.num_seeds}_{timestamp}"
    
    all_df, seed_avg_df, summary_stats = evaluator.save_results(all_seed_results, save_dir)
    
    print(f"\nEvaluation completed! Results saved to: {save_dir}")


if __name__ == "__main__":
    main()

"""
# 簡易定量的評価の実行例
python examples/locomotion/simple_quantitative_eval.py -e go2-walking --ckpt 100 --num_episodes     0 --episode_length 300 --num_seeds 5

# Go2の評価
python examples/locomotion/simple_quantitative_eval.py -e go2-walking --ckpt 100 --robot_name Go2

例  python examples/locomotion/simple_quantitative_eval.py -e minicheetah_to_go2_finetuning --ckpt 120 --num_episodes 5 --num_seeds 5 --episode_length 300 --no_viewer --robot_name Go2

# Minicheetahの評価（環境設定を変更する必要があります）
python examples/locomotion/simple_quantitative_eval.py -e minicheetah-walking --ckpt 100 --robot_name Minicheetah
"""

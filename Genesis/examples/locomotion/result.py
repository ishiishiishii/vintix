#!/usr/bin/env python3
"""
Training Results Analysis and Visualization
複数の訓練結果から平均と標準偏差を計算してグラフを作成
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator


def read_tensorboard_logs(log_dir):
    """
    TensorBoardのログファイルからrewardデータを読み取る
    
    Args:
        log_dir: ログディレクトリのパス
        
    Returns:
        steps: ステップ数のリスト
        rewards: 報酬のリスト
    """
    # TensorBoardのイベントファイルを探す
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    
    if not event_files:
        print(f"Warning: No event files found in {log_dir}")
        return None, None
    
    # 最初のイベントファイルを読み込む
    event_file = event_files[0]
    
    try:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        # 利用可能なスカラー（メトリクス）を確認
        scalar_tags = ea.Tags()['scalars']
        
        # Rewardに関連するタグを探す
        reward_tags = [tag for tag in scalar_tags if 'reward' in tag.lower() or 'episode' in tag.lower()]
        
        if not reward_tags:
            print(f"Available tags: {scalar_tags}")
            # デフォルトで最初のタグを使用
            if scalar_tags:
                reward_tag = scalar_tags[0]
            else:
                return None, None
        else:
            # 最も関連性の高いタグを選択
            reward_tag = reward_tags[0]
        
        print(f"Using tag: {reward_tag}")
        
        # データを取得
        events = ea.Scalars(reward_tag)
        steps = [event.step for event in events]
        rewards = [event.value for event in events]
        
        return np.array(steps), np.array(rewards)
        
    except Exception as e:
        print(f"Error reading {event_file}: {e}")
        return None, None


def collect_multiple_results(log_dirs):
    """
    複数のログディレクトリからデータを収集
    
    Args:
        log_dirs: ログディレクトリのリスト
        
    Returns:
        all_steps: 全実験のステップ数
        all_rewards: 全実験の報酬データ
    """
    all_data = []
    
    for log_dir in log_dirs:
        if not os.path.exists(log_dir):
            print(f"Warning: {log_dir} does not exist")
            continue
        
        steps, rewards = read_tensorboard_logs(log_dir)
        
        if steps is not None and rewards is not None:
            all_data.append({'steps': steps, 'rewards': rewards, 'name': os.path.basename(log_dir)})
            print(f"Loaded {len(steps)} data points from {os.path.basename(log_dir)}")
        else:
            print(f"Failed to load data from {log_dir}")
    
    return all_data


def interpolate_to_common_steps(all_data):
    """
    異なる実験のデータを共通のステップに補間
    
    Args:
        all_data: 各実験のデータのリスト
        
    Returns:
        common_steps: 共通のステップ
        interpolated_rewards: 補間された報酬データ（各実験）
    """
    # 全実験の最小・最大ステップを取得
    min_step = max([data['steps'].min() for data in all_data])
    max_step = min([data['steps'].max() for data in all_data])
    
    # 共通のステップを作成
    num_points = 100
    common_steps = np.linspace(min_step, max_step, num_points)
    
    # 各実験のデータを補間
    interpolated_rewards = []
    for data in all_data:
        interp_rewards = np.interp(common_steps, data['steps'], data['rewards'])
        interpolated_rewards.append(interp_rewards)
    
    return common_steps, np.array(interpolated_rewards)


def plot_results(common_steps, interpolated_rewards, exp_names, output_file='training_results.png'):
    """
    平均と標準偏差を含むグラフを作成
    
    Args:
        common_steps: 共通のステップ
        interpolated_rewards: 補間された報酬データ
        exp_names: 実験名のリスト
        output_file: 出力ファイル名
    """
    # 平均と標準偏差を計算
    mean_rewards = np.mean(interpolated_rewards, axis=0)
    std_rewards = np.std(interpolated_rewards, axis=0)
    
    # グラフを作成
    plt.figure(figsize=(12, 6))
    
    # 平均ライン
    plt.plot(common_steps, mean_rewards, 'b-', linewidth=2, label=f'Mean Reward (n={len(interpolated_rewards)})')
    
    # 標準偏差の帯
    plt.fill_between(
        common_steps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.3,
        color='blue',
        label='±1 Std Dev'
    )
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title(f'Training Results (n={len(interpolated_rewards)} runs)', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # グラフを保存
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nグラフを {output_file} に保存しました")
    
    # 統計情報を表示
    print(f"\n=== 統計情報 ===")
    print(f"実験数: {len(interpolated_rewards)}")
    print(f"最終報酬 (平均 ± 標準偏差): {mean_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}")
    print(f"最大報酬 (平均 ± 標準偏差): {mean_rewards.max():.2f} ± {std_rewards[mean_rewards.argmax()]:.2f}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze and visualize training results')
    parser.add_argument('-d', '--log_dirs', nargs='+', required=True,
                        help='List of log directories to analyze')
    parser.add_argument('-o', '--output', type=str, default='training_results.png',
                        help='Output file name for the plot')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom title for the plot')
    
    args = parser.parse_args()
    
    # ログディレクトリのパスを作成
    log_dirs = []
    for log_dir in args.log_dirs:
        if not os.path.isabs(log_dir):
            # 相対パスの場合、logs/以下として処理
            log_dir = os.path.join('logs', log_dir)
        log_dirs.append(log_dir)
    
    print(f"=== 訓練結果分析 ===")
    print(f"分析対象: {len(log_dirs)} 個の実験")
    
    # データを収集
    all_data = collect_multiple_results(log_dirs)
    
    if not all_data:
        print("Error: No data could be loaded")
        return
    
    print(f"\n成功: {len(all_data)} 個の実験データを読み込みました")
    
    # データを補間
    common_steps, interpolated_rewards = interpolate_to_common_steps(all_data)
    
    # 実験名を取得
    exp_names = [data['name'] for data in all_data]
    
    # グラフを作成
    plot_results(common_steps, interpolated_rewards, exp_names, args.output)


if __name__ == "__main__":
    main()


"""
# 使用例

# 基本的な使用方法
python examples/locomotion/result.py -d go2-walking-seed1 go2-walking-seed2 go2-walking-seed3

# カスタム出力ファイル名
python examples/locomotion/result.py -d go2-walking-seed1 go2-walking-seed2 go2-walking-seed3 -o my_results.png

# 絶対パスを指定
python examples/locomotion/result.py -d logs/go2-walking-seed1 logs/go2-walking-seed2 logs/go2-walking-seed3

# ワイルドカードを使用（シェルで展開）
python examples/locomotion/result.py -d logs/go2-walking-seed*

# 5つのシード結果を分析
python examples/locomotion/result.py -d go2-walking-seed1 go2-walking-seed2 go2-walking-seed3 go2-walking-seed4 go2-walking-seed5 -o go2_5seeds_results.png
"""

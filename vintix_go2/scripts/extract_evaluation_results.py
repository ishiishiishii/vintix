#!/usr/bin/env python3
"""
評価結果のテキストファイルから平均報酬を抽出し、表にまとめるスクリプト
"""
import re
import csv
from pathlib import Path
from typing import Dict, Optional

models = {
    'a1_without': 'A1',
    'go2_without': 'Go2', 
    'go1_without': 'Go1',
    'minicheetah_without': 'Minicheetah'
}

robots = {
    'go1': 'Go1',
    'go2': 'Go2',
    'unitreea1': 'A1',
    'minicheetah': 'Minicheetah'
}

base_path = Path('models/vintix_go2')


def extract_mean_reward(content: str) -> Optional[float]:
    """テキストファイルの内容から平均報酬を抽出"""
    # 様々なパターンで平均報酬を探す
    patterns = [
        r'mean.*?reward.*?(-?\d+\.?\d+)',
        r'average.*?reward.*?(-?\d+\.?\d+)',
        r'reward.*?mean.*?(-?\d+\.?\d+)',
        r'Mean.*?Reward.*?(-?\d+\.?\d+)',
        r'Average.*?Reward.*?(-?\d+\.?\d+)',
        r'mean.*?(-?\d+\.?\d+)',  # より汎用的なパターン
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    # パターンマッチングが失敗した場合、数値が含まれる行を探す
    lines = content.split('\n')
    for line in lines:
        if any(keyword in line.lower() for keyword in ['mean', 'average', 'reward']):
            numbers = re.findall(r'-?\d+\.?\d+', line)
            if numbers:
                try:
                    return float(numbers[0])
                except ValueError:
                    continue
    
    return None


def main():
    results: Dict[str, Dict[str, float]] = {}
    
    # 各モデルの結果を読み込む
    for model_key, model_display in models.items():
        model_path = base_path / model_key / model_key / 'Result'
        if not model_path.exists():
            print(f"Warning: {model_path} does not exist")
            continue
        
        results[model_key] = {}
        for robot_key, robot_display in robots.items():
            robot_dir = model_path / robot_key
            if not robot_dir.exists():
                continue
            
            txt_files = list(robot_dir.glob('*.txt'))
            if not txt_files:
                continue
            
            # 最初のテキストファイルを読む
            with open(txt_files[0], 'r') as f:
                content = f.read()
                mean_reward = extract_mean_reward(content)
                if mean_reward is not None:
                    results[model_key][robot_key] = mean_reward
                    print(f"{model_key}/{robot_key}: {mean_reward}")
    
    # 表を表示
    print("\n" + "="*80)
    print("Evaluation Results Table")
    print("="*80)
    print(f"{'Model (Without)':<20}", end="")
    for robot_display in robots.values():
        print(f"{robot_display:>15}", end="")
    print()
    
    print("-"*80)
    for model_key, model_display in models.items():
        if model_key not in results:
            continue
        print(f"{model_display:<20}", end="")
        for robot_key, robot_display in robots.items():
            if robot_key in results[model_key]:
                print(f"{results[model_key][robot_key]:>15.4f}", end="")
            else:
                print(f"{'N/A':>15}", end="")
        print()
    
    # CSVとして保存
    output_file = Path('evaluation_results_table.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # ヘッダー
        header = ['Model (Without)'] + list(robots.values())
        writer.writerow(header)
        # データ
        for model_key, model_display in models.items():
            if model_key not in results:
                continue
            row = [model_display]
            for robot_key in robots.keys():
                if robot_key in results[model_key]:
                    row.append(results[model_key][robot_key])
                else:
                    row.append('N/A')
            writer.writerow(row)
    
    print(f"\nTable saved to: {output_file}")


if __name__ == "__main__":
    main()

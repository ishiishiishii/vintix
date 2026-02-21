#!/usr/bin/env python3
"""
Generate the Episode Cumulative Reward vs Cumulative Steps plot 
for all trajectory directories.
"""

import subprocess
from pathlib import Path

# Base data directory
data_dir = Path("/home/kawa37/genesis_project/vintix_go2/data")

# Define the directories and their titles
directories_config = [
    {
        "dir": data_dir / "go2_trajectories" / "data_1M",
        "title": "go2_trajectories",
        "output_file": "go2_trajectories_cumulative_reward.pdf"
    },
    {
        "dir": data_dir / "minicheetah_trajectories",
        "title": "minicheetah_trajectories",
        "output_file": "minicheetah_trajectories_cumulative_reward.pdf"
    },
    {
        "dir": data_dir / "go1_trajectories",
        "title": "go1_trajectories",
        "output_file": "go1_trajectories_cumulative_reward.pdf"
    },
    {
        "dir": data_dir / "a1_trajectories",
        "title": "a1_trajectories",
        "output_file": "a1_trajectories_cumulative_reward.pdf"
    }
]

script_path = Path(__file__).parent / "generate_cumulative_reward_plot.py"

for config in directories_config:
    if not config["dir"].exists():
        print(f"⚠ Warning: Directory not found: {config['dir']}")
        continue
    
    output_png = config["dir"] / config["output_file"]
    
    print(f"\nProcessing: {config['dir']}")
    print(f"  Output: {output_png}")
    print(f"  Title: {config['title']}")
    
    # Run the generation script
    cmd = [
        "python3",
        str(script_path),
        str(config["dir"]),
        "--title", config["title"],
        "--output", str(output_png)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error: {e}")
        print(f"  {e.stderr}")

print("\n✅ All plots generated!")

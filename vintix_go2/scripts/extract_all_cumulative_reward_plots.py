#!/usr/bin/env python3
"""
Extract the top-left subplot (Episode Cumulative Reward vs Cumulative Steps) 
from all multi-trajectory analysis PNG files in the data directory.
"""

import subprocess
from pathlib import Path

# Base data directory
data_dir = Path("/home/kawa37/genesis_project/vintix_go2/data")

# Define the directories and their corresponding PNG files and titles
directories_config = [
    {
        "dir": data_dir / "go2_trajectories" / "data_1M",
        "png_file": "visualization_step_based.png",
        "title": "go2_trajectories",
        "output_file": "go2_trajectories_cumulative_reward.png"
    },
    {
        "dir": data_dir / "minicheetah_trajectories",
        "png_file": "minicheetah_trajectories_step_based_analysis.png",
        "title": "minicheetah_trajectories",
        "output_file": "minicheetah_trajectories_cumulative_reward.png"
    },
    {
        "dir": data_dir / "go1_trajectories",
        "png_file": "go1_trajectories_step_based_analysis.png",
        "title": "go1_trajectories",
        "output_file": "go1_trajectories_cumulative_reward.png"
    },
    {
        "dir": data_dir / "a1_trajectories",
        "png_file": "a1_step_based_analysis.png",
        "title": "a1_trajectories",
        "output_file": "a1_trajectories_cumulative_reward.png"
    }
]

script_path = Path(__file__).parent / "extract_cumulative_reward_plot.py"

for config in directories_config:
    input_png = config["dir"] / config["png_file"]
    output_png = config["dir"] / config["output_file"]
    
    if not input_png.exists():
        print(f"⚠ Warning: Input file not found: {input_png}")
        continue
    
    print(f"\nProcessing: {input_png}")
    print(f"  Output: {output_png}")
    print(f"  Title: {config['title']}")
    
    # Run the extraction script
    cmd = [
        "python3",
        str(script_path),
        "--input", str(input_png),
        "--output", str(output_png),
        "--title", config["title"]
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error: {e}")
        print(f"  {e.stderr}")

print("\n✅ All extractions completed!")

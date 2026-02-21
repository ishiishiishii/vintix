#!/usr/bin/env python3
"""
Extract the top-left subplot (Episode Cumulative Reward vs Cumulative Steps) 
from multi-trajectory analysis PNG files and save as separate images.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as mpatches


def extract_top_left_subplot(input_png_path, output_png_path, title):
    """
    Extract the top-left subplot from a 2x2 grid image and save it as a new image.
    
    Args:
        input_png_path: Path to the input PNG file with 2x2 subplots
        output_png_path: Path to save the extracted subplot
        title: Title for the new graph
    """
    # Load the image
    img = Image.open(input_png_path)
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    # For a 2x2 grid, the top-left subplot is approximately:
    # - Left half of width
    # - Top half of height (but need to account for title space)
    # Typically, the title takes about 5-10% of the height
    
    # Estimate title height (usually around 5-8% of total height)
    title_height_ratio = 0.08
    title_height = int(height * title_height_ratio)
    
    # Calculate subplot boundaries
    # Top-left subplot: left half, top half (excluding title)
    subplot_left = 0
    subplot_right = width // 2
    subplot_top = title_height
    subplot_bottom = height // 2 + title_height // 2
    
    # Extract the subplot region
    subplot_img = img_array[subplot_top:subplot_bottom, subplot_left:subplot_right]
    
    # Create a new figure with the extracted subplot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the extracted image
    ax.imshow(subplot_img, aspect='auto')
    ax.axis('off')
    
    # Add title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Remove margins
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    plt.savefig(output_png_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"✓ Extracted subplot saved: {output_png_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract top-left subplot from multi-trajectory analysis PNG files"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input PNG file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PNG file path"
    )
    parser.add_argument(
        "--title",
        type=str,
        required=True,
        help="Title for the extracted graph"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    extract_top_left_subplot(input_path, output_path, args.title)


if __name__ == "__main__":
    main()

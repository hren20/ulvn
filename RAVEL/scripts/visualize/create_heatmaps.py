#!/usr/bin/env python3
"""
scripts/visualize_npy.py
-------------------------

Generates a minimal heatmap visualization for a *single* .npy file and saves it
as an .svg file in the same directory.

This version creates a "minimal" plot:
- No title
- No axis labels (e.g., "X-axis Coordinate")
- No colorbar
- Axis tick labels (the numbers) are shown only every 10 steps.
- Figure size is set to 8x8 inches.
- Axis tick label font size is increased.

Usage:
  python scripts/visualize/create_heatmaps.py --input /path/to/your/data.npy

The script will create '/path/to/your/data.svg'.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the script.
    
    Returns:
        argparse.Namespace: The populated namespace with arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generates a minimal heatmap for a single .npy file."
    )
    
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to the input .npy file to visualize."
    )
    
    args = parser.parse_args()
    
    # Validate that the input file exists
    if not args.input.is_file():
        print(f"[Error] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
        
    return args

def create_heatmap(data: np.ndarray, output_svg_path: Path, filename_stem: str):
    """
    Generates a minimal heatmap and saves it as an SVG file.
    
    Args:
        data (np.ndarray): The 2D numpy array to visualize.
        output_svg_path (Path): The Path object where the SVG will be saved.
        filename_stem (str): The base name of the file (used for logging).
    """
    
    # Check if the loaded data is a 2D array, which is required for a heatmap.
    if data.ndim != 2:
        print(f"    [Warning] Skipping {filename_stem}.npy: Data is not 2D (shape: {data.shape}).")
        return

    print(f"    Generating heatmap for {filename_stem}...")
    
    # --- MODIFICATION 1: Set figure size to 8x8 ---
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # --- MODIFIED HEATMAP CALL ---
    sns.heatmap(
        data, 
        ax=ax, 
        cmap='plasma', 
        annot=False,
        
        # 1. Remove the colorbar
        cbar=False,
        
        # 2. Set tick label (number) spacing to every 10th label
        xticklabels=30, 
        yticklabels=30  
    )
    # --- END HEATMAP MODIFICATION ---
    
    # Explicitly remove titles and axis labels (e.g., 'X-axis', 'Y-axis')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Optional: Ensure x-axis tick labels are horizontal
    plt.setp(ax.get_xticklabels(), rotation=0)

    # --- MODIFICATION 2: Increase tick label font size ---
    # Sets the font size for the '0', '10', '20' labels
    ax.tick_params(axis='both', which='major', labelsize=24) 
    # --- END MODIFICATION ---

    # Save the figure to the specified path
    try:
        # We use bbox_inches='tight' to prevent labels from being cut off
        fig.savefig(output_svg_path, format='svg', bbox_inches='tight')
        print(f"    Successfully saved: {output_svg_path.name}")
    except Exception as e:
        print(f"    [Error] Failed to save {output_svg_path.name}: {e}")
    
    # Close the plot figure to free up memory
    plt.close(fig)

def main():
    """
    Main function to load a single .npy file and process it.
    """
    # 1. Get the single file path from command-line arguments
    args = parse_args()
    file_path = args.input
    
    print(f"Starting to process file: {file_path.resolve()}")
    
    try:
        # 2. Load the numpy array from the file
        data_array = np.load(file_path, allow_pickle=False)
        
        # 3. Define the output path
        # .with_suffix('.svg') automatically replaces .npy with .svg
        output_svg = file_path.with_suffix('.svg')
        
        # 4. Get the file stem (filename without extension) for logging
        file_stem = file_path.stem
        
        # 5. Call the plotting function
        create_heatmap(data_array, output_svg, file_stem)
        
    except Exception as e:
        # Catch any errors during file loading or processing
        print(f"    [Error] Could not process {file_path.name}: {e}")

    print("\n--- File processed. ---")

if __name__ == "__main__":
    # This ensures the main() function runs only when the script is executed directly
    main()
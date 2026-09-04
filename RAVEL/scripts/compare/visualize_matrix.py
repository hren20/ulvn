#!/usr/bin/env python3
"""
scripts/visualize_matrix.py
---------------------

Loads a pre-computed adjacency matrix from a .npy file and generates
a heatmap visualization.

This script only performs visualization and does not compute any matrices.
It uses the same plotting style as 'compute_matrices_flexible.py'.

**Version 2**: Artificially sets the diagonal to the matrix's maximum
value for a more intuitive visualization.

Usage:
  # Visualize a similarity matrix (linear scale)
  python scripts/visualize_matrix.py \
      --matrix_path tmp/sim_matrix_full.npy \
      --out_plot tmp/sim_matrix_visualization.png \
      --title "Feature Similarity (Cosine)"

  # Visualize a LightGlue matrix (log scale)
  python scripts/visualize_matrix.py \
      --matrix_path tmp/lg_matrix_full.npy \
      --out_plot tmp/lg_matrix_visualization.png \
      --title "LightGlue Matches (Full N x N)" \
      --log_scale
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Import visualization library
try:
    import matplotlib.pyplot as plt
except ImportError:
    print(
        "ERROR: matplotlib is required for visualization. "
        "Install with `pip install matplotlib`.",
        file=sys.stderr
    )
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the visualization script.

    Returns:
        argparse.Namespace: The populated namespace with arguments.
    """
    parser = argparse.ArgumentParser(
        description="Visualizes a pre-computed N x N matrix from a .npy file."
    )

    # --- Path Configurations ---
    parser.add_argument(
        "--matrix_path",
        type=str,
        required=True,
        help="Path to the input .npy matrix file to visualize."
    )
    parser.add_argument(
        "--out_plot",
        type=str,
        required=True,
        help="Output path to save the generated heatmap plot (.png)."
    )
    
    # --- Plotting Configurations ---
    parser.add_argument(
        "--title",
        type=str,
        default="Matrix Visualization",
        help="Title for the plot."
    )
    parser.add_argument(
        "--log_scale",
        action="store_true",
        help="Use a log scale for the colorbar (recommended for match counts)."
    )

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)

    return args


def visualize_matrix(
    matrix: np.ndarray,
    title: str,
    out_path: str,
    is_log_scale: bool = False
):
    """Saves a heatmap visualization of a matrix.
    
    (This function is identical to the one in compute_matrices_flexible.py,
     with the addition of filling the diagonal.)
    """
    print(f"Visualizing '{title}'...")
    plt.figure(figsize=(10, 8))
    
    # --- MODIFICATION: Set diagonal to max for visualization ---
    # Create a copy to avoid modifying the original array if it's passed around
    data = matrix.copy()
    if data.size > 0:
        # Find the global maximum value in the matrix
        max_val = np.max(data)
        # Fill the diagonal with this maximum value
        np.fill_diagonal(data, max_val)
    # --- End Modification ---

    cmap = 'viridis'
    label = "Cosine Similarity"
    if is_log_scale:
        data = np.log10(data + 1)  # Add 1 to avoid log(0)
        cmap = 'hot'
        label = "Match Count (log10 scale)"
    
    plt.imshow(data, cmap=cmap, interpolation='nearest', aspect='auto')
    plt.colorbar(label=label)
    plt.title(title)
    plt.xlabel("Image Index")
    plt.ylabel("Image Index")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Saved plot to {out_path}")


def main():
    """Main execution function."""
    args = parse_args()

    # --- 1. Load Data ---
    print(f"[1/2] Loading matrix from: {args.matrix_path}")
    try:
        matrix_to_visualize = np.load(args.matrix_path)
    except FileNotFoundError:
        print(f"Error: Matrix file not found at {args.matrix_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading .npy file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"   Loaded matrix with shape: {matrix_to_visualize.shape}")

    # --- 2. Generate Visualization ---
    print("\n[2/2] Generating Visualization...")
    visualize_matrix(
        matrix_to_visualize,
        args.title,
        args.out_plot,
        args.log_scale
    )
        
    print("\n[OK] Visualization complete.")


if __name__ == "__main__":
    main()

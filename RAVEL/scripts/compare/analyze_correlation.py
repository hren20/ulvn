#!/usr/bin/env python3
"""
scripts/compare/analyze_correlation.py
---------------------

Analyzes the correlation between the global feature similarity matrix
and the LightGlue match count matrix.

This script:
1. Loads the two (N x N) .npy matrices (similarity and LightGlue).
2. Extracts the upper-triangular elements from both (N*(N-1)/2 pairs).
3. Computes the Pearson correlation coefficient between these two vectors.
4. Generates and saves a scatter plot to visualize the relationship.

Usage:
    python scripts/compare/analyze_correlation.py \
        --sim_matrix_path tmp/sim_matrix_full.npy \
        --lg_matrix_path tmp/lg_matrix_full.npy \
        --out_plot_path tmp/correlation_plot.svg
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Import for statistics
try:
    from scipy.stats import pearsonr
except ImportError:
    print(
        "ERROR: scipy is required for correlation analysis. "
        "Install with `pip install scipy`.",
        file=sys.stderr
    )
    sys.exit(1)

# Import for visualization
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
    """Parses command-line arguments for the correlation analysis script.

    Returns:
        argparse.Namespace: The populated namespace with arguments.
    """
    parser = argparse.ArgumentParser(
        description="Analyzes correlation between similarity and match matrices."
    )

    # --- Path Configurations ---
    parser.add_argument(
        "--sim_matrix_path",
        type=str,
        default="tmp/sim_matrix_full.npy",
        help="Path to the (N x N) feature similarity .npy file."
    )
    parser.add_argument(
        "--lg_matrix_path",
        type=str,
        default="tmp/lg_matrix_full.npy",
        help="Path to the (N x N) LightGlue match count .npy file."
    )
    parser.add_argument(
        "--out_plot_path",
        type=str,
        default="tmp/correlation_plot.svg",  # Changed default to .svg
        help="Output path to save the correlation scatter plot."
    )
    
    # --- Plotting Configuration ---
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50000,
        help="Number of points to sample for the scatter plot. "
             "Set to 0 to plot all points (can be very slow)."
    )

    args = parser.parse_args()

    # Ensure output directories exist
    Path(args.out_plot_path).parent.mkdir(parents=True, exist_ok=True)

    return args


def main():
    """Main execution function."""
    args = parse_args()

    # --- 1. Load Matrices ---
    print("1. Loading matrices...")
    try:
        sim_matrix = np.load(args.sim_matrix_path)
    except FileNotFoundError:
        print(f"Error: Similarity matrix not found at {args.sim_matrix_path}", file=sys.stderr)
        sys.exit(1)

    try:
        lg_matrix = np.load(args.lg_matrix_path)
    except FileNotFoundError:
        print(f"Error: LightGlue matrix not found at {args.lg_matrix_path}", file=sys.stderr)
        sys.exit(1)

    if sim_matrix.shape != lg_matrix.shape:
        print(
            f"Error: Matrix shape mismatch! "
            f"Sim: {sim_matrix.shape}, LG: {lg_matrix.shape}",
            file=sys.stderr
        )
        sys.exit(1)
        
    if sim_matrix.ndim != 2 or sim_matrix.shape[0] != sim_matrix.shape[1]:
        print(f"Error: Matrices must be square (N x N). Found {sim_matrix.shape}.", file=sys.stderr)
        sys.exit(1)

    N = sim_matrix.shape[0]
    print(f"  Loaded {N}x{N} matrices.")

    # --- 2. Extract Upper Triangle ---
    print("2. Extracting upper-triangle pairs without the diagonal...")
    
    # Get indices for the upper triangle (k=1 excludes the main diagonal)
    indices = np.triu_indices(N, k=1)
    
    vec_sim = sim_matrix[indices]
    vec_lg = lg_matrix[indices]
    
    total_pairs = len(vec_sim)
    if total_pairs == 0:
        print("Error: No pairs found. Need at least 2 images (N >= 2).", file=sys.stderr)
        sys.exit(1)
        
    print(f"  Extracted {total_pairs} unique pairs.")

    # --- 3. Compute Correlation ---
    print("3. Computing Pearson correlation...")
    
    # Check for zero variance, which would crash pearsonr
    if np.std(vec_sim) == 0 or np.std(vec_lg) == 0:
        print("Error: One of the vectors has zero variance (all values are the same).", file=sys.stderr)
        print("Cannot compute correlation.", file=sys.stderr)
        # Still proceed to plot, it might be informative
        correlation, p_value = 0.0, 1.0
    else:
        # Calculate Pearson r and p-value
        correlation, p_value = pearsonr(vec_sim, vec_lg)

    print("\n--- Correlation Results ---")
    print(f"Total pairs analyzed: {total_pairs}")
    print(f"Pearson Correlation (r): {correlation:.6f}")
    print(f"P-value: {p_value:.2e}")
    if p_value < 0.001:
        print("  -> (Result is statistically significant)")
    else:
        print("  -> (Result may not be statistically significant)")
    print("---------------------------\n")

    # --- 4. Generate Scatter Plot ---
    print("4. Generating scatter plot...")
    
    plot_vec_sim = vec_sim
    plot_vec_lg = vec_lg
    
    # Sample if the dataset is too large to plot
    if args.sample_size > 0 and total_pairs > args.sample_size:
        print(f"  Sampling {args.sample_size} points for plotting...")
        sample_indices = np.random.choice(
            total_pairs, size=args.sample_size, replace=False
        )
        plot_vec_sim = vec_sim[sample_indices]
        plot_vec_lg = vec_lg[sample_indices]

    # --- Plotting Modifications ---
    plt.figure(figsize=(8, 8)) # Set figure size to 8x8 inches
    
    plt.scatter(
        plot_vec_sim, 
        plot_vec_lg, 
        alpha=0.1,  # Use transparency for dense plots
        s=5         # Use small markers
    )
    
    # plt.title(...) # Removed plot title
    plt.xlabel("") # Removed X-axis label
    plt.ylabel("") # Removed Y-axis label
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Set tick label font size
    plt.tick_params(axis='both', which='major', labelsize=28)
    plt.tick_params(axis='both', which='minor', labelsize=28) # For minor ticks too if they exist
    
    # Add the correlation value as text on the plot
    plt.text(
        0.05, 0.95,
        f"Pearson r = {correlation:.4f}",
        transform=plt.gca().transAxes,
        fontsize=28, # Set font size to 14
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
    )
    
    # Set y-axis to log scale if the max count is high
    if np.max(plot_vec_lg) > 100:
        plt.yscale('log')
        # plt.ylabel("LightGlue Match Count (Log Scale)", fontsize=12) # Removed Y-axis label even for log scale
        print("  Using log scale for y-axis (match count).")

    # Save as SVG
    plt.savefig(args.out_plot_path, dpi=300, bbox_inches='tight', format='svg') 
    plt.close()
    
    print(f"  -> Saved plot to {args.out_plot_path}")
    print("\n[OK] Analysis complete.")


if __name__ == "__main__":
    main()

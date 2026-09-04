#!/usr/bin/env python3
"""
scripts/compute_full_matrices.py
---------------------

Computes and visualizes the FULL adjacency matrices for feature similarity
and LightGlue match counts.

This script:
1. Loads features and image paths.
2. Initializes LightGlue.
3. Computes the full (N x N) feature similarity matrix (Cosine Similarity).
4. Computes the FULL (N x N) LightGlue match count matrix by:
   - Iterating through every possible pair of images (i, j) where j > i.
   - Running LightGlue on all N*(N-1)/2 pairs.
   - WARNING: This is extremely computationally expensive.
5. Saves both matrices as .npy files and visualizes them as heatmaps.

Usage:
  python scripts/compute_full_matrices.py --features_path tmp/landmarks_features.npy \
                                          --paths_txt tmp/landmarks_paths.txt
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from tqdm import tqdm

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

# Import LightGlue components
try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image, rbd
except ImportError:
    print(
        "ERROR: lightglue/superpoint not found. "
        "Install with `pip install lightglue`.",
        file=sys.stderr
    )
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the matrix computation script.

    Returns:
        argparse.Namespace: The populated namespace with arguments.
    """
    parser = argparse.ArgumentParser(
        description="Computes and visualizes FULL similarity and match matrices."
    )

    # --- Path Configurations ---
    parser.add_argument(
        "--features_path",
        type=str,
        default="tmp/landmarks_features.npy",
        help="Path to the .npy file containing global image features."
    )
    parser.add_argument(
        "--paths_txt",
        type=str,
        default="tmp/landmarks_paths.txt",
        help="Path to the .txt file listing image paths."
    )
    
    # --- Output Configurations ---
    parser.add_argument(
        "--out_sim_matrix",
        type=str,
        default="tmp/sim_matrix_full.npy",
        help="Output path to save the feature similarity matrix."
    )
    parser.add_argument(
        "--out_lg_matrix",
        type=str,
        default="tmp/lg_matrix_full.npy",
        help="Output path to save the LightGlue match count matrix."
    )
    parser.add_argument(
        "--out_sim_plot",
        type=str,
        default="tmp/sim_matrix_full.png",
        help="Output path to save the feature similarity heatmap."
    )
    parser.add_argument(
        "--out_lg_plot",
        type=str,
        default="tmp/lg_matrix_full.png",
        help="Output path to save the LightGlue match heatmap."
    )

    args = parser.parse_args()

    # Ensure output directories exist
    for out_path in [args.out_sim_matrix, args.out_lg_matrix,
                     args.out_sim_plot, args.out_lg_plot]:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    return args


def find_image_path(idx: int, img_paths_list: List[str]) -> Optional[Path]:
    """Finds the existing path for an image index."""
    if idx >= len(img_paths_list):
        return None
    base_path = Path(img_paths_list[idx])
    if base_path.exists():
        return base_path
    for ext in [".png", ".jpg", ".jpeg"]:
        p = base_path.with_suffix(ext)
        if p.exists():
            return p
    return None


def create_match_counter(
    img_paths_list: List[str]
) -> Callable[[int, int], int]:
    """Initializes LightGlue and returns a function to count matches."""
    print("Initializing LightGlue components...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        extractor = SuperPoint(
            nms_radius=4,
            detection_threshold=0.005,
            max_num_keypoints=1024,
            remove_borders=4
        ).eval().to(device)
        matcher = LightGlue(features="superpoint").eval().to(device)
    except Exception as e:
        print(f"[ERROR] Failed to initialize LightGlue models: {e}", file=sys.stderr)
        sys.exit(1)

    # This is the function that will be returned
    @torch.no_grad()
    def get_match_count(idx0: int, idx1: int) -> int:
        """Calculates the number of LightGlue matches between two images."""
        path0 = find_image_path(idx0, img_paths_list)
        path1 = find_image_path(idx1, img_paths_list)

        if not path0 or not path1:
            return 0

        try:
            image0 = load_image(path0)
            image1 = load_image(path1)
            feats0 = extractor.extract(image0.to(device))
            feats1 = extractor.extract(image1.to(device))
            matches01 = matcher({"image0": feats0, "image1": feats1})
            matches = rbd(matches01)["matches"]
            return len(matches)
        except Exception as e:
            # Treat failed feature matches as non-edges and keep the batch running.
            print(f"[WARN] LightGlue failed for pair ({idx0}, {idx1}). Error: {e}", file=sys.stderr)
            return 0

    return get_match_count


def visualize_matrix(
    matrix: np.ndarray,
    title: str,
    out_path: str,
    is_log_scale: bool = False
):
    """Saves a heatmap visualization of a matrix."""
    print(f"Visualizing '{title}'...")
    plt.figure(figsize=(10, 8))
    
    data = matrix
    cmap = 'viridis'
    label = "Cosine Similarity"
    if is_log_scale:
        data = np.log10(matrix + 1)  # Add 1 to avoid log(0)
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
    print("[1/4] Loading features and paths...")
    try:
        features = np.load(args.features_path).astype('float32')
        # Ensure features are L2 normalized for cosine similarity
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
    except FileNotFoundError:
        print(f"Error: Feature file not found at {args.features_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.paths_txt, "r", encoding="utf-8") as f:
            img_paths = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Error: Paths file not found at {args.paths_txt}", file=sys.stderr)
        sys.exit(1)

    N = len(features)
    if N != len(img_paths):
        print(f"Error: Mismatch! Features={N}, Paths={len(img_paths)}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {N} images/features.")

    # --- 2. Compute Feature Similarity Matrix (Fast) ---
    print("\n[2/4] Computing Feature Similarity Matrix (N x N)...")
    sim_matrix = np.dot(features, features.T)
    sim_matrix = np.clip(sim_matrix, 0, 1)
    
    np.save(args.out_sim_matrix, sim_matrix)
    print(f"   -> Saved similarity matrix to {args.out_sim_matrix}")

    # --- 3. Compute FULL LightGlue Match Matrix (Very Slow) ---
    total_pairs = N * (N - 1) // 2
    print(f"\n[3/4] Computing Full LightGlue Match Matrix (N*(N-1)/2 pairs)...")
    print("   [WARN] This is a very slow operation.")
    print(f"   Total pairs to compute: {total_pairs}")
    
    # Initialize LightGlue
    get_match_count = create_match_counter(img_paths)
    
    # Initialize an empty N x N matrix
    lg_matrix = np.zeros((N, N), dtype=np.int32)
    
    start_time = time.time()
    
    # Iterate over the upper triangle of the matrix (j > i)
    # We use tqdm on the outer loop for progress estimation
    for i in tqdm(range(N), desc="Matching all pairs (Full)"):
        for j in range(i + 1, N):
            # This computes the upper triangle (j > i)
            count = get_match_count(i, j)
            
            # Assign symmetrically
            lg_matrix[i, j] = count
            lg_matrix[j, i] = count

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_pair = total_time / total_pairs if total_pairs > 0 else 0
    
    print(f"   ...Full matching complete in {total_time:.2f} seconds.")
    print(f"   Average time per pair: {avg_time_per_pair * 1000:.2f} ms")
    np.save(args.out_lg_matrix, lg_matrix)
    print(f"   -> Saved LightGlue matrix to {args.out_lg_matrix}")

    # --- 4. Generate Visualizations ---
    print("\n[4/4] Generating Visualizations...")
    
    if args.out_sim_plot:
        visualize_matrix(
            sim_matrix,
            "Feature Similarity (Cosine)",
            args.out_sim_plot,
            is_log_scale=False
        )
    
    if args.out_lg_plot:
        visualize_matrix(
            lg_matrix,
            "LightGlue Matches (Full N x N)", # Updated title
            args.out_lg_plot,
            is_log_scale=True
        )
        
    print("\n[OK] All tasks complete.")


if __name__ == "__main__":
    main()

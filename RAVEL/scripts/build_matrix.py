#!/usr/bin/env python3
"""
build_matrix.py
---------------

Builds the final adjacency matrix and visualizes the topological map graph.

This script acts as the main entry point and performs three major steps:
1.  **Calls `analyze_parameters.py`**: It runs the first-pass analysis
    to dynamically determine the `threshold_tau` (match cutoff) and
    `vpr_dist_thresh` (VPR search distance).
2.  **Builds the Matrix**: It iterates through every image, finds its
    candidate neighbors using FAISS `range_search` (based on VPR distance),
    and runs LightGlue on those pairs. An edge is created if
    the match count > `threshold_tau`.
    - By default, the matrix stores the *actual match count* for valid edges.
    - Use `--binary_matrix` to store 1s instead.
3.  **Builds & Visualizes Graph**: It loads the matrix into NetworkX,
    creates a force-directed layout, and saves the final map visualization.

The final output is saved as:
  - {result_dir}/adjacency_matrix.npy
  - {result_dir}/topological_map_visualization.png

Usage:
  # Run the full pipeline (saves match counts in matrix)
  python build_matrix.py
  
  # Run and save a binary (1/0) matrix
  python build_matrix.py --binary_matrix

  # Or override paths (the analysis script will inherit these)
  python scripts/build_matrix.py --features_path tmp/run/data_features.npy \
                                 --index_path tmp/run/index.faiss \
                                 --paths_txt tmp/run/data_paths.txt \
                                 --result_dir results/run
"""

import argparse
import os
import sys
from pathlib import Path
import concurrent.futures

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

# Import the analysis script as a module
try:
    import analyze_parameters as analyzer
except ImportError:
    print(
        "[ERROR] `analyze_parameters.py` not found.",
        "Please ensure it is in the same directory or Python path.",
        file=sys.stderr
    )
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the entire build process.
    
    Returns:
        argparse.Namespace: The populated namespace with all arguments.
    """
    parser = argparse.ArgumentParser(
        description="Runs the full topological map build process."
    )
    
    # --- Arguments for Step 1 (Analysis) ---
    # These arguments are *shared* and will be passed to the analyzer
    analysis_group = parser.add_argument_group(
        "Step 1: Analysis Parameters (Shared)",
        "Arguments passed to analyze_parameters.py"
    )
    
    analysis_group.add_argument(
        "--features_path", 
        type=Path,  # <-- Use Path
        default="tmp/landmarks_features.npy",
        help="Path to the .npy file containing global image features."
    )
    analysis_group.add_argument(
        "--index_path", 
        type=Path,  # <-- Use Path
        default="tmp/index.faiss",
        help="Path to the FAISS index file."
    )
    analysis_group.add_argument(
        "--paths_txt", 
        type=Path,  # <-- Use Path
        default="tmp/landmarks_paths.txt",
        help="Path to the .txt file listing image paths."
    )
    analysis_group.add_argument(
        "--base_query_idx", 
        type=int, 
        default=0,
        help="The index of the image to use as the base query for the first pass."
    )
    analysis_group.add_argument(
        "--num_clusters", 
        type=int, 
        default=2,
        help="Number of clusters (k) for K-Means (e.g., 'good' and 'bad' matches)."
    )
    analysis_group.add_argument(
        "--vpr_dist_buffer",
        type=float,
        default=0.0,
        help="A safety buffer added to the max 'good' VPR distance to set the threshold."
    )
    analysis_group.add_argument(
        "--out_plot_path",
        type=Path,  # <-- Use Path
        default="tmp/vis/param_analysis_plot.png",
        help="Path to save the cluster analysis visualization."
    )
    analysis_group.add_argument(
        "--disable_ransac",  # <-- Add RANSAC flag
        action="store_true",
        help="If set, skips geometric verification (passed to analyzer)."
    )

    # --- Arguments for Step 2/3 (Build & Visualize) ---
    build_group = parser.add_argument_group(
        "Step 2/3: Build & Visualize Parameters",
        "Arguments for the final matrix and graph"
    )
    build_group.add_argument(
        "--result_dir", 
        type=Path,  # <-- Use Path
        default="results",
        help="Directory to save the final .npy matrix and .png visualization."
    )
    parser.add_argument(
        "--lightglue_weights",
        type=str,
        default="indoor",
        choices=["indoor", "outdoor"],
        help="LightGlue model weights to use (indoor or outdoor)."
    )
    build_group.add_argument(
        "--binary_matrix",
        action="store_true",
        help="If set, saves the adjacency matrix as binary (1/0) instead of "
             "storing the actual LightGlue match counts."
    )
    
    build_group.add_argument(
        "--num_threads",
        type=int,
        default=10,
        help="Number of threads to use for parallel LightGlue checks (Option 1)."
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.result_dir.mkdir(parents=True, exist_ok=True)
    
    return args


def main():
    """Main execution function."""
    args = parse_args()

    # =======================================================
    # 1. RUN PARAMETER ANALYSIS (Calls the first script)
    # =======================================================
    print("--- Step 1: Running Parameter Analysis ---")
    
    try:
        # Pass the shared 'args' namespace to the analysis function
        # The analyzer now returns the validated paths list
        (
            threshold_tau,      # LightGlue match count threshold
            vpr_dist_thresh,    # FAISS distance threshold
            features, 
            index, 
            get_match_count,    # This is now a general-purpose (i, j) function
            N,
            validated_img_paths # Get the pre-validated paths
        ) = analyzer.run_parameter_analysis(args)
    except Exception as e:
        print(f"\n[ERROR] An error occurred during the analysis step: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"\nAnalysis complete.")
    print(f" > Using LightGlue Threshold (tau): {threshold_tau:.2f} matches")
    print(f" > Using VPR Distance Threshold: {vpr_dist_thresh:.4f} (L2 dist)")

    # =======================================================
    # 2. BUILD ADJACENCY MATRIX (Second Pass)
    # =======================================================
    print("\n--- Step 2: Building Adjacency Matrix (using range_search) ---")
    
    # 0 = unknown, -1 = bad match (checked). Good matches will be 1 or > tau.
    # Use int16 to allow for -1 and match counts potentially > 127
    adjacency_matrix = np.zeros((N, N), dtype=np.int16) 
    
    total_lightglue_checks = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        
        for i in tqdm(range(N), desc="Building Adjacency Matrix"):
            try:
                lims, D, I = index.range_search(features[i:i+1], vpr_dist_thresh)
                candidate_indices = I[lims[0]:lims[1]]
            except Exception as e:
                print(f"\n[ERROR] FAISS range_search failed for index {i}: {e}", file=sys.stderr)
                continue
            
            future_to_j = {}
            for j in candidate_indices:
                if i == j:
                    continue
                
                if j < i: 
                    continue

                
                future = executor.submit(get_match_count, i, j)
                future_to_j[future] = j

            if not future_to_j:
                continue

            total_lightglue_checks += len(future_to_j)

            for future in concurrent.futures.as_completed(future_to_j):
                j = future_to_j[future]
                try:
                    match_count = future.result()
                    
                    if match_count > threshold_tau:
                        if args.binary_matrix:
                            value_to_store = 1
                        else:
                            value_to_store = int(round(match_count))
                        
                        adjacency_matrix[i, j] = value_to_store
                        adjacency_matrix[j, i] = value_to_store
                    else:
                        adjacency_matrix[i, j] = -1
                        adjacency_matrix[j, i] = -1

                except Exception as e:
                    print(f"\n[ERROR] LightGlue check failed for pair ({i}, {j}): {e}", file=sys.stderr)
                    adjacency_matrix[i, j] = -1
                    adjacency_matrix[j, i] = -1

    print(f"     Performed {total_lightglue_checks} total LightGlue verifications.")
    print("    Cleaning final matrix (replacing -1s with 0s)...")
    
    # Replace all -1s (checked but bad) with 0s (no connection)
    adjacency_matrix[adjacency_matrix == -1] = 0

    # --- Save the matrix ---
    matrix_save_path = args.result_dir / "adjacency_matrix.npy"
    
    # Select the correct dtype for saving to optimize file size
    if args.binary_matrix:
        print("    Saving as binary (uint8) matrix.")
        save_dtype = np.uint8
    else:
        # uint16 supports match counts up to 65,535
        print(f"    Saving matrix with match counts (uint16).")
        save_dtype = np.uint16
        
    np.save(matrix_save_path, adjacency_matrix.astype(save_dtype))
    print(f"[OK] Adjacency Matrix saved to {matrix_save_path}")

    # =======================================================
    # 3. BUILD AND VISUALIZE NETWORKX GRAPH
    # =======================================================
    print("\n--- Step 3: Building and Visualizing NetworkX Graph ---")

    # --- 3a. Generate Node Labels ---
    # We already have the validated paths list from the analyzer
    img_paths = validated_img_paths

    if N != len(img_paths):
        print(f"Error: Matrix size ({N}) does not match path count ({len(img_paths)}).", file=sys.stderr)
        sys.exit(1)

    node_labels = {}
    for i, path in enumerate(img_paths):
        if path is None:
            continue
        # Extract filename (e.g., "scene_000_0_0")
        filename = path.stem
        # Extract last 4 chars (e.g., "_0_0")
        label = filename[-4:]
        # Map original index (0, 1, 2...) to the new label
        node_labels[i] = label

    # --- 3b. Build NetworkX Graph ---
    print("Building NetworkX Graph...")

    # Create graph from the in-memory numpy matrix
    # nx.from_numpy_array automatically treats non-zero entries as edges
    # and stores the value (match count or 1) as the 'weight' attribute.
    G = nx.from_numpy_array(adjacency_matrix)

    # Relabel nodes from integer indices (0...N-1) to our short string labels
    G = nx.relabel_nodes(G, node_labels, copy=False)

    # Remove isolated nodes (nodes with no connections) to simplify the plot
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    print(f"[OK] Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    if isolated_nodes:
        print(f"    Removed {len(isolated_nodes)} isolated nodes.")

    # --- 3c. Visualize 2D Topological Map (Force-Directed) ---
    print("Visualizing 2D Topological Map (Force-Directed Layout)...")

    plt.figure(figsize=(16, 16))

    # Use spring_layout (a force-directed layout) for visualization
    # This layout does not use edge weights by default, just connectivity
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    nx.draw_networkx_nodes(
        G, pos, 
        node_size=800, 
        node_color='skyblue', 
        alpha=0.9, 
        edgecolors='black', 
        linewidths=1
    )
    nx.draw_networkx_edges(
        G, pos, 
        edge_color='gray', 
        width=1.5, 
        alpha=0.6
    )
    nx.draw_networkx_labels(
        G, pos, 
        font_size=10, 
        font_color='black', 
        font_weight='bold'
    )

    plt.title(
        f'Topological Map of Image Connectivity\n(N={G.number_of_nodes()}, Edges={G.number_of_edges()})',
        fontsize=16
    )
    plt.axis('off')

    # --- 3d. Save Visualization ---
    viz_save_path = args.result_dir / "topological_map_visualization.png"
    plt.savefig(viz_save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Topological Map visualization saved to {viz_save_path}")
    print("\nAll processing finished.")


if __name__ == "__main__":
    main()

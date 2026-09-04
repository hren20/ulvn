#!/usr/bin/env python3
"""
scripts/prune_topomap_nx.py
---------------------------

Prunes a weighted topological map by first finding the
Maximum Spanning Forest (MSF) and then adding back any
strong, redundant edges above a specified threshold.

This script implements a two-stage graph filtering:
1.  Loads a pre-computed weighted adjacency matrix.
2.  Builds a full weighted graph (G) using NetworkX.
3.  Computes the Maximum Spanning Forest (MSF) using NetworkX's
    built-in Kruskal's algorithm. This provides the strongest
    possible "backbone" connecting all components. This result
    is stored in a new graph (G_pruned).
4.  Iterates through all edges in the original graph (G) that
    were *not* included in the MSF.
5.  Any of these "redundant" edges with a weight *above* a
    user-defined threshold (e.g., 300) are added back to G_pruned.
6.  The final, filtered graph (MSF + strong redundant edges) is saved
    as a new weighted matrix and a new 2D force-directed visualization.

Usage:
  python scripts/prune_topomap_nx.py \
      --weighted_matrix results/run/adjacency_matrix.npy \
      --paths_txt tmp/run/data_paths.txt \
      --out_matrix_path results/run/pruned_adjacency_matrix.npy \
      --out_viz_path results/run/pruned_topological_map.png \
      --add_back_threshold 300
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the pruning script.
    
    Returns:
        argparse.Namespace: The populated namespace with arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Prunes a topo map by finding the MSF and adding back "
            "strong edges above a threshold."
        )
    )
    
    # --- Input Files ---
    parser.add_argument(
        "--weighted_matrix", 
        type=Path, 
        default="results/weighted_adjacency_matrix.npy",
        help="Path to the input weighted_adjacency_matrix.npy file."
    )
    parser.add_argument(
        "--paths_txt", 
        type=Path, 
        default="tmp/landmarks_paths.txt",
        help="Path to the .txt file listing image paths (for labels)."
    )
    
    # --- Output Files ---
    parser.add_argument(
        "--out_matrix_path", 
        type=Path, 
        default="results/pruned_adjacency_matrix.npy",
        help="Full path to save the final pruned .npy adjacency matrix."
    )
    parser.add_argument(
        "--out_viz_path", 
        type=Path, 
        default="results/pruned_topological_map.png",
        help="Full path to save the final pruned .png graph visualization."
    )
    
    # --- New Algorithm Parameter ---
    parser.add_argument(
        "--add_back_threshold",
        type=int,
        default=300,
        help=(
            "After finding the MSF, add back any pruned edges "
            "with a weight *above* this threshold."
        )
    )
    
    args = parser.parse_args()

    # --- Create parent directories for outputs ---
    if args.out_matrix_path.parent:
        args.out_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.out_viz_path.parent:
        args.out_viz_path.parent.mkdir(parents=True, exist_ok=True)
        
    return args


def main():
    """Main execution function."""
    args = parse_args()

    # =======================================================
    # 1. LOAD DATA AND BUILD INITIAL GRAPH
    # =======================================================
    print("--- Step 1: Loading Data and Building Initial Graph ---")

    try:
        matrix = np.load(args.weighted_matrix)
        N = matrix.shape[0]
    except FileNotFoundError:
        print(
            f"Error: Weighted matrix not found at {args.weighted_matrix}.",
            file=sys.stderr
        )
        sys.exit(1)

    # Build the weighted graph from the matrix
    # G will hold the *full* graph
    G = nx.Graph()
    G.add_nodes_from(range(N))  # Add all nodes, even if isolated
    for i in range(N):
        for j in range(i + 1, N): # Iterate upper triangle
            weight = int(matrix[i, j]) # Ensure it's a standard int
            if weight > 0:
                G.add_edge(i, j, weight=weight)
    
    print(
        f"Loaded initial graph with {G.number_of_nodes()} nodes "
        f"and {G.number_of_edges()} edges."
    )

    # =======================================================
    # 2. COMPUTE MSF + ADD BACK STRONG EDGES
    # =======================================================
    
    # --- 2a. Compute Maximum Spanning Forest (MSF) ---
    print(
        "\n--- Step 2a: Computing Maximum Spanning Forest (MSF) ---"
    )
    
    # This single line replaces the entire slow "Reverse Kruskal" loop.
    # It uses NetworkX's fast, built-in Kruskal or Prim algorithm.
    # G_pruned will *only* contain the edges of the MSF.
    G_pruned = nx.maximum_spanning_tree(G, weight='weight')
    
    print(
        f"   MSF base graph has {G_pruned.number_of_nodes()} nodes "
        f"and {G_pruned.number_of_edges()} edges."
    )

    # --- 2b. Add back strong, redundant edges ---
    print(
        f"--- Step 2b: Adding back strong edges > {args.add_back_threshold} ---"
    )
    
    # Build a fast lookup set of edges already in the MSF
    # We use frozenset to handle undirected edges (u, v) == (v, u)
    msf_edges = {frozenset(edge) for edge in G_pruned.edges()}
    
    num_added_back = 0
    # Iterate through ALL edges in the *original* graph
    for u, v, data in tqdm(G.edges(data=True), desc="Checking strong edges"):
        edge = frozenset({u, v})
        weight = data['weight']
        
        # If this edge is NOT in the MSF
        if edge not in msf_edges:
            # Check if it's above the user's threshold
            if weight > args.add_back_threshold:
                # Add it back to the pruned graph
                G_pruned.add_edge(u, v, weight=weight)
                num_added_back += 1
                
    print(f"[OK] Added back {num_added_back} strong, redundant edges.")
    print(
        f"   Final graph has {G_pruned.number_of_nodes()} nodes "
        f"and {G_pruned.number_of_edges()} edges."
    )


    # =======================================================
    # 3. SAVE PRUNED MATRIX
    # =======================================================
    print("\n--- Step 3: Saving Pruned Adjacency Matrix ---")
    
    # Convert the final graph back to a weighted numpy array
    pruned_matrix = nx.to_numpy_array(
        G_pruned,
        nodelist=sorted(G.nodes()), # Use original G.nodes() for N x N shape
        weight='weight',
        dtype=np.int32
    )
    
    # Use the full path from arguments
    matrix_save_path = args.out_matrix_path
    np.save(matrix_save_path, pruned_matrix)
    print(f"[OK] Pruned weighted matrix saved to {matrix_save_path}")

    # =======================================================
    # 4. PREPARE & VISUALIZE FINAL GRAPH
    # =======================================================
    print("\n--- Step 4: Visualizing Pruned Topological Map ---")
    
    # --- 4a. Generate Node Labels ---
    try:
        with open(args.paths_txt, "r", encoding="utf-8") as f:
            img_paths = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(
            f"Error: Path file not found at {args.paths_txt}. "
            "Cannot create labels.", file=sys.stderr
        )
        sys.exit(1)

    node_labels = {}
    for i, path in enumerate(img_paths):
        # Check against N to avoid index errors if lists don't match
        if i < N: 
            filename = Path(path).stem
            label = filename[-4:] # Use last 4 chars as label
            node_labels[i] = label

    # --- 4b. Prepare Graph for Visualization ---
    # We use a copy for visualization, relabeling it with strings
    G_viz = G_pruned.copy()
    
    # Relabel nodes from integers (0..N-1) to strings ("_0_0", etc.)
    # Use a dict comprehension to only relabel nodes that exist in the map
    G_viz = nx.relabel_nodes(
        G_viz,
        {k: v for k, v in node_labels.items() if k in G_viz}
    )
    
    # Remove any nodes that are isolated
    # (This can happen if they were isolated in the original graph)
    isolated_nodes = list(nx.isolates(G_viz))
    G_viz.remove_nodes_from(isolated_nodes)
    
    print(
        f"   Visualizing graph with {G_viz.number_of_nodes()} nodes "
        f"and {G_viz.number_of_edges()} edges."
    )

    # --- 4c. Draw Graph ---
    print("   Calculating spring layout...")
    plt.figure(figsize=(16, 16))

    # Use spring_layout (force-directed)
    pos = nx.spring_layout(G_viz, k=0.3, iterations=50, seed=42)

    print("   Drawing graph components...")
    nx.draw_networkx_nodes(
        G_viz, pos, 
        node_size=800, 
        node_color='lightgreen', # Changed color to show it's different
        alpha=0.9, 
        edgecolors='black', 
        linewidths=1
    )

    nx.draw_networkx_edges(
        G_viz, pos, 
        edge_color='gray', 
        width=1.5, 
        alpha=0.6
    )

    nx.draw_networkx_labels(
        G_viz, pos, 
        font_size=10, 
        font_color='black', 
        font_weight='bold'
    )

    plt.title(
        f'Filtered Topological Map (MSF + Edges > {args.add_back_threshold})\n'
        f'(Nodes={G_viz.number_of_nodes()}, Edges={G_viz.number_of_edges()})',
        fontsize=16
    )
    plt.axis('off')

    # --- 4d. Save Visualization ---
    # Use the full path from arguments
    viz_save_path = args.out_viz_path
    plt.savefig(viz_save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Pruned topological map visualization saved to {viz_save_path}")
    print("\nAll processing finished.")


if __name__ == "__main__":
    main()

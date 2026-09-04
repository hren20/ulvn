#!/usr/bin/env python3

"""Dynamically determines optimal parameters for topological map building.

This script performs a first-pass analysis on a subset of the data to find
optimal thresholds (`threshold_tau`, `vpr_dist_thresh`) for building a
topological map based on LightGlue feature matching.

Key Operations:
1.  Loads global features, a FAISS index, and image paths.
2.  Initializes LightGlue (SuperPoint + LightGlue matcher).
3.  Runs a "first pass" by matching two query images against all other images:
    - The base query image (default: index 0).
    - The farthest VPR (FAISS) neighbor from the base query.
4.  Combines the match counts from both query sets.
5.  Uses K-Means clustering (e.g., k=3) on the combined match counts to identify
    the top two clusters (Rank 1: highest mean, Rank 2: second-highest mean).
6.  Calculates `threshold_tau` as the midpoint between the minimum count of
    the Rank 1 cluster and the maximum count of the Rank 2 cluster.
7.  Calculates `vpr_dist_thresh` as the maximum VPR distance found within the
    Rank 1 cluster, plus a small buffer.
8.  Provides a general-purpose match counter function (`get_match_count`)
    for use in the main map-building script.

This script is intended to be run standalone for analysis or, more commonly,
imported as a module by `build_matrix.py`.

Usage:
    As a standalone script for analysis and visualization:
    $ python scripts/analyze_parameters.py \
        --features_path tmp/run/data_features.npy \
        --index_path tmp/run/index.faiss \
        --paths_txt tmp/run/data_paths.txt \
        --out_plot_path tmp/run/param_analysis_plot.png \
        --lightglue_weights outdoor

    As an imported module:
    from scripts import analyze_parameters
    
    # (Assuming argparse.Namespace 'args' is populated by the calling script)
    (
        threshold_tau, 
        vpr_dist_thresh, 
        features, 
        index, 
        get_match_count, 
        N, 
        validated_img_paths
    ) = analyze_parameters.run_parameter_analysis(args)

"""

import cv2
import argparse
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import concurrent.futures

import faiss
import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

# Import visualization libraries (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the analysis script.

    Returns:
        argparse.Namespace: The populated namespace with arguments.
    """
    parser = argparse.ArgumentParser(
        description="Analyzes LightGlue matches to find optimal map-building parameters."
    )
    
    # --- Path Configurations ---
    parser.add_argument(
        "--features_path", 
        type=Path, 
        default="tmp/landmarks_features.npy",
        help="Path to the .npy file containing global image features."
    )
    parser.add_argument(
        "--index_path", 
        type=Path, 
        default="tmp/index.faiss",
        help="Path to the FAISS index file (must be IndexFlatL2 or IndexFlatIP)."
    )
    parser.add_argument(
        "--paths_txt", 
        type=Path, 
        default="tmp/landmarks_paths.txt",
        help="Path to the .txt file listing image paths."
    )
    
    # --- Analysis Configurations ---
    parser.add_argument(
        "--base_query_idx", 
        type=int, 
        default=0,
        help="The index of the image to use as the base query for the first pass."
    )
    parser.add_argument(
        "--num_clusters", 
        type=int, 
        default=2,
        help="Number of clusters (k) for K-Means (e.g., 'good' and 'bad' matches)."
    )
    parser.add_argument(
        "--vpr_dist_buffer",
        type=float,
        default=0.0,
        help="A safety buffer added to the max 'good' VPR distance to set the threshold."
    )
    parser.add_argument(
        "--disable_ransac",
        action="store_true",
        help="If set, skips geometric verification and uses raw LightGlue match counts."
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=20,
        help="Number of threads to use for parallel LightGlue checks in the first pass."
    )
    
    # --- [NEW ARGUMENT] ---
    parser.add_argument(
        "--lightglue_weights",
        type=str,
        default="indoor",
        choices=["indoor", "outdoor"],
        help="LightGlue model weights to use (indoor or outdoor)."
    )
    # --- [END NEW ARGUMENT] ---
    
    # --- Output Plot Configuration ---
    parser.add_argument(
        "--out_plot_path",
        type=Path,
        default="tmp/vis/param_analysis_plot.png",
        help="Path to save the cluster analysis visualization."
    )
    
    # ... (arg parsing logic) ...
    if 'RUNNING_AS_MAIN' in globals():
        args = parser.parse_args()
    else:
        # If imported, parse an empty list to use defaults
        args = parser.parse_args([])

    
    # Ensure output directories exist for all output files
    args.features_path.parent.mkdir(parents=True, exist_ok=True)
    # Assuming index_path is defined in the full script
    # args.index_path.parent.mkdir(parents=True, exist_ok=True) 
    args.paths_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_plot_path.parent.mkdir(parents=True, exist_ok=True)
    
    return args


def pre_validate_image_paths(img_paths_str: List[str]) -> List[Optional[Path]]:
    """Checks for the existence of all image paths from the list.
    
    Attempts to find fallbacks for common extensions if the exact path fails.

    Args:
        img_paths_str: A list of image path strings to check.

    Returns:
        A list where each element is either a valid `Path` object
        or `None` if the file was not found.
    """
    print(" ... Validating image paths...")
    validated_paths: List[Optional[Path]] = []
    
    for path_str in tqdm(img_paths_str, desc="Validating Paths"):
        p = Path(path_str)
        if p.exists():
            validated_paths.append(p)
            continue
            
        # Fallback for common extensions
        found = False
        for ext in [".png", ".jpg", ".jpeg"]:
            p_alt = p.with_suffix(ext)
            if p_alt.exists():
                validated_paths.append(p_alt)
                found = True
                break
                
        if not found:
            print(f"[WARN] Path not found: {path_str}", file=sys.stderr)
            validated_paths.append(None)
            
    return validated_paths


def _create_first_pass_counter(
    validated_paths_list: List[Optional[Path]],
    base_query_idx: int,
    disable_ransac: bool,
    device: torch.device,
    extractor: SuperPoint,
    matcher: LightGlue
) -> Callable[[int], int]:
    """Create an optimized match counter for the first pass.

    This function pre-computes the features for the given `base_query_idx`
    to accelerate matching against all other images.

    Args:
        validated_paths_list: List of validated image paths.
        base_query_idx: The index of the *query* image to pre-compute.
        disable_ransac: If True, skips geometric verification.
        device: The torch device (e.g., 'cuda' or 'cpu').
        extractor: The initialized SuperPoint extractor.
        matcher: The initialized LightGlue matcher.

    Returns:
        A callable function `get_first_pass_match_count(idx1)` that
        efficiently computes matches between the pre-computed query
        and a target image `idx1`.
    """
    # --- RANSAC Parameters ---
    RANSAC_THRESH_PIXELS = 3.0 
    MIN_MATCHES_FOR_RANSAC = 8   
    # ---------------------

    # --- Pre-compute features for the base query image ---
    base_query_path = validated_paths_list[base_query_idx]
    if not base_query_path:
        print(f"[ERROR] Base query image (idx {base_query_idx}) not found at {base_query_path}.", file=sys.stderr)
        sys.exit(1)

    try:
        # Note: This print statement will run for each query in the new loop
        print(f" ... Pre-computing features for base query {base_query_idx}...")
        base_image0 = load_image(base_query_path).to(device)
        with torch.no_grad():
            precomputed_feats0 = extractor.extract(base_image0)
    except Exception as e:
        print(f"[ERROR] Failed to extract features for base query: {e}", file=sys.stderr)
        sys.exit(1)
    # -----------------------------------------------------------------

    @torch.no_grad()
    def get_first_pass_match_count(idx1: int) -> int:
        """Calculates matches between pre-computed query and idx1."""
        path1 = validated_paths_list[idx1]
        if not path1: return 0

        try:
            image1 = load_image(path1)
            with torch.no_grad():
                feats1 = extractor.extract(image1.to(device))
                matches01 = matcher({"image0": precomputed_feats0, "image1": feats1})
                matches = rbd(matches01)["matches"] 
                raw_match_count = len(matches)
                
                if disable_ransac:
                    return raw_match_count
                if raw_match_count < MIN_MATCHES_FOR_RANSAC:
                    return 0

                kpts0 = precomputed_feats0['keypoints'][0][matches[:, 0]].cpu().numpy()
                kpts1 = feats1['keypoints'][0][matches[:, 1]].cpu().numpy()

                try:
                    _, mask = cv2.findFundamentalMat(
                        kpts0, kpts1, 
                        method=getattr(cv2, 'USAC_MAGSAC', cv2.FM_RANSAC), 
                        ransacReprojThreshold=RANSAC_THRESH_PIXELS,
                        confidence=0.999
                    )
                    return 0 if mask is None else int(np.sum(mask))
                except cv2.error:
                    return 0
        except Exception:
            return 0
            
    return get_first_pass_match_count


def create_general_match_counter(
    validated_paths_list: List[Optional[Path]],
    disable_ransac: bool,
    device: torch.device,
    extractor: SuperPoint,
    matcher: LightGlue
) -> Callable[[int, int], int]:
    """[Public] Creates the GENERAL-PURPOSE match counter for the second pass.

    This function creates a counter that takes two arbitrary indices (i, j),
    loads both images, and computes the match count on the fly. It is
    less optimized than `_create_first_pass_counter`.

    Args:
        validated_paths_list: List of validated image paths.
        disable_ransac: If True, skips geometric verification.
        device: The torch device (e.g., 'cuda' or 'cpu').
        extractor: The initialized SuperPoint extractor.
        matcher: The initialized LightGlue matcher.

    Returns:
        A callable function `get_general_match_count(idx0, idx1)` that
        computes matches between any two image indices.
    """
    # --- RANSAC Parameters ---
    RANSAC_THRESH_PIXELS = 3.0 
    MIN_MATCHES_FOR_RANSAC = 8   
    # ---------------------

    @torch.no_grad()
    def get_general_match_count(idx0: int, idx1: int) -> int:
        """Calculates matches between idx0 and idx1."""
        path0 = validated_paths_list[idx0]
        path1 = validated_paths_list[idx1]
        
        if not path0 or not path1:
            return 0

        try:
            image0 = load_image(path0)
            image1 = load_image(path1)

            with torch.no_grad():
                feats0 = extractor.extract(image0.to(device))
                feats1 = extractor.extract(image1.to(device))
                matches01 = matcher({"image0": feats0, "image1": feats1})
                matches = rbd(matches01)["matches"] 
                raw_match_count = len(matches)

                if disable_ransac:
                    return raw_match_count
                if raw_match_count < MIN_MATCHES_FOR_RANSAC:
                    return 0
                
                kpts0 = feats0['keypoints'][0][matches[:, 0]].cpu().numpy()
                kpts1 = feats1['keypoints'][0][matches[:, 1]].cpu().numpy()

                try:
                    _, mask = cv2.findFundamentalMat(
                        kpts0, kpts1, 
                        method=getattr(cv2, 'USAC_MAGSAC', cv2.FM_RANSAC), 
                        ransacReprojThreshold=RANSAC_THRESH_PIXELS,
                        confidence=0.999
                    )
                    return 0 if mask is None else int(np.sum(mask))
                except cv2.error:
                    return 0
        except Exception:
            return 0

    return get_general_match_count


def save_cluster_visualization(
    match_counts: np.ndarray,
    vpr_distances: np.ndarray,
    labels: np.ndarray,
    high_label: int,
    low_label: int,  # This parameter is unused but kept for API compatibility
    out_path: Path
):
    """Generates and saves a visualization of the K-Means clustering results.

    NOTE: This visualization simplifies the view by showing
    'Good Matches' (only the cluster with `high_label`) vs.
    'Bad Matches' (ALL other clusters combined).

    Args:
        match_counts: 1D array of match counts for all pairs.
        vpr_distances: 1D array of VPR distances for all pairs.
        labels: 1D array of cluster labels assigned by K-Means.
        high_label: The integer label of the "Rank 1" (highest mean) cluster.
        low_label: The integer label of the "Rank 2" cluster (unused).
        out_path: The `Path` object where the plot will be saved.
    """
    if not VISUALIZATION_ENABLED:
        print("     (Skipping visualization, matplotlib/seaborn not found.)")
        print("     (Install with: pip install matplotlib seaborn)")
        return

    print(f"     Saving cluster visualization to {out_path}...")
    
    # This logic groups ALL clusters that are NOT high_label into "Bad Matches"
    # This is a simplification for visualization purposes.
    label_names = np.where(
        labels == high_label, 
        "Good Matches (High Cluster)", 
        "Bad Matches (Other Clusters)"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.violinplot(
        x=label_names, y=match_counts,
        order=["Bad Matches (Other Clusters)", "Good Matches (High Cluster)"],
        ax=ax1
    )
    sns.stripplot(
        x=label_names, y=match_counts,
        order=["Bad Matches (Other Clusters)", "Good Matches (High Cluster)"],
        jitter=True, alpha=0.5, color='black', size=3, ax=ax1
    )
    ax1.set_title("K-Means Clustering of LightGlue Match Counts")
    ax1.set_ylabel("Number of LightGlue Matches")
    ax1.set_xlabel("Cluster Assignment (Simplified)")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    sns.violinplot(
        x=label_names, y=vpr_distances,
        order=["Bad Matches (Other Clusters)", "Good Matches (High Cluster)"],
        ax=ax2
    )
    sns.stripplot(
        x=label_names, y=vpr_distances,
        order=["Bad Matches (Other Clusters)", "Good Matches (High Cluster)"],
        jitter=True, alpha=0.5, color='black', size=3, ax=ax2
    )
    ax2.set_title("Distribution of VPR (FAISS) Distances")
    ax2.set_ylabel("VPR (FAISS) Distance (L2)")
    ax2.set_xlabel("Cluster Assignment (Simplified)")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle("Parameter Analysis (First Pass)", fontsize=16)
    
    try:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed to save plot: {e}", file=sys.stderr)


def run_parameter_analysis(args: argparse.Namespace) -> Tuple:
    """Executes the data loading and first-pass analysis.

    Args:
        args: The command-line arguments namespace (passed from __main__
              or `build_matrix.py`).

    Returns:
        A tuple containing:
        - threshold_tau (float): The calculated match count threshold.
        - vpr_dist_thresh (float): The calculated VPR (FAISS) distance threshold.
        - features (np.ndarray): The loaded global features (N, D).
        - index (faiss.Index): The loaded FAISS index.
        - get_match_count (Callable): The GENERAL-PURPOSE (i, j) match counter.
        - N (int): The total number of images.
        - validated_img_paths (List[Optional[Path]]): The validated paths.

    Raises:
        SystemExit: If essential files are not found or data dimensions mismatch.
    """
    print("[1/4] Loading features, index, and paths...")
    try:
        features = np.load(args.features_path).astype('float32')
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
    except FileNotFoundError:
        print(f"Error: Feature file not found at {args.features_path}. Exiting.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.paths_txt, "r", encoding="utf-8") as f:
            img_paths_str = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Error: Paths file not found at {args.paths_txt}. Exiting.", file=sys.stderr)
        sys.exit(1)
        
    validated_img_paths = pre_validate_image_paths(img_paths_str)

    N = len(features)
    if N != len(validated_img_paths):
        print(f"Error: Feature count ({N}) and path count ({len(validated_img_paths)}) mismatch.", file=sys.stderr)
        sys.exit(1)

    # Note: This assumes 'args.index_path' is defined in the calling script's args.
    # The provided snippet for parse_args() omitted it, but the script requires it.
    if not hasattr(args, 'index_path'):
         # Fallback if the arg was truly missing from the parser
        args.index_path = args.features_path.with_suffix(".faiss")
        print(f"[WARN] --index_path not defined in parser. Falling back to: {args.index_path}", file=sys.stderr)

    try:
        index = faiss.read_index(str(args.index_path))
        if index.ntotal != N:
            print(f"Error: Faiss index ntotal ({index.ntotal}) != feature count ({N}).", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read Faiss index at {args.index_path}. {e}", file=sys.stderr)
        sys.exit(1)

    # --- Initialize LightGlue Components ---
    ransac_status = "DISABLED" if args.disable_ransac else "ENABLED"
    print(f"[2/4] Initializing LightGlue components (RANSAC: {ransac_status})...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        extractor = SuperPoint(
            nms_radius=4,
            detection_threshold=0.0005,
            max_num_keypoints=2048,
            remove_borders=4
        ).eval().to(device)
        matcher = LightGlue(
            features="superpoint",
            filter_threshold=0.05,
            depth_confidence=-1,
            weights=args.lightglue_weights
        ).eval().to(device)
    except Exception as e:
        print(f"[ERROR] Failed to initialize LightGlue models: {e}", file=sys.stderr)
        print("Please ensure lightglue and its dependencies are installed.", file=sys.stderr)
        sys.exit(1)

    # --- [NEW LOGIC] Bypass analysis if num_clusters is 0 or 1 ---
    if args.num_clusters <= 1:
        print(f"[WARN] Parameter analysis SKIPPED (num_clusters={args.num_clusters}).")
        print("     This forces all-pairs matching (thresholds set to 0).")
        
        threshold_tau = 0.0
        vpr_dist_thresh = 2.0  # Max L2 distance for normalized vectors
        
        # We must STILL create the general-purpose counter for the next script
        print("[4/4] Creating general-purpose match counter for build script...")
        get_general_match_count = create_general_match_counter(
            validated_img_paths,
            args.disable_ransac,
            device, extractor, matcher
        )
        
        # Return all components needed by build_matrix.py
        return (
            threshold_tau, 
            vpr_dist_thresh, 
            features, 
            index, 
            get_general_match_count,
            N,
            validated_img_paths
        )
    # --- [END NEW LOGIC] ---


    # --- Start Step 3: [MODIFIED] K-Means Clustering and Parameter Determination ---
    print("[3/4] First Pass: Determining query set...")
    
    base_query_idx_0 = args.base_query_idx
    try:
        # Find the farthest neighbor from the base query
        distances_0, indices_0 = index.search(features[base_query_idx_0 : base_query_idx_0+1], N)
    except Exception as e:
        print(f"Error: FAISS search failed for base query. {e}", file=sys.stderr)
        sys.exit(1)

    # --- [FIX] Filter out invalid FAISS results (-1 or max_float) ---
    valid_mask = (indices_0[0] != -1) & (distances_0[0] < np.finfo(np.float32).max)
    
    if not np.any(valid_mask):
            print(f"Error: No valid neighbors found for query {base_query_idx_0}.", file=sys.stderr)
            sys.exit(1)
            
    valid_indices = indices_0[0][valid_mask]
    valid_distances = distances_0[0][valid_mask]
    
    # The farthest neighbor is the last one in the *valid* sorted list
    base_query_idx_1 = valid_indices[-1]
    farthest_dist = valid_distances[-1]
    # --- [END FIX] ---

    print(f" ... Farthest *valid* neighbor to index {base_query_idx_0} is index {base_query_idx_1} with distance {farthest_dist:.4f}.")
    
    query_indices_to_run = [base_query_idx_0]
    if base_query_idx_0 != base_query_idx_1:
        query_indices_to_run.append(base_query_idx_1)
        
    print(f" ... Query set: {query_indices_to_run}")

    all_first_pass_data = []

    # --- [MODIFIED] Loop over the two query indices ---
    for i, current_query_idx in enumerate(query_indices_to_run):
        print(f"First Pass (Set {i+1}/{len(query_indices_to_run)}): Matching query {current_query_idx} against all {N-1} others...")
        
        # --- Create the OPTIMIZED counter for THIS query ---
        get_first_pass_count = _create_first_pass_counter(
            validated_img_paths, 
            current_query_idx, # <-- Pass the current query index
            args.disable_ransac,
            device, extractor, matcher
        )
        
        # Get VPR distances for this query
        try:
            distances_all, indices_all = index.search(features[current_query_idx : current_query_idx+1], N)
        except Exception as e:
            print(f"Error: FAISS search failed for query {current_query_idx}. {e}", file=sys.stderr)
            sys.exit(1)

        neighbor_indices = indices_all[0]
        neighbor_distances = distances_all[0]

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            
            # Dictionary: {future: (target_idx, distance)}
            future_to_data = {}
            
            # 1. Submit all tasks
            for rank, target_idx in enumerate(neighbor_indices):
                if target_idx == current_query_idx:
                    continue
                if target_idx == -1:
                    continue
                
                dist = neighbor_distances[rank]
                
                # Submit task: get_first_pass_count(target_idx)
                future = executor.submit(get_first_pass_count, target_idx)
                future_to_data[future] = (target_idx, dist)

            # 2. Process results as they complete (with TQDM progress bar)
            desc = f"Matching Query {current_query_idx}"
            for future in tqdm(concurrent.futures.as_completed(future_to_data), 
                               total=len(future_to_data), desc=desc):
                
                target_idx, dist = future_to_data[future]
                
                try:
                    count = future.result()  # Get the result from the thread
                    all_first_pass_data.append({"count": count, "dist": dist})
                except Exception as e:
                    # Catch any exceptions during the LightGlue check
                    print(f"\n[WARN] Failed LightGlue check for query {current_query_idx} -> target {target_idx}: {e}", 
                          file=sys.stderr)
                    # Add a 0-count to represent failure
                    all_first_pass_data.append({"count": 0, "dist": dist})
    # --- End of [MODIFIED] loop ---

    # --- (K-Means and threshold logic remains the same, but uses combined data) ---
    print("Running K-Means on combined first-pass data...")
    match_counts_array = np.array([d['count'] for d in all_first_pass_data]).reshape(-1, 1)
    vpr_distances_array = np.array([d['dist'] for d in all_first_pass_data])

    # Check for valid K-Means input
    # Need at least k unique data points for k clusters
    unique_data_points = len(np.unique(match_counts_array))
    if len(match_counts_array) < args.num_clusters or unique_data_points < args.num_clusters:
        print("[WARN] Not enough distinct match counts for K-Means.")
        print(f"   (Found {unique_data_points} unique counts, need {args.num_clusters})")
        print("   Falling back to fixed default parameters (tau=10, vpr_dist=0.3).")
        threshold_tau = 10.0
        vpr_dist_thresh = 0.3
    else:
        # --- [START] MODIFIED K-MEANS LOGIC ---
        kmeans = KMeans(n_clusters=args.num_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(match_counts_array)
        
        cluster_means = [np.mean(match_counts_array[labels == i]) for i in range(args.num_clusters)]
        
        # --- New Logic: Find Rank 1 (Highest) and Rank 2 (Second Highest) ---
        if len(cluster_means) < 2:
            print("[WARN] K-Means returned less than 2 clusters. Using defaults.")
            threshold_tau = 10.0
            vpr_dist_thresh = 0.3
        else:
            # Sort cluster means to find the top two
            # argsort returns indices from lowest to highest
            sorted_indices = np.argsort(cluster_means)
            
            # The label of the cluster with the highest mean (Rank 1)
            rank_1_label = sorted_indices[-1]
            # The label of the cluster with the second-highest mean (Rank 2)
            rank_2_label = sorted_indices[-2]
        
            print(f"   K-Means found {args.num_clusters} clusters.")
            print(f"   Rank 1 Cluster (Good): Label={rank_1_label}, Mean={cluster_means[rank_1_label]:.2f}")
            print(f"   Rank 2 Cluster (Boundary): Label={rank_2_label}, Mean={cluster_means[rank_2_label]:.2f}")

            # Get data for the "Rank 1" (highest) cluster
            rank_1_counts = match_counts_array[labels == rank_1_label]
            rank_1_vpr_dists = vpr_distances_array[labels == rank_1_label]
            
            # Get data for the "Rank 2" (second highest) cluster
            rank_2_counts = match_counts_array[labels == rank_2_label]
        
            # Call visualization
            # The plot will show "Good Matches" (Rank 1) vs. "Bad Matches" (All Others)
            save_cluster_visualization(
                match_counts_array.squeeze(), vpr_distances_array, labels,
                rank_1_label, rank_2_label, args.out_plot_path # Pass Rank 1 as 'high_label'
            )
            
            # --- New Threshold Calculation (Boundary between Rank 1 and Rank 2) ---
            min_rank_1_count = np.min(rank_1_counts) if len(rank_1_counts) > 0 else 0
            max_rank_2_count = np.max(rank_2_counts) if len(rank_2_counts) > 0 else 0
            
            threshold_tau = (min_rank_1_count + max_rank_2_count) / 2.0
            if threshold_tau <= 0:
                threshold_tau = min_rank_1_count * 0.5 if min_rank_1_count > 10 else 10.0
            
            # --- VPR Threshold (Based ONLY on Rank 1 cluster) ---
            max_dist_good_match = 0.0
            if len(rank_1_vpr_dists) > 0:
                max_dist_good_match = np.max(rank_1_vpr_dists)
                vpr_dist_thresh = max_dist_good_match + args.vpr_dist_buffer
            else:
                print("[WARN] No 'good' matches found (Rank 1 cluster is empty). Falling back to default VPR distance (0.3).")
                vpr_dist_thresh = 0.3
        
            print(f"   Min Count in Rank 1 (Good): {min_rank_1_count:.2f}")
            print(f"   Max Count in Rank 2 (Boundary): {max_rank_2_count:.2f}")
            print(f"   Max VPR Distance in Rank 1: {max_dist_good_match:.4f}")
        # --- [END] MODIFIED K-MEANS LOGIC ---

    print(f"[OK] Dynamic LightGlue Threshold (tau): {threshold_tau:.2f} matches")
    print(f"[OK] Dynamic VPR Distance Threshold (vpr_dist_thresh): {vpr_dist_thresh:.4f} (L2 dist)")
    
    # --- Create the GENERAL-PURPOSE counter for the next script ---
    print("[4/4] Creating general-purpose match counter for build script...")
    get_general_match_count = create_general_match_counter(
        validated_img_paths,
        args.disable_ransac,
        device, extractor, matcher
    )
    
    # Return all components needed by build_matrix.py
    return (
        threshold_tau, 
        vpr_dist_thresh, 
        features, 
        index, 
        get_general_match_count,  # <-- The general (i, j) function
        N,
        validated_img_paths       # <-- The validated path list
    )
    
# ==========================
# 4. Standalone Execution
# ==========================
if __name__ == "__main__":
    """
    This block runs only when the script is executed directly.
    `RUNNING_AS_MAIN` is used as a flag for `parse_args` to
    properly parse command-line arguments.
    """
    RUNNING_AS_MAIN = True
    
    # We must call parse_args() *after* setting the global flag
    args = parse_args()
    
    if not VISUALIZATION_ENABLED:
        print("---")
        print("Warning: 'matplotlib' or 'seaborn' not found.")
        print("         Plot generation will be skipped.")
        print("         Install with: pip install matplotlib seaborn")
        print("---")
        
    if args.disable_ransac:
        print("---")
        print("Warning: RANSAC geometric verification is DISABLED.")
        print("         The 'threshold_tau' will be based on raw match counts,")
        print("         which is less robust to outliers.")
        print("---")
    
    print("--- Running Parameter Analysis (Standalone Mode) ---")
    print("--- This will run a full pass for the base query AND its farthest neighbor ---")
    
    # Run the full analysis
    tau, vpr_dist, _, _, _, _, _ = run_parameter_analysis(args)
    
    print("\n--- Analysis Complete ---")
    print("Determined Parameters (based on combined data):")
    print(f"   LightGlue Threshold (tau): {tau:.2f}")
    print(f"   VPR Distance (vpr_dist): {vpr_dist:.4f}")

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ulvn_core.io import load_topomap


def _count_missing(paths: List[str]) -> int:
    return sum(1 for item in paths if not Path(item).exists())


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    topomap = load_topomap(
        descriptors_path=args.descriptors_path,
        adjacency_matrix_path=args.adjacency_matrix_path,
        image_paths_file=args.image_paths_file,
        image_root=args.image_root,
    )
    descriptors = topomap.descriptors
    adjacency = topomap.adjacency_matrix
    positive = adjacency > 0

    errors: List[str] = []
    warnings: List[str] = []

    if not np.all(np.isfinite(descriptors)):
        errors.append("descriptors contain NaN or Inf values")
    if not np.all(np.isfinite(adjacency)):
        errors.append("adjacency matrix contains NaN or Inf values")
    if np.any(adjacency < 0):
        warnings.append("adjacency matrix contains negative entries; only positive entries are treated as edges")

    row_degrees = positive.sum(axis=1)
    col_degrees = positive.sum(axis=0)
    isolated = np.where((row_degrees + col_degrees) == 0)[0].astype(int).tolist()
    if isolated:
        warnings.append(f"{len(isolated)} isolated nodes found")

    image_count = len(topomap.image_paths)
    missing_images = _count_missing(topomap.image_paths) if image_count else 0
    if args.require_images and image_count == 0:
        errors.append("--require-images was set, but no --image-paths-file was provided")
    if args.require_images and missing_images:
        errors.append(f"{missing_images} image paths do not exist")
    elif missing_images:
        warnings.append(f"{missing_images} image paths do not exist")

    symmetric = bool(np.allclose(adjacency, adjacency.T))
    summary: Dict[str, Any] = {
        "nodes": int(descriptors.shape[0]),
        "descriptor_dim": int(descriptors.shape[1]),
        "adjacency_shape": list(adjacency.shape),
        "adjacency_dtype": str(adjacency.dtype),
        "directed_edges": int(np.count_nonzero(positive)),
        "undirected_edge_upper_bound": int(np.count_nonzero(np.triu(positive, k=1))),
        "symmetric": symmetric,
        "isolated_nodes": isolated[:20],
        "isolated_node_count": len(isolated),
        "image_path_count": image_count,
        "missing_image_count": int(missing_images),
        "errors": errors,
        "warnings": warnings,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate ULVN topological map artifacts.")
    parser.add_argument("--descriptors-path", required=True, help="Path to [N, D] descriptor .npy file.")
    parser.add_argument("--adjacency-matrix-path", required=True, help="Path to [N, N] adjacency .npy file.")
    parser.add_argument("--image-paths-file", default=None, help="Optional text file with one node image path per line.")
    parser.add_argument("--image-root", default=None, help="Optional root used to resolve relative image paths.")
    parser.add_argument("--require-images", action="store_true", help="Fail if image paths are missing or not provided.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = validate(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ulvn_core.io import load_topomap
from ulvn_core.navigation import BeliefAwareSubgoalSearch, PlaceRecognition


def _load_feature(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    return np.load(path).astype(np.float32).reshape(-1)


def _path_cost(path, adjacency: np.ndarray, edge_cost: str) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        weight = float(adjacency[u, v])
        if edge_cost == "unweighted":
            total += 1.0
        elif edge_cost == "inverse_weight":
            total += 1.0 / max(weight, 1e-6)
        else:
            raise ValueError(f"Unknown edge_cost: {edge_cost}")
    return total


def run(args: argparse.Namespace) -> Dict[str, Any]:
    topomap = load_topomap(
        descriptors_path=args.descriptors_path,
        adjacency_matrix_path=args.adjacency_matrix_path,
        image_paths_file=args.image_paths_file,
        image_root=args.image_root,
    )
    planner = BeliefAwareSubgoalSearch(topomap.adjacency_matrix, topomap.descriptors)

    start_feature = _load_feature(args.start_feature)
    goal_feature = _load_feature(args.goal_feature)

    if args.start_idx is not None:
        start_idx = int(args.start_idx)
    else:
        start_idx = None
    if args.goal_idx is not None:
        goal_idx = int(args.goal_idx)
    else:
        goal_idx = None

    if start_idx is None or goal_idx is None:
        if start_feature is not None and goal_feature is not None:
            goal_idx, start_idx = planner.initial_localize(goal_feature, start_feature)
        else:
            start_idx = 0 if start_idx is None else start_idx
            goal_idx = topomap.descriptors.shape[0] - 1 if goal_idx is None else goal_idx

    path = planner.plan_path(start_idx=start_idx, goal_idx=goal_idx, edge_cost=args.edge_cost)
    if not path:
        raise RuntimeError(f"No graph path from node {start_idx} to node {goal_idx}")

    subgoal = planner.next_subgoal(path[0], n_ahead=args.n_ahead)
    localizer = PlaceRecognition(topomap.descriptors, topomap.adjacency_matrix)
    localizer.initialize_model(topomap.descriptors[path[0]])
    matched_idx, confidence = localizer.match(topomap.descriptors[path[min(1, len(path) - 1)]])

    result: Dict[str, Any] = {
        "start_idx": int(start_idx),
        "goal_idx": int(goal_idx),
        "path": [int(x) for x in path],
        "path_hops": max(0, len(path) - 1),
        "path_cost": _path_cost(path, topomap.adjacency_matrix, args.edge_cost),
        "edge_cost": args.edge_cost,
        "subgoal": subgoal,
        "bpl_match": {"node": int(matched_idx), "confidence": confidence},
    }
    if topomap.image_paths:
        result["start_image"] = topomap.image_paths[start_idx]
        result["goal_image"] = topomap.image_paths[goal_idx]
        sg = subgoal.get("subgoal_idx")
        result["subgoal_image"] = topomap.image_paths[sg] if sg is not None else None
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an offline ULVN BPL/BASS navigation demo.")
    parser.add_argument("--descriptors-path", required=True, help="Path to [N, D] descriptor .npy file.")
    parser.add_argument("--adjacency-matrix-path", required=True, help="Path to [N, N] adjacency .npy file.")
    parser.add_argument("--image-paths-file", default=None, help="Optional node image path text file.")
    parser.add_argument("--image-root", default=None, help="Optional root for relative image paths.")
    parser.add_argument("--start-idx", type=int, default=None, help="Known start node. Defaults to 0 if no start feature is provided.")
    parser.add_argument("--goal-idx", type=int, default=None, help="Known goal node. Defaults to N-1 if no goal feature is provided.")
    parser.add_argument("--start-feature", default=None, help="Optional observed start descriptor .npy file.")
    parser.add_argument("--goal-feature", default=None, help="Optional goal descriptor .npy file.")
    parser.add_argument("--edge-cost", choices=["unweighted", "inverse_weight"], default="unweighted")
    parser.add_argument("--n-ahead", type=int, default=1, help="How many graph nodes ahead to select as the subgoal.")
    parser.add_argument("--output-json", default=None, help="Optional file to write the JSON result.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

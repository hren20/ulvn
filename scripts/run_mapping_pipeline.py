#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]
RAVEL_DIR = REPO_ROOT / "RAVEL"
RAVEL_SCRIPTS = RAVEL_DIR / "scripts"


def _rel_or_abs(path: Path) -> str:
    return str(path.resolve())


def _run(cmd: List[str], dry_run: bool) -> None:
    print("+ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ULVN/RAVEL visual topological map construction pipeline."
    )
    parser.add_argument("--img-dir", required=True, help="Directory of landmark/topomap images.")
    parser.add_argument("--work-dir", default="outputs/mapping_run", help="Directory for intermediate features and index files.")
    parser.add_argument("--result-dir", default="outputs/mapping_run/results", help="Directory for adjacency and visualization outputs.")
    parser.add_argument("--arch", default="megaloc", choices=["resnet18", "resnet50", "dinov2_vits14", "dinov2_vitb14", "megaloc"])
    parser.add_argument("--img-size", type=int, nargs=2, default=[320, 320], metavar=("W", "H"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--index-type", default="hnsw", choices=["hnsw", "ivfpq", "flat"])
    parser.add_argument("--index-metric", default="l2", choices=["l2", "cos"])
    parser.add_argument("--lightglue-weights", default="outdoor", choices=["indoor", "outdoor"])
    parser.add_argument("--num-threads", type=int, default=10)
    parser.add_argument("--add-back-threshold", type=int, default=300)
    parser.add_argument("--binary-matrix", action="store_true", help="Store valid RAVEL edges as 1 instead of match counts.")
    parser.add_argument("--skip-encode", action="store_true", help="Reuse existing features and paths in --work-dir.")
    parser.add_argument("--skip-prune", action="store_true", help="Stop after adjacency_matrix.npy is built.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    work_dir = (REPO_ROOT / args.work_dir).resolve() if not Path(args.work_dir).is_absolute() else Path(args.work_dir)
    result_dir = (REPO_ROOT / args.result_dir).resolve() if not Path(args.result_dir).is_absolute() else Path(args.result_dir)
    if not args.dry_run:
        work_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

    features = work_dir / "data_features.npy"
    paths_txt = work_dir / "data_paths.txt"
    index_path = work_dir / "index.faiss"
    adjacency = result_dir / "adjacency_matrix.npy"
    pruned = result_dir / "pruned_adjacency_matrix.npy"
    pruned_viz = result_dir / "pruned_topological_map.png"

    if not args.skip_encode:
        _run([
            sys.executable,
            _rel_or_abs(RAVEL_SCRIPTS / "encode_images.py"),
            "--img_dir", args.img_dir,
            "--arch", args.arch,
            "--img_size", str(args.img_size[0]), str(args.img_size[1]),
            "--batch_size", str(args.batch_size),
            "--device", args.device,
            "--num_workers", str(args.num_workers),
            "--out_features", _rel_or_abs(features),
            "--out_paths", _rel_or_abs(paths_txt),
        ], args.dry_run)

    _run([
        sys.executable,
        _rel_or_abs(RAVEL_SCRIPTS / "build_index.py"),
        "--features", _rel_or_abs(features),
        "--out", _rel_or_abs(index_path),
        "--type", args.index_type,
        "--metric", args.index_metric,
    ], args.dry_run)

    build_cmd = [
        sys.executable,
        _rel_or_abs(RAVEL_SCRIPTS / "build_matrix.py"),
        "--features_path", _rel_or_abs(features),
        "--index_path", _rel_or_abs(index_path),
        "--paths_txt", _rel_or_abs(paths_txt),
        "--result_dir", _rel_or_abs(result_dir),
        "--lightglue_weights", args.lightglue_weights,
        "--num_threads", str(args.num_threads),
    ]
    if args.binary_matrix:
        build_cmd.append("--binary_matrix")
    _run(build_cmd, args.dry_run)

    if not args.skip_prune:
        _run([
            sys.executable,
            _rel_or_abs(RAVEL_SCRIPTS / "prune_topomap_nx.py"),
            "--weighted_matrix", _rel_or_abs(adjacency),
            "--paths_txt", _rel_or_abs(paths_txt),
            "--out_matrix_path", _rel_or_abs(pruned),
            "--out_viz_path", _rel_or_abs(pruned_viz),
            "--add_back_threshold", str(args.add_back_threshold),
        ], args.dry_run)

    final_matrix = pruned if not args.skip_prune else adjacency
    _run([
        sys.executable,
        _rel_or_abs(REPO_ROOT / "scripts" / "validate_topomap.py"),
        "--descriptors-path", _rel_or_abs(features),
        "--adjacency-matrix-path", _rel_or_abs(final_matrix),
        "--image-paths-file", _rel_or_abs(paths_txt),
    ], args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

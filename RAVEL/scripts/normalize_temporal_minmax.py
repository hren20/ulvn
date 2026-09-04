#!/usr/bin/env python3
"""Normalize temporal-distance arrays with clipped min-max decay."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def normalize_temporal(input_path: Path, output_path: Path, t_max: float) -> np.ndarray:
    """Convert temporal distances to scores using 1 - clip(distance / t_max, 0, 1)."""
    if t_max <= 0:
        raise ValueError("--t-max must be positive")

    distances = np.load(input_path)
    scores = 1.0 - np.clip(distances / t_max, 0.0, 1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, scores)
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize temporal-distance .npy arrays with clipped min-max decay."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input temporal-distance .npy file.")
    parser.add_argument("--output", required=True, type=Path, help="Output normalized-score .npy file.")
    parser.add_argument("--t-max", default=20.0, type=float, help="Distance mapped to zero score.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = normalize_temporal(args.input, args.output, args.t_max)
    print(f"Saved {scores.shape} normalized scores to {args.output}")


if __name__ == "__main__":
    main()

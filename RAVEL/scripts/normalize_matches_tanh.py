#!/usr/bin/env python3
"""Normalize match-count matrices with tanh scaling."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def normalize_matches(input_path: Path, output_path: Path, scale: float, set_diagonal: bool) -> np.ndarray:
    """Convert match counts to scores using tanh(matches / scale)."""
    if scale <= 0:
        raise ValueError("--scale must be positive")

    matches = np.load(input_path)
    scores = np.tanh(matches / scale)
    if set_diagonal:
        if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
            raise ValueError("--set-diagonal requires a square 2D matrix")
        np.fill_diagonal(scores, 1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, scores)
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize match-count .npy matrices with tanh scaling."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input match-count .npy file.")
    parser.add_argument("--output", required=True, type=Path, help="Output normalized-score .npy file.")
    parser.add_argument("--scale", default=300.0, type=float, help="Tanh denominator for match counts.")
    parser.add_argument("--set-diagonal", action="store_true", help="Set the diagonal to 1.0 after normalization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = normalize_matches(args.input, args.output, args.scale, args.set_diagonal)
    print(f"Saved {scores.shape} normalized scores to {args.output}")


if __name__ == "__main__":
    main()

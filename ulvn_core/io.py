from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


@dataclass(frozen=True)
class TopomapData:
    """Aligned descriptors, graph matrix, and optional node image paths."""

    descriptors: np.ndarray
    adjacency_matrix: np.ndarray
    image_paths: List[str]


PathLike = Union[str, Path]


def read_image_paths(path_file: PathLike) -> List[str]:
    """Read one image path per line, preserving order."""

    path = Path(path_file)
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_image_paths(paths: List[str], image_root: Optional[PathLike] = None) -> List[str]:
    """Resolve relative image paths against an optional image root."""

    if image_root is None:
        return paths

    root = Path(image_root)
    resolved = []
    for item in paths:
        p = Path(item)
        resolved.append(str(p if p.is_absolute() else root / p))
    return resolved


def load_topomap(
    descriptors_path: PathLike,
    adjacency_matrix_path: PathLike,
    image_paths_file: Optional[PathLike] = None,
    image_root: Optional[PathLike] = None,
) -> TopomapData:
    """Load and validate the aligned topological navigation artifacts."""

    descriptors = np.load(descriptors_path).astype(np.float32)
    adjacency = np.load(adjacency_matrix_path)

    if descriptors.ndim != 2:
        raise ValueError(f"descriptors must be 2D [N, D], got {descriptors.shape}")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency_matrix must be square [N, N], got {adjacency.shape}")
    if adjacency.shape[0] != descriptors.shape[0]:
        raise ValueError(
            "node count mismatch: "
            f"descriptors has {descriptors.shape[0]} rows but adjacency has {adjacency.shape[0]} nodes"
        )

    image_paths: List[str] = []
    if image_paths_file is not None:
        image_paths = resolve_image_paths(read_image_paths(image_paths_file), image_root)
        if len(image_paths) != descriptors.shape[0]:
            raise ValueError(
                "node count mismatch: "
                f"image_paths has {len(image_paths)} lines but descriptors has {descriptors.shape[0]} rows"
            )

    return TopomapData(
        descriptors=descriptors,
        adjacency_matrix=adjacency,
        image_paths=image_paths,
    )

#!/usr/bin/env python3
"""
scripts/build_index.py
--------------

Builds a FAISS index for fast approximate nearest neighbor search over
image embeddings. If FAISS is unavailable, it falls back to saving
a 'bruteforce' index (a simple .npz file).

This script generates two files:
  - {out_path}: The FAISS index file (e.g., index.faiss) or .npz fallback.
  - {out_path}.meta.json: A sidecar JSON file with index metadata
    (type, metric, dimension).

Usage:
  # Build a fast HNSW index (default)
  python scripts/build_index.py --features tmp/run/data_features.npy --out tmp/run/index.faiss --type hnsw

  # Build a compressed IVFPQ index for large datasets
  python scripts/build_index.py --features tmp/landmarks_features.npy --out tmp/index.ivfpq \
                        --type ivfpq --nlist 4096 --pq_m 64
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def save_bruteforce(features: np.ndarray, out_path: str, metric: str):
    """Saves a fallback 'brute-force' index as an .npz file.

    This is used when FAISS is not installed. The retrieval code must
    handle this .npz file and perform manual brute-force search.

    Args:
        features: The [N, D] feature array.
        out_path: The target path to save the .npz file.
        metric: The similarity metric ("cos" or "l2").
    """
    meta = {
        "type": "bruteforce",
        "metric": metric,
        "dim": int(features.shape[1])
    }
    # Save features and metadata in a single compressed file
    np.savez(
        out_path,
        features=features.astype("float32"),
        meta=json.dumps(meta)
    )
    print(f"[Fallback] Saved brute-force index to: {out_path}")


def main():
    """Main function to parse args, load data, and build the index."""
    ap = argparse.ArgumentParser(
        description="Builds a FAISS index from feature vectors."
    )

    # --- Core Arguments ---
    ap.add_argument(
        "--features",
        type=str,
        required=True,
        help="Path to feature file (features.npy, [N, D])."
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output index path (e.g., index.faiss or index.npz)."
    )
    ap.add_argument(
        "--type",
        type=str,
        default="hnsw",
        choices=["hnsw", "ivfpq", "flat"],
        help="Type of FAISS index to build. Default: hnsw"
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="l2",
        choices=["cos", "l2"],
        help="Similarity metric. 'cos' expects L2-normalized vectors. Default: l2"
    )

    # --- HNSW Parameters ---
    ap.add_argument(
        "--m",
        type=int,
        default=32,
        help="HNSW M (number of neighbors per layer). Default: 32"
    )
    ap.add_argument(
        "--efc",
        type=int,
        default=200,
        help="HNSW efConstruction (controls build-time quality/speed). Default: 200"
    )

    # --- IVFPQ Parameters ---
    ap.add_argument(
        "--nlist",
        type=int,
        default=4096,
        help="Number of coarse clusters (voronoi cells) for IVF. Default: 4096"
    )
    ap.add_argument(
        "--pq_m",
        type=int,
        default=64,
        help="Number of sub-vectors for Product Quantization (PQ). Default: 64"
    )
    ap.add_argument(
        "--pq_bits",
        type=int,
        default=8,
        help="Bits per PQ code (8 bits = 256 centroids per sub-vector). Default: 8"
    )

    args = ap.parse_args()

    # --- 1. Load Features ---
    print(f"Loading features from: {args.features}")
    try:
        feats = np.load(args.features)
    except FileNotFoundError:
        print(f"[ERROR] Features file not found: {args.features}", file=sys.stderr)
        sys.exit(1)

    # Ensure data is float32 and C-contiguous for FAISS
    feats = np.ascontiguousarray(feats.astype(np.float32))
    
    N, D = feats.shape
    print(f"Loaded features: {feats.shape}, dtype={feats.dtype}, "
          f"contiguous={feats.flags['C_CONTIGUOUS']}")

    # --- 2. Check for FAISS and Set Up ---
    use_faiss = False
    try:
        import faiss  # type: ignore
        use_faiss = True
    except ImportError:
        print(
            "[WARN] FAISS not available. Falling back to brute-force .npz index.",
            file=sys.stderr
        )
    except Exception as e:
        print(
            f"[WARN] Failed to import FAISS ({e}). Falling back to brute-force.",
            file=sys.stderr
        )

    if not use_faiss:
        # Ensure fallback path ends in .npz
        out_path = Path(args.out)
        if out_path.suffix != ".npz":
            out_path = out_path.with_suffix(".npz")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_bruteforce(feats, str(out_path), args.metric)
        return

    # --- 3. Build FAISS Index ---
    if args.metric == "cos":
        # For cosine similarity, use Inner Product on L2-normalized vectors.
        # The encode_images.py script already provides normalized vectors.
        metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        metric_type = faiss.METRIC_L2

    index = None
    if args.type == "flat":
        print("[flat] Building IndexFlat (brute-force L2/IP)...")
        if metric_type == faiss.METRIC_INNER_PRODUCT:
            index = faiss.IndexFlatIP(D)
        else:
            index = faiss.IndexFlatL2(D)
        print("[flat] Adding vectors...")
        index.add(feats)

    elif args.type == "hnsw":
        print(f"[hnsw] Building HNSW index (M={args.m}, efC={args.efc})...")
        index = faiss.IndexHNSWFlat(D, args.m, metric_type)
        index.hnsw.efConstruction = args.efc
        print("[hnsw] Adding vectors...")
        index.add(feats)

    elif args.type == "ivfpq":
        print(f"[ivfpq] Building IVFPQ index (nlist={args.nlist}, "
              f"M={args.pq_m}, bits={args.pq_bits})...")
        
        # Quantizer is the coarse "cell" index
        if metric_type == faiss.METRIC_INNER_PRODUCT:
            quantizer = faiss.IndexFlatIP(D)
        else:
            quantizer = faiss.IndexFlatL2(D)

        index = faiss.IndexIVFPQ(
            quantizer, D, args.nlist, args.pq_m, args.pq_bits, metric_type
        )
        
        print("[ivfpq] Training index...")
        index.train(feats)
        print("[ivfpq] Adding vectors...")
        index.add(feats)
        
        # Set a reasonable default search parameter (number of cells to check)
        index.nprobe = min(32, args.nlist)

    else:
        raise ValueError(f"Unknown index type: {args.type}")

    # --- 4. Save Index and Metadata ---
    out_path = Path(args.out)
    if out_path.suffix != ".faiss":
        out_path = out_path.with_suffix(".faiss")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nWriting index to: {out_path}")
    faiss.write_index(index, str(out_path))

    # Save metadata sidecar file
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta = {
        "type": args.type,
        "metric": args.metric,
        "dim": int(D),
        "count": int(N)
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved FAISS index to: {out_path}")
    print(f"Saved meta to:      {meta_path}")


if __name__ == "__main__":
    main()

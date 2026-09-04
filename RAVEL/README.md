# RAVEL: Visual Landmark Graph Construction

RAVEL builds a topological map from landmark images. The output is an adjacency
matrix whose rows and columns correspond to image nodes. Positive edge weights
represent LightGlue match counts after VPR-based candidate retrieval and
geometric verification.

## Pipeline

1. `scripts/encode_images.py`
   - Recursively scans an image directory.
   - Encodes images with MegaLoc, DINOv2, ResNet-18, or ResNet-50.
   - Saves `features.npy` and an aligned `paths.txt` file.

2. `scripts/build_index.py`
   - Builds a FAISS index over the descriptor array.
   - Falls back to a brute-force `.npz` index when FAISS is unavailable, though
     the current graph-building path expects FAISS.

3. `scripts/analyze_parameters.py`
   - Runs first-pass LightGlue matching.
   - Estimates the LightGlue match threshold and VPR distance threshold using
     K-Means clustering.
   - Can also be run standalone to inspect threshold selection.

4. `scripts/build_matrix.py`
   - Main graph construction entry point.
   - Calls `analyze_parameters.py`, then verifies candidate image pairs with
     SuperPoint + LightGlue.
   - Saves `adjacency_matrix.npy` and `topological_map_visualization.png`.

5. `scripts/prune_topomap_nx.py`
   - Computes a maximum spanning forest backbone.
   - Adds back strong redundant edges above a threshold.
   - Saves `pruned_adjacency_matrix.npy` and a pruned graph visualization.

## Conda environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate ulvn
```

For CUDA acceleration, install the PyTorch build that matches your CUDA driver,
for example:

```bash
conda install -n ulvn pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Recommended workflow

Run the following commands from this `RAVEL/` directory after activating the
`ulvn` conda environment.

### 1. Encode landmark images

```bash
python scripts/encode_images.py \
  --img_dir ../data/landmark \
  --arch megaloc \
  --img_size 320 320 \
  --out_features tmp/run/data_features.npy \
  --out_paths tmp/run/data_paths.txt
```

Alternative encoders:

```bash
python scripts/encode_images.py \
  --img_dir ../data/landmark \
  --arch dinov2_vits14 \
  --img_size 224 224 \
  --out_features tmp/dino/data_features.npy \
  --out_paths tmp/dino/data_paths.txt
```

### 2. Build the retrieval index

```bash
python scripts/build_index.py \
  --features tmp/run/data_features.npy \
  --out tmp/run/index.faiss \
  --type hnsw \
  --metric l2
```

### 3. Build the weighted graph

```bash
python scripts/build_matrix.py \
  --features_path tmp/run/data_features.npy \
  --index_path tmp/run/index.faiss \
  --paths_txt tmp/run/data_paths.txt \
  --result_dir results/run \
  --lightglue_weights outdoor \
  --num_threads 10
```

Use `--binary_matrix` if downstream code should treat every valid edge as `1`
instead of preserving the LightGlue match count.

### 4. Prune the graph

```bash
python scripts/prune_topomap_nx.py \
  --weighted_matrix results/run/adjacency_matrix.npy \
  --paths_txt tmp/run/data_paths.txt \
  --out_matrix_path results/run/pruned_adjacency_matrix.npy \
  --out_viz_path results/run/pruned_topological_map.png \
  --add_back_threshold 300
```

## Outputs

```text
tmp/run/data_features.npy              # [N, D] float32 descriptors
tmp/run/data_paths.txt                 # row-to-image mapping
tmp/run/index.faiss                    # FAISS retrieval index
tmp/run/index.faiss.meta.json          # index metadata
results/run/adjacency_matrix.npy       # weighted or binary graph
results/run/topological_map_visualization.png
results/run/pruned_adjacency_matrix.npy
results/run/pruned_topological_map.png
```

See `../docs/DATA_FORMATS.md` for the expected shape and ordering conventions.

## Utility scripts

- `scripts/compare/compute_distance_matrix.py`: pairwise L2/cosine matrices.
- `scripts/compare/compute_full_matrices.py`: full comparison workflows.
- `scripts/compare/compute_matrices_flexible.py`: configurable comparison
  matrix generation.
- `scripts/compare/analyze_correlation.py`: correlation analysis between
  generated matrices.
- `scripts/visualize/create_heatmaps.py`: heatmap rendering for matrix files.
- `scripts/visualize/visualize_graph.py`: graph visualization from adjacency
  matrices.
- `scripts/normalize_temporal_minmax.py` and `scripts/normalize_matches_tanh.py`:
  matrix normalization utilities.

## Notes and limitations

- MegaLoc and DINOv2 load models through `torch.hub`, which may require network
  access on first use.
- LightGlue weights may also be downloaded if they are not already cached.
- `scripts/build_matrix.py` is the available weighted graph builder. Older
  documentation referred to `build_weighted_topomap.py`, but that file is not
  present in this repository.
- Large datasets can make pairwise geometric verification expensive. Tune
  `--num_threads`, image resolution, and FAISS index type for your hardware.

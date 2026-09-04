# ULVN Pipeline

This document describes the public, repository-relative pipeline for building a
visual topological map, validating it, running BPL/BASS offline, and connecting
the same artifacts to ROS/IsaacSim for online navigation.

The mapping code follows the same high-level design as TopoVisNav-style visual
topological navigation systems: images are graph nodes, global visual place
recognition retrieves candidate neighbors, local feature matching verifies
edges, and a graph pruning step keeps a compact navigation topology.

## 1. Build a visual topological map

Input:

```text
data/landmark_images/
  000000.png
  000001.png
  ...
```

One-command public entry point:

```bash
conda activate ulvn

python scripts/run_mapping_pipeline.py \
  --img-dir data/landmark_images \
  --work-dir outputs/example_topomap/work \
  --result-dir outputs/example_topomap/results \
  --arch megaloc \
  --lightglue-weights outdoor
```

Outputs:

```text
outputs/example_topomap/work/data_features.npy
outputs/example_topomap/work/data_paths.txt
outputs/example_topomap/work/index.faiss
outputs/example_topomap/results/adjacency_matrix.npy
outputs/example_topomap/results/topological_map_visualization.png
outputs/example_topomap/results/pruned_adjacency_matrix.npy
outputs/example_topomap/results/pruned_topological_map.png
```

Equivalent manual steps are documented in `RAVEL/README.md`.

## 2. Validate topomap artifacts

Run this before using a dataset for localization or navigation:

```bash
python scripts/validate_topomap.py \
  --descriptors-path outputs/example_topomap/work/data_features.npy \
  --adjacency-matrix-path outputs/example_topomap/results/pruned_adjacency_matrix.npy \
  --image-paths-file outputs/example_topomap/work/data_paths.txt
```

The validator checks node alignment, matrix shape, finite values, edge counts,
isolated nodes, symmetry, and optional image-path existence.

## 3. Run offline BPL/BASS navigation

This step uses only NumPy. It does not require ROS, IsaacSim, Torch, FAISS, or
local planner checkpoints.

```bash
python scripts/offline_navigation_demo.py \
  --descriptors-path outputs/example_topomap/work/data_features.npy \
  --adjacency-matrix-path outputs/example_topomap/results/pruned_adjacency_matrix.npy \
  --image-paths-file outputs/example_topomap/work/data_paths.txt \
  --start-idx 0 \
  --goal-idx 20 \
  --n-ahead 2
```

If `--start-idx` or `--goal-idx` are omitted, the script defaults to node `0`
and node `N-1`. If `--start-feature` and `--goal-feature` are provided, it first
localizes the start and goal descriptors against the node descriptor matrix.

## 4. Run online ROS navigation

Online navigation has two ROS-facing processes:

```text
camera image topic -> bass.py -> /goal_path -> local_planner.py -> waypoint topic
```

Default BASS interface:

```bash
python "BPL & BASS/bass.py" \
  --img_dir data/landmark_images \
  --descriptors_path outputs/example_topomap/work/data_features.npy \
  --adjacency_matrix_path outputs/example_topomap/results/pruned_adjacency_matrix.npy \
  --image_paths_file outputs/example_topomap/work/data_paths.txt \
  --goal_image_index 20 \
  --image_topic /isaac_node/camera0/image_raw
```

Default local planner interface:

```bash
python "BPL & BASS/local_planner.py" \
  --model nomad_ulvn \
  --config config/models.yaml \
  --robot-config config/robot.yaml \
  --image-topic /isaac_node/camera0/image_raw \
  --goal-path-topic /goal_path \
  --num-samples 8 \
  --waypoint 3
```

ROS, `config/models.yaml`, and ViNT/NoMaD checkpoints are not required for
offline validation, but they are required for online execution. The ViNT/NoMaD
inference code and matching `nomad_ulvn` / `vint_ulvn` model configs are
included in this repository.

## 5. IsaacSim data-generation interface

The IsaacSim-side mapping pipeline is an external interface, not a dependency
of the core package. The public contract is:

```text
USD/scene assets
  -> occupancy voxel map .npy
  -> passable-area .png/.json metadata
  -> skeleton sample .npy
  -> captured topomap images
  -> RAVEL descriptors and adjacency matrix
```

The IsaacSim scripts are included in `isaacsim_pipeline/` and use
repository-relative or CLI-provided output directories. See
`isaacsim_pipeline/README.md` and `docs/REPRODUCTION.md` for commands and ROS
topics.

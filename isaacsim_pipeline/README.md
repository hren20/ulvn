# IsaacSim Data Collection Pipeline

This directory contains the IsaacSim data-collection scripts used before RAVEL topological map
construction:

```text
USD scene
  -> 3D occupancy voxel map
  -> 2D passable-area map
  -> skeleton graph and uniformly spaced oriented sample points
  -> image capture at skeleton samples through IsaacSim ROS topics
```

Large generated artifacts are intentionally not committed. Outputs default to
`isaacsim_pipeline/outputs/`, which is ignored by `.gitignore`.

## Files

- `export_voxel_map.py`: load a USD scene in IsaacSim and export a 3D
  occupancy grid plus metadata.
- `export_passable_area.py`: convert a 3D occupancy grid into a 2D
  passable-area map with robot-radius erosion.
- `export_skeleton_points.py`: extract a skeleton from the passable map
  and sample points with heading directions.
- `capture_photos.py`: capture images with collision-aware forward adjustment
  at sampled poses through ROS.
- `run_map_generation_batch.py`: batch USD -> occupancy -> passable map ->
  skeleton points.
- `extended_mapping/`, `sdg_roadmap/`, `navigation_utils/`: helper modules used
  by skeleton extraction.

## Environment

Run IsaacSim-specific scripts with IsaacLab's Python, for example:

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p /path/to/ulvn/isaacsim_pipeline/export_voxel_map.py --help
```

Install Python-side helper packages into that environment if missing:

```bash
/path/to/IsaacLab/_isaac_sim/python.sh -m pip install -r /path/to/ulvn/requirements-isaacsim.txt
```

The image-capture scripts additionally require ROS Python packages such as
`rospy`, `cv_bridge`, `geometry_msgs`, `sensor_msgs`, `nav_msgs`, and `std_msgs`.

## Single-Scene Workflow

### 1. Export 3D occupancy voxels

```bash
python isaacsim_pipeline/export_voxel_map.py \
  --usd_path /path/to/scene/start_result_navigation.usd \
  --z_low 0.2 \
  --z_high 1.2 \
  --cell_size 0.05 \
  --margin 25.0 \
  --output_name scene_000 \
  --output_dir isaacsim_pipeline/outputs/occupancy_data
```

Outputs:

```text
occupancy_3d_scene_000.npy
occupancy_3d_scene_000_metadata.npy
```

### 2. Export 2D passable area

```bash
python isaacsim_pipeline/export_passable_area.py \
  --npy_path isaacsim_pipeline/outputs/occupancy_data/occupancy_3d_scene_000.npy \
  --z_low 0.2 \
  --z_high 1.2 \
  --robot_radius 0.3 \
  --output_dir isaacsim_pipeline/outputs/passable_area
```

### 3. Export skeleton samples and directions

```bash
python isaacsim_pipeline/export_skeleton_points.py \
  --map_path isaacsim_pipeline/outputs/passable_area/scene_000_max_passable_area.png \
  --metadata_path isaacsim_pipeline/outputs/occupancy_data/occupancy_3d_scene_000_metadata.npy \
  --spacing 0.5 \
  --angle_interval 60 \
  --output_dir isaacsim_pipeline/outputs/skeleton_points
```

### 4. Capture images at skeleton samples

Start `roscore` and the Isaac node first, then run:

```bash
python isaacsim_pipeline/capture_photos.py \
  --npy_path isaacsim_pipeline/outputs/skeleton_points/scene_000_skeleton_points_with_directions.npy \
  --output_dir data/ulvn_sim/images/scene_000 \
  --height 0.4 \
  --use_hover \
  --advance_distance 0.1 \
  --max_advance_attempts 5 \
  --stabilization_time 2.0
```

## Batch Workflow

```bash
python isaacsim_pipeline/run_map_generation_batch.py \
  --usd_base_dir assets/GRScenes-100/home_scenes/scenes \
  --output_voxel_dir isaacsim_pipeline/outputs/occupancy_data \
  --output_passable_dir isaacsim_pipeline/outputs/passable_area \
  --output_skeleton_dir isaacsim_pipeline/outputs/skeleton_points \
  --z_low 0.2 \
  --z_high 1.2 \
  --cell_size 0.05 \
  --margin 25.0 \
  --robot_radius 0.3 \
  --spacing 0.5 \
  --angle_interval 60
```

## Notes

- Put USD scene assets under `assets/` or pass explicit paths through CLI
  arguments.
- Generated `.npy`, image, zip, and output files are ignored by Git.

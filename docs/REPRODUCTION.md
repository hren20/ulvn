# Reproduction Guide

This guide documents the full repository-relative reproducibility path for
ULVN. Source code lives in Git; datasets, scene assets, and learned checkpoints
are distributed as external release assets.

## Required assets

A full run uses four artifact groups:

1. Source code in this repository.
2. ULVN data archives containing images, descriptors, adjacency matrices, and
   task definitions.
3. Isaac Sim scene assets, or instructions for obtaining compatible GRScenes
   assets.
4. Navigation model code and checkpoints used by the local planner.

Use repository-relative paths instead of machine-specific paths.

```text
data/ulvn_sim/
data/ulvn_real/
assets/GRScenes-100/
isaacsim_pipeline/
BPL & BASS/
```

Use the layout below for downloaded or generated assets.

## Recommended public layout

```text
data/
  ulvn_sim/
    images/
    features/
    adjacency_matrix/
    tasks/
  ulvn_real/
    images/
    features/
    adjacency_matrix/
assets/
  GRScenes-100/
    home_scenes/
    commercial_scenes/
checkpoints/
  local_planner/
    nomad_ulvn.pth
    vint_ulvn.pth
```

Keep the large directories out of Git. Publish them as release assets, Zenodo,
Hugging Face datasets, or another durable storage backend, and document the
download commands with the release.

## Environments

### RAVEL / map construction

```bash
conda env create -f environment.yml
conda activate ulvn
```

### Online navigation

Create the navigation environment with Python 3.8:

```bash
conda env create -f environment-navigation.yml
conda activate ulvn_nav
```

Prepare local runtime config files:

```bash
cp config/models.example.yaml config/models.yaml
cp config/robot.example.yaml config/robot.yaml
```

`config/models.example.yaml` contains ULVN deployment-compatible entries for
`nomad_ulvn` and `vint_ulvn`. Their per-checkpoint model configs live in
`config/model_configs/`; only `ckpt_path` needs to be changed after downloading
the released weights.

Install and source ROS separately. The ROS node expects standard packages such
as `rospy`, `sensor_msgs`, `std_msgs`, `geometry_msgs`, `nav_msgs`,
`visualization_msgs`, and TF packages.

### Isaac Sim / IsaacLab

Isaac Sim should be run with IsaacLab's bundled Python, not the conda
environment:

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/RL/isaac_node.py \
  --enable_cameras \
  --scene_group GRS_home \
  --scene_idx 0
```

Use `--headless` when no viewer window is needed. Install IsaacSim-side helper
packages into IsaacLab's bundled Python when needed:

```bash
/path/to/IsaacLab/_isaac_sim/python.sh -m pip install rospkg
```

## IsaacSim startup

Start ROS first:

```bash
roscore
```

Start IsaacSim / IsaacLab node:

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/RL/isaac_node.py \
  --scene_group GRS_home \
  --scene_idx 0 \
  --enable_cameras
```

Scene group examples:

```bash
--scene_group demo --scene_idx 0
--scene_group Carla --scene_idx 7
--scene_group GRS_home --scene_idx 0
--scene_group GRS_commercial --scene_idx 0
```

`Carla` above is an IsaacLab scene-group label. The active ROS interface uses
the Isaac node topics below, not CARLA ROS bridge topics.

The Isaac node publishes and subscribes to these key ROS interfaces:

```text
Subscribed:
  /cmd_vel_mux/input/navi        geometry_msgs/Twist
  /isaac_node/set_pose_simple    geometry_msgs/Pose
  /isaac_node/command            std_msgs/String JSON command

Published:
  /isaac_node/camera0/image_raw  sensor_msgs/Image
  /isaac_node/odom               nav_msgs/Odometry
  /isaac_node/collision_detected std_msgs/Bool
  /isaac_node/collision_force    geometry_msgs/Vector3Stamped
  /isaac_node/command_status     std_msgs/String JSON status
  /tf                            odom -> base_link
```

Command examples:

```bash
rostopic pub /isaac_node/command std_msgs/String \
  "data: '{\"command\": \"start\"}'"

rostopic pub /isaac_node/command std_msgs/String \
  "data: '{\"command\": \"reset\", \"scene_group\": \"GRS_home\", \"scene_idx\": 0, \"position\": {\"x\": 1.0, \"y\": 2.0, \"z\": 0.4}, \"orientation\": {\"qx\": 0.0, \"qy\": 0.0, \"qz\": 0.0, \"qw\": 1.0}}'"
```

## Map and image generation pipeline

The IsaacSim data-generation scripts in `isaacsim_pipeline/` follow this flow:

```text
USD scene
  -> 3D occupancy voxel map
  -> 2D maximum passable area
  -> skeleton graph and uniformly spaced oriented sample points
  -> multi-direction image capture through Isaac node ROS topics
```

Included scripts:

```text
export_voxel_map.py
export_passable_area.py
export_skeleton_points.py
capture_photos.py
run_map_generation_batch.py
```

### 1. Export 3D occupancy voxels

```bash
cd /path/to/ulvn

python isaacsim_pipeline/export_voxel_map.py \
  --usd_path /path/to/scene.usd \
  --z_low 0.2 \
  --z_high 1.2 \
  --cell_size 0.05 \
  --margin 25.0 \
  --output_name scene_name \
  --output_dir isaacsim_pipeline/outputs/occupancy_data
```

Output goes to the script's occupancy output directory. The expected voxel
semantics are:

```text
100 occupied
0   free
-1  unknown
```

### 2. Export maximum passable area

```bash
python isaacsim_pipeline/export_passable_area.py \
  --npy_path /path/to/occupancy_3d_scene_name.npy \
  --z_low 0.2 \
  --z_high 1.2 \
  --robot_radius 0.3 \
  --output_dir isaacsim_pipeline/outputs/passable_area
```

### 3. Export skeleton points and orientations

```bash
python isaacsim_pipeline/export_skeleton_points.py \
  --map_path /path/to/max_passable_area.png \
  --metadata_path /path/to/metadata.json \
  --spacing 0.5 \
  --angle_interval 60 \
  --output_dir isaacsim_pipeline/outputs/skeleton_points
```

The resulting coordinate `.npy` is consumed by the capture scripts.

### 4. Capture images at skeleton samples

Start `roscore` and `isaac_node.py` first, then run:

```bash
python isaacsim_pipeline/capture_photos.py \
  --npy_path /path/to/skeleton_points.npy \
  --output_dir /path/to/output_images \
  --height 0.4 \
  --use_hover \
  --advance_distance 0.1 \
  --max_advance_attempts 5 \
  --stabilization_time 2.0
```

### 5. Batch automation

The batch script `run_map_generation_batch.py` exposes these parameters:

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

The scripts default to repository-relative `isaacsim_pipeline/outputs/` paths;
large generated files remain ignored by Git.

## Build visual topological artifacts

If using released images only, run the RAVEL pipeline from `RAVEL/README.md`.

If using released precomputed data, point BPL/BASS directly at the matching
feature and adjacency files. Released feature arrays use shape `[N, 8448]`;
adjacency matrices use shape `[N, N]`.

## Run online navigation

`launch_ulvn.sh` starts `bass.py` and `local_planner.py` in tmux:

```bash
sudo apt-get install tmux
```

```bash
cd "BPL & BASS"
ULVN_CONDA_ENV=ulvn_nav bash launch_ulvn.sh \
  --bass \
    --img_dir ../data/ulvn_sim/images/taoyuan1 \
    --descriptors_path ../data/ulvn_sim/features/data_features1.npy \
    --adjacency_matrix_path ../data/ulvn_sim/adjacency_matrix/adjacency_matrix_directional.npy \
    --image_paths_file ../data/ulvn_sim/images/taoyuan1_paths.txt \
    --goal_image_index 239 \
    --image_topic /isaac_node/camera0/image_raw \
  --nav \
    --image-topic /isaac_node/camera0/image_raw \
    --goal-path-topic /goal_path \
    --model nomad_ulvn \
    --config ../config/models.yaml \
    --robot-config ../config/robot.yaml \
    --num-samples 8 \
    --waypoint 3
```

## Reproduction checklist

1. Build or download topomap artifacts for one released scene.
2. Validate node alignment before navigation:
   ```bash
   python scripts/validate_topomap.py \
     --descriptors-path <features.npy> \
     --adjacency-matrix-path <adjacency.npy> \
     --image-paths-file <paths.txt>
   ```
3. Run offline BPL/BASS on the released artifacts:
   ```bash
   python scripts/offline_navigation_demo.py \
     --descriptors-path <features.npy> \
     --adjacency-matrix-path <adjacency.npy> \
     --start-idx 0 \
     --goal-idx <goal>
   ```
4. Prepare `config/models.yaml`, `config/robot.yaml`, and local planner
   checkpoints under `checkpoints/local_planner/`.
5. Start ROS:
   ```bash
   roscore
   ```
6. Start Isaac node and verify topics:
   ```bash
   rostopic list | grep isaac_node
   rostopic echo /isaac_node/camera0/image_raw -n 1
   rostopic echo /isaac_node/odom -n 1
   ```
7. Send `start` and `reset` commands to `/isaac_node/command`.
8. Run `capture_photos.py` on a small released skeleton-point file.
9. Run `launch_ulvn.sh` on one short released task path.

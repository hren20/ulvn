<div align="center">

# ULVN: Unordered Landmark Visual Navigation

<a href="https://arxiv.org/abs/2608.06833"><img src="https://img.shields.io/badge/arXiv-2608.06833-b31b1b" alt="arXiv"></a>
<a href="https://hren20.github.io/ulvn-website/"><img src="https://img.shields.io/badge/Project-ULVN-blue" alt="Project Page"></a>
<a href="https://github.com/hren20/ulvn"><img src="https://img.shields.io/badge/GitHub-ULVN-black?logo=github" alt="GitHub Code"></a>


ECCV 2026 (Spotlight paper with Oral presentation)

Hao Ren<sup>1</sup>,
Junzhe Zhu<sup>1</sup>,
Yihan Li<sup>1</sup>,
Zetong Bi<sup>1</sup>,
Le Zheng<sup>1</sup>,
Zhi Li<sup>1</sup>,
Yiqing Yuan<sup>1</sup>,
Zhaoliang Wan<sup>2</sup>,
Dizhe Zhang<sup>2</sup>,
Lu Qi<sup>2</sup>,
Hui Cheng<sup>1</sup>

<sup>1</sup>Sun Yat-sen University, China
<sup>2</sup>Insta360 Research, China


</div>

---

## TLDR

ULVN builds a navigable topological graph directly from unordered RGB images,
then performs vision-only localization, planning, and dynamic replanning without
odometry, depth, LiDAR, or temporal capture priors.

---

This repository contains research code for navigation from unordered visual
landmarks. It is organized around these components:

- `RAVEL/`: visual landmark graph construction from image observations.
- `ulvn_core/`: ROS-free NumPy implementations of topomap loading,
  localization, graph planning, and subgoal selection.
- `scripts/`: public command-line entry points for mapping, validation, and
  offline navigation demos.
- `isaacsim_pipeline/`: IsaacSim data-collection scripts for occupancy maps,
  passable-area maps, skeleton samples, and image capture.
- `inference_utils/`, `vint_train/`, `diffusion_policy/`: ViNT and NoMaD
  inference code used by the ROS local planner.
- `BPL & BASS/`: belief propagation localization (BPL), belief-aware subgoal
  search (BASS), and a ROS local planner interface.

This release includes the mapping, localization, planning, local-planner
inference, and IsaacSim data-generation interfaces needed to reproduce ULVN with
the separately released assets.

## Repository layout

- `RAVEL/`: visual topological map construction, including image encoding,
  FAISS retrieval, LightGlue verification, graph construction, and pruning.
- `BPL & BASS/`: BPL/BASS global planner, subgoal publisher, and ROS local
  planner entry point.
- `isaacsim_pipeline/`: IsaacSim scripts for USD scene processing, map
  extraction, skeleton sampling, and image capture.
- `ulvn_core/` and `scripts/`: ROS-free utilities and public CLI wrappers for
  validating topomaps and running offline navigation.
- `inference_utils/`, `vint_train/`, `diffusion_policy/`: ViNT/NoMaD inference
  support used by the local planner.
- `config/` and `docs/`: example runtime configs plus data-format, pipeline,
  and reproduction notes.

## Method components

### RAVEL

RAVEL builds a topological graph from a directory of landmark images:

1. Encode images into global VPR descriptors.
2. Build a FAISS index for nearest-neighbor retrieval.
3. Estimate LightGlue and VPR thresholds from the dataset.
4. Build a weighted adjacency matrix from geometrically verified matches.
5. Prune redundant edges to produce a cleaner navigation graph.

Start with `RAVEL/README.md` for the full command sequence.

The root-level wrapper `scripts/run_mapping_pipeline.py` provides the same
pipeline as one public entry point: image encoding, FAISS index creation,
LightGlue-verified adjacency construction, pruning, and final artifact
validation.

### BPL and BASS

`BPL & BASS/bass.py` contains:

- `PlaceRecognition`: belief propagation localization over the topological map.
- `BeliefAwareSubgoalSearch`: graph-based path planning and dynamic subgoal
  selection.
- A ROS node entry point that publishes the current subgoal image path.

For non-ROS testing, use `ulvn_core` and `scripts/offline_navigation_demo.py`.
Those paths only require NumPy and can be run on a machine without IsaacSim,
ROS, Torch, FAISS, or local-planner checkpoints.

`BPL & BASS/local_planner.py` consumes the subgoal image path and produces local
waypoints through ViNT or NoMaD. The inference code is included; online
execution still requires ROS, `config/models.yaml`, and matching model
checkpoints.

## Conda environments

Create the core RAVEL / analysis environment from the repository root:

```bash
conda env create -f environment.yml
conda activate ulvn
```

For CUDA acceleration, install the PyTorch build that matches your local CUDA
driver after creating the environment. For example:

```bash
conda install -n ulvn pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

Create the online navigation environment for ROS/local-planner scripts:

```bash
conda env create -f environment-navigation.yml
conda activate ulvn_nav
```

ROS packages such as `rospy`, `sensor_msgs`, `std_msgs`, and
`visualization_msgs` must be installed through your ROS distribution. ViNT and
NoMaD checkpoints are distributed outside Git; place the released files under
`checkpoints/local_planner/` or update `config/models.yaml`.

The example local-planner configs define the ULVN NoMaD and ViNT model entries:

- `config/model_configs/nomad_ulvn.yaml` for the ULVN NoMaD checkpoint.
- `config/model_configs/vint_ulvn.yaml` for the ULVN ViNT checkpoint.

Copy `config/models.example.yaml` to `config/models.yaml`, then point
`ckpt_path` to the released checkpoint files under
`checkpoints/local_planner/`.

## Model checkpoints

Large datasets, scene assets, and model checkpoints are distributed outside Git.
Fill in the ULVN asset URLs before publishing the repository:

| Asset | Destination | Download URL |
| --- | --- | --- |
| ULVN datasets | `data/ulvn_sim/`, `data/ulvn_real/` | `<ADD_DATASET_URL>` |
| GRScenes assets | `assets/GRScenes-100/` | [GRScenes on Hugging Face](https://huggingface.co/datasets/InternRobotics/GRScenes) |
| NoMaD checkpoint | `checkpoints/local_planner/nomad_ulvn.pth` | `<ADD_NOMAD_ULVN_CHECKPOINT_URL>` |
| ViNT checkpoint | `checkpoints/local_planner/vint_ulvn.pth` | `<ADD_VINT_ULVN_CHECKPOINT_URL>` |

Use the official links or loaders below to prepare third-party checkpoints.

| Component | Used by | Checkpoint source |
| --- | --- | --- |
| ULVN NoMaD local planner | `--model nomad_ulvn` | Use the ULVN NoMaD checkpoint listed in the release-asset table above, not the upstream NoMaD baseline checkpoint. |
| ULVN ViNT local planner | `--model vint_ulvn` | Use the ULVN ViNT checkpoint listed in the release-asset table above, not the upstream ViNT baseline checkpoint. |
| MegaLoc VPR encoder | `--arch megaloc` | Loaded through `torch.hub.load("gmberton/MegaLoc", "get_trained_model")`; official model page: https://huggingface.co/gberton/MegaLoc; direct weight: https://huggingface.co/gberton/MegaLoc/resolve/main/model.safetensors. |
| DINOv2 ViT-S/14 | `--arch dinov2_vits14` | Loaded through `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`; direct official weight: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth. |
| DINOv2 ViT-B/14 | `--arch dinov2_vitb14` | Loaded through `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")`; direct official weight: https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth. |
| ResNet-18 / ResNet-50 | `--arch resnet18`, `--arch resnet50` | Loaded by TorchVision `ResNet*_Weights.DEFAULT`; official weights are downloaded from PyTorch model URLs, e.g. https://download.pytorch.org/models/resnet18-f37072fd.pth and https://download.pytorch.org/models/resnet50-11ad3fa6.pth. |
| SuperPoint + LightGlue | RAVEL geometric verification | Official LightGlue release assets: https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth and https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth. |

The `--lightglue-weights indoor|outdoor` option is kept for compatibility with
earlier experiments. The current RAVEL path initializes
`LightGlue(features="superpoint")`, whose vendored implementation downloads the
official SuperPoint and SuperPoint-LightGlue weights listed above.

## Quick start: build a topological map

```bash
conda activate ulvn

python scripts/run_mapping_pipeline.py \
  --img-dir data/landmark \
  --work-dir outputs/example_topomap/work \
  --result-dir outputs/example_topomap/results \
  --arch megaloc \
  --lightglue-weights outdoor
```

Manual RAVEL commands are documented in `RAVEL/README.md`.

## Quick start: validate and run offline navigation

Run BPL/BASS over a real topomap artifact set:

```bash
python scripts/validate_topomap.py \
  --descriptors-path outputs/example_topomap/work/data_features.npy \
  --adjacency-matrix-path outputs/example_topomap/results/pruned_adjacency_matrix.npy \
  --image-paths-file outputs/example_topomap/work/data_paths.txt

python scripts/offline_navigation_demo.py \
  --descriptors-path outputs/example_topomap/work/data_features.npy \
  --adjacency-matrix-path outputs/example_topomap/results/pruned_adjacency_matrix.npy \
  --image-paths-file outputs/example_topomap/work/data_paths.txt \
  --start-idx 0 \
  --goal-idx 20 \
  --n-ahead 2
```

See `docs/PIPELINE.md` for the complete mapping, validation, offline planning,
ROS, and IsaacSim interface contract.

## Quick start: run BPL/BASS online

Prepare the files described in `docs/DATA_FORMATS.md`, then create local config
files from the examples:

```bash
cp config/robot.example.yaml config/robot.yaml
cp config/models.example.yaml config/models.yaml
```

For IsaacSim experiments, `config/robot_isaac.example.yaml` matches the tested
simulator velocity topics and limits.

Launch the global planner node:

```bash
conda activate ulvn

python "BPL & BASS/bass.py" \
  --img_dir data/landmark \
  --descriptors_path continuous_images/global-feats-test.npy \
  --adjacency_matrix_path continuous_images/adjacency_matrix.npy \
  --image_paths_file continuous_images/image_paths.txt \
  --goal_image_index 0 \
  --arch megaloc \
  --image_topic /isaac_node/camera0/image_raw
```

Launch the local planner node after ROS, model configs, and checkpoints are
available:

```bash
conda activate ulvn

python "BPL & BASS/local_planner.py" \
  --model nomad_ulvn \
  --config config/models.yaml \
  --robot-config config/robot.yaml \
  --num-samples 8 \
  --waypoint 3 \
  --image-topic /isaac_node/camera0/image_raw \
  --goal-path-topic /goal_path
```

## Data

After downloading the released archives, place them under the layouts documented
in `docs/REPRODUCTION.md`.

Expected file formats and node-ordering conventions are documented in
`docs/DATA_FORMATS.md`.

The end-to-end IsaacSim and navigation reproduction path is documented in
`docs/REPRODUCTION.md`.

The IsaacSim data-collection scripts are vendored in `isaacsim_pipeline/`; see
`isaacsim_pipeline/README.md` for the USD -> occupancy -> passable area ->
skeleton samples -> image capture workflow.

## Citation

```bibtex
@article{ren2026unordered,
  title   = {Unordered Landmark Visual Navigation},
  author  = {Ren, Hao and Zhu, Junzhe and Li, Yihan and Bi, Zetong and Zheng, Le and Li, Zhi and Yuan, Yiqing and Wan, Zhaoliang and Zhang, Dizhe and Qi, Lu and Cheng, Hui},
  journal = {arXiv preprint arXiv:2608.06833},
  year    = {2026}
}
```

## License

This project is released under the MIT License. See `LICENSE`.

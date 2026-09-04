# Data Formats

This repository expects image datasets and generated graph artifacts to follow
the conventions below.

## RAVEL inputs

`RAVEL/scripts/encode_images.py` recursively scans an image directory.
Supported image extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`.

Example input layout:

```text
data/landmark/
  node_000.png
  node_001.png
  node_002.png
```

## RAVEL generated artifacts

`landmarks_features.npy` or `data_features.npy`:

- NumPy array with shape `[N, D]`.
- `float32` descriptors.
- Rows are L2-normalized by the encoder.

`landmarks_paths.txt` or `data_paths.txt`:

- UTF-8 text file.
- One image path per line.
- Line `i` maps to row `i` in the feature array and node `i` in the graph.

`adjacency_matrix.npy`:

- NumPy array with shape `[N, N]`.
- `0` means no edge.
- Positive values are LightGlue match counts, unless `--binary_matrix` was used,
  in which case positive values are `1`.

`pruned_adjacency_matrix.npy`:

- NumPy array with shape `[N, N]`.
- Produced by `RAVEL/scripts/prune_topomap_nx.py` from a weighted adjacency
  matrix.

## BPL and BASS inputs

`BPL & BASS/bass.py` expects:

- `--descriptors_path`: node descriptor array, shape `[N, D]`.
- `--adjacency_matrix_path`: graph adjacency matrix, shape `[N, N]`.
- `--image_paths_file`: one image path per line, where line `i` corresponds to
  graph node `i`.
- `--img_dir`: base image directory used when `image_paths_file` contains
  relative paths.

The descriptor rows, adjacency rows/columns, and image path lines must use the
same node ordering.

## Released dataset layout

Released data uses this repository-relative structure:

```text
data/
  ulvn_sim/
    images/
      taoyuan1/
      taoyuan2/
      taoyuan3/
      taoyuan4/
      taoyuan6/
      taoyuan_commercial_2/
      taoyuan_commercial_4/
      taoyuan_commercial_5/
      taoyuan_commercial_6/
      taoyuan_commercial_7/
    features/
    adjacency_matrix/
    tasks/
  ulvn_real/
    images/
      roundabout/
      street1/
      street2/
    features/
    adjacency_matrix/
```

Released dataset dimensions:

```text
Simulation:
  taoyuan1:             features [540, 8448], adjacency [540, 540]
  taoyuan2:             features [226, 8448], adjacency [226, 226]
  taoyuan3:             features [332, 8448], adjacency [332, 332]
  taoyuan4:             features [216, 8448], adjacency [216, 216]
  taoyuan6:             features [342, 8448], adjacency [342, 342]
  taoyuan_commercial_2: features [474, 8448], adjacency [474, 474]
  taoyuan_commercial_4: features [398, 8448], adjacency [398, 398]
  taoyuan_commercial_5: features [298, 8448], adjacency [298, 298]
  taoyuan_commercial_6: features [388, 8448], adjacency [388, 388]
  taoyuan_commercial_7: features [380, 8448], adjacency [380, 380]

Real robot:
  roundabout: features [71, 8448], adjacency [71, 71]
  street1:    features [60, 8448], adjacency [60, 60]
  street2:    features [53, 8448], adjacency [53, 53]
```

Simulation task-path image directories follow this naming convention:

```text
tasks/path_images_<scene_id>_<path_id>/
```

Each task image name follows:

```text
<path_step_index>_<origin_topomap_node_index>.png
```

For example, `1_19.png` means path step `1`, whose source topomap node index is
`19`.

Task-length `.npy` files should store rows in this format:

```text
[start_node_index, goal_node_index, path_length]
```

## ROS topics

`bass.py` subscribes to the camera image topic and publishes:

- `/goal_path` (`std_msgs/String`): current subgoal image path.
- `/topoplan/reached_goal` (`std_msgs/Bool`): goal reached flag.

`local_planner.py` subscribes to `/goal_path` and the camera image topic, then
publishes local waypoints and visualization messages. Topic names are exposed as
CLI arguments.

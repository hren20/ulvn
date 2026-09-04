from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image as PILImage


def msg_to_pil(msg: Any) -> PILImage.Image:
    """Convert a ROS-like Image message into an RGB PIL image."""

    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    channels = img.shape[2]
    encoding = getattr(msg, "encoding", "").lower()

    if channels == 3:
        if encoding.startswith("bgr"):
            img = img[..., ::-1]
    elif channels == 4:
        if encoding.startswith("bgra"):
            img = img[..., [2, 1, 0, 3]]
        img = img[..., :3]
    elif channels == 1:
        img = np.repeat(img, 3, axis=2)
    else:
        raise ValueError(f"Unsupported image channel count: {channels}")

    return PILImage.fromarray(np.ascontiguousarray(img), mode="RGB")


def create_marker_from_points(
    points: Sequence[np.ndarray],
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    scale: float = 0.1,
    frame_id: str = "base_link",
    z_value: float = 0.0,
    marker_id: int = 0,
    namespace: str = "points",
    enforce_eight_points: bool = True,
):
    """Create an RViz POINTS marker from 2D points.

    ROS imports are intentionally delayed so the package can be imported on
    non-ROS machines for config checks and offline tests.
    """

    try:
        import rospy
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker
    except ImportError as exc:
        raise RuntimeError("ROS visualization packages are required to create markers.") from exc

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.POINTS
    marker.action = Marker.ADD
    marker.scale.x = scale
    marker.scale.y = scale
    marker.color.a = 1.0
    marker.color.r = float(color[0])
    marker.color.g = float(color[1])
    marker.color.b = float(color[2])

    pts = np.asarray(points, dtype=np.float32)
    if enforce_eight_points and len(pts) < 8:
        pts = np.vstack([pts, np.zeros((8 - len(pts), 2), dtype=np.float32)])

    for pt in pts:
        p = Point()
        p.x = float(pt[0])
        p.y = float(pt[1])
        p.z = float(z_value)
        marker.points.append(p)
    return marker


def load_config(model_key: str, config_path: str):
    """Load a model config and checkpoint path from config/models.yaml.

    Supported schemas:

    1. Preferred release schema:
       model_key:
         config_path: path/to/model_config.yaml
         ckpt_path: path/to/checkpoint.pth

    2. Inline schema:
       model_key:
         model_type: nomad
         checkpoint: path/to/checkpoint.pth
         ... model config fields ...
    """

    root = Path(config_path).resolve().parent
    with Path(config_path).open("r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f) or {}

    if model_key not in full_config:
        raise KeyError(f"Model key '{model_key}' not found in {config_path}")

    model_info = dict(full_config[model_key] or {})
    ckpt_path = model_info.get("ckpt_path") or model_info.get("checkpoint")
    if not ckpt_path:
        raise KeyError(f"Model key '{model_key}' must define ckpt_path or checkpoint")

    ckpt_path = str((root / ckpt_path).resolve()) if not Path(str(ckpt_path)).is_absolute() else str(ckpt_path)

    if "config_path" in model_info:
        model_config_path = Path(model_info["config_path"])
        if not model_config_path.is_absolute():
            model_config_path = root / model_config_path
        with model_config_path.open("r", encoding="utf-8-sig") as f:
            model_config = yaml.safe_load(f) or {}
        for key, value in model_info.items():
            if key not in {"config_path", "ckpt_path", "checkpoint"}:
                model_config[key] = value
    else:
        model_config = {k: v for k, v in model_info.items() if k not in {"ckpt_path", "checkpoint"}}

    if "model_type" not in model_config:
        model_config["model_type"] = model_key
    return model_config, ckpt_path


def inference_config_init(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    config = dict(config)
    config["device"] = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    config["train"] = False
    config["num_samples"] = int(getattr(args, "num_samples", config.get("num_samples", 8)))
    config["close_threshold"] = int(getattr(args, "close_threshold", config.get("close_threshold", 3)))
    return config


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()

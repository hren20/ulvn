# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import heapq
import os
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    _torch_no_grad = torch.no_grad
except ImportError:
    torch = None
    F = None
    transforms = None

    class _MissingModuleBase:
        pass

    class _MissingNN:
        Module = _MissingModuleBase

    nn = _MissingNN()

    def _torch_no_grad():
        def decorator(func):
            return func
        return decorator
import argparse
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import sys
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from RAVEL.scripts.encode_images import DINOv2Embeddings, ResNetEmbeddings
except ImportError:
    DINOv2Embeddings = None
    ResNetEmbeddings = None

try:
    import rospy
    from sensor_msgs.msg import Image as ROSImage
    from std_msgs.msg import Bool, String
except ImportError:
    rospy = None
    ROSImage = None
    Bool = None
    String = None

try:
    from inference_utils.common import msg_to_pil
except ImportError:
    msg_to_pil = None


def _require_bass_runtime_dependencies():
    missing = []
    if torch is None or transforms is None:
        missing.append("PyTorch and TorchVision")
    if rospy is None or ROSImage is None or Bool is None or String is None:
        missing.append("ROS Python packages: rospy, sensor_msgs, std_msgs")
    if msg_to_pil is None:
        missing.append("inference_utils.common.msg_to_pil")
    if missing:
        raise RuntimeError(
            "bass.py requires additional runtime dependencies for the ROS node: "
            + "; ".join(missing)
            + ". Install/provide them before launching the online navigation node."
        )

class VPRModelBase:
    def encode(self, image) -> np.ndarray:
        raise NotImplementedError

class MyVPR(VPRModelBase):
    def __init__(self, model: nn.Module, transform: transforms.Compose, device: torch.device):
        self.model = model.to(device).eval()
        self.transform = transform
        self.device = device

    @_torch_no_grad()
    def encode(self, image: Image.Image) -> torch.Tensor:
        img_rgb = image.convert("RGB")
        tensor = self.transform(img_rgb)

        tensor = tensor.unsqueeze(0).to(self.device, non_blocking=True)

        features = self.model(tensor)

        features = F.normalize(features, p=2, dim=1)
        return features.squeeze(0).cpu().numpy()

def load_vpr_encoder(
    arch: str = "megaloc",
    img_size: Tuple[int, int] = (320, 320),
    pretrained: bool = True,
    device_str: str = "auto",
    dinov2_local: Optional[str] = None,
    hub_force_reload: bool = False,
) -> MyVPR:
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    model = build_encoder(
        arch=arch,
        pretrained=pretrained,
        dinov2_local=dinov2_local,
        hub_force_reload=hub_force_reload,
    )
    
    transform = build_transform(img_size[0], img_size[1], arch)
    
    print(f"Loading VPR Encoder '{arch}' to device: {device}")
    
    return MyVPR(model, transform, device)


class PlaceRecognition:
    def __init__(self, descriptors: np.ndarray, adjacency_matrix: np.ndarray, max_steps=3, delta=5):
        self.descriptors = descriptors
        self.adjacency_matrix = np.array(adjacency_matrix)
        
        self.max_steps = max_steps
        self.delta = delta
        
        self.transition_matrix = self._build_transition_matrix()
        
        self.belief = None

    def _build_transition_matrix(self):
        n = self.adjacency_matrix.shape[0]
        cumulative_matrix = np.eye(n)
        current_power = np.eye(n)

        for step in range(1, self.max_steps + 1):
            current_power = current_power @ self.adjacency_matrix
            cumulative_matrix += current_power

        row_sums = cumulative_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return cumulative_matrix / row_sums

    def _compute_belief_entropy(self):
        belief_safe = np.clip(self.belief, 1e-10, 1.0)
        entropy = -np.sum(self.belief * np.log(belief_safe))
        max_entropy = np.log(len(self.belief))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        return normalized_entropy

    def _get_fusion_weights(self, entropy):
        if entropy > 0.7:
            w_prediction = 0.3
            w_observation = 0.7
        elif entropy < 0.3:
            w_prediction = 0.6
            w_observation = 0.4
        else:
            w_prediction = 0.6 - 0.75 * (entropy - 0.3)
            w_observation = 1.0 - w_prediction
        return w_prediction, w_observation

    def initialize_model(self, query_desc: np.ndarray):
        """
        Initialize the filter belief distribution based on the input goal descriptor.
        """
        dists = safe_cosine_distance(self.descriptors, query_desc)  # Compute distance between the query and every node

        # Compute the lambda1 parameter used by the likelihood function
        descriptor_quantiles = np.quantile(dists, [0.025, 0.975])
        if descriptor_quantiles[1] == descriptor_quantiles[0]:
            self.lambda1 = 1.0
        else:
            self.lambda1 = np.log(self.delta) / (descriptor_quantiles[1] - descriptor_quantiles[0])

        self.belief = np.exp(-self.lambda1 * dists)  # Update the belief distribution
        self.belief /= self.belief.sum()  # Normalize the belief distribution

    def obs_lhood(self, descriptor: np.ndarray) -> np.ndarray:
        """
        Compute observation likelihood measuring similarity between an observation and database nodes.
        """
        return safe_obs_likelihood(self.descriptors, descriptor, self.lambda1)

    def match(self, query_desc: np.ndarray) -> Tuple[int, float]:
        """
        Match a query descriptor to the topological graph and return the most likely node index and confidence.
        """
        # Compute the entropy of the current belief distribution
        entropy = self._compute_belief_entropy()

        # Adjust fusion weights based on entropy
        w_pred, w_obs = self._get_fusion_weights(entropy)

        # Prediction step
        belief_prior = self.belief @ self.transition_matrix
        belief_prior /= belief_prior.sum()  # Normalize

        # Compute observation likelihood
        obs_lhood = self.obs_lhood(query_desc)
        obs_lhood /= obs_lhood.sum()  # Normalize to a probability distribution

        # Fuse prediction and observation with adaptive weights
        self.belief = np.power(belief_prior, w_pred) * np.power(obs_lhood, w_obs)
        self.belief /= self.belief.sum()

        # Return the most likely node and its confidence
        max_bel = np.argmax(self.belief)
        score = self.belief[max_bel]

        return max_bel, score

    def external_localizer_index(self, query_desc: np.ndarray) -> int:
        """
        Return the most likely node index for the current observation descriptor using belief updates.
        """
        # Execute matching and return the most probable node index
        pred_node, _ = self.match(query_desc)

        return pred_node


# Safe cosine distance computation helper
def safe_cosine_distance(descriptors: np.ndarray, query_desc: np.ndarray) -> np.ndarray:
    """
    Compute cosine distance while keeping numerical stability in check.
    """
    cos_sim = np.dot(descriptors, query_desc)
    # Clamp cosine similarity to the valid range
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    # Convert to distance and ensure non-negative values
    distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cos_sim))
    return distances

def safe_obs_likelihood(descriptors: np.ndarray, query_desc: np.ndarray, lambda1: float) -> np.ndarray:
    """
    Compute observation likelihood in a numerically stable way.
    """
    distances = safe_cosine_distance(descriptors, query_desc)
    return np.exp(-lambda1 * distances)

def _patch_pep604_to_optional(root: Path) -> int:
    """
    Replace Python 3.10+ union syntax such as 'float | None' with Optional[...] across files.
    Automatically adds 'from typing import Optional' if it is missing.
    Returns the number of modified files.
    """
    if not root.exists():
        return 0

    n_modified = 0
    for py in root.rglob("*.py"):
        try:
            text = py.read_text(encoding="utf-8")
        except Exception:
            continue

        original = text
        # Ensure Optional is imported
        if "from typing import Optional" not in text:
            text = "from typing import Optional\n" + text

        # Replace common union patterns (extendable)
        text = re.sub(r"\bfloat\s*\|\s*None\b", "Optional[float]", text)
        text = re.sub(r"\bint\s*\|\s*None\b",   "Optional[int]",   text)
        text = re.sub(r"\bbool\s*\|\s*None\b",  "Optional[bool]",  text)
        text = re.sub(r"\bstr\s*\|\s*None\b",   "Optional[str]",   text)

        if text != original:
            py.write_text(text, encoding="utf-8")
            n_modified += 1
    return n_modified


class Dinov2CompatHubLoader:
    """
    Context manager that temporarily wraps torch.hub.load to keep dinov2 compatible with Python < 3.10.
    - When loading 'facebookresearch/dinov2' fails due to union-type syntax, patch the local sources and retry.
    - Optionally prioritizes loading from a local dinov2 clone by setting local_dir (source='local').
    """

    def __init__(self, local_dir: Optional[str] = None, force_reload: bool = False):
        self.local_dir = Path(local_dir).resolve() if local_dir else None
        self.force_reload = force_reload
        self._orig_load = None

    def __enter__(self):
        self._orig_load = torch.hub.load

        def _wrapped_load(repo_or_dir, model, *args, **kwargs):
            is_dinov2 = False
            # Detect several common repo descriptions
            if isinstance(repo_or_dir, str):
                low = repo_or_dir.lower()
                is_dinov2 = ("facebookresearch/dinov2" in low) or ("dinov2" == Path(low).name)

            # Non-dinov2 requests go straight through
            if not is_dinov2:
                return self._orig_load(repo_or_dir, model, *args, **kwargs)

            # Python >= 3.10 does not need the compatibility shim
            if sys.version_info >= (3, 10):
                return self._orig_load(repo_or_dir, model, *args, **kwargs)

            # Python < 3.10: patch and retry if union syntax triggers TypeError
            try:
                return self._orig_load(repo_or_dir, model, *args, **kwargs)
            except TypeError as e:
                msg = str(e)
                if "unsupported operand type(s) for |" not in msg:
                    raise  # Not a union syntax error; re-raise

                # Choose a directory to patch: prefer local clone, otherwise torch.hub cache
                target_dir = None
                if self.local_dir and self.local_dir.exists():
                    target_dir = self.local_dir
                else:
                    # For example ~/.cache/torch/hub/facebookresearch_dinov2_main/dinov2
                    hub_dir = Path(torch.hub.get_dir())
                    # Support multiple plausible branch names
                    for folder in ["facebookresearch_dinov2_main", "facebookresearch_dinov2_master", "facebookresearch_dinov2"]:
                        candidate = hub_dir / folder / "dinov2"
                        if candidate.exists():
                            target_dir = candidate
                            break

                if target_dir is None:
                    # If cache is missing, trigger a fetch once before patching
                    _ = self._orig_load(repo_or_dir, model, *args, **kwargs)
                    hub_dir = Path(torch.hub.get_dir())
                    for folder in ["facebookresearch_dinov2_main", "facebookresearch_dinov2_master", "facebookresearch_dinov2"]:
                        candidate = hub_dir / folder / "dinov2"
                        if candidate.exists():
                            target_dir = candidate
                            break

                if target_dir is None:
                    raise RuntimeError(
                        "Cannot locate the dinov2 source directory for compatibility patching."
                        "Provide --dinov2-local or ensure the torch.hub cache is writable."
                    )

                n = _patch_pep604_to_optional(target_dir)
                print(f"[Dinov2Compat] Patched {n} file(s) under: {target_dir}")

                # Reload and optionally force refresh
                kwargs2 = dict(kwargs)
                if self.force_reload:
                    kwargs2["force_reload"] = True

                # Enforce local loading if a local directory was provided
                if self.local_dir and self.local_dir.exists():
                    kwargs2["source"] = "local"
                    repo_or_dir2 = str(self.local_dir)
                else:
                    repo_or_dir2 = repo_or_dir

                return self._orig_load(repo_or_dir2, model, *args, **kwargs2)

        torch.hub.load = _wrapped_load
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig_load is not None:
            torch.hub.load = self._orig_load
        return False  # Propagate any exception



class MegaLocEmbeddings(nn.Module):
    """MegaLoc wrapper for feature extraction.

    Loads the SOTA VPR model from torch.hub and L2-normalizes its output.

    Attributes:
        arch (str): The model architecture (just "megaloc").
        backbone (nn.Module): The MegaLoc model.
        out_dim (int): The output feature dimension (fixed at 512).
    """
    def __init__(self, arch: str = "megaloc",
                 dinov2_local: Optional[str] = None,
                 hub_force_reload: bool = False):
        super().__init__()
        if arch != "megaloc":
            raise ValueError("This class only supports 'megaloc'.")

        self.arch = arch
        print("Loading 'MegaLoc' model from torch.hub 'gmberton/MegaLoc'...")

        # Wrap hub.load to ensure dinov2 is patched automatically when needed
        with Dinov2CompatHubLoader(
            local_dir=dinov2_local,
            force_reload=hub_force_reload,
        ):
            last_err = None
            for entry in ("get_trained_model", "megaloc"):
                try:
                    # MegaLoc repo commonly exposes get_trained_model with pretrained weights
                    # Some branches may call the entry "megaloc", so try both
                    self.backbone = torch.hub.load('gmberton/MegaLoc', entry)
                    print(f"[MegaLoc] Loaded entry: {entry}")
                    break
                except Exception as e:
                    last_err = e
                    print(f"[MegaLoc] Fallback: entry '{entry}' not usable: {e}")
                    self.backbone = None

            if self.backbone is None:
                raise RuntimeError(
                    "Failed to load MegaLoc from gmberton/MegaLoc. "
                    "Tried entries: ['get_trained_model','megaloc']"
                ) from last_err



        # MegaLoc paper specifies a 512-dimensional output
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, returns L2-normalized descriptor."""
        feats = self.backbone(x)  # (N, D)
        feats = F.normalize(feats, p=2, dim=1)  # L2 normalize
        return feats
    

def build_encoder(arch: str = "megaloc", pretrained: bool = True,
                  dinov2_local: Optional[str] = None,
                  hub_force_reload: bool = False) -> nn.Module:
    """
    Factory function to build the raw nn.Module encoder.

    Args:
        arch: The architecture name (e.g., "resnet18", "megaloc").
        pretrained: Whether to load pretrained weights.

    Returns:
        A torch.nn.Module instance (the encoder).
    """
    arch = arch.lower()

    if arch in ["resnet18", "resnet50"]:
        if ResNetEmbeddings is None:
            raise RuntimeError(
                "ResNet encoder support requires RAVEL/scripts/encode_images.py "
                "to be available from the repository root."
            )
        print(f"Building torchvision model: {arch}")
        return ResNetEmbeddings(arch=arch, pretrained=pretrained)

    elif arch.startswith("dinov2_"):
        if DINOv2Embeddings is None:
            raise RuntimeError(
                "DINOv2 encoder support requires RAVEL/scripts/encode_images.py "
                "to be available from the repository root."
            )
        print(f"Building DINOv2 model: {arch}")
        if not pretrained:
            print(
                "WARN: DINOv2 is always pretrained. '--no_pretrained' is ignored.",
                file=sys.stderr
            )
        return DINOv2Embeddings(arch=arch)

    elif arch == "megaloc":
        print(f"Building MegaLoc SOTA VPR model: {arch}")
        if not pretrained:
            print("WARN: MegaLoc is always pretrained. '--no_pretrained' is ignored.", file=sys.stderr)
        return MegaLocEmbeddings(arch=arch,
                                 dinov2_local=dinov2_local,
                                 hub_force_reload=hub_force_reload)

    else:
        raise ValueError(f"Unknown architecture: {arch}")


def build_transform(
    img_w: int, img_h: int, arch: str = "megaloc"
) -> transforms.Compose:
    """Builds the image preprocessing pipeline based on the model arch.

    Args:
        img_w: Target image width.
        img_h: Target image height.
        arch: The model architecture (to select correct normalization).

    Returns:
        torchvision.transforms.Compose: The transformation pipeline.
    """
    arch = arch.lower()
    
    # All models listed (ResNet, DINOv2, MegaLoc) use standard ImageNet stats.
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if arch.startswith("dinov2_"):
        print("Using ImageNet normalization constants for DINOv2.")
        if img_w < 224 or img_h < 224:
            print(
                f"WARN: DINOv2 performs best at 224x224 or larger, "
                f"but running at {img_w}x{img_h}.",
                file=sys.stderr
            )
    elif arch == "megaloc":
        print("Using ImageNet normalization constants for MegaLoc.")
        # MegaLoc paper suggests training at 320x320
        if (img_w, img_h) != (320, 320):
             print(
                f"INFO: MegaLoc was trained at 320x320, "
                f"but running at {img_w}x{img_h}.",
                file=sys.stderr
            )
    else:
        # Default (ResNet)
        print("Using ImageNet normalization constants for ResNet.")

    return transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

# -------------------------------
# BASS: Satisfy the three required functions
# -------------------------------
@dataclass
class PlanState:
    goal_idx: Optional[int] = None
    start_idx: Optional[int] = None
    path: List[int] = None   # Planned route as node indices
    edge_cost: str = "unweighted"


class BeliefAwareSubgoalSearch:
    """
    - initial_localize(goal_image, obs_image) -> (goal_idx, start_idx)
    - plan_path(start_idx, goal_idx, edge_cost="unweighted") -> List[int], storing the path in self.state
    - next_subgoal(current_idx, n_ahead=1, replan_if_offpath=True) -> dict {subgoal_idx, replanned, path}
    """
    def __init__(
        self,
        # graph: TopoGraph,
        adjacency_matrix: np.ndarray,
        node_embeddings: np.ndarray,   # Shape [N, d], ideally already L2-normalized
        vpr_encoder,       # Must expose encode(image)->embedding
        l2_normalize_inputs: bool = True
    ):
        # self.G = graph
        self.W = np.array(adjacency_matrix, dtype=np.float32)
        self.N = self.W.shape[0]
        self.Z = np.asarray(node_embeddings, dtype=np.float32)
        assert self.Z.ndim == 2 and self.Z.shape[0] == self.N, "Number of graph nodes does not match descriptor count"
        self.d = self.Z.shape[1]
        self.vpr = vpr_encoder
        self.norm_inputs = l2_normalize_inputs
        self.state = PlanState(path=[])

        # Enforce L2 normalization to make cosine similarity reliable
        Z_norm = np.linalg.norm(self.Z, axis=1, keepdims=True) + 1e-12
        self.Z = (self.Z / Z_norm).astype(np.float32)

    # ---------- 1) Initial localization ----------
    def initial_localize(self, goal_image, obs_image) -> Tuple[int, int]:
        """
        Use VPR to localize goal and starting nodes in the topological graph.
        Returns (goal_idx, start_idx) and updates planner state.
        """
        z_g = self._to_embedding(goal_image)
        z_o = self._to_embedding(obs_image)

        # Z is L2-normalized, so the dot product equals cosine similarity.
        goal_idx = int(np.argmax(self.Z @ z_g))
        start_idx = int(np.argmax(self.Z @ z_o))

        self.state.goal_idx = goal_idx
        self.state.start_idx = start_idx
        return goal_idx, start_idx

    # ---------- 2) Dijkstra path planning ----------
    def plan_path(
        self,
        start_idx: Optional[int] = None,
        goal_idx: Optional[int] = None,
        edge_cost: str = "unweighted"
    ) -> List[int]:
        """
        Run Dijkstra shortest path and store the resulting path inside self.state.
        - start_idx/goal_idx fallback to indices from initial_localize if omitted.
        - edge_cost: "unweighted" (default) or "inverse_weight".
        """
        if start_idx is None:
            start_idx = self.state.start_idx
        if goal_idx is None:
            goal_idx = self.state.goal_idx
        if start_idx is None or goal_idx is None:
            raise ValueError("start_idx / goal_idx not set; run initial_localize or pass indices explicitly.")

        path = self._dijkstra(int(start_idx), int(goal_idx), edge_cost=edge_cost)
        self.state.start_idx = int(start_idx)
        self.state.goal_idx  = int(goal_idx)
        self.state.path = path
        self.state.edge_cost = edge_cost
        return path

    def _dijkstra(
        self, 
        start: int, 
        goal: int, 
        edge_cost: str = "unweighted"
    ) -> List[int]:
        """
        Dijkstra shortest path algorithm.

        Args:
            start: start node index
            goal: goal node index
            edge_cost: edge cost mode

        Returns:
            Path [start, ..., goal], or [] if unreachable.
        """
        if start == goal:
            return [start]
        
        # Edge cost helper
        def _edge_len(weight: float) -> float:
            if edge_cost == "unweighted":
                return 1.0
            elif edge_cost == "inverse_weight":
                return 1.0 / max(float(weight), 1e-6)
            else:
                raise ValueError(f"Unknown edge_cost: {edge_cost}")
        
        # Initialize Dijkstra data structures
        dist = {start: 0.0}
        prev = {start: None}
        pq = [(0.0, start)]
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if u == goal:
                break
            
            if d > dist.get(u, float('inf')):
                continue
            
            # Iterate over neighbors with positive weight
            for v in range(self.N):
                if self.W[u, v] > 0:  # Edge exists
                    nd = d + _edge_len(self.W[u, v])
                    if nd < dist.get(v, float('inf')):
                        dist[v] = nd
                        prev[v] = u
                        heapq.heappush(pq, (nd, v))
        
        # Unreachable goal
        if goal not in prev:
            return []
        
        # Backtrack to reconstruct the path
        path = [goal]
        while path[-1] != start:
            path.append(prev[path[-1]])
        path.reverse()
        
        return path

    # ---------- 3) Navigation logic (subgoal selection + replanning) ----------
    def next_subgoal(
        self,
        current_idx: int,
        n_ahead: int = 1,
        replan_if_offpath: bool = True
    ) -> Dict:
        """
        Advance navigation:
          - If current_idx lies outside the stored path and replanning is allowed, replan from current_idx to goal_idx.
          - If current_idx is on the stored path, choose the node n_ahead steps ahead (bounded by goal).
        Returns:
          {
            "subgoal_idx": Optional[int],  # Subgoal to follow (None if unreachable/no path)
            "replanned": bool,            # Whether replanning occurred
            "path": List[int],            # Latest full path
            "whereami_on_path": Optional[int],  # Index of current node on the path if applicable
          }
        """
        if self.state.goal_idx is None:
            raise ValueError("Goal index not set; call initial_localize / plan_path first.")

        # If no path exists yet or it is empty, compute one first
        if not self.state.path:
            self.plan_path(start_idx=current_idx, goal_idx=self.state.goal_idx, edge_cost=self.state.edge_cost)

        path = self.state.path
        replanned = False

        # Map current_idx onto the stored path
        whereami = None
        pos_map = {node: i for i, node in enumerate(path)}
        if current_idx in pos_map:
            whereami = pos_map[current_idx]
        else:
            # Replan when the current node falls outside the stored path
            if replan_if_offpath:
                path = self.plan_path(start_idx=current_idx, goal_idx=self.state.goal_idx, edge_cost=self.state.edge_cost)
                replanned = True
                pos_map = {node: i for i, node in enumerate(path)}
                whereami = pos_map.get(current_idx, None)

        # No route available after attempting to plan
        if not path:
            return dict(subgoal_idx=None, replanned=replanned, path=[], whereami_on_path=None)

        # Choose a subgoal: take the node n steps ahead along the path.
        # If current_idx is not on the path (e.g., due to localization noise), fall back to the closest path node.
        if whereami is None:
            # Fallback strategy: find the path node with minimal geodesic distance
            anchor_idx = self._nearest_on_path(current_idx, path)
            whereami = anchor_idx

        # Subgoal index = min(whereami + n_ahead, len(path)-1)
        j = min(whereami + max(1, int(n_ahead)), len(path) - 1)
        subgoal_idx = int(path[j])

        # If current_idx sits beyond the proposed subgoal, bump subgoal forward by one step
        if whereami >= j and whereami < len(path) - 1:
            j = min(whereami + 1, len(path) - 1)
            subgoal_idx = int(path[j])

        # Update planner start index so subsequent planning uses the latest position
        self.state.start_idx = int(current_idx)
        self.state.path = path
        return dict(subgoal_idx=subgoal_idx, replanned=replanned, path=list(path), whereami_on_path=whereami)

    # ---------- Helpers ----------
    def _to_embedding(self, img_or_feat) -> np.ndarray:
        """
        Accept a 1D feature vector or an image object.
        Returns a column vector of shape [d, 1] with L2 normalization.
        """
        if isinstance(img_or_feat, np.ndarray) and img_or_feat.ndim == 1:
            z = img_or_feat.astype(np.float32)
        else:
            z = np.asarray(self.vpr.encode(img_or_feat), dtype=np.float32)
        if self.norm_inputs:
            z = z / (np.linalg.norm(z) + 1e-12)
        return z.reshape(-1, 1)

    def _nearest_on_path(self, node_idx: int, path: List[int]) -> int:
        """
        Find the path index whose node is closest in graph distance to node_idx.
        """
        # Simple implementation: fast enough for small-degree graphs; cache APSP for faster lookups if needed.
        best_j, best_d = 0, float("inf")
        for j, u in enumerate(path):
            d = self._geodesic_distance_quick(node_idx, u)
            if d < best_d:
                best_d = d
                best_j = j
        return best_j

    def _geodesic_distance_quick(self, a: int, b: int, cutoff: Optional[int] = None) -> int:
        """
        Approximate unweighted BFS distance for local planning decisions.
        """
        a, b = int(a), int(b)
        if a == b:
            return 0
        visited = {a}
        q = [(a, 0)]
        while q:
            u, d = q.pop(0)
            nd = d + 1
            # for v in self.G.neighbors(u).keys():
            for v in range(self.N):
                if v == b:
                    return nd
                if v not in visited:
                    visited.add(v)
                    if cutoff is None or nd <= cutoff:
                        q.append((v, nd))
        return int(1e9)  # Treat as a large number meaning unreachable


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the BPL/BASS global planner node.

    Returns:
        argparse.Namespace: The populated namespace with arguments.
    """
    ap = argparse.ArgumentParser(
        description="BPL/BASS global planner node for ULVN online navigation."
    )

    # --- Input Configuration ---
    ap.add_argument(
        "--img_dir",
        type=str,
        default="data",
        help="Directory containing images (searched recursively). Default: ./data"
    )
    

    ap.add_argument(
        "--descriptors_path",
        type=str,
        default="continuous_images/global-feats-test.npy",
        help="Path to node descriptors file (.npy). Default: continuous_images/global-feats-test.npy"
    )

    ap.add_argument(
        "--adjacency_matrix_path",
        type=str,
        default="continuous_images/adjacency_matrix.npy",
        help="Path to adjacency matrix file (.npy). Default: continuous_images/adjacency_matrix.npy"
    )

    ap.add_argument(
        "--goal_image_index",
        type=int,
        default=0,
        help="Index of goal image. Default: 0"
    )

    ap.add_argument(
        "--image_paths_file",
        type=str,
        default="continuous_images/image_paths.txt",
        help="Path to text file containing image paths (one per line, line n = node n). Default: continuous_images/image_paths.txt"
    )

    # --- Output Configuration ---
    default_tmp_dir = Path("./tmp")
    default_feat_name = "data_features.npy"
    default_paths_name = "data_paths.txt"

    ap.add_argument(
        "--out_features",
        type=str,
        default=str(default_tmp_dir / default_feat_name),
        help=f"Path to save feature file. Default: {default_tmp_dir / default_feat_name}"
    )
    ap.add_argument(
        "--out_paths",
        type=str,
        default=str(default_tmp_dir / default_paths_name),
        help=f"Path to save image paths file. Default: {default_tmp_dir / default_paths_name}"
    )

    # --- Model & Encoding Configuration ---
    ap.add_argument(
        "--arch",
        type=str,
        default="megaloc",  # <-- Default changed to megaloc
        choices=[
            "resnet18", "resnet50",
            "dinov2_vits14", "dinov2_vitb14",
            "megaloc"
        ],
        help="Backbone architecture. Default: megaloc"
    )
    ap.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Use randomly initialized weights (if supported by the model)."
    )
    ap.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[320, 320],  # <-- Default changed to match megaloc
        metavar=("W", "H"),
        help="Resize each image to (width, height). Default: 320 320"
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for encoding. Reduce if OOM (out of memory). Default: 128"
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use. 'auto' prefers GPU if available. Default: auto"
    )
    ap.add_argument(
    "--image_topic",
    type=str,
    default="/isaac_node/camera0/image_raw",
    help="ROS topic for camera image"
)

    ap.add_argument(
        "--dinov2-local",
        type=str,
        default=None,
        help="Prefer loading from a local facebookresearch/dinov2 repository (for limited networks); point to the repo root."
    )
    ap.add_argument(
        "--hub-force-reload",
        action="store_true",
        help="Force torch.hub to reload (useful when cache is corrupted or freshly patched)."
    )


    args = ap.parse_args()

    # Ensure output directories exist before starting.
    Path(args.out_features).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_paths).parent.mkdir(parents=True, exist_ok=True)

    return args

# ========== ROS Callbacks ==========
def image_callback(msg):
    global current_image
    if msg_to_pil is None:
        raise RuntimeError("Missing inference_utils.common.msg_to_pil; cannot convert ROS images.")
    current_image = msg_to_pil(msg)

current_image = None


# -------------------------------
# Usage example
if __name__ == "__main__":
    args = parse_args()

    _require_bass_runtime_dependencies()

    rospy.init_node("bass_node", anonymous=False)
    rospy.Subscriber(args.image_topic, ROSImage, image_callback, queue_size=1)

    goal_path_pub = rospy.Publisher("/goal_path", String, queue_size=1)
    goal_status_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)

    descriptors = np.load(args.descriptors_path)
    adjacency_matrix = np.load(args.adjacency_matrix_path)

    with open(args.image_paths_file, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]

    W, H = args.img_size
    vpr = load_vpr_encoder(
                arch=args.arch,
                img_size=(W, H),
                pretrained=(not args.no_pretrained),
                device_str=args.device,
                dinov2_local=args.dinov2_local, hub_force_reload=args.hub_force_reload
            )
    bass = BeliefAwareSubgoalSearch(adjacency_matrix=adjacency_matrix, node_embeddings=descriptors, vpr_encoder=vpr)

    goal_img_path = os.path.join(args.img_dir, image_paths[int(args.goal_image_index)])
    if not os.path.exists(goal_img_path):
        rospy.logfatal("Goal image not found: %s", goal_img_path)
        sys.exit(1)
    I_g = Image.open(goal_img_path)

    # Wait for the first subscribed frame
    rospy.loginfo("Waiting for first camera image on topic: %s", args.image_topic)
    while not rospy.is_shutdown() and current_image is None:
        rospy.loginfo_throttle(5.0, "Waiting for camera image...")
        rospy.sleep(0.1)
    if rospy.is_shutdown():
        sys.exit(0)

    goal_idx, start_idx = bass.initial_localize(I_g, current_image)  # I_g and current_image denote goal and current observations

    # 5) Plan route (unweighted shortest path; use edge_cost="inverse_weight" to account for edge strength)
    path = bass.plan_path(start_idx=start_idx, goal_idx=goal_idx, edge_cost="unweighted")
    print("planned path:", path)        # Also accessible via bass.state.path

    # 6) Instantiate PlaceRecognition
    place_recognition = PlaceRecognition(
        descriptors=descriptors,
        adjacency_matrix=adjacency_matrix,
        max_steps=3,
        delta=5
    )

    # Initialize PlaceRecognition belief distribution
    initial_obs_desc = vpr.encode(current_image)  # Use the initial observation image
    place_recognition.initialize_model(initial_obs_desc)

    # 7) Online navigation (external localization produces current_idx in real time)

    n_ahead = 2  # Example: look two subgoals ahead

    rate = rospy.Rate(30)

    goal_reached = False

    while not rospy.is_shutdown():
        if current_image is None:
            rate.sleep()
            continue
        
        current_feature = vpr.encode(current_image)
        current_idx = place_recognition.external_localizer_index(current_feature)  
        print("current_idx", current_idx)
        
        nav = bass.next_subgoal(current_idx=current_idx, n_ahead=n_ahead, replan_if_offpath=True)
        subgoal_idx = nav["subgoal_idx"]
        if subgoal_idx is None:
            continue

        print("subgoal_idx", subgoal_idx)
        subgoal_name = image_paths[subgoal_idx]
        subgoal_path = os.path.join(args.img_dir, subgoal_name)

        goal_path_pub.publish(String(data=subgoal_path))

        if subgoal_idx == goal_idx and current_idx == goal_idx:
            goal_reached = True

        if goal_reached:
            goal_status_pub.publish(Bool(data=True))

        rate.sleep()

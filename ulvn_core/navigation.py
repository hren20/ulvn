from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import heapq
import numpy as np


def _as_l2_normalized(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    denom = np.linalg.norm(features, axis=-1, keepdims=True) + 1e-12
    return features / denom


def safe_cosine_distance(descriptors: np.ndarray, query_desc: np.ndarray) -> np.ndarray:
    """Compute cosine-derived L2 distance after L2-normalizing inputs."""

    query = _as_l2_normalized(np.asarray(query_desc, dtype=np.float32).reshape(1, -1))[0]
    descriptors = _as_l2_normalized(np.asarray(descriptors, dtype=np.float32))
    cos_sim = descriptors @ query
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cos_sim))


def safe_obs_likelihood(
    descriptors: np.ndarray,
    query_desc: np.ndarray,
    lambda1: float,
) -> np.ndarray:
    distances = safe_cosine_distance(descriptors, query_desc)
    return np.exp(-lambda1 * distances)


class PlaceRecognition:
    """Belief propagation localization over a topological graph.

    Inputs are plain NumPy arrays. Descriptors are L2-normalized on load, so
    callers may provide raw or normalized feature vectors.
    """

    def __init__(
        self,
        descriptors: np.ndarray,
        adjacency_matrix: np.ndarray,
        max_steps: int = 3,
        delta: float = 5.0,
    ):
        self.descriptors = _as_l2_normalized(descriptors)
        self.adjacency_matrix = np.asarray(adjacency_matrix, dtype=np.float32)
        if self.adjacency_matrix.shape != (self.descriptors.shape[0], self.descriptors.shape[0]):
            raise ValueError(
                "adjacency_matrix must be [N, N] and align with descriptors; "
                f"got {self.adjacency_matrix.shape} for {self.descriptors.shape[0]} descriptors"
            )

        self.max_steps = int(max_steps)
        self.delta = float(delta)
        self.transition_matrix = self._build_transition_matrix()
        self.belief: Optional[np.ndarray] = None
        self.lambda1 = 1.0

    def _build_transition_matrix(self) -> np.ndarray:
        n = self.adjacency_matrix.shape[0]
        binary_adj = (self.adjacency_matrix > 0).astype(np.float32)
        cumulative = np.eye(n, dtype=np.float32)
        current_power = np.eye(n, dtype=np.float32)
        for _ in range(1, self.max_steps + 1):
            current_power = current_power @ binary_adj
            cumulative += current_power
        row_sums = cumulative.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return cumulative / row_sums

    def initialize_model(self, query_desc: np.ndarray) -> None:
        query = _as_l2_normalized(np.asarray(query_desc, dtype=np.float32).reshape(1, -1))[0]
        dists = safe_cosine_distance(self.descriptors, query)
        q_low, q_high = np.quantile(dists, [0.025, 0.975])
        self.lambda1 = 1.0 if q_high == q_low else float(np.log(self.delta) / (q_high - q_low))
        belief = np.exp(-self.lambda1 * dists)
        total = belief.sum()
        self.belief = belief / total if total > 0 else np.full(len(belief), 1.0 / len(belief))

    def _compute_belief_entropy(self) -> float:
        if self.belief is None:
            raise RuntimeError("PlaceRecognition must be initialized before match().")
        belief_safe = np.clip(self.belief, 1e-10, 1.0)
        entropy = -np.sum(self.belief * np.log(belief_safe))
        max_entropy = np.log(len(self.belief))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    @staticmethod
    def _get_fusion_weights(entropy: float) -> Tuple[float, float]:
        if entropy > 0.7:
            return 0.3, 0.7
        if entropy < 0.3:
            return 0.6, 0.4
        w_prediction = 0.6 - 0.75 * (entropy - 0.3)
        return float(w_prediction), float(1.0 - w_prediction)

    def obs_lhood(self, descriptor: np.ndarray) -> np.ndarray:
        query = _as_l2_normalized(np.asarray(descriptor, dtype=np.float32).reshape(1, -1))[0]
        return safe_obs_likelihood(self.descriptors, query, self.lambda1)

    def match(self, query_desc: np.ndarray) -> Tuple[int, float]:
        if self.belief is None:
            self.initialize_model(query_desc)

        entropy = self._compute_belief_entropy()
        w_pred, w_obs = self._get_fusion_weights(entropy)

        belief_prior = self.belief @ self.transition_matrix
        belief_prior /= belief_prior.sum()

        obs = self.obs_lhood(query_desc)
        obs_sum = obs.sum()
        obs_lhood = obs / obs_sum if obs_sum > 0 else np.full_like(obs, 1.0 / len(obs))

        self.belief = np.power(belief_prior, w_pred) * np.power(obs_lhood, w_obs)
        self.belief /= self.belief.sum()

        node = int(np.argmax(self.belief))
        return node, float(self.belief[node])

    def external_localizer_index(self, query_desc: np.ndarray) -> int:
        node, _ = self.match(query_desc)
        return node


@dataclass
class PlanState:
    goal_idx: Optional[int] = None
    start_idx: Optional[int] = None
    path: List[int] = field(default_factory=list)
    edge_cost: str = "unweighted"


class BeliefAwareSubgoalSearch:
    """BASS planner and subgoal selector over a weighted adjacency matrix."""

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        node_embeddings: np.ndarray,
        vpr_encoder: Optional[Any] = None,
        l2_normalize_inputs: bool = True,
    ):
        self.W = np.asarray(adjacency_matrix, dtype=np.float32)
        if self.W.ndim != 2 or self.W.shape[0] != self.W.shape[1]:
            raise ValueError(f"adjacency_matrix must be square [N, N], got {self.W.shape}")

        self.N = self.W.shape[0]
        self.Z = np.asarray(node_embeddings, dtype=np.float32)
        if self.Z.ndim != 2 or self.Z.shape[0] != self.N:
            raise ValueError("Number of graph nodes does not match descriptor count")

        self.vpr = vpr_encoder
        self.norm_inputs = bool(l2_normalize_inputs)
        self.Z = _as_l2_normalized(self.Z)
        self.state = PlanState()

    def initial_localize(self, goal_image_or_feature: Any, obs_image_or_feature: Any) -> Tuple[int, int]:
        z_goal = self._to_embedding(goal_image_or_feature)
        z_obs = self._to_embedding(obs_image_or_feature)
        goal_idx = int(np.argmax(self.Z @ z_goal))
        start_idx = int(np.argmax(self.Z @ z_obs))
        self.state.goal_idx = goal_idx
        self.state.start_idx = start_idx
        return goal_idx, start_idx

    def plan_path(
        self,
        start_idx: Optional[int] = None,
        goal_idx: Optional[int] = None,
        edge_cost: str = "unweighted",
    ) -> List[int]:
        if start_idx is None:
            start_idx = self.state.start_idx
        if goal_idx is None:
            goal_idx = self.state.goal_idx
        if start_idx is None or goal_idx is None:
            raise ValueError("start_idx / goal_idx not set; run initial_localize or pass indices explicitly.")

        path = self._dijkstra(int(start_idx), int(goal_idx), edge_cost=edge_cost)
        self.state.start_idx = int(start_idx)
        self.state.goal_idx = int(goal_idx)
        self.state.path = path
        self.state.edge_cost = edge_cost
        return path

    def next_subgoal(
        self,
        current_idx: int,
        n_ahead: int = 1,
        replan_if_offpath: bool = True,
    ) -> Dict[str, Any]:
        if self.state.goal_idx is None:
            raise ValueError("Goal index not set; call initial_localize / plan_path first.")
        if not self.state.path:
            self.plan_path(start_idx=current_idx, goal_idx=self.state.goal_idx, edge_cost=self.state.edge_cost)

        path = self.state.path
        replanned = False
        position_by_node = {node: i for i, node in enumerate(path)}
        whereami = position_by_node.get(int(current_idx))

        if whereami is None and replan_if_offpath:
            path = self.plan_path(start_idx=current_idx, goal_idx=self.state.goal_idx, edge_cost=self.state.edge_cost)
            replanned = True
            position_by_node = {node: i for i, node in enumerate(path)}
            whereami = position_by_node.get(int(current_idx))

        if not path:
            return {"subgoal_idx": None, "replanned": replanned, "path": [], "whereami_on_path": None}

        if whereami is None:
            whereami = self._nearest_on_path(int(current_idx), path)

        j = min(whereami + max(1, int(n_ahead)), len(path) - 1)
        subgoal_idx = int(path[j])
        self.state.start_idx = int(current_idx)
        self.state.path = path
        return {
            "subgoal_idx": subgoal_idx,
            "replanned": replanned,
            "path": list(path),
            "whereami_on_path": whereami,
        }

    def _to_embedding(self, img_or_feat: Any) -> np.ndarray:
        if isinstance(img_or_feat, np.ndarray):
            z = np.asarray(img_or_feat, dtype=np.float32).reshape(-1)
        elif self.vpr is not None:
            z = np.asarray(self.vpr.encode(img_or_feat), dtype=np.float32).reshape(-1)
        else:
            raise ValueError("Pass a 1D feature vector or provide a vpr_encoder with encode().")
        if self.norm_inputs:
            z = _as_l2_normalized(z.reshape(1, -1))[0]
        return z

    def _edge_len(self, weight: float, edge_cost: str) -> float:
        if edge_cost == "unweighted":
            return 1.0
        if edge_cost == "inverse_weight":
            return 1.0 / max(float(weight), 1e-6)
        raise ValueError(f"Unknown edge_cost: {edge_cost}")

    def _dijkstra(self, start: int, goal: int, edge_cost: str = "unweighted") -> List[int]:
        if start == goal:
            return [start]

        dist = {start: 0.0}
        prev: Dict[int, Optional[int]] = {start: None}
        pq = [(0.0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            if u == goal:
                break
            if d > dist.get(u, float("inf")):
                continue
            for v in np.flatnonzero(self.W[u] > 0):
                nd = d + self._edge_len(self.W[u, v], edge_cost)
                v_int = int(v)
                if nd < dist.get(v_int, float("inf")):
                    dist[v_int] = nd
                    prev[v_int] = u
                    heapq.heappush(pq, (nd, v_int))

        if goal not in prev:
            return []

        path = [goal]
        while path[-1] != start:
            parent = prev[path[-1]]
            if parent is None:
                return []
            path.append(parent)
        path.reverse()
        return path

    def _nearest_on_path(self, node_idx: int, path: List[int]) -> int:
        best_j, best_d = 0, float("inf")
        for j, node in enumerate(path):
            d = self._bfs_distance(node_idx, int(node), cutoff=int(best_d) if best_d < float("inf") else None)
            if d < best_d:
                best_j, best_d = j, d
        return best_j

    def _bfs_distance(self, start: int, goal: int, cutoff: Optional[int] = None) -> int:
        if start == goal:
            return 0
        visited = {start}
        queue: deque[Tuple[int, int]] = deque([(start, 0)])
        while queue:
            u, d = queue.popleft()
            nd = d + 1
            if cutoff is not None and nd > cutoff:
                continue
            for v in np.flatnonzero(self.W[u] > 0):
                v_int = int(v)
                if v_int == goal:
                    return nd
                if v_int not in visited:
                    visited.add(v_int)
                    queue.append((v_int, nd))
        return int(1e9)

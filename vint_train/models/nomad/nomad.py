from __future__ import annotations

import torch.nn as nn


class NoMaD(nn.Module):
    def __init__(self, vision_encoder, noise_pred_net, dist_pred_net) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net

    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder":
            return self.vision_encoder(
                kwargs["obs_img"],
                kwargs["goal_img"],
                input_goal_mask=kwargs.get("input_goal_mask"),
            )
        if func_name == "noise_pred_net":
            return self.noise_pred_net(
                sample=kwargs["sample"],
                timestep=kwargs["timestep"],
                global_cond=kwargs["global_cond"],
            )
        if func_name == "dist_pred_net":
            return self.dist_pred_net(kwargs["obsgoal_cond"])
        raise NotImplementedError(f"Unknown NoMaD function: {func_name}")


class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 4, self.embedding_dim // 16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 16, 1),
        )

    def forward(self, x):
        x = x.reshape((-1, self.embedding_dim))
        return self.network(x)

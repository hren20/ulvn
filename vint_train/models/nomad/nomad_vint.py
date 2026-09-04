from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from vint_train.models.vint.self_attention import PositionalEncoding


class NoMaD_ViNT(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        super().__init__()
        self.obs_encoding_size = int(obs_encoding_size)
        self.goal_encoding_size = int(obs_encoding_size)
        self.context_size = int(context_size)

        if obs_encoder.split("-")[0] != "efficientnet":
            raise NotImplementedError("NoMaD_ViNT release model supports efficientnet observation encoders.")
        self.obs_encoder = replace_bn_with_gn(EfficientNet.from_name(obs_encoder, in_channels=3))
        self.num_obs_features = self.obs_encoder._fc.in_features
        self.goal_encoder = replace_bn_with_gn(EfficientNet.from_name("efficientnet-b0", in_channels=6))
        self.num_goal_features = self.goal_encoder._fc.in_features

        self.compress_obs_enc = (
            nn.Linear(self.num_obs_features, self.obs_encoding_size)
            if self.num_obs_features != self.obs_encoding_size
            else nn.Identity()
        )
        self.compress_goal_enc = (
            nn.Linear(self.num_goal_features, self.goal_encoding_size)
            if self.num_goal_features != self.goal_encoding_size
            else nn.Identity()
        )

        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=self.context_size + 2)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size,
            nhead=mha_num_attention_heads,
            dim_feedforward=mha_ff_dim_factor * self.obs_encoding_size,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)

        goal_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        goal_mask[:, -1] = True
        no_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        self.register_buffer("all_masks", torch.cat([no_mask, goal_mask], dim=0), persistent=False)
        avg_pool_mask = torch.cat([
            1 - no_mask.float(),
            (1 - goal_mask.float()) * ((self.context_size + 2) / (self.context_size + 1)),
        ], dim=0)
        self.register_buffer("avg_pool_mask", avg_pool_mask, persistent=False)

    def forward(self, obs_img: torch.Tensor, goal_img: torch.Tensor, input_goal_mask: torch.Tensor = None) -> torch.Tensor:
        device = obs_img.device
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)
        else:
            goal_mask = None

        obsgoal_img = torch.cat([obs_img[:, 3 * self.context_size :, :, :], goal_img], dim=1)
        goal_encoding = self.goal_encoder.extract_features(obsgoal_img)
        goal_encoding = self.goal_encoder._avg_pooling(goal_encoding)
        if self.goal_encoder._global_params.include_top:
            goal_encoding = goal_encoding.flatten(start_dim=1)
            goal_encoding = self.goal_encoder._dropout(goal_encoding)
        goal_encoding = self.compress_goal_enc(goal_encoding)
        if len(goal_encoding.shape) == 2:
            goal_encoding = goal_encoding.unsqueeze(1)

        obs_parts = torch.split(obs_img, 3, dim=1)
        obs_stack = torch.concat(obs_parts, dim=0)
        obs_encoding = self.obs_encoder.extract_features(obs_stack)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.unsqueeze(1)
        obs_encoding = obs_encoding.reshape((self.context_size + 1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        obs_encoding = torch.cat((obs_encoding, goal_encoding), dim=1)

        if goal_mask is not None:
            src_key_padding_mask = torch.index_select(self.all_masks.to(device), 0, goal_mask.long())
        else:
            src_key_padding_mask = None

        tokens = self.positional_encoding(obs_encoding)
        tokens = self.sa_encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            avg_mask = torch.index_select(self.avg_pool_mask.to(device), 0, goal_mask.long()).unsqueeze(-1)
            tokens = tokens * avg_mask
        return torch.mean(tokens, dim=1)


def replace_bn_with_gn(root_module: nn.Module, features_per_group: int = 16) -> nn.Module:
    return replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=max(1, x.num_features // features_per_group),
            num_channels=x.num_features,
        ),
    )


def replace_submodules(root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    targets = [name.split(".") for name, module in root_module.named_modules(remove_duplicate=True) if predicate(module)]
    for *parent, key in targets:
        parent_module = root_module.get_submodule(".".join(parent)) if parent else root_module
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(key)] = func(parent_module[int(key)])
        else:
            setattr(parent_module, key, func(getattr(parent_module, key)))
    return root_module

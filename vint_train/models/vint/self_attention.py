from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 6) -> None:
        super().__init__()
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pos_enc", pos_enc.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_enc[:, : x.size(1), :]


class MultiLayerDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        seq_len: int = 6,
        output_layers=None,
        nhead: int = 8,
        num_layers: int = 8,
        ff_dim_factor: int = 4,
    ) -> None:
        super().__init__()
        if output_layers is None:
            output_layers = [256, 128, 64]
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim_factor * embed_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        layers = [nn.Linear(seq_len * embed_dim, embed_dim), nn.Linear(embed_dim, output_layers[0])]
        for i in range(len(output_layers) - 1):
            layers.append(nn.Linear(output_layers[i], output_layers[i + 1]))
        self.output_layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        x = x.reshape(x.shape[0], -1)
        for layer in self.output_layers:
            x = F.relu(layer(x))
        return x

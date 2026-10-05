from __future__ import annotations

import torch
from torch import nn

from .base import SequentialRecommender, reset_parameters


class FilterLayer(nn.Module):
    def __init__(self, max_length: int, embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        frequency_bins = max_length // 2 + 1
        self.complex_weight = nn.Parameter(torch.randn(frequency_bins, embedding_dim, 2) * 0.02)
        self.filter_norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embedding_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(values, dim=1, norm="ortho")
        weights = torch.view_as_complex(self.complex_weight[: spectrum.size(1)].contiguous())
        filtered = torch.fft.irfft(spectrum * weights.unsqueeze(0), n=values.size(1), dim=1, norm="ortho")
        values = self.filter_norm(values + filtered)
        return self.ffn_norm(values + self.ffn(values))


class FMLP(SequentialRecommender):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        max_length: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        **_: object,
    ):
        super().__init__(num_items, embedding_dim, max_length, dropout)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.layers = nn.ModuleList(
            [FilterLayer(max_length, embedding_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.apply(reset_parameters)

    def encode_sequence(self, items: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(items.size(1), device=items.device).unsqueeze(0)
        values = self.input_norm(self.item_embedding(items) + self.position_embedding(positions))
        values = self.dropout(values)
        target_width = self.max_length
        if values.size(1) < target_width:
            values = torch.nn.functional.pad(values, (0, 0, 0, target_width - values.size(1)))
        elif values.size(1) > target_width:
            values = values[:, -target_width:]
        for layer in self.layers:
            values = layer(values)
        return values[:, : items.size(1)]


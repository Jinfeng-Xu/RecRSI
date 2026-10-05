from __future__ import annotations

import torch
from torch import nn

from .base import SequentialRecommender, reset_parameters


class SASRec(SequentialRecommender):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        max_length: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.5,
        **_: object,
    ):
        super().__init__(num_items, embedding_dim, max_length, dropout)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.apply(reset_parameters)

    def encode_sequence(self, items: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(items.size(1), device=items.device).unsqueeze(0)
        values = self.item_embedding(items) + self.position_embedding(positions)
        causal = torch.triu(
            torch.ones(items.size(1), items.size(1), dtype=torch.bool, device=items.device), diagonal=1
        )
        padding = items.eq(0)
        output = self.encoder(self.dropout(values), mask=causal, src_key_padding_mask=padding)
        return self.norm(output)


from __future__ import annotations

import torch
from torch import nn

from .base import SequentialRecommender, reset_parameters


class GRU4Rec(SequentialRecommender):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        max_length: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        **_: object,
    ):
        super().__init__(num_items, embedding_dim, max_length, dropout)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(hidden_dim, embedding_dim) if hidden_dim != embedding_dim else nn.Identity()
        self.apply(reset_parameters)

    def encode_sequence(self, items: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.item_embedding(items))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, total_length=items.size(1)
        )
        return self.projection(output)


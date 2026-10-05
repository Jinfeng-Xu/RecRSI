from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn


class SequentialRecommender(nn.Module, ABC):
    def __init__(self, num_items: int, embedding_dim: int, max_length: int, dropout: float = 0.0):
        super().__init__()
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.max_length = int(max_length)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    @abstractmethod
    def encode_sequence(self, items: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return a representation for every sequence position, shape [B, L, D]."""

    def encode(self, items: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_sequence(items, lengths)
        row = torch.arange(items.size(0), device=items.device)
        return encoded[row, lengths - 1]

    def score_all(self, items: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.encode(items, lengths) @ self.item_embedding.weight.t()

    def score_items(self, query: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.item_embedding(item_ids)
        if item_ids.ndim == 1:
            return (query * embeddings).sum(dim=-1)
        return torch.einsum("bd,bnd->bn", query, embeddings)

    def checkpoint(self, config: Dict, step: int) -> Dict:
        return {
            "config": config,
            "step": int(step),
            "num_items": self.num_items,
            "state_dict": self.state_dict(),
        }


def reset_parameters(module: nn.Module) -> None:
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


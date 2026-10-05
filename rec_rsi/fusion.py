from __future__ import annotations

import torch


def masked_log_probabilities(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != mask.shape:
        raise ValueError("logits and mask must have the same shape")
    safe = logits.masked_fill(mask, -torch.inf)
    if torch.any(torch.isneginf(safe).all(dim=1)):
        raise ValueError("At least one candidate must remain unmasked")
    return torch.log_softmax(safe, dim=1)


def fuse_log_probabilities(
    left_logits: torch.Tensor,
    right_logits: torch.Tensor,
    beta: float = 0.5,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """Geometrically mix two full-catalog predictive distributions."""
    if left_logits.shape != right_logits.shape:
        raise ValueError("Both models must score the same catalog")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if mask is None:
        mask = torch.zeros_like(left_logits, dtype=torch.bool)
    left = masked_log_probabilities(left_logits, mask)
    right = masked_log_probabilities(right_logits, mask)
    return ((1.0 - beta) * left + beta * right).masked_fill(mask, -torch.inf)


def history_mask(histories, num_items: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(len(histories), num_items, dtype=torch.bool, device=device)
    mask[:, 0] = True
    for row, history in enumerate(histories):
        if history:
            mask[row, torch.tensor(tuple(set(history)), device=device)] = True
    return mask


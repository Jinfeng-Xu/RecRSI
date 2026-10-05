from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence

import torch


def ranking_metrics(top_items: torch.Tensor, targets: torch.Tensor, cutoffs: Iterable[int]) -> Dict[str, float]:
    if top_items.ndim != 2 or targets.ndim != 1:
        raise ValueError("top_items must be [N,K] and targets must be [N]")
    output: Dict[str, float] = {}
    for cutoff in cutoffs:
        width = min(int(cutoff), top_items.size(1))
        hits = top_items[:, :width].eq(targets[:, None])
        recall = hits.any(dim=1).float()
        discounts = 1.0 / torch.log2(torch.arange(2, width + 2, device=top_items.device).float())
        ndcg = (hits.float() * discounts).sum(dim=1)
        output[f"recall@{cutoff}"] = float(recall.mean())
        output[f"ndcg@{cutoff}"] = float(ndcg.mean())
    return output


def topk_overlap(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("Rank lists must have the same shape")
    matches = left[:, :, None].eq(right[:, None, :]).any(dim=2)
    return matches.float().sum(dim=1) / left.size(1)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return float(sum(values) / len(values))


def reciprocal_log_rank(rank: int) -> float:
    return 1.0 / math.log2(rank + 2)


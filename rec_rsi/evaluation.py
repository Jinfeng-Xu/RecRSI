from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

from .data import batched, collate_evaluation
from .fusion import fuse_log_probabilities, history_mask
from .metrics import mean, ranking_metrics, topk_overlap
from .models import SequentialRecommender


CROSS_FAMILY: Tuple[Tuple[str, str], ...] = (("P", "A"), ("P", "Aprime"), ("B", "A"), ("B", "Aprime"))
WITHIN_FAMILY: Tuple[Tuple[str, str], ...] = (("P", "B"), ("A", "Aprime"))


@dataclass
class PairOutput:
    metrics: Dict[str, float]
    overlap: float
    left_only_hit_rate: float
    right_only_hit_rate: float
    both_hit_rate: float


def cross_generation_advantage(
    pair_metrics: Mapping[str, Mapping[str, float]], metric: str = "ndcg@20"
) -> float:
    cross = [pair_metrics[pair_key(*pair)][metric] for pair in CROSS_FAMILY]
    within = [pair_metrics[pair_key(*pair)][metric] for pair in WITHIN_FAMILY]
    return mean(cross) - mean(within)


def family_average(
    pair_metrics: Mapping[str, Mapping[str, float]],
    family: Sequence[Tuple[str, str]],
    metric: str,
) -> float:
    return mean([pair_metrics[pair_key(left, right)][metric] for left, right in family])


def rank_separation(
    pair_metrics: Mapping[str, Mapping[str, float]],
    cross_family: Sequence[Tuple[str, str]] = CROSS_FAMILY,
    within_family: Sequence[Tuple[str, str]] = WITHIN_FAMILY,
) -> float:
    """Within-family top-k overlap minus cross-family top-k overlap."""
    within = mean([pair_metrics[pair_key(*pair)]["overlap"] for pair in within_family])
    cross = mean([pair_metrics[pair_key(*pair)]["overlap"] for pair in cross_family])
    return within - cross


def pair_key(left: str, right: str) -> str:
    return f"{left}+{right}"


@torch.no_grad()
def collect_logits(
    models: Mapping[str, SequentialRecommender],
    examples: Sequence,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    outputs: Dict[str, List[torch.Tensor]] = {name: [] for name in models}
    masks: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    for model in models.values():
        model.eval()
    for rows in batched(examples, batch_size):
        batch = collate_evaluation(rows, max_length)
        items = batch["items"].to(device)
        lengths = batch["lengths"].to(device)
        mask = history_mask(batch["histories"], next(iter(models.values())).num_items, device)
        masks.append(mask.cpu())
        targets.append(batch["targets"])
        for name, model in models.items():
            logits = model.score_all(items, lengths).masked_fill(mask, -torch.inf)
            outputs[name].append(logits.cpu())
    return (
        {name: torch.cat(chunks) for name, chunks in outputs.items()},
        torch.cat(masks),
        torch.cat(targets),
    )


def evaluate_models(
    models: Mapping[str, SequentialRecommender],
    examples: Sequence,
    max_length: int,
    cutoffs: Iterable[int] = (10, 20),
    beta: float = 0.5,
    batch_size: int = 512,
    device: torch.device = torch.device("cpu"),
    cross_family: Sequence[Tuple[str, str]] = CROSS_FAMILY,
    within_family: Sequence[Tuple[str, str]] = WITHIN_FAMILY,
) -> Dict:
    if not examples:
        raise ValueError("Evaluation window is empty")
    logits, mask, targets = collect_logits(models, examples, max_length, batch_size, device)
    cutoffs = tuple(sorted(set(map(int, cutoffs))))
    top_width = min(max(cutoffs), next(iter(models.values())).num_items - 1)
    singles: Dict[str, Dict[str, float]] = {}
    rankings: Dict[str, torch.Tensor] = {}
    for name, scores in logits.items():
        rankings[name] = scores.topk(top_width, dim=1).indices
        singles[name] = ranking_metrics(rankings[name], targets, cutoffs)

    pairs: Dict[str, Dict[str, float]] = {}
    for left, right in tuple(cross_family) + tuple(within_family):
        key = pair_key(left, right)
        mixed = fuse_log_probabilities(logits[left], logits[right], beta=beta, mask=mask)
        top = mixed.topk(top_width, dim=1).indices
        row = ranking_metrics(top, targets, cutoffs)
        overlap = topk_overlap(rankings[left], rankings[right])
        left_hit = rankings[left].eq(targets[:, None]).any(dim=1)
        right_hit = rankings[right].eq(targets[:, None]).any(dim=1)
        row.update(
            overlap=float(overlap.mean()),
            left_only_hit_rate=float((left_hit & ~right_hit).float().mean()),
            right_only_hit_rate=float((right_hit & ~left_hit).float().mean()),
            both_hit_rate=float((left_hit & right_hit).float().mean()),
        )
        pairs[key] = row

    if not cross_family and not within_family:
        return {"n": len(targets), "single": singles, "pairs": {}}

    primary = f"ndcg@{max(cutoffs)}"
    cross_mean = family_average(pairs, cross_family, primary)
    within_mean = family_average(pairs, within_family, primary)
    separation = rank_separation(pairs, cross_family, within_family)
    return {
        "n": len(targets),
        "beta": beta,
        "single": singles,
        "pairs": pairs,
        "family": {
            "metric": primary,
            "cross": cross_mean,
            "within": within_mean,
            "cga": cross_mean - within_mean,
            "rank_separation": separation,
            "selected": "cross" if separation > 0 else "within",
        },
    }


def adjacent_families(older: Tuple[str, str], newer: Tuple[str, str]):
    cross = tuple((left, right) for left in older for right in newer)
    within = ((older[0], older[1]), (newer[0], newer[1]))
    return cross, within

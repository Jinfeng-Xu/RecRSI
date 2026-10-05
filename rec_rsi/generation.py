from __future__ import annotations

import random
from typing import List, Sequence, Set, Tuple

import torch

from .data import SequenceCorpus
from .models import SequentialRecommender


@torch.no_grad()
def generate_sequences(
    model: SequentialRecommender,
    corpus: SequenceCorpus,
    samples_per_history: int = 5,
    fidelity_k: int = 20,
    local_probability: float = 0.5,
    min_length: int = 3,
    max_length: int = 50,
    seed: int = 2023,
    device: torch.device = torch.device("cpu"),
) -> List[List[int]]:
    """Generate rank-faithful continuations from training histories.

    A continuation remains active while at least one yet-unvisited item from the
    source history lies within the parent's top-``fidelity_k`` catalog ranks.
    Decoding alternates between the source-item pool and the full catalog.
    """
    if samples_per_history < 1 or fidelity_k < 1:
        raise ValueError("samples_per_history and fidelity_k must be positive")
    rng = random.Random(seed)
    model.eval()
    generated: Set[Tuple[int, ...]] = set()
    for row in corpus.users:
        source = tuple(row.fit[-max_length:])
        if len(source) < min_length:
            continue
        for _ in range(samples_per_history):
            sequence = [source[0]]
            while len(sequence) < min(max_length, len(source)):
                items = torch.tensor([sequence], dtype=torch.long, device=device)
                lengths = torch.tensor([len(sequence)], dtype=torch.long, device=device)
                scores = model.score_all(items, lengths)[0]
                scores[0] = -torch.inf
                scores[torch.tensor(sequence, device=device)] = -torch.inf
                remaining = sorted(set(source).difference(sequence))
                if not remaining:
                    break
                ranks = torch.argsort(torch.argsort(scores, descending=True)) + 1
                if not bool((ranks[torch.tensor(remaining, device=device)] <= fidelity_k).any()):
                    break
                if rng.random() < local_probability:
                    candidates = remaining
                else:
                    candidates = [item for item in range(1, corpus.num_items) if item not in sequence]
                if not candidates:
                    break
                candidate_tensor = torch.tensor(candidates, device=device)
                choice = int(candidate_tensor[scores[candidate_tensor].argmax()])
                sequence.append(choice)
            if len(sequence) >= min_length:
                generated.add(tuple(sequence))
    return [list(sequence) for sequence in sorted(generated)]


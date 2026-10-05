from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import NextItemDataset, SequenceCorpus, collate_next_item
from .evaluation import evaluate_models
from .models import SequentialRecommender, build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def sample_negatives(
    histories: torch.Tensor,
    targets: torch.Tensor,
    num_items: int,
    count: int,
) -> torch.Tensor:
    batch = histories.size(0)
    weights = torch.ones(batch, num_items, device=histories.device)
    weights[:, 0] = 0.0
    weights.scatter_(1, histories, 0.0)
    weights.scatter_(1, targets[:, None], 0.0)
    available = weights.sum(dim=1)
    if torch.any(available < 1):
        raise ValueError("A training row masks the entire item catalog")
    return torch.multinomial(weights, count, replacement=True)


def sampled_softmax_loss(
    model: SequentialRecommender,
    items: torch.Tensor,
    lengths: torch.Tensor,
    targets: torch.Tensor,
    num_negatives: int,
) -> torch.Tensor:
    query = model.encode(items, lengths)
    negatives = sample_negatives(items, targets, model.num_items, num_negatives)
    positive_score = model.score_items(query, targets)[:, None]
    negative_score = model.score_items(query, negatives)
    logits = torch.cat((positive_score, negative_score), dim=1)
    labels = torch.zeros(len(targets), dtype=torch.long, device=targets.device)
    return nn.functional.cross_entropy(logits, labels)


@dataclass
class TrainingResult:
    model: SequentialRecommender
    best_checkpoint: Path
    last_checkpoint: Path
    half_checkpoint: Path
    history: Sequence[Dict]


class Trainer:
    def __init__(self, config: Dict, corpus: SequenceCorpus):
        self.config = config
        self.corpus = corpus
        self.device = resolve_device(str(config.get("device", "cuda")))

    def train(
        self,
        role: str,
        seed: int,
        output_dir: Path,
        augmented: Optional[Sequence[Sequence[int]]] = None,
    ) -> TrainingResult:
        seed_everything(seed)
        output_dir.mkdir(parents=True, exist_ok=False)
        model = build_model(self.corpus.num_items, self.config).to(self.device)
        train_cfg = self.config["training"]
        dataset = NextItemDataset(
            self.corpus.training_sequences(augmented),
            max_length=int(self.config["data"]["max_length"]),
        )
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            drop_last=False,
            collate_fn=collate_next_item,
            generator=generator,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        steps = int(train_cfg["steps"])
        eval_every = int(train_cfg["eval_every"])
        half_step = max(1, steps // 2)
        iterator = iter(loader)
        history = []
        best_metric = -float("inf")
        best_path = output_dir / "best.pt"
        last_path = output_dir / "last.pt"
        half_path = output_dir / "half.pt"
        select_examples = self.corpus.evaluation_examples("select")

        for step in range(1, steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            model.train()
            items = batch["items"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            targets = batch["targets"].to(self.device)
            optimizer.zero_grad(set_to_none=True)
            loss = sampled_softmax_loss(
                model, items, lengths, targets, int(train_cfg.get("num_negatives", 256))
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            loss.backward()
            optimizer.step()

            if step == half_step:
                torch.save(model.checkpoint(self.config, step), half_path)
            if step % eval_every == 0 or step == steps:
                selected = evaluate_models(
                    {role: model},
                    select_examples,
                    max_length=int(self.config["data"]["max_length"]),
                    cutoffs=self.config["evaluation"]["cutoffs"],
                    batch_size=int(self.config["evaluation"].get("batch_size", 512)),
                    device=self.device,
                    cross_family=(),
                    within_family=(),
                )["single"][role]
                primary = selected[f"ndcg@{max(self.config['evaluation']['cutoffs'])}"]
                row = {"step": step, "loss": float(loss.detach()), **selected}
                history.append(row)
                if primary > best_metric:
                    best_metric = primary
                    torch.save(model.checkpoint(self.config, step), best_path)

        torch.save(model.checkpoint(self.config, steps), last_path)
        if not half_path.exists():
            torch.save(model.checkpoint(self.config, steps), half_path)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        (output_dir / "run.json").write_text(
            json.dumps(
                {
                    "role": role,
                    "seed": seed,
                    "training_examples": len(dataset),
                    "generated_sequences": len(augmented or []),
                    "best_step": torch.load(best_path, map_location="cpu")["step"],
                },
                indent=2,
            )
            + "\n"
        )
        load_checkpoint(model, best_path, self.device)
        return TrainingResult(model, best_path, last_path, half_path, history)


def load_model(path: Path, device: torch.device) -> SequentialRecommender:
    checkpoint = torch.load(path, map_location=device)
    model = build_model(int(checkpoint["num_items"]), checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_checkpoint(model: SequentialRecommender, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()


from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from .data import NextItemDataset, SequenceCorpus, collate_next_item
from .fusion import fuse_log_probabilities, history_mask
from .models import SequentialRecommender, build_model
from .training import load_model, sampled_softmax_loss, seed_everything


ROUTES = {"hard-data", "full-kl", "listwise", "complementary", "interpolate"}


@torch.no_grad()
def interpolate_models(
    left: SequentialRecommender,
    right: SequentialRecommender,
    alpha: float = 0.5,
) -> Dict[str, torch.Tensor]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    left_state, right_state = left.state_dict(), right.state_dict()
    if left_state.keys() != right_state.keys():
        raise ValueError("Interpolation requires identical architectures")
    mixed = {}
    for key in left_state:
        if left_state[key].shape != right_state[key].shape:
            raise ValueError(f"Shape mismatch for {key}")
        if torch.is_floating_point(left_state[key]):
            mixed[key] = (1.0 - alpha) * left_state[key] + alpha * right_state[key]
        else:
            mixed[key] = left_state[key].clone()
    return mixed


def transfer_loss(
    route: str,
    student_logits: torch.Tensor,
    left_logits: torch.Tensor,
    right_logits: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
    temperature: float,
    topk: int,
) -> torch.Tensor:
    teacher = fuse_log_probabilities(left_logits, right_logits, beta=beta, mask=mask)
    student = torch.log_softmax((student_logits / temperature).masked_fill(mask, -torch.inf), dim=1)
    if route == "full-kl":
        probabilities = torch.softmax(teacher / temperature, dim=1)
        return torch.nn.functional.kl_div(student, probabilities, reduction="batchmean") * temperature**2
    if route == "listwise":
        indices = teacher.topk(min(topk, teacher.size(1) - 1), dim=1).indices
        target = torch.softmax(teacher.gather(1, indices) / temperature, dim=1)
        prediction = torch.log_softmax(student_logits.gather(1, indices) / temperature, dim=1)
        return -(target * prediction).sum(dim=1).mean() * temperature**2
    if route == "complementary":
        width = min(topk, teacher.size(1) - 1)
        left_top = left_logits.masked_fill(mask, -torch.inf).topk(width, dim=1).indices
        right_top = right_logits.masked_fill(mask, -torch.inf).topk(width, dim=1).indices
        targets = []
        for row in range(len(left_top)):
            left_only = set(left_top[row].tolist()).difference(right_top[row].tolist())
            right_only = set(right_top[row].tolist()).difference(left_top[row].tolist())
            candidates = sorted(left_only | right_only)
            if not candidates:
                targets.append(int(teacher[row].argmax()))
            else:
                index = torch.tensor(candidates, device=teacher.device)
                targets.append(int(index[teacher[row, index].argmax()]))
        return torch.nn.functional.cross_entropy(
            student_logits.masked_fill(mask, -torch.inf),
            torch.tensor(targets, device=student_logits.device),
        )
    raise ValueError(f"Unknown differentiable route: {route}")


def consolidate(
    config: Dict,
    corpus: SequenceCorpus,
    left_checkpoint: Path,
    right_checkpoint: Path,
    route: str,
    output_dir: Path,
    seed: int,
    generated: Optional[Sequence[Sequence[int]]] = None,
    transfer_weight: float = 1.0,
    temperature: float = 1.0,
    topk: int = 50,
    alpha: float = 0.5,
) -> Path:
    if route not in ROUTES:
        raise ValueError(f"route must be one of {sorted(ROUTES)}")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    device = torch.device(config.get("device", "cuda"))
    left = load_model(left_checkpoint, device)
    right = load_model(right_checkpoint, device)

    if route == "interpolate":
        student = build_model(corpus.num_items, config).to(device)
        student.load_state_dict(interpolate_models(left, right, alpha))
        path = output_dir / "model.pt"
        torch.save(student.checkpoint(config, 0), path)
        _write_manifest(output_dir, route, seed, left_checkpoint, right_checkpoint, alpha=alpha)
        return path

    if route == "hard-data":
        if not generated:
            raise ValueError("hard-data consolidation requires generated sequences")
        sequences = corpus.training_sequences(generated)
    else:
        sequences = corpus.training_sequences()
    dataset = NextItemDataset(sequences, int(config["data"]["max_length"]))
    seed_everything(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        collate_fn=collate_next_item,
        generator=torch.Generator().manual_seed(seed),
    )
    student = build_model(corpus.num_items, config).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=float(config["training"]["learning_rate"]))
    iterator = iter(loader)
    history = []
    for step in range(1, int(config["training"]["steps"]) + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        items = batch["items"].to(device)
        lengths = batch["lengths"].to(device)
        targets = batch["targets"].to(device)
        student.train()
        optimizer.zero_grad(set_to_none=True)
        supervised = sampled_softmax_loss(
            student, items, lengths, targets, int(config["training"].get("num_negatives", 256))
        )
        transfer = torch.zeros((), device=device)
        if route != "hard-data":
            with torch.no_grad():
                left_logits = left.score_all(items, lengths)
                right_logits = right.score_all(items, lengths)
            logits = student.score_all(items, lengths)
            histories = [tuple(row[: int(length)].tolist()) for row, length in zip(items, lengths)]
            mask = history_mask(histories, corpus.num_items, device)
            transfer = transfer_loss(
                route,
                logits,
                left_logits,
                right_logits,
                mask,
                float(config["evaluation"].get("beta", 0.5)),
                temperature,
                topk,
            )
        loss = supervised + transfer_weight * transfer
        loss.backward()
        optimizer.step()
        if step % int(config["training"]["eval_every"]) == 0 or step == int(config["training"]["steps"]):
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "supervised": float(supervised.detach()),
                    "transfer": float(transfer.detach()),
                }
            )
    path = output_dir / "model.pt"
    torch.save(student.checkpoint(config, int(config["training"]["steps"])), path)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    _write_manifest(
        output_dir,
        route,
        seed,
        left_checkpoint,
        right_checkpoint,
        transfer_weight=transfer_weight,
        temperature=temperature,
        topk=topk,
        generated_sequences=len(generated or []),
    )
    return path


def _write_manifest(output_dir: Path, route: str, seed: int, left: Path, right: Path, **kwargs) -> None:
    payload = {"route": route, "seed": seed, "left": str(left), "right": str(right), **kwargs}
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch

from .data import SequenceCorpus, UserSequence
from .evaluation import (
    CROSS_FAMILY,
    WITHIN_FAMILY,
    adjacent_families,
    evaluate_models,
    family_average,
    pair_key,
)
from .generation import generate_sequences
from .training import Trainer, TrainingResult, load_model


class RecursiveProtocol:
    """Run the P/B/A/A' protocol and its optional second update."""

    def __init__(self, config: Dict):
        self.config = config
        self.corpus = SequenceCorpus.load(config["data"]["corpus"])
        self.trainer = Trainer(config, self.corpus)
        self.output = Path(config["output_dir"])
        self.device = self.trainer.device

    def run(self) -> Dict:
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        reports = []
        for trajectory in self.config["protocol"]["trajectories"]:
            reports.append(self.run_trajectory(int(trajectory)))
        report = {
            "format": "rec-rsi-protocol-v1",
            "config": self.config,
            "trajectories": reports,
            "summary": summarize_trajectories(reports),
        }
        (self.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    def run_trajectory(self, trajectory: int) -> Dict:
        root = self.output / f"trajectory_{trajectory}"
        root.mkdir(parents=True, exist_ok=False)
        stride = int(self.config["protocol"].get("seed_stride", 1000))
        base = int(self.config.get("seed", 2023)) + trajectory * stride

        p = self.trainer.train("P", base + 1, root / "P")
        generated_1 = self._generate(p.model, base + 11)
        (root / "generated_1.json").write_text(json.dumps(generated_1) + "\n")
        a = self.trainer.train("A", base + 2, root / "A", generated_1)
        b = self.trainer.train("B", base + 2, root / "B")
        aprime = self.trainer.train("Aprime", base + 3, root / "Aprime", generated_1)
        models = {"P": p.model, "B": b.model, "A": a.model, "Aprime": aprime.model}
        windows = self._evaluate_windows(models, CROSS_FAMILY, WITHIN_FAMILY)
        snapshots = self._snapshot_control({"P": p, "A": a})
        result: Dict = {
            "trajectory": trajectory,
            "seeds": {"P": base + 1, "A": base + 2, "B": base + 2, "Aprime": base + 3},
            "generated_1": len(generated_1),
            "first_update": windows,
            "snapshot_control": snapshots,
        }

        if bool(self.config["protocol"].get("second_update", False)):
            generation_corpus = SequenceCorpus(
                [
                    UserSequence(user=index + 1, fit=tuple(sequence))
                    for index, sequence in enumerate(self.corpus.training_sequences(generated_1))
                ],
                self.corpus.num_items,
            )
            generated_2 = self._generate(a.model, base + 21, generation_corpus)
            (root / "generated_2.json").write_text(json.dumps(generated_2) + "\n")
            combined = list(generated_1) + list(generated_2)
            c = self.trainer.train("C", base + 4, root / "C", combined)
            cprime = self.trainer.train("Cprime", base + 5, root / "Cprime", combined)
            second_models = {"A": a.model, "Aprime": aprime.model, "C": c.model, "Cprime": cprime.model}
            cross, within = adjacent_families(("A", "Aprime"), ("C", "Cprime"))
            result["generated_2"] = len(generated_2)
            result["second_update"] = self._evaluate_windows(second_models, cross, within)
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    def _generate(
        self,
        model,
        seed: int,
        corpus: SequenceCorpus = None,
    ) -> List[List[int]]:
        generation = self.config["generation"]
        return generate_sequences(
            model,
            corpus or self.corpus,
            samples_per_history=int(generation["samples_per_history"]),
            fidelity_k=int(generation["fidelity_k"]),
            local_probability=float(generation.get("local_probability", 0.5)),
            min_length=int(generation.get("min_length", 3)),
            max_length=int(self.config["data"]["max_length"]),
            seed=seed,
            device=self.device,
        )

    def _evaluate_windows(self, models, cross_family, within_family) -> Dict:
        output = {}
        for window in ("select", "audit", "confirm", "test"):
            output[window] = evaluate_models(
                models,
                self.corpus.evaluation_examples(window),
                max_length=int(self.config["data"]["max_length"]),
                cutoffs=self.config["evaluation"]["cutoffs"],
                beta=float(self.config["evaluation"].get("beta", 0.5)),
                batch_size=int(self.config["evaluation"].get("batch_size", 512)),
                device=self.device,
                cross_family=cross_family,
                within_family=within_family,
            )
        choice = output["select"]["family"]["selected"]
        primary = output["test"]["family"]["metric"]
        successor = "A" if "A" in output["test"]["single"] else "C"
        output["selection"] = {
            "indicator_window": "select",
            "selected_family": choice,
            "test_selected_utility": output["test"]["family"][choice],
            "test_successor_utility": output["test"]["single"][successor][primary],
        }
        return output

    def _snapshot_control(self, results: Mapping[str, TrainingResult]) -> Dict:
        metric = f"ndcg@{max(self.config['evaluation']['cutoffs'])}"
        candidates = {}
        examples = self.corpus.evaluation_examples("confirm")
        for role, result in results.items():
            checkpoint_pairs = {
                "half+last": (result.half_checkpoint, result.last_checkpoint),
                "best+last": (result.best_checkpoint, result.last_checkpoint),
            }
            for label, (left_path, right_path) in checkpoint_pairs.items():
                models = {
                    "left": load_model(left_path, self.device),
                    "right": load_model(right_path, self.device),
                }
                evaluated = evaluate_models(
                    models,
                    examples,
                    max_length=int(self.config["data"]["max_length"]),
                    cutoffs=self.config["evaluation"]["cutoffs"],
                    beta=float(self.config["evaluation"].get("beta", 0.5)),
                    batch_size=int(self.config["evaluation"].get("batch_size", 512)),
                    device=self.device,
                    cross_family=(("left", "right"),),
                    within_family=(("left", "right"),),
                )
                candidates[f"{role}:{label}"] = evaluated["pairs"][pair_key("left", "right")][metric]
        best = max(candidates, key=candidates.get)
        return {"window": "confirm", "metric": metric, "candidates": candidates, "best": best, "value": candidates[best]}


def summarize_trajectories(reports: Sequence[Dict]) -> Dict:
    first_cga = [row["first_update"]["test"]["family"]["cga"] for row in reports]
    selected_gain = [
        row["first_update"]["selection"]["test_selected_utility"]
        - row["first_update"]["selection"]["test_successor_utility"]
        for row in reports
    ]
    summary = {
        "trajectories": len(reports),
        "first_update_test_cga_mean": sum(first_cga) / len(first_cga),
        "first_update_test_cga_positive": sum(value > 0 for value in first_cga),
        "selected_family_minus_successor_mean": sum(selected_gain) / len(selected_gain),
        "selected_family_beats_successor": sum(value > 0 for value in selected_gain),
    }
    second = [row for row in reports if "second_update" in row]
    if second:
        values = [row["second_update"]["confirm"]["family"]["cga"] for row in second]
        summary.update(
            second_update_confirm_cga_mean=sum(values) / len(values),
            second_update_confirm_cga_positive=sum(value > 0 for value in values),
        )
    return summary

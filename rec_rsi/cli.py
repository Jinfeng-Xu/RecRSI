from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import prepare_interactions
from .consolidation import ROUTES, consolidate
from .data import SequenceCorpus
from .evaluation import evaluate_models
from .generation import generate_sequences
from .protocol import RecursiveProtocol
from .training import load_model, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rec-rsi", description="Recursive recommendation state-retention experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build a RecRSI corpus from chronological interactions")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--user-column", default="user_id")
    prepare.add_argument("--item-column", default="item_id")
    prepare.add_argument("--time-column", default="timestamp")
    prepare.add_argument("--min-events", type=int, default=8)

    run = subparsers.add_parser("run", help="Run the first-update and optional second-update protocol")
    run.add_argument("--config", required=True)

    generate = subparsers.add_parser("generate", help="Generate recursive training sequences from one checkpoint")
    generate.add_argument("--config", required=True)
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--seed", type=int, default=2023)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a P/B/A/A' checkpoint family")
    evaluate.add_argument("--config", required=True)
    for role in ("P", "B", "A", "Aprime"):
        evaluate.add_argument(f"--{role}", required=True)
    evaluate.add_argument("--window", choices=["select", "audit", "confirm", "test"], default="test")
    evaluate.add_argument("--output", required=True)

    summarize = subparsers.add_parser("summarize", help="Print the compact summary from a completed run")
    summarize.add_argument("result")

    merge = subparsers.add_parser("consolidate", help="Train or construct a single-model consolidation route")
    merge.add_argument("--config", required=True)
    merge.add_argument("--left", required=True)
    merge.add_argument("--right", required=True)
    merge.add_argument("--route", choices=sorted(ROUTES), required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--generated", help="JSON list of generated sequences for hard-data")
    merge.add_argument("--seed", type=int, default=2023)
    merge.add_argument("--transfer-weight", type=float, default=1.0)
    merge.add_argument("--temperature", type=float, default=1.0)
    merge.add_argument("--topk", type=int, default=50)
    merge.add_argument("--alpha", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        corpus = prepare_interactions(
            args.input,
            args.output,
            user_column=args.user_column,
            item_column=args.item_column,
            time_column=args.time_column,
            min_events=args.min_events,
        )
        print(json.dumps({"users": len(corpus.users), "num_items": corpus.num_items, "output": args.output}))
    elif args.command == "run":
        report = RecursiveProtocol(load_config(args.config)).run()
        print(json.dumps(report["summary"], indent=2))
    elif args.command == "generate":
        config = load_config(args.config)
        corpus = SequenceCorpus.load(config["data"]["corpus"])
        device = resolve_device(str(config.get("device", "cuda")))
        model = load_model(Path(args.checkpoint), device)
        generation = config["generation"]
        rows = generate_sequences(
            model,
            corpus,
            samples_per_history=int(generation["samples_per_history"]),
            fidelity_k=int(generation["fidelity_k"]),
            local_probability=float(generation.get("local_probability", 0.5)),
            min_length=int(generation.get("min_length", 3)),
            max_length=int(config["data"]["max_length"]),
            seed=args.seed,
            device=device,
        )
        Path(args.output).write_text(json.dumps(rows) + "\n")
        print(json.dumps({"generated": len(rows), "output": args.output}))
    elif args.command == "evaluate":
        config = load_config(args.config)
        corpus = SequenceCorpus.load(config["data"]["corpus"])
        device = resolve_device(str(config.get("device", "cuda")))
        models = {role: load_model(Path(getattr(args, role)), device) for role in ("P", "B", "A", "Aprime")}
        report = evaluate_models(
            models,
            corpus.evaluation_examples(args.window),
            max_length=int(config["data"]["max_length"]),
            cutoffs=config["evaluation"]["cutoffs"],
            beta=float(config["evaluation"].get("beta", 0.5)),
            batch_size=int(config["evaluation"].get("batch_size", 512)),
            device=device,
        )
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["family"], indent=2))
    elif args.command == "summarize":
        report = json.loads(Path(args.result).read_text(encoding="utf-8"))
        print(json.dumps(report["summary"], indent=2))
    elif args.command == "consolidate":
        config = load_config(args.config)
        corpus = SequenceCorpus.load(config["data"]["corpus"])
        generated = json.loads(Path(args.generated).read_text()) if args.generated else None
        path = consolidate(
            config,
            corpus,
            Path(args.left),
            Path(args.right),
            args.route,
            Path(args.output),
            args.seed,
            generated=generated,
            transfer_weight=args.transfer_weight,
            temperature=args.temperature,
            topk=args.topk,
            alpha=args.alpha,
        )
        print(json.dumps({"checkpoint": str(path), "route": args.route}))


if __name__ == "__main__":
    main()

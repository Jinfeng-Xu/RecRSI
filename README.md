# RecRSI

Official research code for **state retention in recommendation recursive self-improvement**. The repository implements the complete experimental object used in the accompanying paper: recursive sequence generation, pre/post-update model families, cross-generation advantage (CGA), rank separation, snapshot controls, a second recursive update, and five single-model consolidation routes.

## What is implemented

- Three sequential recommenders: GRU4Rec, SASRec, and FMLP.
- The `P/B/A/A'` matched design. `P` and `B` are independently trained before an update; `P` generates the data used by post-update models `A` and `A'`.
- Rank-faithful recursive generation with `m` continuations and a top-`k` fidelity condition.
- Full-catalog log-probability fusion, Recall/NDCG at arbitrary cutoffs, top-k overlap, unique-hit diagnostics, CGA, and label-free rank separation.
- Same-model snapshot controls with equal two-model inference cost.
- An optional second update `A/A' -> C/C'`.
- Hard-data retraining, full-catalog KL distillation, top-k listwise distillation, complementary targets, and parameter interpolation.

## Repository layout

```text
rec_rsi/                 core library
  models/                GRU4Rec, SASRec, FMLP
  generation.py          recursive sequence generator
  evaluation.py          matched families, CGA, rank separation
  protocol.py            first- and second-update experiment runner
  consolidation.py       five single-model transfer routes
configs/                 paper-scale and smoke configurations
examples/                deterministic end-to-end example
```

## Installation

Python 3.9+ and PyTorch 2.0+ are supported.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data

The input is a chronological interaction CSV with user, item, and timestamp columns. The preparation command maps raw ids, reserves the final four interactions as selection, audit, confirmation, and held-out test windows, and stores the earlier prefix for training.

```bash
rec-rsi prepare \
  --input data/raw/interactions.csv \
  --output data/processed/amazon-toys.json \
  --min-events 8
```

Column names can be changed with `--user-column`, `--item-column`, and `--time-column`. The processed schema is documented in [DATA.md](DATA.md).

## Run the recursive protocol

Set the corpus path and model in a YAML file, then run:

```bash
rec-rsi run --config configs/paper.yaml
```

`results.json` contains every model, pair, family contrast, selection decision, and snapshot control for each trajectory. Set `protocol.second_update: true` to train `C/C'` from a second generated dataset. The compact CPU example runs the same path:

```bash
python examples/make_tiny_data.py
rec-rsi prepare --input examples/tiny_interactions.csv --output examples/tiny_corpus.json --min-events 8
rec-rsi run --config configs/smoke.yaml
```

## Single-model consolidation

Each route consumes two compatible teacher checkpoints. `hard-data` additionally consumes the generated-sequence JSON emitted by the protocol.

```bash
rec-rsi consolidate \
  --config configs/paper.yaml \
  --left runs/paper/trajectory_1/P/best.pt \
  --right runs/paper/trajectory_1/A/best.pt \
  --route listwise \
  --output runs/consolidation/listwise
```

Available routes are `hard-data`, `full-kl`, `listwise`, `complementary`, and `interpolate`.
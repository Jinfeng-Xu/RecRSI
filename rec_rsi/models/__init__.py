from __future__ import annotations

from typing import Any, Dict

from .base import SequentialRecommender
from .fmlp import FMLP
from .gru4rec import GRU4Rec
from .sasrec import SASRec

MODELS = {"gru4rec": GRU4Rec, "sasrec": SASRec, "fmlp": FMLP}


def build_model(num_items: int, config: Dict[str, Any]) -> SequentialRecommender:
    name = config["model"]["name"].lower()
    model_config = config["model"]
    return MODELS[name](
        num_items=num_items,
        embedding_dim=int(model_config.get("embedding_dim", 64)),
        max_length=int(config["data"].get("max_length", 50)),
        hidden_dim=int(model_config.get("hidden_dim", 256)),
        num_layers=int(model_config.get("num_layers", 2)),
        num_heads=int(model_config.get("num_heads", 2)),
        dropout=float(model_config.get("dropout", 0.2)),
    )


__all__ = ["SequentialRecommender", "GRU4Rec", "SASRec", "FMLP", "build_model"]


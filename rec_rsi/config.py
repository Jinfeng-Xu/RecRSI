from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a mapping")

    base = config_path.parent
    for section, key in (("data", "corpus"),):
        value = config.get(section, {}).get(key)
        if value and not Path(value).is_absolute():
            candidate = (base / value).resolve()
            if not candidate.exists():
                candidate = (config_path.parent.parent / value).resolve()
            config[section][key] = str(candidate)
    output = config.get("output_dir", "runs/default")
    if not Path(output).is_absolute():
        config["output_dir"] = str((config_path.parent.parent / output).resolve())
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    required = ("data", "model", "training", "generation", "evaluation", "protocol")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError("Missing configuration sections: " + ", ".join(missing))
    if config["model"]["name"].lower() not in {"gru4rec", "sasrec", "fmlp"}:
        raise ValueError("model.name must be gru4rec, sasrec, or fmlp")
    beta = float(config["evaluation"].get("beta", 0.5))
    if not 0.0 <= beta <= 1.0:
        raise ValueError("evaluation.beta must be in [0, 1]")
    if int(config["generation"].get("fidelity_k", 20)) < 1:
        raise ValueError("generation.fidelity_k must be positive")


def with_overrides(config: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    updated = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        cursor = updated
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    validate_config(updated)
    return updated


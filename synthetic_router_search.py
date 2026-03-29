"""YAML-backed experiment config loading and random sampling."""

import math
import random
from pathlib import Path
from typing import Any, Dict

import yaml

from synthetic_router_settings import DEFAULT_OPTIMIZERS


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return 10 ** rng.uniform(math.log10(low), math.log10(high))


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object at top level in {path}")
    return payload


def validate_distribution_spec(name: str, spec: Any) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError(f"Search space '{name}' must be a mapping.")
    spec_type = spec.get("type")
    if spec_type not in {"choice", "uniform", "log_uniform"}:
        raise ValueError(f"Search space '{name}' has unsupported type '{spec_type}'.")
    if spec_type == "choice":
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Search space '{name}' choice spec must define non-empty 'values'.")
    else:
        low = spec.get("low")
        high = spec.get("high")
        if low is None or high is None:
            raise ValueError(f"Search space '{name}' {spec_type} spec must define 'low' and 'high'.")
        if float(low) >= float(high):
            raise ValueError(f"Search space '{name}' requires low < high.")
    return spec


def load_search_space(payload_or_path) -> Dict[str, Any]:
    if isinstance(payload_or_path, (str, Path)):
        payload = load_yaml(Path(payload_or_path))
    else:
        payload = payload_or_path

    search_space = payload.get("search_space", payload)
    if not isinstance(search_space, dict):
        raise ValueError("Expected 'search_space' to be a mapping in the config YAML.")
    shared = search_space.get("shared")
    families = search_space.get("families")
    if not isinstance(shared, dict):
        raise ValueError("Search space YAML must contain a 'shared' mapping.")
    if not isinstance(families, dict):
        raise ValueError("Search space YAML must contain a 'families' mapping.")

    expert_num_options = shared.get("expert_num_options")
    if not isinstance(expert_num_options, list) or not expert_num_options:
        raise ValueError("'shared.expert_num_options' must be a non-empty list.")

    for name in [
        "expert_lr",
        "expert_momentum",
        "expert_weight_decay",
        "router_momentum",
        "router_weight_decay",
    ]:
        validate_distribution_spec(f"shared.{name}", shared.get(name))

    for family in DEFAULT_OPTIMIZERS:
        family_spec = families.get(family)
        if not isinstance(family_spec, dict):
            raise ValueError(f"Search space YAML must define families.{family}.")
        validate_distribution_spec(f"families.{family}.router_lr", family_spec.get("router_lr"))
        if family == "muon":
            validate_distribution_spec(f"families.{family}.ns_steps", family_spec.get("ns_steps"))

    return {"shared": shared, "families": families}


def sample_from_spec(name: str, spec: Dict[str, Any], rng: random.Random) -> Any:
    spec = validate_distribution_spec(name, spec)
    spec_type = spec["type"]
    if spec_type == "choice":
        return rng.choice(spec["values"])
    if spec_type == "uniform":
        return rng.uniform(float(spec["low"]), float(spec["high"]))
    if spec_type == "log_uniform":
        return log_uniform(rng, float(spec["low"]), float(spec["high"]))
    raise ValueError(f"Unsupported spec type '{spec_type}' for '{name}'.")


def sample_search_config(
    family: str,
    search_space: Dict[str, Any],
    rng: random.Random,
    epochs: int,
    load_balancing: bool,
    early_stopping: bool,
) -> Dict[str, Any]:
    shared = search_space["shared"]
    family_space = search_space["families"][family]
    config = {
        "optimizer_family": family,
        "expert_num": int(rng.choice(list(shared["expert_num_options"]))),
        "epochs": epochs,
        "load_balancing": load_balancing,
        "early_stopping": early_stopping,
        "expert_lr": round(float(sample_from_spec("shared.expert_lr", shared["expert_lr"], rng)), 8),
        "expert_momentum": round(float(sample_from_spec("shared.expert_momentum", shared["expert_momentum"], rng)), 6),
        "expert_weight_decay": round(
            float(sample_from_spec("shared.expert_weight_decay", shared["expert_weight_decay"], rng)), 8
        ),
        "router_momentum": round(float(sample_from_spec("shared.router_momentum", shared["router_momentum"], rng)), 6),
        "router_weight_decay": round(
            float(sample_from_spec("shared.router_weight_decay", shared["router_weight_decay"], rng)), 8
        ),
        "router_lr": round(float(sample_from_spec(f"families.{family}.router_lr", family_space["router_lr"], rng)), 8),
    }
    if family == "muon":
        config["ns_steps"] = int(sample_from_spec("families.muon.ns_steps", family_space["ns_steps"], rng))
    return config

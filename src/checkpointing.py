from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import torch

from config import ExpConfig
from src.models import build_model

MODEL_CONFIG_FIELDS = {
    "num_ues",
    "num_trps",
    "antennas_per_trp",
    "embed_dim",
    "dropout",
    "eta",
    "fairness_alpha",
    "eps",
    "ssa_learned_correction_scale",
    "ssa_future_rate_residual_scale",
}


def _coerce_like(value: Any, reference: Any) -> Any:
    if isinstance(reference, bool):
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    if isinstance(reference, tuple):
        return tuple(value)
    if isinstance(reference, Path):
        return Path(value)
    return value


def config_for_checkpoint(runtime_cfg: ExpConfig, saved_cfg: dict[str, Any] | None) -> ExpConfig:
    """Merge saved model settings into the runtime configuration."""
    model_cfg = replace(runtime_cfg)
    if not saved_cfg:
        return model_cfg
    valid_fields = {f.name for f in fields(ExpConfig)}
    for key in MODEL_CONFIG_FIELDS:
        if key in saved_cfg and key in valid_fields:
            current = getattr(model_cfg, key)
            setattr(model_cfg, key, _coerce_like(saved_cfg[key], current))
    return model_cfg


def validate_dataset_meta(model_cfg: ExpConfig, meta: dict[str, Any]) -> None:
    expected = {
        "num_ues": model_cfg.num_ues,
        "pmi_real_dim": model_cfg.pmi_real_dim,
        "feature_dim": model_cfg.feature_dim,
    }
    mismatches = []
    for key, exp in expected.items():
        if key in meta and int(meta[key]) != int(exp):
            mismatches.append(f"{key}: checkpoint expects {exp}, dataset has {meta[key]}")
    if mismatches:
        raise ValueError("Checkpoint/dataset incompatibility: " + "; ".join(mismatches))


def load_model_from_checkpoint(
    model_name: str,
    ckpt_path: Path | str,
    runtime_cfg: ExpConfig,
    device: torch.device,
):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_name = str(ckpt.get("model_name", model_name)).lower()
    requested = model_name.lower()
    def canonical(name: str) -> str:
        normalized = name.lower().replace("-", "_")
        if normalized in {"ssa", "ssanet"}:
            return "ssa"
        return normalized

    if canonical(saved_name) != canonical(requested):
        raise ValueError(f"Checkpoint model '{saved_name}' does not match requested model '{model_name}'.")
    model_cfg = config_for_checkpoint(runtime_cfg, ckpt.get("cfg"))
    model = build_model(model_name, model_cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model, model_cfg, ckpt

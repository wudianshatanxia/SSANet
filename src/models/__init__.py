from __future__ import annotations

from config import ExpConfig
from src.models.lstm import LSTMScoreNet
from src.models.ssa_net import SSANet


def build_model(name: str, cfg: ExpConfig):
    name = name.lower()
    if name == "lstm":
        return LSTMScoreNet(cfg.pmi_real_dim, hidden_dim=cfg.embed_dim, dropout=cfg.dropout)
    if name in {"ssa", "ssa-net", "ssanet"}:
        return SSANet(
            cfg.pmi_real_dim,
            cfg.num_trps,
            cfg.antennas_per_trp,
            cfg.embed_dim,
            cfg.dropout,
            eta=cfg.eta,
            fairness_alpha=cfg.fairness_alpha,
            eps=cfg.eps,
            learned_correction_scale=getattr(cfg, "ssa_learned_correction_scale", 0.45),
            future_rate_residual_scale=getattr(cfg, "ssa_future_rate_residual_scale", 0.35),
        )
    raise ValueError(f"Unknown model name: {name}")

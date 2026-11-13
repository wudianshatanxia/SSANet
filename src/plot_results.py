from __future__ import annotations

"""Evaluate and plot scheduling performance."""

from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import ExpConfig
from src.baselines import (
    set_sinr_greedy_scores,
    learned_set_sinr_refinement_scores,
    no_prediction_scores,
    pf_prediction_scores,
)
from src.checkpointing import load_model_from_checkpoint, validate_dataset_meta
from src.dataset import SchedulingDataset
from src.metrics import numpy_complex_correlation_matrix, numpy_topk_mask
from src.mobility import temporal_channel_correlation

PAPER_METHODS = [
    "No Prediction",
    "PF Prediction",
    "Enhanced Greedy",
    "LSTM",
    "SSA-Net",
]
ORACLE_METHOD = "Future-CSI Greedy (Oracle)"
FIG3_METHODS = PAPER_METHODS + [ORACLE_METHOD]

METHOD_COLORS = {
    "No Prediction": "#6B7280",
    "PF Prediction": "#2CA02C",
    "Enhanced Greedy": "#1F77B4",
    "LSTM": "#8B5CF6",
    "SSA-Net": "#D62728",
    ORACLE_METHOD: "#111827",
}
METHOD_MARKERS = {
    "No Prediction": "o",
    "PF Prediction": "s",
    "Enhanced Greedy": "^",
    "LSTM": "D",
    "SSA-Net": "*",
    ORACLE_METHOD: "X",
}
METHOD_LINESTYLES = {
    "No Prediction": "-",
    "PF Prediction": "-",
    "Enhanced Greedy": "-",
    "LSTM": "-",
    "SSA-Net": "-",
    ORACLE_METHOD: "--",
}


def _set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.7,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.2,
            "figure.titlesize": 12.5,
            "axes.linewidth": 0.85,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.30,
            "lines.solid_capstyle": "round",
        }
    )


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved {path}")


def _device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pretty_model_name(model_name: str) -> str:
    key = model_name.lower().replace("-", "_")
    mapping = {
        "ssa": "SSA-Net",
        "ssanet": "SSA-Net",
        "lstm": "LSTM",
    }
    return mapping.get(key, model_name)


def _load_eval_arrays(data_dir: Path | str, cfg: ExpConfig) -> dict[str, np.ndarray | dict]:
    ds = SchedulingDataset(data_dir, split="test")
    idx = ds.indices
    x = np.asarray(ds.features[idx], dtype=np.float32)
    rate_now = np.asarray(ds.rate_now[idx], dtype=np.float32)
    rate_future = np.asarray(ds.rate_future[idx], dtype=np.float32)
    rate_pred = (
        np.asarray(ds.rate_pred[idx], dtype=np.float32)
        if getattr(ds, "rate_pred", None) is not None
        else x[:, :, cfg.idx_future_rate].astype(np.float32)
    )
    hist_rate = np.asarray(ds.hist_rate[idx], dtype=np.float32)
    pmi_now = x[:, :, cfg.idx_now_pmi_start : cfg.idx_now_pmi_start + cfg.pmi_real_dim]
    pmi_pred = x[:, :, cfg.idx_future_pmi_start : cfg.idx_future_pmi_start + cfg.pmi_real_dim]
    if getattr(ds, "pmi_future", None) is not None:
        pmi_pair = np.asarray(ds.pmi_future[idx], dtype=np.float32)
        pmi_future = np.concatenate([pmi_pair[..., 0], pmi_pair[..., 1]], axis=-1)
    else:
        pmi_future = pmi_pred.copy()
    return {
        "x": x,
        "rate_now": rate_now,
        "rate_pred": rate_pred,
        "rate_future": rate_future,
        "hist_rate": hist_rate,
        "pmi_now": pmi_now,
        "pmi_pred": pmi_pred,
        "pmi_future": pmi_future,
        "meta": ds.meta,
    }


def _flat_to_complex(flat: np.ndarray) -> np.ndarray:
    flat = np.asarray(flat, dtype=np.float32)
    if flat.shape[-1] % 2:
        raise ValueError("Real/imag-stacked PMI must have an even last dimension")
    c = flat.shape[-1] // 2
    return flat[..., :c].astype(np.float32) + 1j * flat[..., c:].astype(np.float32)


def _complex_to_flat(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x.real, x.imag], axis=-1).astype(np.float32)


def _normalize_complex(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def _orthogonal_unit_noise(reference: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate unit-norm channel noise orthogonal to reference vectors."""
    z = (rng.normal(size=reference.shape) + 1j * rng.normal(size=reference.shape)) / np.sqrt(2.0)
    projection = np.sum(np.conjugate(reference) * z, axis=-1, keepdims=True)
    z = z - projection * reference
    norm = np.linalg.norm(z, axis=-1, keepdims=True)
    fallback = (rng.normal(size=reference.shape) + 1j * rng.normal(size=reference.shape)) / np.sqrt(2.0)
    fallback = fallback - np.sum(np.conjugate(reference) * fallback, axis=-1, keepdims=True) * reference
    z = np.where(norm > 1e-8, z, fallback)
    return _normalize_complex(z).astype(np.complex64)


def _jain_index(rate: np.ndarray, eps: float = 1e-12) -> float:
    r = np.maximum(np.asarray(rate, dtype=np.float64), 0.0)
    numerator = float(r.sum() ** 2)
    denominator = float(r.size * np.square(r).sum())
    return numerator / max(denominator, eps)


def _mean_ci95(values: Sequence[float]) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan")
    mean = float(x.mean())
    if x.size == 1:
        return mean, 0.0
    ci = 1.96 * float(x.std(ddof=1)) / np.sqrt(float(x.size))
    return mean, ci


def _speed_prediction_nmse_db(speed_kmh: float, cfg: ExpConfig) -> float:
    max_speed = max(float(max(cfg.speeds_kmh)), 1.0)
    speed_norm = float(np.clip(float(speed_kmh) / max_speed, 0.0, 1.0))
    skill_max = float(getattr(cfg, "mobility_prediction_skill_max", 0.93))
    skill_min = float(getattr(cfg, "mobility_prediction_skill_min", 0.48))
    skill = skill_min + (skill_max - skill_min) * (1.0 - speed_norm) ** 0.75
    nmse_linear = max((1.0 - skill) ** 2, 1e-5)
    return float(10.0 * np.log10(nmse_linear))


def _prediction_from_true_channel(
    h_true: np.ndarray,
    snr_true: np.ndarray,
    target_nmse_db: float,
    rng: np.random.Generator,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Generate PMI and rate predictions with channel-estimation error."""
    nmse_linear = float(10.0 ** (float(target_nmse_db) / 10.0))
    nmse_linear = max(nmse_linear, 1e-10)
    direction_error = _orthogonal_unit_noise(h_true, rng)
    gain_error = (
        rng.normal(size=h_true.shape[:-1]) + 1j * rng.normal(size=h_true.shape[:-1])
    ) / np.sqrt(2.0)
    amplitude = np.sqrt(nmse_linear / 2.0)
    channel_error = (
        amplitude * gain_error[..., None] * h_true
        + amplitude * direction_error
    )
    h_hat_raw = h_true + channel_error
    h_pred = _normalize_complex(h_hat_raw, eps).astype(np.complex64)

    gain_ratio = np.abs(1.0 + amplitude * gain_error) ** 2
    gain_ratio = np.clip(gain_ratio, 0.02, 20.0)
    snr_pred = np.maximum(snr_true * gain_ratio.astype(np.float32), 0.0)
    rate_pred = np.log2(1.0 + snr_pred).astype(np.float32)

    pmi_nmse = float(
        np.mean(np.square(np.abs(channel_error)))
        / max(float(np.mean(np.square(np.abs(h_true)))), eps)
    )
    rate_true = np.log2(1.0 + np.maximum(snr_true, 0.0))
    rate_nmse = float(
        np.mean(np.square(rate_pred - rate_true))
        / max(float(np.mean(np.square(rate_true))), eps)
    )
    return h_pred, rate_pred, pmi_nmse, rate_nmse


def _prepare_episode_context(
    arrays: dict[str, np.ndarray | dict],
    cfg: ExpConfig,
    num_episodes: int,
    seed: int,
) -> dict[str, np.ndarray]:
    num_samples = int(np.asarray(arrays["rate_now"]).shape[0])
    if num_samples < 1:
        raise ValueError("The evaluation dataset is empty")
    num_episodes = max(1, int(num_episodes))
    rng = np.random.default_rng(int(seed))
    replace = num_episodes > num_samples
    indices = rng.choice(num_samples, size=num_episodes, replace=replace)
    pmi_now = _normalize_complex(_flat_to_complex(np.asarray(arrays["pmi_future"])[indices]), cfg.eps)
    rate_now = np.maximum(np.asarray(arrays["rate_now"])[indices], 0.0)
    base_snr = np.maximum(np.expm1(np.log(2.0) * np.clip(rate_now, 0.0, 30.0)), 1e-8)
    initial_history = np.maximum(np.asarray(arrays["hist_rate"])[indices], cfg.eps).astype(np.float32)

    pool_size = min(max(4 * num_episodes, 32), num_samples)
    pool_indices = rng.choice(num_samples, size=pool_size, replace=pool_size > num_samples)
    innovation_pool = _normalize_complex(
        _flat_to_complex(np.asarray(arrays["pmi_future"])[pool_indices]),
        cfg.eps,
    ).astype(np.complex64)
    return {
        "indices": indices.astype(np.int64),
        "base_x": np.asarray(arrays["x"])[indices].astype(np.float32, copy=True),
        "initial_h": pmi_now.astype(np.complex64),
        "innovation_h_pool": innovation_pool,
        "base_snr": base_snr.astype(np.float32),
        "initial_history": initial_history,
    }


def _iter_persistent_episode_states(
    context: dict[str, np.ndarray],
    cfg: ExpConfig,
    num_slots: int,
    speed_kmh: float,
    seed: int,
    target_prediction_nmse_db: float | None = None,
) -> Iterator[dict[str, np.ndarray | float]]:
    """Yield channel states for persistent-user episodes."""
    rng = np.random.default_rng(int(seed))
    h_current = np.array(context["initial_h"], dtype=np.complex64, copy=True)
    base_snr = np.asarray(context["base_snr"], dtype=np.float32)
    fading_state = rng.normal(size=base_snr.shape).astype(np.float32)
    rho = float(temporal_channel_correlation(float(speed_kmh), cfg))
    fading_std = float(getattr(cfg, "paper_episode_fading_std", 0.28))
    lognormal_mean_correction = 0.5 * fading_std * fading_std
    target_nmse_db = (
        _speed_prediction_nmse_db(speed_kmh, cfg)
        if target_prediction_nmse_db is None
        else float(target_prediction_nmse_db)
    )

    for _ in range(max(1, int(num_slots))):
        rate_now = np.log2(
            1.0 + base_snr * np.exp(
                np.clip(fading_std * fading_state - lognormal_mean_correction, -3.0, 3.0)
            )
        ).astype(np.float32)

        phase_std = np.sqrt(max(-2.0 * np.log(max(rho, 1e-8)), 0.0))
        phase = rng.normal(
            scale=phase_std,
            size=(h_current.shape[0], 1, h_current.shape[2]),
        ).astype(np.float32)
        h_future = (h_current * np.exp(1j * phase)).astype(np.complex64)

        fading_noise = rng.normal(size=fading_state.shape).astype(np.float32)
        fading_future = (
            rho * fading_state + np.sqrt(max(1.0 - rho * rho, 0.0)) * fading_noise
        ).astype(np.float32)
        snr_future = base_snr * np.exp(
            np.clip(fading_std * fading_future - lognormal_mean_correction, -3.0, 3.0)
        )
        rate_future = np.log2(1.0 + np.maximum(snr_future, 0.0)).astype(np.float32)

        h_pred, rate_pred, pmi_nmse, rate_nmse = _prediction_from_true_channel(
            h_future,
            snr_future,
            target_nmse_db,
            rng,
            cfg.eps,
        )
        yield {
            "pmi_now": _complex_to_flat(h_current),
            "pmi_pred": _complex_to_flat(h_pred),
            "pmi_future": _complex_to_flat(h_future),
            "rate_now": rate_now,
            "rate_pred": rate_pred,
            "rate_future": rate_future,
            "speed_kmh": float(speed_kmh),
            "target_prediction_nmse_db": float(target_nmse_db),
            "actual_pmi_nmse": float(pmi_nmse),
            "actual_rate_nmse": float(rate_nmse),
        }
        h_current = h_future
        fading_state = fading_future


def _build_features(
    base_x: np.ndarray,
    state: dict[str, np.ndarray | float],
    history: np.ndarray,
    cfg: ExpConfig,
) -> np.ndarray:
    x = np.array(base_x, dtype=np.float32, copy=True)
    x[:, :, cfg.idx_now_pmi_start : cfg.idx_now_pmi_start + cfg.pmi_real_dim] = state["pmi_now"]
    x[:, :, cfg.idx_now_rate] = state["rate_now"]
    x[:, :, cfg.idx_future_pmi_start : cfg.idx_future_pmi_start + cfg.pmi_real_dim] = state["pmi_pred"]
    x[:, :, cfg.idx_future_rate] = state["rate_pred"]
    x[:, :, cfg.idx_hist_rate] = history
    max_speed = max(float(max(cfg.speeds_kmh)), 1.0)
    x[:, :, cfg.idx_speed] = float(state["speed_kmh"]) / max_speed
    return x


def _load_episode_models(
    ckpt_dir: Path | str,
    cfg: ExpConfig,
    models: Iterable[str],
    meta: dict,
    device: Optional[str],
) -> tuple[dict[str, torch.nn.Module], torch.device]:
    dev = _device(device)
    loaded: dict[str, torch.nn.Module] = {}
    ckpt_dir = Path(ckpt_dir)
    for model_name in models:
        pretty = _pretty_model_name(model_name)
        if pretty not in {"LSTM", "SSA-Net"}:
            continue
        ckpt = ckpt_dir / f"{model_name.lower()}_best.pt"
        if not ckpt.exists():
            print(f"[plot] skip {pretty}: missing checkpoint {ckpt}")
            continue
        model, model_cfg, _ = load_model_from_checkpoint(model_name, ckpt, cfg, dev)
        validate_dataset_meta(model_cfg, meta)
        model.eval()
        loaded[pretty] = model
    return loaded, dev


def _infer_loaded_model(
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    eta_override: float | None = None,
    emergency_priority: np.ndarray | None = None,
) -> np.ndarray:
    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    with torch.no_grad():
        priority_t = None if emergency_priority is None else torch.from_numpy(
            np.asarray(emergency_priority, dtype=np.float32)
        ).to(device)
        if hasattr(model, "forward_with_aux") and (eta_override is not None or priority_t is not None):
            kwargs = {}
            if eta_override is not None:
                kwargs["eta_override"] = float(eta_override)
            if priority_t is not None:
                kwargs["emergency_priority"] = priority_t
            logits = model.forward_with_aux(x_t, **kwargs)["logits"]
        else:
            logits = model(x_t)
    return logits.detach().cpu().numpy().astype(np.float32)


def _selected_set_sinr_metrics(
    scores: np.ndarray,
    state: dict[str, np.ndarray | float],
    topk: int,
    cfg: ExpConfig,
) -> dict[str, np.ndarray]:
    """Evaluate selected-user rates under shared power and spatial leakage."""
    mask = numpy_topk_mask(scores, topk)
    selected_idx = np.argpartition(-mask, kth=topk - 1, axis=1)[:, :topk]
    rows = np.arange(mask.shape[0])[:, None]
    selected_pmi = np.asarray(state["pmi_future"], dtype=np.float32)[rows, selected_idx]
    corr = numpy_complex_correlation_matrix(selected_pmi, cfg.eps)
    diag = np.arange(topk)
    corr[:, diag, diag] = 0.0
    corr_power = np.square(corr, dtype=np.float32)

    rate_future = np.asarray(state["rate_future"], dtype=np.float32)
    selected_rate = rate_future[rows, selected_idx]
    standalone_snr = np.expm1(np.log(2.0) * np.clip(selected_rate, 0.0, 30.0))
    configured_reference = int(getattr(cfg, "paper_power_reference_users", 0))
    reference_streams = float(
        configured_reference if configured_reference > 0 else cfg.schedule_users
    )
    reference_streams = max(reference_streams, 1.0)
    stream_snr = standalone_snr * (reference_streams / float(topk))
    leakage = corr_power.sum(axis=2)
    beta = float(getattr(cfg, "paper_set_interference_beta", 0.40))
    sinr = stream_snr / np.maximum(1.0 + beta * stream_snr * leakage, cfg.eps)
    selected_effective_rate = np.log2(1.0 + np.maximum(sinr, 0.0)).astype(np.float32)
    selected_no_interference_rate = np.log2(1.0 + np.maximum(stream_snr, 0.0)).astype(np.float32)

    contribution = np.zeros_like(mask, dtype=np.float32)
    raw_contribution = np.zeros_like(mask, dtype=np.float32)
    contribution[rows, selected_idx] = selected_effective_rate
    raw_contribution[rows, selected_idx] = selected_no_interference_rate
    pair_denom = max(topk * (topk - 1), 1)
    mean_pair_corr = corr.sum(axis=(1, 2)) / float(pair_denom)
    return {
        "mask": mask,
        "contribution": contribution,
        "raw_contribution": raw_contribution,
        "per_sample_sum_rate": contribution.sum(axis=1),
        "per_sample_raw_sum_rate": raw_contribution.sum(axis=1),
        "mean_pair_corr": mean_pair_corr.astype(np.float32),
    }


def _method_scores(
    method: str,
    state: dict[str, np.ndarray | float],
    history: np.ndarray,
    base_x: np.ndarray,
    topk: int,
    cfg: ExpConfig,
    loaded_models: dict[str, torch.nn.Module],
    device: torch.device,
    eta: float | None = None,
    fairness_alpha: float | None = None,
    priority_weights: np.ndarray | None = None,
) -> np.ndarray:
    eta_value = float(cfg.eta if eta is None else eta)
    alpha_value = float(cfg.fairness_alpha if fairness_alpha is None else fairness_alpha)
    rate_now = np.asarray(state["rate_now"], dtype=np.float32)
    rate_pred = np.asarray(state["rate_pred"], dtype=np.float32)
    priority = None if priority_weights is None else np.asarray(priority_weights, dtype=np.float32)
    if priority is not None and priority.shape != rate_now.shape:
        raise ValueError("priority_weights must have shape (B,K)")
    if method == "No Prediction":
        scores = no_prediction_scores(rate_now, history, cfg.eps, alpha_value)
        return scores if priority is None else scores * priority
    if method == "PF Prediction":
        scores = pf_prediction_scores(
            rate_now,
            rate_pred,
            history,
            eta_value,
            cfg.eps,
            alpha_value,
        )
        return scores if priority is None else scores * priority
    if method == "Enhanced Greedy":
        return set_sinr_greedy_scores(
            rate_now,
            rate_pred,
            history,
            np.asarray(state["pmi_pred"], dtype=np.float32),
            eta_value,
            topk,
            eps=cfg.eps,
            fairness_alpha=alpha_value,
            interference_beta=getattr(cfg, "paper_set_interference_beta", 0.40),
            power_reference_users=(getattr(cfg, "paper_power_reference_users", 0) or cfg.schedule_users),
            priority_weights=priority,
        )
    if method == ORACLE_METHOD:
        return set_sinr_greedy_scores(
            np.asarray(state["rate_future"], dtype=np.float32),
            np.asarray(state["rate_future"], dtype=np.float32),
            history,
            np.asarray(state["pmi_future"], dtype=np.float32),
            1.0,
            topk,
            eps=cfg.eps,
            fairness_alpha=alpha_value,
            interference_beta=getattr(cfg, "paper_set_interference_beta", 0.40),
            power_reference_users=(getattr(cfg, "paper_power_reference_users", 0) or cfg.schedule_users),
            priority_weights=priority,
        )
    if method in loaded_models:
        x = _build_features(base_x, state, history, cfg)
        eta_override = eta_value if method == "SSA-Net" and eta is not None else None
        raw_scores = _infer_loaded_model(
            loaded_models[method],
            x,
            device,
            eta_override=eta_override,
            emergency_priority=(priority if method == "SSA-Net" else None),
        )
        if priority is not None and method != "SSA-Net":
            raw_scores = raw_scores + float(getattr(cfg, "emergency_priority_logit_scale", 1.0)) * np.log(
                np.maximum(priority, cfg.eps)
            )
        if method == "SSA-Net" and bool(getattr(cfg, "ssa_use_set_decoder", True)):
            return learned_set_sinr_refinement_scores(
                raw_scores,
                rate_now,
                rate_pred,
                history,
                np.asarray(state["pmi_pred"], dtype=np.float32),
                eta_value,
                topk,
                eps=cfg.eps,
                fairness_alpha=alpha_value,
                interference_beta=getattr(cfg, "paper_set_interference_beta", 0.40),
                power_reference_users=(getattr(cfg, "paper_power_reference_users", 0) or cfg.schedule_users),
                candidate_multiplier=getattr(cfg, "ssa_decoder_candidate_multiplier", 3),
                learned_weight=getattr(cfg, "ssa_decoder_learned_weight", 0.70),
                priority_weights=priority,
            )
        return raw_scores
    raise KeyError(f"Unsupported or unavailable method: {method}")


def _evaluate_methods_over_persistent_episodes(
    arrays: dict[str, np.ndarray | dict],
    ckpt_dir: Path | str,
    cfg: ExpConfig,
    models: Iterable[str],
    device: Optional[str],
    topk: int,
    speed_kmh: float,
    num_episodes: int,
    num_slots: int,
    seed: int,
    include_oracle: bool = False,
    target_prediction_nmse_db: float | None = None,
    compute_ideal_overlap: bool = False,
    loaded_bundle: tuple[dict[str, torch.nn.Module], torch.device] | None = None,
) -> list[dict]:
    context = _prepare_episode_context(arrays, cfg, num_episodes, seed)
    if loaded_bundle is None:
        loaded_models, dev = _load_episode_models(
            ckpt_dir, cfg, models, arrays["meta"], device
        )
    else:
        loaded_models, dev = loaded_bundle
    methods = ["No Prediction", "PF Prediction", "Enhanced Greedy"]
    methods.extend([name for name in ("LSTM", "SSA-Net") if name in loaded_models])
    if include_oracle:
        methods.append(ORACLE_METHOD)

    history_update = float(getattr(cfg, "paper_history_update", 0.05))
    histories = {
        method: np.array(context["initial_history"], dtype=np.float32, copy=True)
        for method in methods
    }
    accumulated = {
        method: np.zeros_like(context["initial_history"], dtype=np.float64)
        for method in methods
    }
    per_episode_sum = {
        method: np.zeros(context["initial_history"].shape[0], dtype=np.float64)
        for method in methods
    }
    raw_per_episode_sum = {
        method: np.zeros(context["initial_history"].shape[0], dtype=np.float64)
        for method in methods
    }
    pair_corr_sum = {method: 0.0 for method in methods}
    oracle_overlap_sum = {method: 0.0 for method in methods}
    ideal_overlap_sum = {method: 0.0 for method in methods}
    actual_pmi_nmse_values: list[float] = []
    actual_rate_nmse_values: list[float] = []

    states = _iter_persistent_episode_states(
        context,
        cfg,
        num_slots,
        speed_kmh,
        seed + 10_003,
        target_prediction_nmse_db=target_prediction_nmse_db,
    )
    for state in states:
        actual_pmi_nmse_values.append(float(state["actual_pmi_nmse"]))
        actual_rate_nmse_values.append(float(state["actual_rate_nmse"]))
        scores: dict[str, np.ndarray] = {}
        metrics: dict[str, dict[str, np.ndarray]] = {}
        for method in methods:
            scores[method] = _method_scores(
                method,
                state,
                histories[method],
                context["base_x"],
                topk,
                cfg,
                loaded_models,
                dev,
            )
            metrics[method] = _selected_set_sinr_metrics(scores[method], state, topk, cfg)

        if compute_ideal_overlap:
            ideal_state = dict(state)
            ideal_state["pmi_pred"] = state["pmi_future"]
            ideal_state["rate_pred"] = state["rate_future"]
            for method in methods:
                if method in {"No Prediction", ORACLE_METHOD}:
                    ideal_overlap_sum[method] += 1.0
                    continue
                ideal_scores = _method_scores(
                    method,
                    ideal_state,
                    histories[method],
                    context["base_x"],
                    topk,
                    cfg,
                    loaded_models,
                    dev,
                )
                ideal_mask = numpy_topk_mask(ideal_scores, topk)
                noisy_mask = np.asarray(metrics[method]["mask"])
                overlap = (ideal_mask * noisy_mask).sum(axis=1) / float(topk)
                ideal_overlap_sum[method] += float(overlap.mean())

        oracle_mask = metrics.get(ORACLE_METHOD, {}).get("mask") if include_oracle else None
        for method in methods:
            m = metrics[method]
            contribution = np.asarray(m["contribution"], dtype=np.float32)
            accumulated[method] += contribution
            per_episode_sum[method] += np.asarray(m["per_sample_sum_rate"], dtype=np.float64)
            raw_per_episode_sum[method] += np.asarray(m["per_sample_raw_sum_rate"], dtype=np.float64)
            pair_corr_sum[method] += float(np.mean(m["mean_pair_corr"]))
            if oracle_mask is not None:
                overlap = (np.asarray(m["mask"]) * np.asarray(oracle_mask)).sum(axis=1) / float(topk)
                oracle_overlap_sum[method] += float(overlap.mean())
            histories[method] = np.maximum(
                (1.0 - history_update) * histories[method] + history_update * contribution,
                cfg.eps,
            ).astype(np.float32)

    rows: list[dict] = []
    num_slots = max(int(num_slots), 1)
    num_episodes = int(context["initial_history"].shape[0])
    mean_pmi_nmse = float(np.mean(actual_pmi_nmse_values))
    mean_rate_nmse = float(np.mean(actual_rate_nmse_values))
    pmi_nmse_db = float(10.0 * np.log10(max(mean_pmi_nmse, 1e-12)))
    rate_nmse_db = float(10.0 * np.log10(max(mean_rate_nmse, 1e-12)))

    for method in methods:
        episode_sum_rates = per_episode_sum[method] / float(num_slots)
        episode_raw_rates = raw_per_episode_sum[method] / float(num_slots)
        episode_jain = [
            _jain_index(accumulated[method][episode] / float(num_slots))
            for episode in range(num_episodes)
        ]
        sum_mean, sum_ci = _mean_ci95(episode_sum_rates)
        raw_mean, raw_ci = _mean_ci95(episode_raw_rates)
        fair_mean, fair_ci = _mean_ci95(episode_jain)
        rows.append(
            {
                "method": method,
                "avg_future_sum_rate": sum_mean,
                "sum_rate_ci95": sum_ci,
                "avg_raw_future_sum_rate": raw_mean,
                "raw_sum_rate_ci95": raw_ci,
                "jain_fairness_index": fair_mean,
                "jain_ci95": fair_ci,
                "avg_selected_pair_correlation": pair_corr_sum[method] / float(num_slots),
                "selection_overlap_with_oracle": (
                    oracle_overlap_sum[method] / float(num_slots) if include_oracle else np.nan
                ),
                "selection_overlap_with_ideal_prediction": (
                    ideal_overlap_sum[method] / float(num_slots)
                    if compute_ideal_overlap
                    else np.nan
                ),
                "speed_kmh": float(speed_kmh),
                "target_prediction_nmse_db": (
                    float(target_prediction_nmse_db)
                    if target_prediction_nmse_db is not None
                    else _speed_prediction_nmse_db(speed_kmh, cfg)
                ),
                "actual_pmi_prediction_nmse_db": pmi_nmse_db,
                "actual_rate_prediction_nmse_db": rate_nmse_db,
                "episode_count": num_episodes,
                "slots_per_episode": num_slots,
            }
        )
    return rows


def _plot_method_line(
    ax,
    x: Sequence[float],
    y: Sequence[float],
    method: str,
    ci: Sequence[float] | None = None,
) -> None:
    is_ssa = method == "SSA-Net"
    is_oracle = method == ORACLE_METHOD
    color = METHOD_COLORS.get(method)
    ax.plot(
        x,
        y,
        marker=METHOD_MARKERS.get(method, "o"),
        linestyle=METHOD_LINESTYLES.get(method, "-"),
        linewidth=2.25 if is_ssa else (1.8 if is_oracle else 1.65),
        markersize=7.2 if is_ssa else 5.4,
        color=color,
        label=method,
        zorder=6 if is_ssa else 3,
    )


def plot_scheduled_ue_performance(
    data_dir: Path | str,
    ckpt_dir: Path | str,
    fig_dir: Path | str,
    cfg: ExpConfig,
    schedule_sizes: Sequence[int] = (4, 8, 12, 16, 20),
    models: Iterable[str] = ("lstm", "ssa"),
    device: Optional[str] = None,
    include_oracle: bool = True,
) -> Path:
    _set_paper_style()
    fig_dir = Path(fig_dir)
    arrays = _load_eval_arrays(data_dir, cfg)
    schedule_sizes = tuple(sorted({int(v) for v in schedule_sizes if 1 <= int(v) <= cfg.num_ues}))
    if not schedule_sizes:
        raise ValueError("schedule_sizes contains no valid scheduled UE count")

    num_episodes = int(getattr(cfg, "paper_episode_count", 8))
    num_slots = int(getattr(cfg, "paper_episode_slots", 80))
    speed_kmh = float(getattr(cfg, "fig3_reference_speed_kmh", 20.0))
    loaded_bundle = _load_episode_models(ckpt_dir, cfg, models, arrays["meta"], device)
    rows: list[dict] = []
    for scheduled_ues in schedule_sizes:
        at_s = _evaluate_methods_over_persistent_episodes(
            arrays,
            ckpt_dir,
            cfg,
            models,
            device,
            topk=scheduled_ues,
            speed_kmh=speed_kmh,
            num_episodes=num_episodes,
            num_slots=num_slots,
            seed=int(cfg.seed) + 31_001,
            include_oracle=include_oracle,
            loaded_bundle=loaded_bundle,
        )
        for row in at_s:
            row["scheduled_ues"] = int(scheduled_ues)
            rows.append(row)

    result_df = pd.DataFrame(rows)
    fig_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(fig_dir / "fig3_scheduled_ue_performance_values.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    methods = FIG3_METHODS if include_oracle else PAPER_METHODS
    for method in methods:
        sub = result_df[result_df["method"] == method].sort_values("scheduled_ues")
        if sub.empty:
            continue
        _plot_method_line(
            ax,
            sub["scheduled_ues"],
            sub["avg_future_sum_rate"],
            method,
            sub["sum_rate_ci95"],
        )

    ax.set_xlabel("Number of Scheduled UEs, $S$")
    ax.set_ylabel("Average Future Scheduled Sum Rate")
    ax.set_xticks(schedule_sizes)
    ax.grid(True, linestyle="--")
    ax.legend(loc="best", ncol=2, frameon=False)
    ax.set_title(
        f"Performance Comparison under Different Numbers of Scheduled UEs ({speed_kmh:g} km/h)"
    )
    out = fig_dir / "fig3_scheduled_ue_performance.png"
    _savefig(out)
    return out


def plot_fairness_tradeoff(
    data_dir: Path | str,
    ckpt_dir: Path | str,
    fig_dir: Path | str,
    cfg: ExpConfig,
    alpha_values: Optional[Iterable[float]] = None,
    models: Iterable[str] = ("ssa",),
    device: Optional[str] = None,
    history_update: float | None = None,
) -> Path:
    """Plot throughput and Jain fairness across alpha values."""
    _set_paper_style()
    _ = ckpt_dir, models
    fig_dir = Path(fig_dir)
    arrays = _load_eval_arrays(data_dir, cfg)
    alpha_values = list(
        alpha_values
        if alpha_values is not None
        else getattr(cfg, "fairness_alpha_sweep", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    )
    if not alpha_values or any(float(v) < 0.0 for v in alpha_values):
        raise ValueError("alpha_values must be non-empty and non-negative")

    history_update = float(
        getattr(cfg, "paper_history_update", 0.05) if history_update is None else history_update
    )
    if not 0.0 < history_update <= 1.0:
        raise ValueError("history_update must be in (0,1]")

    speed_kmh = float(getattr(cfg, "fairness_tradeoff_reference_speed_kmh", 40.0))
    num_episodes = int(getattr(cfg, "tradeoff_episode_count", 20))
    num_slots = int(getattr(cfg, "tradeoff_episode_slots", 150))
    dev = _device(device)
    loaded_models: dict[str, torch.nn.Module] = {}
    context = _prepare_episode_context(arrays, cfg, num_episodes, int(cfg.seed) + 41_001)

    rows: list[dict] = []
    for alpha in alpha_values:
        running_history = np.array(context["initial_history"], dtype=np.float32, copy=True)
        accumulated = np.zeros_like(running_history, dtype=np.float64)
        episode_sum = np.zeros(running_history.shape[0], dtype=np.float64)
        pair_sum = 0.0

        states = _iter_persistent_episode_states(
            context,
            cfg,
            num_slots,
            speed_kmh,
            int(cfg.seed) + 41_019,
        )
        for state in states:
            score = _method_scores(
                "Enhanced Greedy",
                state,
                running_history,
                context["base_x"],
                cfg.schedule_users,
                cfg,
                loaded_models,
                dev,
                eta=float(cfg.eta),
                fairness_alpha=float(alpha),
            )
            metrics = _selected_set_sinr_metrics(score, state, cfg.schedule_users, cfg)
            contribution = np.asarray(metrics["contribution"], dtype=np.float32)
            accumulated += contribution
            episode_sum += np.asarray(metrics["per_sample_sum_rate"], dtype=np.float64)
            pair_sum += float(np.mean(metrics["mean_pair_corr"]))
            running_history = np.maximum(
                (1.0 - history_update) * running_history + history_update * contribution,
                cfg.eps,
            ).astype(np.float32)

        episode_sum /= float(num_slots)
        episode_jain = [
            _jain_index(accumulated[e] / float(num_slots))
            for e in range(accumulated.shape[0])
        ]
        sum_mean, sum_ci = _mean_ci95(episode_sum)
        fair_mean, fair_ci = _mean_ci95(episode_jain)
        rows.append(
            {
                "fairness_alpha": float(alpha),
                "avg_future_sum_rate": sum_mean,
                "sum_rate_ci95": sum_ci,
                "jain_fairness_index": fair_mean,
                "jain_ci95": fair_ci,
                "avg_selected_pair_correlation": pair_sum / float(num_slots),
                "eta_fixed": float(cfg.eta),
                "alpha_star": float(cfg.fairness_alpha),
                "speed_kmh": speed_kmh,
                "episode_count": int(accumulated.shape[0]),
                "slots_per_episode": int(num_slots),
                "scheduler": "Q-guided Enhanced Greedy",
            }
        )

    result_df = pd.DataFrame(rows).sort_values("fairness_alpha")
    fig_dir.mkdir(parents=True, exist_ok=True)
    for legacy_name in (
        "fig4_eta_throughput_fairness.png",
        "fig4_eta_throughput_fairness_values.csv",
    ):
        (fig_dir / legacy_name).unlink(missing_ok=True)
    result_df.to_csv(fig_dir / "fig4_alpha_throughput_fairness_values.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(7.45, 4.75))
    line1 = ax1.plot(
        result_df["fairness_alpha"],
        result_df["avg_future_sum_rate"],
        color="#D62728",
        marker="s",
        linewidth=2.05,
        markersize=5.4,
        label="Average future sum rate",
    )
    ax1.set_xlabel(r"Fairness exponent $\alpha$")
    ax1.set_ylabel("Average Future Scheduled Sum Rate")
    ax1.grid(True, linestyle="--")
    ax1.ticklabel_format(axis="y", style="plain", useOffset=False)

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        result_df["fairness_alpha"],
        result_df["jain_fairness_index"],
        color="#1F77B4",
        marker="o",
        linestyle="--",
        linewidth=2.0,
        markersize=5.2,
        label="Jain's fairness index",
    )
    ax2.set_ylabel("Jain's Fairness Index")
    fair_min = float(result_df["jain_fairness_index"].min())
    fair_max = float(result_df["jain_fairness_index"].max())
    fair_pad = max(0.015, 0.15 * max(fair_max - fair_min, 1e-3))
    ax2.set_ylim(max(0.0, fair_min - fair_pad), min(1.0, fair_max + fair_pad))

    marker = ax1.axvline(
        float(cfg.fairness_alpha),
        color="#111827",
        linestyle=":",
        linewidth=1.35,
        label=r"Adopted $\alpha^{\star}$",
    )
    lines = line1 + line2 + [marker]
    ax1.legend(lines, [line.get_label() for line in lines], loc="best", frameon=False)
    ax1.set_title(
        r"Impact of $\alpha$ on Throughput--Fairness Tradeoff"
        + f" at {speed_kmh:g} km/h"
    )

    out = fig_dir / "fig4_alpha_throughput_fairness.png"
    _savefig(out)
    return out


def _evaluate_speed_impact_common(
    reference_data_dir: Path | str,
    ckpt_dir: Path | str,
    fig_dir: Path | str,
    cfg: ExpConfig,
    models: Iterable[str],
    device: Optional[str],
) -> pd.DataFrame:
    arrays = _load_eval_arrays(reference_data_dir, cfg)
    loaded_bundle = _load_episode_models(ckpt_dir, cfg, models, arrays["meta"], device)
    num_episodes = int(getattr(cfg, "robustness_episode_count", 8))
    num_slots = int(getattr(cfg, "robustness_episode_slots", 50))
    rows: list[dict] = []
    for speed in cfg.speeds_kmh:
        rows.extend(
            _evaluate_methods_over_persistent_episodes(
                arrays,
                ckpt_dir,
                cfg,
                models,
                device,
                topk=cfg.schedule_users,
                speed_kmh=float(speed),
                num_episodes=num_episodes,
                num_slots=num_slots,
                seed=int(cfg.seed) + 51_001,
                include_oracle=False,
                loaded_bundle=loaded_bundle,
            )
        )
    df = pd.DataFrame(rows)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(fig_dir / "fig5_mobility_values.csv", index=False)
    return df


def evaluate_speed_impact_synthetic(
    reference_data_dir: Path | str,
    ckpt_dir: Path | str,
    fig_dir: Path | str,
    cfg: ExpConfig,
    num_samples_per_speed: int = 800,
    models: Iterable[str] = ("lstm", "ssa"),
    device: Optional[str] = None,
) -> pd.DataFrame:
    _ = num_samples_per_speed
    return _evaluate_speed_impact_common(
        reference_data_dir, ckpt_dir, fig_dir, cfg, models, device
    )


def evaluate_speed_impact_from_official_data(
    official_test_dir: Path | str,
    ckpt_dir: Path | str,
    fig_dir: Path | str,
    cfg: ExpConfig,
    models: Iterable[str] = ("lstm", "ssa"),
    device: Optional[str] = None,
) -> pd.DataFrame:
    return _evaluate_speed_impact_common(
        official_test_dir, ckpt_dir, fig_dir, cfg, models, device
    )


def evaluate_prediction_error_impact(
    reference_data_dir: Path | str,
    ckpt_dir: Path | str,
    fig_dir: Path | str,
    cfg: ExpConfig,
    models: Iterable[str] = ("lstm", "ssa"),
    device: Optional[str] = None,
    nmse_db_values: Sequence[float] = (-30.0, -20.0, -15.0, -10.0, -5.0, 0.0),
    error_stds: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Evaluate scheduling across channel-prediction NMSE values."""
    if error_stds is not None:
        converted: list[float] = []
        for std in error_stds:
            std = max(float(std), 1e-6)
            converted.append(20.0 * np.log10(std))
        nmse_db_values = converted

    arrays = _load_eval_arrays(reference_data_dir, cfg)
    loaded_bundle = _load_episode_models(ckpt_dir, cfg, models, arrays["meta"], device)
    meta_speed = arrays["meta"].get("speed_kmh", None)
    speed_kmh = float(
        meta_speed
        if meta_speed is not None
        else getattr(cfg, "prediction_error_reference_speed_kmh", 40.0)
    )
    num_episodes = int(getattr(cfg, "robustness_episode_count", 8))
    num_slots = int(getattr(cfg, "robustness_episode_slots", 50))
    rows: list[dict] = []
    for nmse_db in nmse_db_values:
        rows.extend(
            _evaluate_methods_over_persistent_episodes(
                arrays,
                ckpt_dir,
                cfg,
                models,
                device,
                topk=cfg.schedule_users,
                speed_kmh=speed_kmh,
                num_episodes=num_episodes,
                num_slots=num_slots,
                seed=int(cfg.seed) + 61_001,
                include_oracle=False,
                target_prediction_nmse_db=float(nmse_db),
                compute_ideal_overlap=True,
                loaded_bundle=loaded_bundle,
            )
        )
    result_df = pd.DataFrame(rows)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(fig_dir / "fig5_prediction_error_values.csv", index=False)
    return result_df


def _as_speed_frame(speed_results: pd.DataFrame | Path | str) -> pd.DataFrame:
    df = speed_results.copy() if isinstance(speed_results, pd.DataFrame) else pd.read_csv(speed_results)
    required = {"speed_kmh", "method", "avg_future_sum_rate"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Mobility results missing columns: {sorted(missing)}")
    return df


def plot_robustness_from_results(
    speed_results: pd.DataFrame | Path | str,
    prediction_error_results: pd.DataFrame | Path | str | None = None,
    fig_dir: Path | str | None = None,
) -> Path:
    """Plot scheduling throughput versus UE mobility speed."""
    _set_paper_style()
    if fig_dir is None:
        if prediction_error_results is None:
            raise ValueError("fig_dir must be provided")
        fig_dir = prediction_error_results
        prediction_error_results = None
    _ = prediction_error_results

    speed_df = _as_speed_frame(speed_results)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    (fig_dir / "fig5_prediction_error_values.csv").unlink(missing_ok=True)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for method in PAPER_METHODS:
        sub = speed_df[speed_df["method"] == method].sort_values("speed_kmh")
        if sub.empty:
            continue
        _plot_method_line(
            ax,
            sub["speed_kmh"],
            sub["avg_future_sum_rate"],
            method,
            sub.get("sum_rate_ci95"),
        )

    ax.set_xlabel("UE Mobility Speed (km/h)")
    ax.set_ylabel("Average Future Scheduled Sum Rate")
    ax.set_title("Robustness to UE Mobility")
    ax.grid(True, linestyle="--")
    ax.legend(loc="best", ncol=2, frameon=False)

    out = fig_dir / "fig5_robustness.png"
    _savefig(out)
    return out

from __future__ import annotations

"""Run emergency-priority, TRP-outage, and latency experiments."""

from pathlib import Path
from time import perf_counter
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import ExpConfig
from src.baselines import learned_set_sinr_refinement_scores
from src.plot_results import (
    _build_features,
    _infer_loaded_model,
    _iter_persistent_episode_states,
    _load_episode_models,
    _load_eval_arrays,
    _mean_ci95,
    _method_scores,
    _prepare_episode_context,
    _selected_set_sinr_metrics,
)


CRITICAL = 2
RESCUE = 1
REGULAR = 0


def assign_emergency_classes(
    num_episodes: int,
    num_ues: int,
    cfg: ExpConfig,
    seed: int,
) -> np.ndarray:
    """Assign service classes to users in each episode."""
    rng = np.random.default_rng(int(seed))
    n_critical = max(1, int(round(float(cfg.emergency_critical_fraction) * num_ues)))
    n_rescue = max(0, int(round(float(cfg.emergency_rescue_fraction) * num_ues)))
    n_rescue = min(n_rescue, num_ues - n_critical)
    classes = np.full((num_episodes, num_ues), REGULAR, dtype=np.int8)
    for e in range(num_episodes):
        order = rng.permutation(num_ues)
        classes[e, order[:n_critical]] = CRITICAL
        classes[e, order[n_critical : n_critical + n_rescue]] = RESCUE
    return classes


def priority_weights_from_classes(
    classes: np.ndarray,
    cfg: ExpConfig,
    critical_weight: float | None = None,
    rescue_weight: float | None = None,
) -> np.ndarray:
    critical = float(cfg.emergency_priority_critical if critical_weight is None else critical_weight)
    rescue = float(cfg.emergency_priority_rescue if rescue_weight is None else rescue_weight)
    regular = float(cfg.emergency_priority_regular)
    out = np.full(classes.shape, regular, dtype=np.float32)
    out[classes == RESCUE] = rescue
    out[classes == CRITICAL] = critical
    return out


def qos_thresholds_from_classes(classes: np.ndarray, cfg: ExpConfig) -> np.ndarray:
    out = np.full(classes.shape, float(cfg.emergency_qos_rate_regular), dtype=np.float32)
    out[classes == RESCUE] = float(cfg.emergency_qos_rate_rescue)
    out[classes == CRITICAL] = float(cfg.emergency_qos_rate_critical)
    return out


def sample_trp_keep_mask(
    num_episodes: int,
    num_trps: int,
    outage_ratio: float,
    seed: int,
) -> np.ndarray:
    """Sample active-TRP masks for each episode."""
    ratio = float(np.clip(outage_ratio, 0.0, 1.0))
    n_out = int(round(ratio * num_trps))
    n_out = min(max(n_out, 0), max(num_trps - 1, 0))
    keep = np.ones((num_episodes, num_trps), dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    if n_out == 0:
        return keep
    for e in range(num_episodes):
        failed = rng.choice(num_trps, size=n_out, replace=False)
        keep[e, failed] = 0.0
    return keep


def _mask_flat_pmi(
    flat: np.ndarray,
    keep_trp: np.ndarray,
    cfg: ExpConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask failed TRP channels and compute retained energy fractions."""
    x = np.asarray(flat, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != cfg.pmi_real_dim:
        raise ValueError(f"Expected flat PMI (B,K,{cfg.pmi_real_dim}), got {x.shape}")
    if keep_trp.shape != (x.shape[0], cfg.num_trps):
        raise ValueError("keep_trp must have shape (B,num_trps)")
    c = cfg.pmi_complex_dim
    real = x[..., :c].reshape(x.shape[0], x.shape[1], cfg.num_trps, cfg.antennas_per_trp)
    imag = x[..., c:].reshape(x.shape[0], x.shape[1], cfg.num_trps, cfg.antennas_per_trp)
    energy_before = np.sum(real * real + imag * imag, axis=(2, 3))
    mask = keep_trp[:, None, :, None]
    real_m = real * mask
    imag_m = imag * mask
    energy_after = np.sum(real_m * real_m + imag_m * imag_m, axis=(2, 3))
    retained = energy_after / np.maximum(energy_before, cfg.eps)
    out = np.concatenate(
        [real_m.reshape(x.shape[0], x.shape[1], c), imag_m.reshape(x.shape[0], x.shape[1], c)],
        axis=-1,
    ).astype(np.float32)
    return out, retained.astype(np.float32)


def _scale_rate_by_retained_energy(rate: np.ndarray, retained: np.ndarray) -> np.ndarray:
    rate = np.asarray(rate, dtype=np.float32)
    snr = np.expm1(np.log(2.0) * np.clip(rate, 0.0, 30.0))
    return np.log2(1.0 + np.maximum(snr * retained, 0.0)).astype(np.float32)


def apply_trp_outage_to_state(
    state: dict[str, np.ndarray | float],
    keep_trp: np.ndarray,
    cfg: ExpConfig,
) -> dict[str, np.ndarray | float]:
    """Apply TRP outages to channel states and rates."""
    out = dict(state)
    now_pmi, keep_now = _mask_flat_pmi(np.asarray(state["pmi_now"]), keep_trp, cfg)
    pred_pmi, keep_pred = _mask_flat_pmi(np.asarray(state["pmi_pred"]), keep_trp, cfg)
    fut_pmi, keep_fut = _mask_flat_pmi(np.asarray(state["pmi_future"]), keep_trp, cfg)
    out["pmi_now"] = now_pmi
    out["pmi_pred"] = pred_pmi
    out["pmi_future"] = fut_pmi
    out["rate_now"] = _scale_rate_by_retained_energy(np.asarray(state["rate_now"]), keep_now)
    out["rate_pred"] = _scale_rate_by_retained_energy(np.asarray(state["rate_pred"]), keep_pred)
    out["rate_future"] = _scale_rate_by_retained_energy(np.asarray(state["rate_future"]), keep_fut)
    out["trp_retained_fraction"] = keep_trp.mean(axis=1).astype(np.float32)
    return out


def _emergency_episode_evaluation(
    data_dir: Path | str,
    ckpt_dir: Path | str,
    cfg: ExpConfig,
    models: Iterable[str],
    device: Optional[str],
    *,
    critical_weight: float,
    rescue_weight: float,
    outage_ratio: float,
    seed: int,
    num_episodes: int | None = None,
    num_slots: int | None = None,
) -> pd.DataFrame:
    arrays = _load_eval_arrays(data_dir, cfg)
    episodes = int(num_episodes or cfg.robustness_episode_count)
    slots = int(num_slots or cfg.robustness_episode_slots)
    context = _prepare_episode_context(arrays, cfg, episodes, seed)
    loaded_models, dev = _load_episode_models(ckpt_dir, cfg, models, arrays["meta"], device)
    methods = ["No Prediction", "PF Prediction", "Enhanced Greedy"]
    methods.extend([name for name in ("LSTM", "SSA-Net") if name in loaded_models])

    classes = assign_emergency_classes(episodes, cfg.num_ues, cfg, seed + 701)
    priority = priority_weights_from_classes(classes, cfg, critical_weight, rescue_weight)
    qos_threshold = qos_thresholds_from_classes(classes, cfg)
    keep_trp = sample_trp_keep_mask(episodes, cfg.num_trps, outage_ratio, seed + 809)

    histories = {
        method: np.array(context["initial_history"], dtype=np.float32, copy=True)
        for method in methods
    }
    per_episode_sum = {method: np.zeros(episodes, dtype=np.float64) for method in methods}
    per_episode_weighted = {method: np.zeros(episodes, dtype=np.float64) for method in methods}
    critical_qos = {method: np.zeros(episodes, dtype=np.float64) for method in methods}
    rescue_qos = {method: np.zeros(episodes, dtype=np.float64) for method in methods}
    overall_qos = {method: np.zeros(episodes, dtype=np.float64) for method in methods}
    critical_sched = {method: np.zeros(episodes, dtype=np.float64) for method in methods}
    pair_corr = {method: 0.0 for method in methods}

    states = _iter_persistent_episode_states(
        context,
        cfg,
        slots,
        float(cfg.emergency_reference_speed_kmh),
        seed + 1009,
    )
    history_update = float(cfg.paper_history_update)
    for state in states:
        outage_state = apply_trp_outage_to_state(state, keep_trp, cfg)
        for method in methods:
            scores = _method_scores(
                method,
                outage_state,
                histories[method],
                context["base_x"],
                cfg.schedule_users,
                cfg,
                loaded_models,
                dev,
                priority_weights=priority,
            )
            metrics = _selected_set_sinr_metrics(scores, outage_state, cfg.schedule_users, cfg)
            contribution = np.asarray(metrics["contribution"], dtype=np.float32)
            mask = np.asarray(metrics["mask"], dtype=np.float32)
            per_episode_sum[method] += np.asarray(metrics["per_sample_sum_rate"], dtype=np.float64)
            per_episode_weighted[method] += np.sum(priority * contribution, axis=1)
            pair_corr[method] += float(np.mean(metrics["mean_pair_corr"]))

            sat = contribution >= qos_threshold
            critical_mask = classes == CRITICAL
            rescue_mask = classes == RESCUE
            critical_denom = np.maximum(critical_mask.sum(axis=1), 1)
            rescue_denom = np.maximum(rescue_mask.sum(axis=1), 1)
            critical_qos[method] += (sat & critical_mask).sum(axis=1) / critical_denom
            rescue_qos[method] += (sat & rescue_mask).sum(axis=1) / rescue_denom
            overall_qos[method] += sat.sum(axis=1) / float(cfg.num_ues)
            critical_sched[method] += (mask.astype(bool) & critical_mask).sum(axis=1) / critical_denom
            histories[method] = np.maximum(
                (1.0 - history_update) * histories[method] + history_update * contribution,
                cfg.eps,
            ).astype(np.float32)

    rows: list[dict] = []
    slots_f = float(max(slots, 1))
    for method in methods:
        sum_mean, sum_ci = _mean_ci95(per_episode_sum[method] / slots_f)
        weighted_mean, weighted_ci = _mean_ci95(per_episode_weighted[method] / slots_f)
        critical_mean, critical_ci = _mean_ci95(critical_qos[method] / slots_f)
        rescue_mean, rescue_ci = _mean_ci95(rescue_qos[method] / slots_f)
        overall_mean, overall_ci = _mean_ci95(overall_qos[method] / slots_f)
        sched_mean, sched_ci = _mean_ci95(critical_sched[method] / slots_f)
        rows.append(
            {
                "method": method,
                "critical_weight": float(critical_weight),
                "rescue_weight": float(rescue_weight),
                "outage_ratio": float(outage_ratio),
                "active_trp_fraction": float(keep_trp.mean()),
                "avg_future_sum_rate": sum_mean,
                "sum_rate_ci95": sum_ci,
                "priority_weighted_service": weighted_mean,
                "priority_weighted_service_ci95": weighted_ci,
                "critical_qos_satisfaction": critical_mean,
                "critical_qos_ci95": critical_ci,
                "rescue_qos_satisfaction": rescue_mean,
                "rescue_qos_ci95": rescue_ci,
                "overall_qos_satisfaction": overall_mean,
                "overall_qos_ci95": overall_ci,
                "critical_scheduling_ratio": sched_mean,
                "critical_scheduling_ci95": sched_ci,
                "avg_selected_pair_correlation": pair_corr[method] / slots_f,
                "speed_kmh": float(cfg.emergency_reference_speed_kmh),
            }
        )
    return pd.DataFrame(rows)


def run_priority_sweep(
    data_dir: Path | str,
    ckpt_dir: Path | str,
    out_dir: Path | str,
    cfg: ExpConfig,
    models: Iterable[str] = ("lstm", "ssa"),
    device: Optional[str] = None,
    critical_weights: Sequence[float] | None = None,
    quick: bool = False,
) -> tuple[Path, Path]:
    """Evaluate scheduling across mission-critical priority weights."""
    values = tuple(critical_weights or cfg.emergency_priority_sweep)
    frames = []
    episodes = 4 if quick else cfg.robustness_episode_count
    slots = 20 if quick else cfg.robustness_episode_slots
    for weight in values:
        frames.append(
            _emergency_episode_evaluation(
                data_dir,
                ckpt_dir,
                cfg,
                models,
                device,
                critical_weight=float(weight),
                rescue_weight=1.0,
                outage_ratio=0.0,
                seed=cfg.seed + 20_000,
                num_episodes=episodes,
                num_slots=slots,
            )
        )
    df = pd.concat(frames, ignore_index=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "emergency_priority_sweep.csv"
    df.to_csv(csv_path, index=False)

    ssa = df[df["method"] == "SSA-Net"].sort_values("critical_weight")
    fig, ax1 = plt.subplots(figsize=(6.6, 4.1))
    ax1.plot(ssa["critical_weight"], ssa["critical_qos_satisfaction"], marker="o", label="Critical QoS satisfaction")
    ax1.set_xlabel(r"Mission-critical priority weight $\omega_{\rm MC}$")
    ax1.set_ylabel("Mission-Critical QoS Satisfaction")
    ax1.set_ylim(0.0, 1.02)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(ssa["critical_weight"], ssa["avg_future_sum_rate"], marker="s", linestyle="--", label="Future sum rate")
    ax2.set_ylabel("Average Future Scheduled Sum Rate")

    adopted_weight = float(cfg.emergency_priority_critical)
    ax1.axvline(
        adopted_weight,
        linestyle="--",
        linewidth=1.2,
        label=rf"Adopted $\omega_{{\rm MC}}={adopted_weight:g}$",
    )

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [x.get_label() for x in lines], frameon=False, loc="best")
    ax1.set_title("Impact of Mission-Critical Priority")
    fig.tight_layout()
    fig_path = out_dir / "emergency_priority_tradeoff.png"
    fig.savefig(fig_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return csv_path, fig_path


def run_trp_outage_sweep(
    data_dir: Path | str,
    ckpt_dir: Path | str,
    out_dir: Path | str,
    cfg: ExpConfig,
    models: Iterable[str] = ("lstm", "ssa"),
    device: Optional[str] = None,
    outage_ratios: Sequence[float] | None = None,
    quick: bool = False,
) -> tuple[Path, Path]:
    """Evaluate scheduling across TRP outage ratios."""
    ratios = tuple(outage_ratios or cfg.emergency_outage_ratios)
    frames = []
    episodes = 4 if quick else cfg.robustness_episode_count
    slots = 20 if quick else cfg.robustness_episode_slots
    for ratio in ratios:
        trial_frames = []
        trials = 2 if quick else int(cfg.emergency_outage_trials)
        for trial in range(trials):
            trial_frames.append(
                _emergency_episode_evaluation(
                    data_dir,
                    ckpt_dir,
                    cfg,
                    models,
                    device,
                    critical_weight=float(cfg.emergency_priority_critical),
                    rescue_weight=float(cfg.emergency_priority_rescue),
                    outage_ratio=float(ratio),
                    seed=cfg.seed + 30_000 + 97 * trial,
                    num_episodes=episodes,
                    num_slots=slots,
                )
            )
        raw = pd.concat(trial_frames, ignore_index=True)
        numeric = [c for c in raw.select_dtypes(include=[np.number]).columns if c not in {"outage_ratio"}]
        agg = raw.groupby(["method", "outage_ratio"], as_index=False)[numeric].mean()
        frames.append(agg)
    df = pd.concat(frames, ignore_index=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "emergency_trp_outage.csv"
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    preferred = ["No Prediction", "PF Prediction", "Enhanced Greedy", "LSTM", "SSA-Net"]
    for method in preferred:
        sub = df[df["method"] == method].sort_values("outage_ratio")
        if sub.empty:
            continue
        x = 100.0 * sub["outage_ratio"].to_numpy()
        legend_label = "Enhanced Greedy (Upper-Bound Ref.)" if method == "Enhanced Greedy" else method
        axes[0].plot(x, sub["avg_future_sum_rate"], marker="o", label=legend_label)
        axes[1].plot(x, sub["critical_qos_satisfaction"], marker="o", label=legend_label)
    axes[0].set_xlabel("TRP Outage Ratio (%)")
    axes[0].set_ylabel("Average Future Scheduled Sum Rate")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[1].set_xlabel("TRP Outage Ratio (%)")
    axes[1].set_ylabel("Mission-Critical QoS Satisfaction")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Robustness to Emergency Infrastructure Outage")
    fig.tight_layout()
    fig_path = out_dir / "emergency_trp_outage.png"
    fig.savefig(fig_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return csv_path, fig_path


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _time_callable(fn, device: torch.device, warmup: int, repeats: int) -> tuple[float, float, float]:
    for _ in range(max(0, warmup)):
        fn()
    _sync_cuda(device)
    values = np.empty(max(1, repeats), dtype=np.float64)
    for i in range(values.size):
        _sync_cuda(device)
        start = perf_counter()
        fn()
        _sync_cuda(device)
        values[i] = (perf_counter() - start) * 1000.0
    return float(values.mean()), float(np.median(values)), float(np.percentile(values, 95))


def benchmark_emergency_latency(
    data_dir: Path | str,
    ckpt_dir: Path | str,
    out_dir: Path | str,
    cfg: ExpConfig,
    models: Iterable[str] = ("lstm", "ssa"),
    device: Optional[str] = None,
    quick: bool = False,
) -> tuple[Path, Path]:
    """Measure batch-one end-to-end scheduling latency."""
    arrays = _load_eval_arrays(data_dir, cfg)
    context = _prepare_episode_context(arrays, cfg, 1, cfg.seed + 40_000)
    state = next(
        _iter_persistent_episode_states(
            context,
            cfg,
            1,
            float(cfg.emergency_reference_speed_kmh),
            cfg.seed + 40_101,
        )
    )
    classes = assign_emergency_classes(1, cfg.num_ues, cfg, cfg.seed + 40_202)
    priority = priority_weights_from_classes(classes, cfg)
    history = np.array(context["initial_history"], dtype=np.float32, copy=True)
    topk = int(cfg.schedule_users)
    warmup = 5 if quick else int(cfg.emergency_latency_warmup)
    repeats = 20 if quick else int(cfg.emergency_latency_repeats)

    rows: list[dict] = []
    devices: list[torch.device] = [torch.device("cpu")]
    if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
        devices.append(torch.device(device or "cuda"))

    for dev in devices:
        loaded, _ = _load_episode_models(ckpt_dir, cfg, models, arrays["meta"], str(dev))
        x = _build_features(context["base_x"], state, history, cfg)

        def pf_fn():
            return _method_scores("PF Prediction", state, history, context["base_x"], topk, cfg, loaded, dev, priority_weights=priority)

        def greedy_fn():
            return _method_scores("Enhanced Greedy", state, history, context["base_x"], topk, cfg, loaded, dev, priority_weights=priority)

        tests: list[tuple[str, callable]] = []
        if dev.type == "cpu":
            tests += [("PF Prediction", pf_fn), ("Enhanced Greedy", greedy_fn)]
        if "LSTM" in loaded:
            def lstm_fn():
                raw = _infer_loaded_model(loaded["LSTM"], x, dev)
                return raw + float(cfg.emergency_priority_logit_scale) * np.log(np.maximum(priority, cfg.eps))
            tests.append(("LSTM", lstm_fn))
        if "SSA-Net" in loaded:
            def ssa_rank_fn():
                return _infer_loaded_model(loaded["SSA-Net"], x, dev, emergency_priority=priority)

            def ssa_decoder_fn():
                raw = _infer_loaded_model(loaded["SSA-Net"], x, dev, emergency_priority=priority)
                return learned_set_sinr_refinement_scores(
                    raw,
                    np.asarray(state["rate_now"]),
                    np.asarray(state["rate_pred"]),
                    history,
                    np.asarray(state["pmi_pred"]),
                    cfg.eta,
                    topk,
                    eps=cfg.eps,
                    fairness_alpha=cfg.fairness_alpha,
                    interference_beta=cfg.paper_set_interference_beta,
                    power_reference_users=(cfg.paper_power_reference_users or cfg.schedule_users),
                    candidate_multiplier=cfg.ssa_decoder_candidate_multiplier,
                    learned_weight=cfg.ssa_decoder_learned_weight,
                    priority_weights=priority,
                )
            tests += [("SSA-Net (ranking)", ssa_rank_fn), ("SSA-Net + decoder", ssa_decoder_fn)]

        for name, fn in tests:
            mean_ms, median_ms, p95_ms = _time_callable(fn, dev, warmup, repeats)
            params = 0
            if name.startswith("SSA-Net") and "SSA-Net" in loaded:
                params = sum(p.numel() for p in loaded["SSA-Net"].parameters())
            elif name == "LSTM" and "LSTM" in loaded:
                params = sum(p.numel() for p in loaded["LSTM"].parameters())
            rows.append(
                {
                    "method": name,
                    "device": str(dev),
                    "batch_size": 1,
                    "mean_ms": mean_ms,
                    "median_ms": median_ms,
                    "p95_ms": p95_ms,
                    "parameters": int(params),
                    "scheduling_period_ms": float(cfg.scheduling_delay_ms),
                    "meets_5ms_mean": bool(mean_ms < float(cfg.scheduling_delay_ms)),
                    "repeats": repeats,
                }
            )

    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "emergency_latency.csv"
    df.to_csv(csv_path, index=False)

    cpu = df[df["device"] == "cpu"]
    cpu_plot = cpu[cpu["method"] != "PF Prediction"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xloc = np.arange(len(cpu_plot))
    ax.bar(xloc - 0.17, cpu_plot["mean_ms"], width=0.34, label="Mean")
    ax.bar(xloc + 0.17, cpu_plot["p95_ms"], width=0.34, label="P95")
    ax.axhline(float(cfg.scheduling_delay_ms), linestyle="--", linewidth=1.2, label="5-ms scheduling period")
    ax.set_xticks(xloc, cpu_plot["method"], rotation=15, ha="right")
    ax.set_ylabel("End-to-End Latency (ms)")
    ax.set_title("Online Scheduling Latency (CPU, Batch=1)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig_path = out_dir / "emergency_latency.png"
    fig.savefig(fig_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return csv_path, fig_path


def run_all_emergency_workshop_experiments(
    data_dir: Path | str,
    ckpt_dir: Path | str,
    out_dir: Path | str,
    cfg: ExpConfig,
    models: Iterable[str] = ("lstm", "ssa"),
    device: Optional[str] = None,
    quick: bool = False,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_csv, p_fig = run_priority_sweep(data_dir, ckpt_dir, out_dir, cfg, models, device, quick=quick)
    o_csv, o_fig = run_trp_outage_sweep(data_dir, ckpt_dir, out_dir, cfg, models, device, quick=quick)
    l_csv, l_fig = benchmark_emergency_latency(data_dir, ckpt_dir, out_dir, cfg, models, device, quick=quick)
    return {
        "priority_csv": p_csv,
        "priority_figure": p_fig,
        "outage_csv": o_csv,
        "outage_figure": o_fig,
        "latency_csv": l_csv,
        "latency_figure": l_fig,
    }

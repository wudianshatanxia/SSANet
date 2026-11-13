"""Generate synthetic user-scheduling datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from config import ExpConfig
from src.baselines import set_sinr_greedy_scores


def _speed_to_correlation(speed_kmh: np.ndarray) -> np.ndarray:
    """Convert UE speeds to temporal channel correlations."""
    rho = np.exp(-speed_kmh / 140.0)
    return np.clip(rho, 0.35, 0.995)


def _complex_to_realimag_features(x_complex: np.ndarray) -> np.ndarray:
    """Concatenate real and imaginary channel features."""
    return np.concatenate([x_complex.real, x_complex.imag], axis=-1)


def _pf_utility(
    rate_now: np.ndarray,
    rate_future_like: np.ndarray,
    hist_rate: np.ndarray,
    eta: float,
    fairness_alpha: float,
    eps: float,
) -> np.ndarray:
    """Compute prediction-aware proportional-fair utility."""
    denom = np.power(hist_rate + eps, fairness_alpha)
    return (eta * rate_now + (1.0 - eta) * rate_future_like) / denom


def _normalize_complex(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _make_clustered_pmi(
    rng: np.random.Generator,
    batch_size: int,
    num_ues: int,
    complex_dim: int,
    num_clusters: int,
    cluster_strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate clustered PMI directions and cluster assignments."""
    proto = (
        rng.normal(0, 1, size=(batch_size, num_clusters, complex_dim)).astype(np.float32)
        + 1j * rng.normal(0, 1, size=(batch_size, num_clusters, complex_dim)).astype(np.float32)
    ) / np.sqrt(2.0)
    proto = _normalize_complex(proto).astype(np.complex64)

    cluster_id = rng.integers(0, num_clusters, size=(batch_size, num_ues), endpoint=False)
    user_noise = (
        rng.normal(0, 1, size=(batch_size, num_ues, complex_dim)).astype(np.float32)
        + 1j * rng.normal(0, 1, size=(batch_size, num_ues, complex_dim)).astype(np.float32)
    ) / np.sqrt(2.0)
    user_noise = _normalize_complex(user_noise).astype(np.complex64)

    proto_for_user = proto[np.arange(batch_size)[:, None], cluster_id]
    s = float(np.clip(cluster_strength, 0.0, 0.98))
    now_complex = s * proto_for_user + np.sqrt(max(1.0 - s * s, 1e-6)) * user_noise
    now_complex = _normalize_complex(now_complex).astype(np.complex64)
    return now_complex, cluster_id.astype(np.int16), proto


def _cluster_interference_penalty(cluster_id: np.ndarray, strength: float) -> np.ndarray:
    """Compute rate penalties from cluster loads."""
    B, K = cluster_id.shape
    penalty = np.ones((B, K), dtype=np.float32)
    for b in range(B):
        counts = np.bincount(cluster_id[b], minlength=int(cluster_id.max()) + 1).astype(np.float32)
        load = counts[cluster_id[b]] - 1.0
        penalty[b] = 1.0 / (1.0 + strength * load / max(K / max(len(counts), 1), 1.0))
    return penalty


def generate_synthetic_dataset(
    out_dir: Path | str,
    cfg: ExpConfig,
    num_samples: Optional[int] = None,
    fixed_speed_kmh: Optional[float] = None,
    overwrite: bool = False,
) -> Path:
    """Generate and save a synthetic scheduling dataset."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(num_samples or cfg.num_samples)

    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not overwrite:
        print(f"[data] Dataset already exists at {out_dir}. Use --overwrite to regenerate.")
        return out_dir

    rng = np.random.default_rng(cfg.seed + (0 if fixed_speed_kmh is None else int(fixed_speed_kmh * 10)))

    features = np.lib.format.open_memmap(out_dir / "features.npy", mode="w+", dtype=np.float32, shape=(n, cfg.num_ues, cfg.feature_dim))
    labels = np.lib.format.open_memmap(out_dir / "labels.npy", mode="w+", dtype=np.float32, shape=(n, cfg.num_ues))
    target_scores = np.lib.format.open_memmap(out_dir / "target_scores.npy", mode="w+", dtype=np.float32, shape=(n, cfg.num_ues))
    rate_now_all = np.lib.format.open_memmap(out_dir / "rate_now.npy", mode="w+", dtype=np.float32, shape=(n, cfg.num_ues))
    rate_future_all = np.lib.format.open_memmap(out_dir / "rate_future.npy", mode="w+", dtype=np.float32, shape=(n, cfg.num_ues))
    rate_pred_all = np.lib.format.open_memmap(out_dir / "rate_pred.npy", mode="w+", dtype=np.float32, shape=(n, cfg.num_ues))
    hist_rate_all = np.lib.format.open_memmap(out_dir / "hist_rate.npy", mode="w+", dtype=np.float32, shape=(n, cfg.num_ues))
    speed_all = np.lib.format.open_memmap(out_dir / "speed_kmh.npy", mode="w+", dtype=np.float32, shape=(n,))

    snr_linear = 10.0 ** (cfg.snr_db / 10.0)
    K = cfg.num_ues
    C = cfg.pmi_complex_dim
    G = int(getattr(cfg, "num_user_clusters", 12))

    for start in tqdm(range(0, n, cfg.chunk_size), desc=f"Generating {out_dir.name}"):
        end = min(start + cfg.chunk_size, n)
        b = end - start

        if fixed_speed_kmh is None:
            speed = rng.choice(np.array(cfg.speeds_kmh, dtype=np.float32), size=b)
        else:
            speed = np.full(b, float(fixed_speed_kmh), dtype=np.float32)
        speed_norm = (speed / max(cfg.speeds_kmh)).astype(np.float32)
        rho = _speed_to_correlation(speed).astype(np.float32)[:, None, None]

        large_scale = rng.lognormal(mean=0.0, sigma=0.45, size=(b, K)).astype(np.float32)
        hist_rate = rng.gamma(shape=2.0, scale=1.2, size=(b, K)).astype(np.float32) + 0.2
        drift_sigma = (0.04 + 0.28 * speed_norm)[:, None].astype(np.float32)
        future_quality_drift = rng.lognormal(mean=-0.5 * drift_sigma**2, sigma=drift_sigma, size=(b, K)).astype(np.float32)
        pred_drift_noise = rng.normal(0.0, 0.25 * drift_sigma + 0.02, size=(b, K)).astype(np.float32)
        predicted_quality_drift = np.clip(future_quality_drift * np.exp(pred_drift_noise), 0.35, 2.8).astype(np.float32)

        now_complex, cluster_id, _ = _make_clustered_pmi(
            rng,
            b,
            K,
            C,
            G,
            getattr(cfg, "cluster_strength", 0.78),
        )

        noise_future, _, _ = _make_clustered_pmi(
            rng,
            b,
            K,
            C,
            G,
            getattr(cfg, "cluster_strength", 0.78),
        )
        future_complex_true = rho * now_complex + np.sqrt(np.maximum(1.0 - rho**2, 1e-6)) * noise_future
        future_complex_true = _normalize_complex(future_complex_true).astype(np.complex64)

        pred_noise_scale = (cfg.future_pred_noise_base + cfg.future_pred_noise_speed * speed_norm)[:, None, None].astype(np.float32)
        pred_noise = (
            rng.normal(0, 1, size=(b, K, C)).astype(np.float32)
            + 1j * rng.normal(0, 1, size=(b, K, C)).astype(np.float32)
        ) / np.sqrt(2.0)
        pred_noise = _normalize_complex(pred_noise).astype(np.complex64)
        future_complex_pred = future_complex_true + pred_noise_scale * pred_noise
        future_complex_pred = _normalize_complex(future_complex_pred).astype(np.complex64)

        cluster_penalty = _cluster_interference_penalty(cluster_id, getattr(cfg, "cluster_interference_strength", 0.35))

        align_now = np.abs(np.mean(now_complex, axis=-1)).astype(np.float32)
        align_future = np.abs(np.mean(future_complex_true, axis=-1)).astype(np.float32)
        align_pred = np.abs(np.mean(future_complex_pred, axis=-1)).astype(np.float32)
        gain_now = large_scale * (0.65 + 3.0 * align_now) * cluster_penalty
        gain_future_true = large_scale * future_quality_drift * (0.65 + 3.0 * align_future) * cluster_penalty
        gain_future_pred = large_scale * predicted_quality_drift * (0.65 + 3.0 * align_pred) * cluster_penalty

        rate_now = np.log2(1.0 + snr_linear * gain_now).astype(np.float32)
        rate_future = np.log2(1.0 + snr_linear * gain_future_true).astype(np.float32)
        rate_pred = np.log2(1.0 + snr_linear * gain_future_pred).astype(np.float32)

        now_feat = _complex_to_realimag_features(now_complex).astype(np.float32)
        future_feat = _complex_to_realimag_features(future_complex_pred).astype(np.float32)
        teacher_future_feat = _complex_to_realimag_features(future_complex_true).astype(np.float32)
        y_score = set_sinr_greedy_scores(
            rate_now,
            rate_future,
            hist_rate,
            teacher_future_feat,
            cfg.eta,
            cfg.schedule_users,
            eps=cfg.eps,
            fairness_alpha=cfg.fairness_alpha,
            interference_beta=getattr(cfg, "paper_set_interference_beta", 0.40),
            power_reference_users=(getattr(cfg, "paper_power_reference_users", 0) or cfg.schedule_users),
        )
        y = (y_score > 0).astype(np.float32)

        x = np.zeros((b, K, cfg.feature_dim), dtype=np.float32)
        x[:, :, cfg.idx_now_pmi_start : cfg.idx_now_pmi_start + cfg.pmi_real_dim] = now_feat
        x[:, :, cfg.idx_now_rate] = rate_now
        x[:, :, cfg.idx_future_pmi_start : cfg.idx_future_pmi_start + cfg.pmi_real_dim] = future_feat
        x[:, :, cfg.idx_future_rate] = rate_pred
        x[:, :, cfg.idx_hist_rate] = hist_rate
        x[:, :, cfg.idx_speed] = speed_norm[:, None]

        features[start:end] = x
        labels[start:end] = y
        target_scores[start:end] = y_score
        rate_now_all[start:end] = rate_now
        rate_future_all[start:end] = rate_future
        rate_pred_all[start:end] = rate_pred
        hist_rate_all[start:end] = hist_rate
        speed_all[start:end] = speed

    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    split = {"train": idx[:n_train].tolist(), "val": idx[n_train : n_train + n_val].tolist(), "test": idx[n_train + n_val :].tolist()}
    with open(out_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(split, f)

    meta = {
        "num_samples": n,
        "num_ues": cfg.num_ues,
        "num_trps": cfg.num_trps,
        "antennas_per_trp": cfg.antennas_per_trp,
        "pmi_complex_dim": cfg.pmi_complex_dim,
        "pmi_real_dim": cfg.pmi_real_dim,
        "feature_dim": cfg.feature_dim,
        "schedule_users": cfg.schedule_users,
        "eta": cfg.eta,
        "fairness_alpha": cfg.fairness_alpha,
        "num_user_clusters": getattr(cfg, "num_user_clusters", 12),
        "cluster_strength": getattr(cfg, "cluster_strength", 0.78),
        "cluster_interference_strength": getattr(cfg, "cluster_interference_strength", 0.35),
        "scheduling_delay_ms": cfg.scheduling_delay_ms,
        "fixed_speed_kmh": fixed_speed_kmh,
        "description": "Synthetic task-aligned dataset: feature future fields are predicted 5-ms PMI/rate; rate_future.npy stores true future rates for evaluation; labels include correlation-aware greedy diversity.",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[data] Generated dataset at {out_dir} with {n} samples.")
    return out_dir

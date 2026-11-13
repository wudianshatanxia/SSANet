from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import j0

from config import ExpConfig
from src.baselines import set_sinr_greedy_scores
from src.metrics import compute_q

_C = 299_792_458.0


def temporal_channel_correlation(speed_kmh: float, cfg: ExpConfig) -> float:
    speed_mps = max(float(speed_kmh), 0.0) / 3.6
    f_d = speed_mps * float(cfg.carrier_frequency_hz) / _C
    x = 2.0 * np.pi * f_d * float(cfg.mobility_time_step_s)
    model = str(getattr(cfg, "mobility_correlation_model", "gaussian_doppler")).lower()
    if model == "jakes":
        rho = abs(float(j0(x)))
    elif model == "gaussian_doppler":
        scale = max(float(getattr(cfg, "mobility_doppler_scale", 2.5)), 1e-6)
        rho = float(np.exp(-0.5 * (x / scale) ** 2))
    else:
        raise ValueError(f"Unknown mobility_correlation_model={model}")
    return float(np.clip(rho, 0.0, 0.999999))


def _flat_to_complex(flat: np.ndarray) -> np.ndarray:
    if flat.shape[-1] % 2:
        raise ValueError("Real/imag-stacked PMI must have an even last dimension")
    c = flat.shape[-1] // 2
    return flat[..., :c].astype(np.float32) + 1j * flat[..., c:].astype(np.float32)


def _pair_to_complex(pair: np.ndarray) -> np.ndarray:
    return pair[..., 0].astype(np.float32) + 1j * pair[..., 1].astype(np.float32)


def _complex_to_flat(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x.real, x.imag], axis=-1).astype(np.float32)


def _complex_to_pair(x: np.ndarray) -> np.ndarray:
    return np.stack([x.real, x.imag], axis=-1).astype(np.float32)


def _normalize_complex(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def _standardize_users(x: np.ndarray, eps: float = 1e-6):
    mean = x.mean(axis=1, keepdims=True)
    std = np.maximum(x.std(axis=1, keepdims=True), eps)
    return (x - mean) / std, mean, std


def _orthogonal_complex_innovation(base: np.ndarray, reference: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    base = _normalize_complex(base)
    reference = _normalize_complex(reference)
    projection = np.sum(np.conjugate(reference) * base, axis=-1, keepdims=True)
    innovation = base - projection * reference
    norm = np.linalg.norm(innovation, axis=-1, keepdims=True)
    fallback = (rng.normal(size=reference.shape) + 1j * rng.normal(size=reference.shape)) / np.sqrt(2.0)
    fallback = fallback - np.sum(np.conjugate(reference) * fallback, axis=-1, keepdims=True) * reference
    innovation = np.where(norm > 1e-6, innovation, fallback)
    return _normalize_complex(innovation).astype(np.complex64)


def _independent_standardized_noise(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=shape).astype(np.float32)
    return _standardize_users(z)[0]


def _speed_parameters(speed_kmh: np.ndarray, cfg: ExpConfig):
    speed = np.asarray(speed_kmh, dtype=np.float32).reshape(-1)
    max_speed = max(float(max(cfg.speeds_kmh)), 1.0)
    speed_norm = np.clip(speed / max_speed, 0.0, 1.0)
    rho = np.array([temporal_channel_correlation(float(v), cfg) for v in speed], dtype=np.float32)
    skill_max = float(getattr(cfg, "mobility_prediction_skill_max", 0.95))
    skill_min = float(getattr(cfg, "mobility_prediction_skill_min", 0.62))
    pred_skill = skill_min + (skill_max - skill_min) * np.power(1.0 - speed_norm, 0.75)
    return speed_norm.astype(np.float32), rho.astype(np.float32), pred_skill.astype(np.float32)


def augment_mobility_chunk(
    pmi0_pair: np.ndarray,
    pmi5_pair: np.ndarray,
    rate_now: np.ndarray,
    rate_future_base: np.ndarray,
    speed_kmh: np.ndarray,
    cfg: ExpConfig,
    rng: np.random.Generator,
):
    """Generate future channel states and predictions at assigned speeds."""
    h_now = _normalize_complex(_pair_to_complex(pmi0_pair)).astype(np.complex64)
    h_base_future = _normalize_complex(_pair_to_complex(pmi5_pair)).astype(np.complex64)
    speed_norm, rho, pred_skill = _speed_parameters(speed_kmh, cfg)
    rho_h = rho[:, None, None]
    skill_h = pred_skill[:, None, None]

    innovation = _orthogonal_complex_innovation(h_base_future, h_now, rng)
    h_true = _normalize_complex(rho_h * h_now + np.sqrt(np.maximum(1.0 - rho_h**2, 0.0)) * innovation).astype(np.complex64)
    pred_noise = (rng.normal(size=h_true.shape) + 1j * rng.normal(size=h_true.shape)) / np.sqrt(2.0)
    pred_noise = _orthogonal_complex_innovation(pred_noise, h_true, rng)
    h_pred = _normalize_complex(skill_h * h_true + np.sqrt(np.maximum(1.0 - skill_h**2, 0.0)) * pred_noise).astype(np.complex64)

    snr_now = np.maximum(np.expm1(np.log(2.0) * np.maximum(rate_now, 0.0)), 1e-8)
    snr_future_base = np.maximum(np.expm1(np.log(2.0) * np.maximum(rate_future_base, 0.0)), 1e-8)
    z_now, _, _ = _standardize_users(np.log(snr_now))
    z_base, mu_future, std_future = _standardize_users(np.log(snr_future_base))
    corr = np.clip((z_now * z_base).mean(axis=1, keepdims=True), -0.98, 0.98)
    innovation_rate = z_base - corr * z_now
    innovation_rate, _, _ = _standardize_users(innovation_rate)
    z_true = rho[:, None] * z_now + np.sqrt(np.maximum(1.0 - rho[:, None] ** 2, 0.0)) * innovation_rate
    log_snr_true = mu_future + std_future * z_true
    rate_true = np.log2(1.0 + np.exp(np.clip(log_snr_true, -20.0, 20.0))).astype(np.float32)

    independent = _independent_standardized_noise(z_true.shape, rng)
    current_bias = z_now - z_true
    noise_mix = 0.70 * independent + 0.30 * current_bias
    noise_mix = _standardize_users(noise_mix)[0]
    z_pred = pred_skill[:, None] * z_true + np.sqrt(np.maximum(1.0 - pred_skill[:, None] ** 2, 0.0)) * noise_mix
    rate_pred = np.log2(1.0 + np.exp(np.clip(mu_future + std_future * z_pred, -20.0, 20.0))).astype(np.float32)

    return {
        "pmi_true_pair": _complex_to_pair(h_true),
        "pmi_pred_flat": _complex_to_flat(h_pred),
        "rate_true": rate_true,
        "rate_pred": rate_pred,
        "speed_norm": speed_norm,
        "rho": rho,
        "pred_skill": pred_skill,
    }


def write_mobility_variant(base_data_dir: Path | str, out_dir: Path | str, cfg: ExpConfig, speed_kmh: float) -> Path:
    base_data_dir, out_dir = Path(base_data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(base_data_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    features = np.asarray(np.load(base_data_dir / "features.npy", mmap_mode="r"), dtype=np.float32)
    rate_now = np.asarray(np.load(base_data_dir / "rate_now.npy", mmap_mode="r"), dtype=np.float32)
    rate_future_base = np.asarray(np.load(base_data_dir / "rate_future.npy", mmap_mode="r"), dtype=np.float32)
    hist_rate = np.asarray(np.load(base_data_dir / "hist_rate.npy", mmap_mode="r"), dtype=np.float32)
    pmi0_pair = np.asarray(np.load(base_data_dir / "pmi_0ms.npy", mmap_mode="r"), dtype=np.float32)
    pmi5_pair = np.asarray(np.load(base_data_dir / "pmi_5ms.npy", mmap_mode="r"), dtype=np.float32)

    rng = np.random.default_rng(int(cfg.seed) + 10_007 + int(round(float(speed_kmh) * 100)))
    speeds = np.full((features.shape[0],), float(speed_kmh), dtype=np.float32)
    aug = augment_mobility_chunk(pmi0_pair, pmi5_pair, rate_now, rate_future_base, speeds, cfg, rng)

    x = features.copy()
    x[:, :, cfg.idx_future_pmi_start : cfg.idx_future_pmi_start + cfg.pmi_real_dim] = aug["pmi_pred_flat"]
    x[:, :, cfg.idx_future_rate] = aug["rate_pred"]
    x[:, :, cfg.idx_speed] = aug["speed_norm"][:, None] if getattr(cfg, "mobility_use_speed_feature", True) else 0.0
    pmi_true_flat = np.concatenate([aug["pmi_true_pair"][..., 0], aug["pmi_true_pair"][..., 1]], axis=-1)

    greedy_scores = set_sinr_greedy_scores(
        rate_now,
        aug["rate_true"],
        hist_rate,
        pmi_true_flat,
        cfg.eta,
        cfg.schedule_users,
        eps=cfg.eps,
        fairness_alpha=cfg.fairness_alpha,
        interference_beta=getattr(cfg, "paper_set_interference_beta", 0.40),
        power_reference_users=(getattr(cfg, "paper_power_reference_users", 0) or cfg.schedule_users),
    )
    labels = (greedy_scores > 0).astype(np.float32)
    q = compute_q(rate_now, aug["rate_true"], hist_rate, cfg.eta, cfg.eps, cfg.fairness_alpha)
    qn = (q - q.mean(axis=1, keepdims=True)) / (q.std(axis=1, keepdims=True) + cfg.eps)
    target_scores = greedy_scores + 0.05 * np.tanh(qn).astype(np.float32)

    np.save(out_dir / "features.npy", x.astype(np.float32))
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "target_scores.npy", target_scores.astype(np.float32))
    np.save(out_dir / "rate_now.npy", rate_now)
    np.save(out_dir / "rate_pred.npy", aug["rate_pred"])
    np.save(out_dir / "rate_future.npy", aug["rate_true"])
    np.save(out_dir / "hist_rate.npy", hist_rate)
    np.save(out_dir / "speed_kmh.npy", speeds)
    np.save(out_dir / "pmi_0ms.npy", pmi0_pair)
    np.save(out_dir / "pmi_5ms.npy", aug["pmi_true_pair"])
    with open(out_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump({"train": [], "val": [], "test": list(range(features.shape[0]))}, f, indent=2)
    meta.update({
        "source": "official_mat_mobility_variant",
        "base_data_dir": str(base_data_dir),
        "speed_kmh": float(speed_kmh),
        "temporal_correlation": float(np.mean(aug["rho"])),
        "prediction_skill": float(np.mean(aug["pred_skill"])),
        "mobility_model": getattr(cfg, "mobility_correlation_model", "gaussian_doppler"),
        "labels_recomputed": True,
    })
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return out_dir

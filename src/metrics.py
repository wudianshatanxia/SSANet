from __future__ import annotations

from typing import Dict, Literal

import numpy as np
import torch

Reduction = Literal["mean", "sum"]


def _safe_history(hist_rate, eps: float):
    if isinstance(hist_rate, torch.Tensor):
        return torch.clamp(hist_rate, min=float(eps))
    return np.maximum(np.asarray(hist_rate), float(eps))


def torch_or_numpy_power(x, alpha: float):
    if isinstance(x, torch.Tensor):
        return torch.pow(x, alpha)
    return np.power(x, alpha)


def compute_q(rate_now, rate_future, hist_rate, eta: float, eps: float = 1e-6, fairness_alpha: float = 1.0):
    """Compute prediction-aware proportional-fair utility."""
    eta = float(eta)
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must be in [0,1], got {eta}")
    if fairness_alpha < 0:
        raise ValueError("fairness_alpha must be non-negative")
    hist_safe = _safe_history(hist_rate, eps)
    return (eta * rate_now + (1.0 - eta) * rate_future) / torch_or_numpy_power(hist_safe, fairness_alpha)


def _validate_topk(k: int, num_users: int) -> int:
    k = int(k)
    if k < 1 or k > int(num_users):
        raise ValueError(f"top-k must satisfy 1 <= k <= {num_users}, got {k}")
    return k


def _validate_realimag_dim(last_dim: int) -> int:
    if last_dim < 2 or last_dim % 2 != 0:
        raise ValueError("PMI features must be [all real parts, all imaginary parts]")
    return last_dim // 2


def numpy_complex_correlation_matrix(pmi_features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pmi = np.asarray(pmi_features, dtype=np.float32)
    if pmi.ndim != 3:
        raise ValueError(f"Expected PMI shape (B,K,2C), got {pmi.shape}")
    c = _validate_realimag_dim(pmi.shape[-1])
    real, imag = pmi[..., :c], pmi[..., c:]
    if c >= 64 and c % 8 == 0:
        real = real.reshape(*real.shape[:-1], c // 8, 8).mean(axis=-1)
        imag = imag.reshape(*imag.shape[:-1], c // 8, 8).mean(axis=-1)
    dot_real = np.einsum("bkc,blc->bkl", real, real, optimize=True)
    dot_real += np.einsum("bkc,blc->bkl", imag, imag, optimize=True)
    dot_imag = np.einsum("bkc,blc->bkl", real, imag, optimize=True)
    dot_imag -= np.einsum("bkc,blc->bkl", imag, real, optimize=True)
    magnitude = np.sqrt(np.maximum(dot_real * dot_real + dot_imag * dot_imag, 0.0))
    norm = np.sqrt(np.maximum((real * real + imag * imag).sum(axis=-1), 0.0))
    denom = norm[:, :, None] * norm[:, None, :]
    return np.clip(magnitude / np.maximum(denom, float(eps)), 0.0, 1.0).astype(np.float32)


def torch_complex_correlation_matrix(pmi_features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if pmi_features.ndim != 3:
        raise ValueError(f"Expected PMI shape (B,K,2C), got {tuple(pmi_features.shape)}")
    c = _validate_realimag_dim(pmi_features.shape[-1])
    real, imag = pmi_features[..., :c], pmi_features[..., c:]
    if c >= 64 and c % 8 == 0:
        real = real.reshape(*real.shape[:-1], c // 8, 8).mean(dim=-1)
        imag = imag.reshape(*imag.shape[:-1], c // 8, 8).mean(dim=-1)
    dot_real = torch.bmm(real, real.transpose(1, 2)) + torch.bmm(imag, imag.transpose(1, 2))
    dot_imag = torch.bmm(real, imag.transpose(1, 2)) - torch.bmm(imag, real.transpose(1, 2))
    magnitude = torch.sqrt(torch.clamp(dot_real.square() + dot_imag.square(), min=0.0))
    norm = torch.sqrt(torch.clamp((real.square() + imag.square()).sum(dim=-1), min=0.0))
    denom = norm.unsqueeze(2) * norm.unsqueeze(1)
    return torch.clamp(magnitude / torch.clamp(denom, min=float(eps)), 0.0, 1.0)


def topk_mask_from_scores(scores: torch.Tensor, k: int) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (B,K), got {tuple(scores.shape)}")
    k = _validate_topk(k, scores.shape[1])
    idx = torch.topk(scores, k=k, dim=1).indices
    mask = torch.zeros_like(scores)
    mask.scatter_(1, idx, 1.0)
    return mask


def numpy_topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (B,K), got {scores.shape}")
    k = _validate_topk(k, scores.shape[1])
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    mask = np.zeros_like(scores, dtype=np.float32)
    rows = np.arange(scores.shape[0])[:, None]
    mask[rows, idx] = 1.0
    return mask


def torch_selected_pair_statistics(mask: torch.Tensor, pmi_features: torch.Tensor, topk: int, eps: float = 1e-6):
    """Compute selected-user correlation statistics with PyTorch."""
    topk = _validate_topk(topk, mask.shape[1])
    corr = torch_complex_correlation_matrix(pmi_features, eps)
    eye = torch.eye(corr.size(1), dtype=torch.bool, device=corr.device).unsqueeze(0)
    corr = corr.masked_fill(eye, 0.0)
    pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
    selected_corr_sum = (corr * pair_mask).sum(dim=2)
    mean_corr_per_user = selected_corr_sum / float(max(topk - 1, 1))
    mean_pair_corr = selected_corr_sum.sum(dim=1) / float(max(topk * (topk - 1), 1))
    return mean_pair_corr, mean_corr_per_user


def numpy_selected_pair_statistics(mask: np.ndarray, pmi_features: np.ndarray, topk: int, eps: float = 1e-6):
    """Compute selected-user correlation statistics with NumPy."""
    mask = np.asarray(mask)
    topk = _validate_topk(topk, mask.shape[1])
    selected_idx = np.argpartition(-mask, kth=topk - 1, axis=1)[:, :topk]
    rows = np.arange(mask.shape[0])[:, None]
    selected_pmi = np.asarray(pmi_features, dtype=np.float32)[rows, selected_idx]
    corr = numpy_complex_correlation_matrix(selected_pmi, eps)
    diag = np.arange(topk)
    corr[:, diag, diag] = 0.0
    mean_pair_corr = corr.sum(axis=(1, 2)) / float(max(topk * (topk - 1), 1))
    per_selected = corr.sum(axis=2) / float(max(topk - 1, 1))
    per_user = np.zeros_like(mask, dtype=np.float32)
    per_user[rows, selected_idx] = per_selected
    return mean_pair_corr.astype(np.float32), per_user


def torch_selected_corr_penalty(mask: torch.Tensor, pmi_features: torch.Tensor, topk: int, eps: float = 1e-6) -> torch.Tensor:
    return torch_selected_pair_statistics(mask, pmi_features, topk, eps)[0]


def numpy_selected_corr_penalty(mask: np.ndarray, pmi_features: np.ndarray, topk: int, eps: float = 1e-6) -> np.ndarray:
    return numpy_selected_pair_statistics(mask, pmi_features, topk, eps)[0]


def _reduce_selected(values, mask, topk: int, reduction: Reduction):
    selected_sum = (mask * values).sum(dim=1) if isinstance(values, torch.Tensor) else (mask * values).sum(axis=1)
    if reduction == "mean":
        return selected_sum / float(topk)
    if reduction == "sum":
        return selected_sum
    raise ValueError(f"Unsupported utility reduction: {reduction}")


def torch_set_sinr_contribution(
    scores: torch.Tensor,
    rate_future: torch.Tensor,
    pmi_future_features: torch.Tensor,
    topk: int,
    eps: float = 1e-6,
    interference_beta: float = 0.40,
    power_reference_users: int | None = None,
):
    """Compute per-user rates under shared power and spatial leakage."""
    mask = topk_mask_from_scores(scores, topk)
    idx = torch.topk(scores, k=int(topk), dim=1).indices
    rows = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
    selected_pmi = pmi_future_features[rows, idx]
    corr = torch_complex_correlation_matrix(selected_pmi, eps)
    eye = torch.eye(int(topk), dtype=torch.bool, device=scores.device).unsqueeze(0)
    corr = corr.masked_fill(eye, 0.0)
    corr_power = corr.square()

    selected_rate = rate_future[rows, idx]
    standalone_snr = torch.expm1(torch.log(torch.tensor(2.0, device=scores.device, dtype=scores.dtype)) * torch.clamp(selected_rate, 0.0, 30.0))
    reference = float(power_reference_users if power_reference_users and power_reference_users > 0 else topk)
    stream_snr = standalone_snr * (reference / float(topk))
    leakage = corr_power.sum(dim=2)
    sinr = stream_snr / torch.clamp(1.0 + float(interference_beta) * stream_snr * leakage, min=float(eps))
    selected_effective_rate = torch.log2(1.0 + torch.clamp(sinr, min=0.0))
    selected_raw_rate = torch.log2(1.0 + torch.clamp(stream_snr, min=0.0))

    contribution = torch.zeros_like(mask)
    raw_contribution = torch.zeros_like(mask)
    contribution.scatter_(1, idx, selected_effective_rate)
    raw_contribution.scatter_(1, idx, selected_raw_rate)
    mean_pair_corr = corr.sum(dim=(1, 2)) / float(max(int(topk) * (int(topk) - 1), 1))
    return mask, contribution, raw_contribution, mean_pair_corr


def torch_soft_set_utility_loss(
    logits: torch.Tensor,
    rate_future: torch.Tensor,
    hist_rate: torch.Tensor,
    pmi_future_features: torch.Tensor,
    topk: int,
    eps: float = 1e-6,
    interference_beta: float = 0.40,
    power_reference_users: int | None = None,
    fairness_alpha: float = 0.5,
    temperature: float = 0.75,
) -> torch.Tensor:
    """Compute a differentiable set-throughput loss."""
    k = _validate_topk(topk, logits.shape[1])
    tau = max(float(temperature), 1e-3)
    weights = torch.softmax(logits / tau, dim=1) * float(k)
    overflow_penalty = torch.relu(weights - 1.0).square().mean()

    corr = torch_complex_correlation_matrix(pmi_future_features, eps)
    eye = torch.eye(corr.shape[1], dtype=torch.bool, device=logits.device).unsqueeze(0)
    corr_power = corr.masked_fill(eye, 0.0).square()
    leakage = torch.bmm(corr_power, weights.unsqueeze(-1)).squeeze(-1)

    standalone_snr = torch.expm1(
        torch.log(torch.tensor(2.0, device=logits.device, dtype=logits.dtype))
        * torch.clamp(rate_future, 0.0, 30.0)
    )
    reference = float(power_reference_users if power_reference_users and power_reference_users > 0 else k)
    stream_snr = standalone_snr * (reference / float(k))
    sinr = stream_snr / torch.clamp(
        1.0 + float(interference_beta) * stream_snr * leakage,
        min=float(eps),
    )
    effective_rate = torch.log2(1.0 + torch.clamp(sinr, min=0.0))
    hist = torch.clamp(hist_rate, min=float(eps)).pow(float(fairness_alpha))
    utility = (weights * effective_rate / hist).sum(dim=1)
    return -utility.mean() / float(k) + 0.05 * overflow_penalty


def torch_metrics_from_scores(
    scores: torch.Tensor,
    rate_now: torch.Tensor,
    rate_future: torch.Tensor,
    hist_rate: torch.Tensor,
    eta: float,
    topk: int,
    eps: float = 1e-6,
    fairness_alpha: float = 1.0,
    pmi_features: torch.Tensor | None = None,
    pmi_future_features: torch.Tensor | None = None,
    corr_penalty_weight: float = 0.0,
    q_reduction: Reduction = "mean",
    interference_beta: float = 0.0,
    power_reference_users: int | None = None,
) -> Dict[str, float]:
    q = compute_q(rate_now, rate_future, hist_rate, eta, eps, fairness_alpha)
    corr_source = pmi_future_features if pmi_future_features is not None else pmi_features
    if corr_source is not None and float(interference_beta) > 0.0:
        mask, contribution, raw_contribution, mean_pair_corr = torch_set_sinr_contribution(
            scores, rate_future, corr_source, topk, eps, interference_beta, power_reference_users
        )
        effective_sum_rate = contribution.sum(dim=1)
        raw_sum_rate = raw_contribution.sum(dim=1)
    else:
        mask = topk_mask_from_scores(scores, topk)
        raw_sum_rate = (mask * rate_future).sum(dim=1)
        effective_sum_rate = raw_sum_rate
        mean_pair_corr = torch.zeros_like(raw_sum_rate)
    raw_q = _reduce_selected(q, mask, topk, q_reduction)
    corr_penalty = float(corr_penalty_weight) * mean_pair_corr
    spatial_q = raw_q - corr_penalty
    hist_selected = (mask * hist_rate).sum(dim=1) / float(topk)
    return {
        "avg_sum_rate": float(effective_sum_rate.mean().detach().cpu()),
        "avg_effective_sum_rate": float(effective_sum_rate.mean().detach().cpu()),
        "avg_raw_sum_rate": float(raw_sum_rate.mean().detach().cpu()),
        "avg_q": float(spatial_q.mean().detach().cpu()),
        "avg_raw_q": float(raw_q.mean().detach().cpu()),
        "avg_corr_penalty": float(corr_penalty.mean().detach().cpu()),
        "avg_pair_corr": float(mean_pair_corr.mean().detach().cpu()),
        "avg_selected_hist_rate": float(hist_selected.mean().detach().cpu()),
    }


def numpy_metrics_from_scores(
    scores: np.ndarray,
    rate_now: np.ndarray,
    rate_future: np.ndarray,
    hist_rate: np.ndarray,
    eta: float,
    topk: int,
    eps: float = 1e-6,
    fairness_alpha: float = 1.0,
    pmi_features: np.ndarray | None = None,
    pmi_future_features: np.ndarray | None = None,
    corr_penalty_weight: float = 0.0,
    q_reduction: Reduction = "mean",
    interference_beta: float = 0.0,
    power_reference_users: int | None = None,
) -> Dict[str, float]:
    mask = numpy_topk_mask(scores, topk)
    q = compute_q(rate_now, rate_future, hist_rate, eta, eps, fairness_alpha)
    corr_source = pmi_future_features if pmi_future_features is not None else pmi_features
    if corr_source is not None and float(interference_beta) > 0.0:
        selected_idx = np.argpartition(-scores, kth=int(topk) - 1, axis=1)[:, : int(topk)]
        rows = np.arange(scores.shape[0])[:, None]
        selected_pmi = np.asarray(corr_source, dtype=np.float32)[rows, selected_idx]
        corr = numpy_complex_correlation_matrix(selected_pmi, eps)
        diag = np.arange(int(topk))
        corr[:, diag, diag] = 0.0
        selected_rate = np.asarray(rate_future, dtype=np.float32)[rows, selected_idx]
        standalone_snr = np.expm1(np.log(2.0) * np.clip(selected_rate, 0.0, 30.0))
        reference = float(power_reference_users if power_reference_users and power_reference_users > 0 else topk)
        stream_snr = standalone_snr * (reference / float(topk))
        leakage = np.square(corr, dtype=np.float32).sum(axis=2)
        sinr = stream_snr / np.maximum(1.0 + float(interference_beta) * stream_snr * leakage, float(eps))
        selected_effective = np.log2(1.0 + np.maximum(sinr, 0.0)).astype(np.float32)
        selected_raw = np.log2(1.0 + np.maximum(stream_snr, 0.0)).astype(np.float32)
        contribution = np.zeros_like(mask, dtype=np.float32)
        raw_contribution = np.zeros_like(mask, dtype=np.float32)
        contribution[rows, selected_idx] = selected_effective
        raw_contribution[rows, selected_idx] = selected_raw
        effective_sum_rate = contribution.sum(axis=1)
        raw_sum_rate = raw_contribution.sum(axis=1)
        mean_pair_corr = corr.sum(axis=(1, 2)) / float(max(int(topk) * (int(topk) - 1), 1))
    else:
        raw_sum_rate = (mask * rate_future).sum(axis=1)
        effective_sum_rate = raw_sum_rate
        mean_pair_corr = np.zeros(raw_sum_rate.shape, dtype=np.float32)
    raw_q = _reduce_selected(q, mask, topk, q_reduction)
    corr_penalty = float(corr_penalty_weight) * mean_pair_corr
    return {
        "avg_sum_rate": float(effective_sum_rate.mean()),
        "avg_effective_sum_rate": float(effective_sum_rate.mean()),
        "avg_raw_sum_rate": float(raw_sum_rate.mean()),
        "avg_q": float((raw_q - corr_penalty).mean()),
        "avg_raw_q": float(raw_q.mean()),
        "avg_corr_penalty": float(corr_penalty.mean()),
        "avg_pair_corr": float(mean_pair_corr.mean()),
        "avg_selected_hist_rate": float(((mask * hist_rate).sum(axis=1) / float(topk)).mean()),
    }

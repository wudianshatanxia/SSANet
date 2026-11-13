from __future__ import annotations

"""Compute official-style scheduling scores."""

from pathlib import Path
from typing import Dict, Optional

import numpy as np


def selected_indices_from_scores(scores: np.ndarray, topk: int) -> np.ndarray:
    topk = min(max(int(topk), 1), scores.shape[1])
    return np.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]


def official_score_from_indices(
    schedule_indices: np.ndarray,
    pmi_0ms: np.ndarray,
    pmi_5ms: np.ndarray,
    eigenvalue_5ms: np.ndarray,
    rate_old: np.ndarray,
    noise: float = 5.9716e-14,
    eps: float = 1e-12,
) -> Dict[str, float | np.ndarray]:
    """Compute official-style scores for selected user indices."""
    B = int(schedule_indices.shape[0])
    topk = int(schedule_indices.shape[1])
    sample_scores = np.zeros(B, dtype=np.float64)
    per_user_score_sum = np.zeros(120, dtype=np.float64)
    per_user_count = np.zeros(120, dtype=np.int64)

    for sample_index in range(B):
        selected = schedule_indices[sample_index].astype(np.int64)
        pmi_sample = pmi_0ms[sample_index, selected, :, :]
        pmi_future_sample = pmi_5ms[sample_index, selected, :, :]
        eigen_future = eigenvalue_5ms[sample_index, selected]
        hist = rate_old[sample_index, selected]

        P = (pmi_sample[:, :, 0] + 1j * pmi_sample[:, :, 1]).T.astype(np.complex64)
        gram = P.conj().T @ P
        W = P @ np.linalg.pinv(gram)
        col_norm = np.abs(W).sum(axis=0, keepdims=True)
        W = W / np.maximum(col_norm, eps)

        for j in range(topk):
            P_j = (pmi_future_sample[j, :, 0] + 1j * pmi_future_sample[j, :, 1]).astype(np.complex64)
            desired = np.abs(P_j.conj().T @ W[:, j]) ** 2 * eigen_future[j]
            interf = 0.0
            for s in range(topk):
                if s == j:
                    continue
                interf += np.abs(P_j.conj().T @ W[:, s]) ** 2 * eigen_future[j]
            sinr = desired / (noise + interf + eps)
            rate_new = np.log2(1.0 + sinr)
            score = np.log2(max(rate_new + hist[j], eps))
            sample_scores[sample_index] += float(score)
            per_user_score_sum[selected[j]] += float(score)
            per_user_count[selected[j]] += 1

    return {
        "avg_official_score": float(sample_scores.mean()),
        "sum_official_score": float(sample_scores.sum()),
        "sample_scores": sample_scores,
        "per_user_score_sum": per_user_score_sum,
        "per_user_count": per_user_count,
    }


def official_score_from_scores(
    scores: np.ndarray,
    data_dir: Path | str,
    topk: int,
    split_indices: Optional[np.ndarray] = None,
    noise: float = 5.9716e-14,
) -> Dict[str, float | np.ndarray]:
    data_dir = Path(data_dir)
    required = ["pmi_0ms.npy", "pmi_5ms.npy", "rate_future.npy", "hist_rate.npy"]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing official score arrays in {data_dir}: {missing}. Re-run conversion with --store_pmi_for_official_score."
        )
    idx = selected_indices_from_scores(scores, topk)
    pmi0 = np.load(data_dir / "pmi_0ms.npy", mmap_mode="r")
    pmi5 = np.load(data_dir / "pmi_5ms.npy", mmap_mode="r")
    r5 = np.load(data_dir / "rate_future.npy", mmap_mode="r")
    hist = np.load(data_dir / "hist_rate.npy", mmap_mode="r")
    if split_indices is not None:
        pmi0 = pmi0[split_indices]
        pmi5 = pmi5[split_indices]
        r5 = r5[split_indices]
        hist = hist[split_indices]
    return official_score_from_indices(idx, np.asarray(pmi0), np.asarray(pmi5), np.asarray(r5), np.asarray(hist), noise=noise)

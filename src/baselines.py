from __future__ import annotations

import numpy as np

from src.metrics import compute_q, numpy_complex_correlation_matrix


def _safe_hist(hist_rate: np.ndarray, eps: float) -> np.ndarray:
    return np.maximum(np.asarray(hist_rate, dtype=np.float32), float(eps))


def no_prediction_scores(rate_now: np.ndarray, hist_rate: np.ndarray, eps: float = 1e-6, fairness_alpha: float = 1.0) -> np.ndarray:
    """Score users from current rates and rate history."""
    return np.asarray(rate_now, dtype=np.float32) / np.power(_safe_hist(hist_rate, eps), fairness_alpha)


def pf_prediction_scores(
    rate_now: np.ndarray,
    rate_pred: np.ndarray,
    hist_rate: np.ndarray,
    eta: float,
    eps: float = 1e-6,
    fairness_alpha: float = 1.0,
) -> np.ndarray:
    """Score users with prediction-aware proportional fairness."""
    return compute_q(rate_now, rate_pred, hist_rate, eta, eps, fairness_alpha).astype(np.float32)


def set_sinr_greedy_scores(
    rate_now: np.ndarray,
    rate_future_for_score: np.ndarray,
    hist_rate: np.ndarray,
    pmi_for_score: np.ndarray,
    eta: float,
    topk: int,
    *,
    eps: float = 1e-6,
    fairness_alpha: float = 0.5,
    interference_beta: float = 0.40,
    power_reference_users: int | None = None,
    chunk_size: int = 32,
    priority_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Rank users by greedy set-SINR utility."""
    rate_now = np.asarray(rate_now, dtype=np.float32)
    rate_future = np.asarray(rate_future_for_score, dtype=np.float32)
    hist = _safe_hist(hist_rate, eps)
    pmi = np.asarray(pmi_for_score, dtype=np.float32)
    if rate_now.shape != rate_future.shape or rate_now.shape != hist.shape:
        raise ValueError("rate_now, rate_future_for_score and hist_rate must have identical shapes")
    if pmi.ndim != 3 or pmi.shape[:2] != rate_now.shape:
        raise ValueError("pmi_for_score must have shape (B,K,2C)")
    bsz, num_users = rate_now.shape
    if not 1 <= int(topk) <= num_users:
        raise ValueError(f"topk must be in [1,{num_users}], got {topk}")

    q_rate = float(eta) * rate_now + (1.0 - float(eta)) * rate_future
    standalone_snr = np.expm1(np.log(2.0) * np.clip(q_rate, 0.0, 30.0)).astype(np.float32)
    hist_weight = np.power(hist, float(fairness_alpha)).astype(np.float32)
    priority = np.ones_like(rate_now, dtype=np.float32) if priority_weights is None else np.asarray(priority_weights, dtype=np.float32)
    if priority.shape != rate_now.shape:
        raise ValueError("priority_weights must have shape (B,K)")
    priority = np.maximum(priority, float(eps))
    reference = float(power_reference_users if power_reference_users and power_reference_users > 0 else topk)
    out = np.zeros((bsz, num_users), dtype=np.float32)

    for start in range(0, bsz, int(chunk_size)):
        end = min(start + int(chunk_size), bsz)
        corr = numpy_complex_correlation_matrix(pmi[start:end], eps)
        corr2 = np.square(corr, dtype=np.float32)
        for local, b in enumerate(range(start, end)):
            selected: list[int] = []
            available = np.ones(num_users, dtype=bool)
            for rank in range(int(topk)):
                candidates = np.flatnonzero(available)
                new_size = rank + 1
                stream_snr_all = standalone_snr[b] * (reference / float(new_size))
                if not selected:
                    candidate_sinr = stream_snr_all[candidates]
                    candidate_rate = np.log2(1.0 + np.maximum(candidate_sinr, 0.0))
                    objective = priority[b, candidates] * candidate_rate / hist_weight[b, candidates]
                else:
                    sel = np.asarray(selected, dtype=np.int64)
                    existing_base_leak = corr2[local][np.ix_(sel, sel)].sum(axis=1)
                    candidate_to_selected = corr2[local][np.ix_(candidates, sel)]
                    candidate_leak = candidate_to_selected.sum(axis=1)
                    candidate_snr = stream_snr_all[candidates]
                    candidate_sinr = candidate_snr / np.maximum(
                        1.0 + float(interference_beta) * candidate_snr * candidate_leak,
                        float(eps),
                    )
                    candidate_rate = np.log2(1.0 + np.maximum(candidate_sinr, 0.0))

                    existing_leak = existing_base_leak[None, :] + candidate_to_selected
                    existing_snr = stream_snr_all[sel][None, :]
                    existing_sinr = existing_snr / np.maximum(
                        1.0 + float(interference_beta) * existing_snr * existing_leak,
                        float(eps),
                    )
                    existing_rate = np.log2(1.0 + np.maximum(existing_sinr, 0.0))
                    objective = (
                        priority[b, sel][None, :] * existing_rate / hist_weight[b, sel][None, :]
                    ).sum(axis=1) + priority[b, candidates] * candidate_rate / hist_weight[b, candidates]

                pick = int(candidates[int(np.argmax(objective))])
                selected.append(pick)
                available[pick] = False
                out[b, pick] = 1.0 - rank / float(max(int(topk), 1))
    return out


def learned_set_sinr_refinement_scores(
    learned_scores: np.ndarray,
    rate_now: np.ndarray,
    rate_pred: np.ndarray,
    hist_rate: np.ndarray,
    pmi_pred: np.ndarray,
    eta: float,
    topk: int,
    *,
    eps: float = 1e-6,
    fairness_alpha: float = 0.5,
    interference_beta: float = 0.40,
    power_reference_users: int | None = None,
    candidate_multiplier: int = 3,
    learned_weight: float = 0.70,
    priority_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Refine learned rankings with predicted set-SINR utility."""
    learned = np.asarray(learned_scores, dtype=np.float32)
    rate_now = np.asarray(rate_now, dtype=np.float32)
    rate_pred = np.asarray(rate_pred, dtype=np.float32)
    hist = _safe_hist(hist_rate, eps)
    pmi = np.asarray(pmi_pred, dtype=np.float32)
    if learned.shape != rate_now.shape or learned.shape != rate_pred.shape or learned.shape != hist.shape:
        raise ValueError("learned_scores and scalar inputs must have identical (B,K) shapes")
    bsz, num_users = learned.shape
    if not 1 <= int(topk) <= num_users:
        raise ValueError(f"topk must be in [1,{num_users}], got {topk}")
    pool_size = min(num_users, max(int(topk), int(candidate_multiplier) * int(topk)))
    learned_norm = (learned - learned.mean(axis=1, keepdims=True)) / (learned.std(axis=1, keepdims=True) + eps)
    q_rate = float(eta) * rate_now + (1.0 - float(eta)) * rate_pred
    snr = np.expm1(np.log(2.0) * np.clip(q_rate, 0.0, 30.0)).astype(np.float32)
    hist_weight = np.power(hist, float(fairness_alpha)).astype(np.float32)
    priority = np.ones_like(rate_now, dtype=np.float32) if priority_weights is None else np.asarray(priority_weights, dtype=np.float32)
    if priority.shape != rate_now.shape:
        raise ValueError("priority_weights must have shape (B,K)")
    priority = np.maximum(priority, float(eps))
    reference = float(power_reference_users if power_reference_users and power_reference_users > 0 else topk)
    output = np.zeros_like(learned, dtype=np.float32)

    corr = numpy_complex_correlation_matrix(pmi, eps)
    corr2 = np.square(corr, dtype=np.float32)
    for b in range(bsz):
        pool = np.argpartition(-learned[b], kth=pool_size - 1)[:pool_size]
        available = np.ones(pool_size, dtype=bool)
        selected: list[int] = []
        for rank in range(int(topk)):
            candidate_pos = np.flatnonzero(available)
            candidates = pool[candidate_pos]
            new_size = rank + 1
            stream_snr = snr[b] * (reference / float(new_size))
            if selected:
                sel = np.asarray(selected, dtype=np.int64)
                base_leak = corr2[b][np.ix_(sel, sel)].sum(axis=1)
                cand_to_sel = corr2[b][np.ix_(candidates, sel)]
                cand_leak = cand_to_sel.sum(axis=1)
                cand_sinr = stream_snr[candidates] / np.maximum(
                    1.0 + float(interference_beta) * stream_snr[candidates] * cand_leak,
                    float(eps),
                )
                cand_rate = np.log2(1.0 + np.maximum(cand_sinr, 0.0))
                existing_leak = base_leak[None, :] + cand_to_sel
                existing_snr = stream_snr[sel][None, :]
                existing_sinr = existing_snr / np.maximum(
                    1.0 + float(interference_beta) * existing_snr * existing_leak,
                    float(eps),
                )
                existing_rate = np.log2(1.0 + np.maximum(existing_sinr, 0.0))
                set_utility = (priority[b, sel][None, :] * existing_rate / hist_weight[b, sel][None, :]).sum(axis=1)
                set_utility += priority[b, candidates] * cand_rate / hist_weight[b, candidates]
            else:
                cand_rate = np.log2(1.0 + np.maximum(stream_snr[candidates], 0.0))
                set_utility = priority[b, candidates] * cand_rate / hist_weight[b, candidates]
            utility_norm = (set_utility - set_utility.mean()) / (set_utility.std() + eps)
            combined = float(learned_weight) * learned_norm[b, candidates] + (1.0 - float(learned_weight)) * utility_norm
            local_pick = int(np.argmax(combined))
            pick = int(candidates[local_pick])
            selected.append(pick)
            available[candidate_pos[local_pick]] = False
            output[b, pick] = 1.0 - rank / float(max(int(topk), 1))
    return output

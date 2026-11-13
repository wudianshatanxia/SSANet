from __future__ import annotations

import numpy as np

from config import ExpConfig
from src.plot_results import (
    _prediction_from_true_channel,
    _selected_set_sinr_metrics,
)


def test_set_dependent_rate_is_not_plain_linear_sum():
    cfg = ExpConfig(num_ues=4, num_trps=1, antennas_per_trp=2, schedule_users=4)
    cfg.paper_power_reference_users = 4
    cfg.paper_set_interference_beta = 0.5
    pmi = np.zeros((1, 4, cfg.pmi_real_dim), dtype=np.float32)
    pmi[..., 0] = 1.0
    state = {
        "pmi_future": pmi,
        "rate_future": np.full((1, 4), 2.0, dtype=np.float32),
    }
    scores = np.array([[4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
    m2 = _selected_set_sinr_metrics(scores, state, 2, cfg)
    m4 = _selected_set_sinr_metrics(scores, state, 4, cfg)
    sum2 = float(m2["per_sample_sum_rate"][0])
    sum4 = float(m4["per_sample_sum_rate"][0])
    assert sum4 < 2.0 * sum2
    assert float(m4["per_sample_raw_sum_rate"][0]) > sum4


def test_channel_error_hits_requested_nmse_and_changes_rate_prediction():
    cfg = ExpConfig(num_ues=4, num_trps=1, antennas_per_trp=2)
    rng = np.random.default_rng(7)
    h = rng.normal(size=(16, 4, 2)) + 1j * rng.normal(size=(16, 4, 2))
    h = h / np.linalg.norm(h, axis=-1, keepdims=True)
    snr = np.full((16, 4), 10.0, dtype=np.float32)
    _, rate_pred, pmi_nmse, _ = _prediction_from_true_channel(
        h.astype(np.complex64), snr, -10.0, rng, cfg.eps
    )
    measured_db = 10.0 * np.log10(pmi_nmse)
    assert abs(measured_db - (-10.0)) < 1.0
    true_rate = np.log2(1.0 + snr)
    assert not np.allclose(rate_pred, true_rate)


def test_ssa_eta_override_changes_q_prior_without_new_checkpoint_parameters():
    import torch
    from src.models import build_model

    cfg = ExpConfig(num_ues=6, num_trps=1, antennas_per_trp=2, embed_dim=16)
    model = build_model("ssa", cfg).eval()
    x = torch.randn(2, cfg.num_ues, cfg.feature_dim)
    x[:, :, cfg.idx_now_rate] = torch.linspace(1.0, 3.0, cfg.num_ues)
    x[:, :, cfg.idx_future_rate] = torch.linspace(3.0, 1.0, cfg.num_ues)
    x[:, :, cfg.idx_hist_rate] = 1.0
    with torch.no_grad():
        a = model.forward_with_aux(x, eta_override=0.1)["logits"]
        b = model.forward_with_aux(x, eta_override=0.9)["logits"]
    assert not torch.allclose(a, b)


def test_set_sinr_greedy_avoids_redundant_high_rate_pair():
    from src.baselines import set_sinr_greedy_scores

    cfg = ExpConfig(num_ues=3, num_trps=1, antennas_per_trp=2, schedule_users=2)
    pmi = np.array(
        [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    rate = np.array([[4.0, 3.9, 3.7]], dtype=np.float32)
    hist = np.ones_like(rate)
    scores = set_sinr_greedy_scores(
        rate, rate, hist, pmi, 1.0, 2,
        interference_beta=1.0, power_reference_users=2,
    )
    selected = set(np.flatnonzero(scores[0] > 0).tolist())
    assert 0 in selected and 2 in selected

import numpy as np

from config import ExpConfig
from src.emergency_experiments import (
    CRITICAL,
    apply_trp_outage_to_state,
    assign_emergency_classes,
    priority_weights_from_classes,
    sample_trp_keep_mask,
)


def _random_flat(batch, users, cfg, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch, users, cfg.pmi_real_dim)).astype(np.float32)
    c = cfg.pmi_complex_dim
    complex_x = x[..., :c] + 1j * x[..., c:]
    complex_x /= np.maximum(np.linalg.norm(complex_x, axis=-1, keepdims=True), cfg.eps)
    return np.concatenate([complex_x.real, complex_x.imag], axis=-1).astype(np.float32)


def test_priority_assignment_and_weights():
    cfg = ExpConfig()
    classes = assign_emergency_classes(3, cfg.num_ues, cfg, 7)
    weights = priority_weights_from_classes(classes, cfg)
    assert classes.shape == (3, cfg.num_ues)
    assert weights.shape == classes.shape
    assert np.all(weights[classes == CRITICAL] == cfg.emergency_priority_critical)
    assert np.all(weights >= cfg.emergency_priority_regular)


def test_outage_mask_and_rate_degradation():
    cfg = ExpConfig()
    batch = 2
    flat = _random_flat(batch, cfg.num_ues, cfg, 11)
    rate = np.full((batch, cfg.num_ues), 3.0, dtype=np.float32)
    state = {
        "pmi_now": flat,
        "pmi_pred": flat.copy(),
        "pmi_future": flat.copy(),
        "rate_now": rate,
        "rate_pred": rate.copy(),
        "rate_future": rate.copy(),
    }
    keep = sample_trp_keep_mask(batch, cfg.num_trps, 0.5, 13)
    out = apply_trp_outage_to_state(state, keep, cfg)
    assert out["pmi_now"].shape == flat.shape
    assert np.all(out["rate_future"] <= rate + 1e-6)
    assert np.any(out["rate_future"] < rate - 1e-4)
    assert np.allclose(out["trp_retained_fraction"], keep.mean(axis=1))

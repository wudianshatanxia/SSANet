from __future__ import annotations

import unittest

import numpy as np
import torch

torch.set_num_threads(1)

from config import ExpConfig
from src.checkpointing import config_for_checkpoint
from src.metrics import (
    compute_q,
    numpy_complex_correlation_matrix,
    numpy_topk_mask,
    torch_complex_correlation_matrix,
)
from src.models import build_model
from src.train import listwise_rank_loss, normalize_rank_targets


class CoreLogicTests(unittest.TestCase):
    def test_complex_correlation_keeps_imaginary_inner_product(self):
        pmi = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        corr_np = numpy_complex_correlation_matrix(pmi)
        self.assertAlmostEqual(float(corr_np[0, 0, 1]), 1.0, places=6)
        corr_t = torch_complex_correlation_matrix(torch.from_numpy(pmi))
        self.assertAlmostEqual(float(corr_t[0, 0, 1]), 1.0, places=6)

    def test_compute_q_clamps_invalid_history(self):
        now = np.array([[1.0, 2.0]], dtype=np.float32)
        future = np.array([[2.0, 3.0]], dtype=np.float32)
        hist = np.array([[0.0, -1.0]], dtype=np.float32)
        q = compute_q(now, future, hist, eta=0.4, eps=1e-3, fairness_alpha=0.5)
        self.assertTrue(np.isfinite(q).all())

    def test_topk_validation(self):
        with self.assertRaises(ValueError):
            numpy_topk_mask(np.ones((2, 3), dtype=np.float32), 0)
        with self.assertRaises(ValueError):
            numpy_topk_mask(np.ones((2, 3), dtype=np.float32), 4)

    def test_rank_targets_are_bounded(self):
        target = torch.tensor([[20.0, 19.0, 1.0, 0.0]])
        normalized = normalize_rank_targets(target)
        self.assertGreaterEqual(float(normalized.min()), 0.0)
        self.assertLessEqual(float(normalized.max()), 1.0)
        logits = torch.zeros_like(target, requires_grad=True)
        loss = listwise_rank_loss(logits, target, tau=0.25, normalize_targets=True)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_checkpoint_model_config_merge(self):
        cfg = ExpConfig()
        merged = config_for_checkpoint(cfg, {"embed_dim": 64, "eta": 0.25, "batch_size": 999})
        self.assertEqual(merged.embed_dim, 64)
        self.assertAlmostEqual(merged.eta, 0.25)
        self.assertEqual(merged.batch_size, cfg.batch_size)

    def test_ssa_forward(self):
        cfg = ExpConfig(num_ues=4, num_trps=2, antennas_per_trp=2, embed_dim=16)
        x = torch.randn(1, cfg.num_ues, cfg.feature_dim)
        x[:, :, cfg.idx_hist_rate] = torch.rand(1, cfg.num_ues) + 0.1
        model = build_model("ssa", cfg)
        y = model(x)
        self.assertEqual(tuple(y.shape), (1, cfg.num_ues))
        self.assertTrue(torch.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()

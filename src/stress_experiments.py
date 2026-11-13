from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import ExpConfig
from src.baselines import set_sinr_greedy_scores, pf_prediction_scores
from src.checkpointing import load_model_from_checkpoint, validate_dataset_meta
from src.dataset import SchedulingDataset
from src.metrics import numpy_metrics_from_scores


def _load_eval_arrays(data_dir: Path | str, cfg: ExpConfig, split: str):
    ds = SchedulingDataset(data_dir, split=split)
    x = np.asarray(ds.features[ds.indices], dtype=np.float32)
    rate_now = np.asarray(ds.rate_now[ds.indices], dtype=np.float32)
    rate_future = np.asarray(ds.rate_future[ds.indices], dtype=np.float32)
    hist_rate = np.asarray(ds.hist_rate[ds.indices], dtype=np.float32)
    rate_pred = (
        np.asarray(ds.rate_pred[ds.indices], dtype=np.float32)
        if getattr(ds, "rate_pred", None) is not None
        else x[:, :, cfg.idx_future_rate].astype(np.float32)
    )
    return ds, x, rate_now, rate_pred, rate_future, hist_rate


def _metric_row(
    name: str,
    group: str,
    scores: np.ndarray,
    x: np.ndarray,
    rate_now: np.ndarray,
    rate_future: np.ndarray,
    hist_rate: np.ndarray,
    cfg: ExpConfig,
) -> dict:
    metrics = numpy_metrics_from_scores(
        scores,
        rate_now,
        rate_future,
        hist_rate,
        cfg.eta,
        cfg.schedule_users,
        cfg.eps,
        cfg.fairness_alpha,
        pmi_features=x[:, :, : cfg.pmi_real_dim],
        corr_penalty_weight=getattr(cfg, "eval_corr_penalty", 0.0),
        q_reduction=getattr(cfg, "utility_reduction", "mean"),
    )
    return {"experiment": group, "variant": name, **metrics}


def _time_checkpoint(
    model_name: str,
    ckpt_path: Path,
    data_dir: Path | str,
    cfg: ExpConfig,
    split: str,
    device: Optional[str],
) -> dict:
    device_t = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = SchedulingDataset(data_dir, split=split)
    model, model_cfg, _ = load_model_from_checkpoint(model_name, ckpt_path, cfg, device_t)
    validate_dataset_meta(model_cfg, ds.meta)
    model.eval()
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    try:
        batch = next(iter(loader))["x"].to(device_t)
    except StopIteration as exc:
        raise ValueError("Cannot time an empty split") from exc

    with torch.no_grad():
        for _ in range(10):
            _ = model(batch)
        if device_t.type == "cuda":
            torch.cuda.synchronize()
        repeats = 50
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(batch)
        if device_t.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    processed = repeats * int(batch.shape[0])
    return {
        "experiment": "inference_time",
        "variant": model_name,
        "device": str(device_t),
        "batch_size": int(batch.shape[0]),
        "warmup_batches": 10,
        "timed_batches": repeats,
        "ms_per_sample": 1000.0 * elapsed / max(processed, 1),
    }


def run_stress_experiments(
    data_dir: Path | str,
    ckpt_dir: Path | str,
    result_dir: Path | str,
    cfg: ExpConfig,
    models: Iterable[str] = ("lstm", "ssa"),
    split: str = "test",
    device: Optional[str] = None,
) -> tuple[Path, Path]:
    """Export prediction-error and inference-latency metrics."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    _, x, rate_now, rate_pred, rate_future, hist_rate = _load_eval_arrays(data_dir, cfg, split)

    stress_rows: list[dict] = []
    for error in (0.0, 0.15, 0.30, 0.45, 0.60):
        stressed_pred = (1.0 - error) * rate_pred + error * rate_now
        pf_scores = pf_prediction_scores(rate_now, stressed_pred, hist_rate, cfg.eta, cfg.eps, cfg.fairness_alpha)
        greedy_scores = set_sinr_greedy_scores(
            rate_now,
            stressed_pred,
            hist_rate,
            x[:, :, cfg.idx_future_pmi_start:cfg.idx_future_pmi_start + cfg.pmi_real_dim],
            cfg.eta,
            cfg.schedule_users,
            eps=cfg.eps,
            fairness_alpha=cfg.fairness_alpha,
            interference_beta=cfg.paper_set_interference_beta,
            power_reference_users=cfg.paper_power_reference_users or cfg.schedule_users,
        )
        for name, scores in (("PF Prediction", pf_scores), ("Enhanced Greedy", greedy_scores)):
            row = _metric_row(name, "prediction_error_stress", scores, x, rate_now, rate_future, hist_rate, cfg)
            row["prediction_error"] = error
            stress_rows.append(row)

    stress_out = result_dir / f"stress_prediction_error_{split}.csv"
    pd.DataFrame(stress_rows).to_csv(stress_out, index=False)

    timing_rows: list[dict] = []
    for model_name in models:
        ckpt = Path(ckpt_dir) / f"{model_name.lower()}_best.pt"
        if ckpt.exists():
            timing_rows.append(_time_checkpoint(model_name, ckpt, data_dir, cfg, split, device))
    timing_out = result_dir / f"stress_inference_time_{split}.csv"
    pd.DataFrame(timing_rows).to_csv(timing_out, index=False)
    print(f"[stress] Saved {stress_out} and {timing_out}")
    return stress_out, timing_out

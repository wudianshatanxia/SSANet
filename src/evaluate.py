from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import ExpConfig
from src.baselines import (
    set_sinr_greedy_scores,
    no_prediction_scores,
    pf_prediction_scores,
)
from src.checkpointing import load_model_from_checkpoint, validate_dataset_meta
from src.dataset import SchedulingDataset
from src.metrics import (
    compute_q,
    numpy_metrics_from_scores,
    numpy_topk_mask,
    torch_metrics_from_scores,
)
from src.official_score import official_score_from_scores


def _device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pretty_name(model_name: str) -> str:
    mapping = {
        "ssa": "SSA-Net",
        "ssanet": "SSA-Net",
        "ssa-net": "SSA-Net",
        "lstm": "LSTM",
    }
    return mapping.get(model_name.lower(), model_name.upper())


def _maybe_add_official_score(row: dict, scores: np.ndarray, data_dir: Path, ds: SchedulingDataset, cfg: ExpConfig) -> dict:
    if not getattr(cfg, "use_official_score", False):
        return row
    if not ((data_dir / "pmi_0ms.npy").exists() and (data_dir / "pmi_5ms.npy").exists()):
        return row
    if len(ds.indices) > getattr(cfg, "official_score_max_samples", 2000):
        row["avg_official_score"] = np.nan
        return row
    try:
        official = official_score_from_scores(
            scores,
            data_dir,
            topk=getattr(cfg, "official_schedule_users", cfg.schedule_users),
            split_indices=ds.indices,
            noise=getattr(cfg, "official_score_noise", 5.9716e-14),
        )
        row["avg_official_score"] = official["avg_official_score"]
    except Exception as exc:
        print(f"[eval] Official score skipped: {exc}")
        row["avg_official_score"] = np.nan
    return row


def _per_sample_metrics_from_scores(
    scores: np.ndarray,
    x: np.ndarray,
    rate_now: np.ndarray,
    rate_future: np.ndarray,
    hist_rate: np.ndarray,
    cfg: ExpConfig,
    pmi_future: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    mask = numpy_topk_mask(scores, cfg.schedule_users)
    q = compute_q(rate_now, rate_future, hist_rate, cfg.eta, cfg.eps, cfg.fairness_alpha)
    raw_q_sum = (mask * q).sum(axis=1)
    reduction = getattr(cfg, "utility_reduction", "mean")
    raw_q = raw_q_sum / float(cfg.schedule_users) if reduction == "mean" else raw_q_sum
    corr_source = pmi_future if pmi_future is not None else x[:, :, : cfg.pmi_real_dim]
    from src.metrics import numpy_selected_pair_statistics
    mean_pair_corr, per_user_corr = numpy_selected_pair_statistics(mask, corr_source, cfg.schedule_users, cfg.eps)
    corr_penalty = getattr(cfg, "eval_corr_penalty", 0.0) * mean_pair_corr
    raw_sum_rate = (mask * rate_future).sum(axis=1)
    effective_rate = rate_future / (1.0 + getattr(cfg, "interference_beta", 0.0) * per_user_corr)
    return {
        "sum_rate": (mask * effective_rate).sum(axis=1),
        "raw_sum_rate": raw_sum_rate,
        "q": raw_q - corr_penalty,
        "raw_q": raw_q,
        "corr_penalty": corr_penalty,
        "pair_corr": mean_pair_corr,
        "selected_hist_rate": (mask * hist_rate).sum(axis=1) / float(cfg.schedule_users),
    }


def _per_sample_frame(method: str, method_type: str, sample_indices: np.ndarray, per_sample: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_idx": sample_indices,
            "method": method,
            "type": method_type,
            "sum_rate": per_sample["sum_rate"],
            "raw_sum_rate": per_sample["raw_sum_rate"],
            "q": per_sample["q"],
            "raw_q": per_sample["raw_q"],
            "corr_penalty": per_sample["corr_penalty"],
            "pair_corr": per_sample["pair_corr"],
            "selected_hist_rate": per_sample["selected_hist_rate"],
        }
    )


def _selection_overlap(scores_a: np.ndarray, scores_b: np.ndarray, topk: int) -> float:
    a = numpy_topk_mask(scores_a, topk)
    b = numpy_topk_mask(scores_b, topk)
    return float((a * b).sum(axis=1).mean() / float(topk))


def evaluate_baselines(
    data_dir: Path | str,
    cfg: ExpConfig,
    split: str = "test",
    return_per_sample: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    ds = SchedulingDataset(data_dir, split=split)
    x = np.asarray(ds.features[ds.indices], dtype=np.float32)
    rate_now = np.asarray(ds.rate_now[ds.indices], dtype=np.float32)
    rate_future = np.asarray(ds.rate_future[ds.indices], dtype=np.float32)
    rate_pred = (
        np.asarray(ds.rate_pred[ds.indices], dtype=np.float32)
        if getattr(ds, "rate_pred", None) is not None
        else x[:, :, cfg.idx_future_rate].astype(np.float32)
    )
    hist_rate = np.asarray(ds.hist_rate[ds.indices], dtype=np.float32)
    if getattr(ds, "pmi_future", None) is not None:
        pmi_pair = np.asarray(ds.pmi_future[ds.indices], dtype=np.float32)
        pmi_future = np.concatenate([pmi_pair[..., 0], pmi_pair[..., 1]], axis=-1)
    else:
        pmi_future = x[:, :, cfg.idx_future_pmi_start : cfg.idx_future_pmi_start + cfg.pmi_real_dim]
    pmi_pred = x[:, :, cfg.idx_future_pmi_start : cfg.idx_future_pmi_start + cfg.pmi_real_dim]

    methods = {
        "No Prediction": no_prediction_scores(rate_now, hist_rate, cfg.eps, cfg.fairness_alpha),
        "PF Prediction": pf_prediction_scores(rate_now, rate_pred, hist_rate, cfg.eta, cfg.eps, cfg.fairness_alpha),
        "Enhanced Greedy": set_sinr_greedy_scores(
            rate_now,
            rate_pred,
            hist_rate,
            pmi_pred,
            cfg.eta,
            cfg.schedule_users,
            eps=cfg.eps,
            fairness_alpha=cfg.fairness_alpha,
            interference_beta=getattr(cfg, "paper_set_interference_beta", 0.40),
            power_reference_users=(getattr(cfg, "paper_power_reference_users", 0) or cfg.schedule_users),
        ),
        "Future-CSI Greedy (Oracle)": set_sinr_greedy_scores(
            rate_future,
            rate_future,
            hist_rate,
            pmi_future,
            1.0,
            cfg.schedule_users,
            eps=cfg.eps,
            fairness_alpha=cfg.fairness_alpha,
            interference_beta=cfg.paper_set_interference_beta,
            power_reference_users=cfg.paper_power_reference_users or cfg.schedule_users,
        ),
    }
    pf_greedy_overlap = _selection_overlap(methods["PF Prediction"], methods["Enhanced Greedy"], cfg.schedule_users)

    rows: list[dict] = []
    per_sample_rows: list[pd.DataFrame] = []
    for name, scores in methods.items():
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
            pmi_future_features=pmi_future,
            corr_penalty_weight=getattr(cfg, "eval_corr_penalty", 0.0),
            q_reduction=getattr(cfg, "utility_reduction", "mean"),
            interference_beta=getattr(cfg, "paper_set_interference_beta", 0.40),
            power_reference_users=(getattr(cfg, "paper_power_reference_users", 0) or cfg.schedule_users),
        )
        row = {"method": name, "type": "baseline", **metrics}
        if name == "Enhanced Greedy":
            row["selection_overlap_with_pf"] = pf_greedy_overlap
        row = _maybe_add_official_score(row, scores, data_dir, ds, cfg)
        rows.append(row)
        if return_per_sample:
            per_sample = _per_sample_metrics_from_scores(scores, x, rate_now, rate_future, hist_rate, cfg, pmi_future)
            per_sample_rows.append(_per_sample_frame(name, "baseline", ds.indices, per_sample))

    summary = pd.DataFrame(rows)
    if return_per_sample:
        return summary, pd.concat(per_sample_rows, ignore_index=True)
    return summary


@torch.no_grad()
def evaluate_checkpoint(
    model_name: str,
    ckpt_path: Path | str,
    data_dir: Path | str,
    cfg: ExpConfig,
    split: str = "test",
    device: Optional[str] = None,
    return_per_sample: bool = False,
) -> dict | tuple[dict, pd.DataFrame]:
    torch.set_num_threads(getattr(cfg, "torch_num_threads", 2))
    data_dir = Path(data_dir)
    device_t = _device(device)
    ds = SchedulingDataset(data_dir, split=split)
    model, model_cfg, _ = load_model_from_checkpoint(model_name, ckpt_path, cfg, device_t)
    validate_dataset_meta(model_cfg, ds.meta)
    model.eval()

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device_t.type == "cuda",
    )
    metric_keys = [
        "avg_sum_rate", "avg_effective_sum_rate", "avg_raw_sum_rate",
        "avg_q", "avg_raw_q", "avg_corr_penalty", "avg_pair_corr",
        "avg_selected_hist_rate",
    ]
    total = {key: 0.0 for key in metric_keys}
    n = 0
    all_scores: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device_t, non_blocking=True)
        logits = model(x)
        all_scores.append(logits.detach().cpu().numpy())
        metrics = torch_metrics_from_scores(
            logits,
            batch["rate_now"].to(device_t, non_blocking=True),
            batch["rate_future"].to(device_t, non_blocking=True),
            batch["hist_rate"].to(device_t, non_blocking=True),
            cfg.eta,
            cfg.schedule_users,
            cfg.eps,
            cfg.fairness_alpha,
            pmi_features=x[:, :, : cfg.pmi_real_dim],
            pmi_future_features=batch["pmi_future"].to(device_t, non_blocking=True),
            corr_penalty_weight=getattr(cfg, "eval_corr_penalty", 0.0),
            q_reduction=getattr(cfg, "utility_reduction", "mean"),
            interference_beta=getattr(cfg, "paper_set_interference_beta", 0.40),
            power_reference_users=(getattr(cfg, "paper_power_reference_users", 0) or cfg.schedule_users),
        )
        batch_size = x.size(0)
        for key in total:
            total[key] += metrics[key] * batch_size
        n += batch_size

    row = {key: value / max(n, 1) for key, value in total.items()}
    row.update({"method": _pretty_name(model_name), "type": "neural"})
    if all_scores:
        scores_np = np.concatenate(all_scores, axis=0)
        row = _maybe_add_official_score(row, scores_np, data_dir, ds, cfg)
        if return_per_sample:
            rate_now = np.asarray(ds.rate_now[ds.indices], dtype=np.float32)
            rate_future = np.asarray(ds.rate_future[ds.indices], dtype=np.float32)
            hist_rate = np.asarray(ds.hist_rate[ds.indices], dtype=np.float32)
            x_np = np.asarray(ds.features[ds.indices], dtype=np.float32)
            if getattr(ds, "pmi_future", None) is not None:
                pmi_pair = np.asarray(ds.pmi_future[ds.indices], dtype=np.float32)
                pmi_future = np.concatenate([pmi_pair[..., 0], pmi_pair[..., 1]], axis=-1)
            else:
                pmi_future = None
            per_sample = _per_sample_metrics_from_scores(scores_np, x_np, rate_now, rate_future, hist_rate, cfg, pmi_future)
            return row, _per_sample_frame(row["method"], row["type"], ds.indices, per_sample)
    if return_per_sample:
        return row, pd.DataFrame()
    return row


def evaluate_all(
    data_dir: Path | str,
    ckpt_dir: Path | str,
    result_dir: Path | str,
    cfg: ExpConfig,
    models: Iterable[str] = ("lstm", "ssa"),
    split: str = "test",
    device: Optional[str] = None,
) -> Path:
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    baseline_df, baseline_per_sample = evaluate_baselines(data_dir, cfg, split, return_per_sample=True)
    rows = [baseline_df]
    per_sample_rows = [baseline_per_sample]
    neural_rows: list[dict] = []
    for model_name in models:
        ckpt_path = Path(ckpt_dir) / f"{model_name.lower()}_best.pt"
        if ckpt_path.exists():
            neural_row, neural_per_sample = evaluate_checkpoint(
                model_name,
                ckpt_path,
                data_dir,
                cfg,
                split,
                device,
                return_per_sample=True,
            )
            neural_rows.append(neural_row)
            if not neural_per_sample.empty:
                per_sample_rows.append(neural_per_sample)
        else:
            print(f"[eval] Skip {model_name}: missing checkpoint {ckpt_path}")
    if neural_rows:
        rows.append(pd.DataFrame(neural_rows))
    df = pd.concat(rows, ignore_index=True)
    out_csv = result_dir / f"metrics_{split}.csv"
    df.to_csv(out_csv, index=False)
    per_sample_csv = result_dir / f"metrics_{split}_per_sample.csv"
    pd.concat(per_sample_rows, ignore_index=True).to_csv(per_sample_csv, index=False)
    print(f"[eval] Saved metrics to {out_csv}")
    print(f"[eval] Saved per-sample metrics to {per_sample_csv}")
    print(df)
    return out_csv

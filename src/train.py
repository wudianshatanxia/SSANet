from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ExpConfig
from src.dataset import SchedulingDataset
from src.metrics import torch_metrics_from_scores, torch_soft_set_utility_loss
from src.models import build_model
from src.reproducibility import seed_everything, seed_worker


def _device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_seed(base_seed: int, model_name: str) -> int:
    return int(base_seed) + sum((i + 1) * ord(ch) for i, ch in enumerate(model_name.lower())) % 100_000


def normalize_rank_targets(target_score: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize per-sample ranking targets to the unit interval."""
    lo = target_score.min(dim=1, keepdim=True).values
    hi = target_score.max(dim=1, keepdim=True).values
    return (target_score - lo) / torch.clamp(hi - lo, min=float(eps))


def listwise_rank_loss(
    logits: torch.Tensor,
    target_score: torch.Tensor,
    tau: float = 0.25,
    normalize_targets: bool = True,
) -> torch.Tensor:
    """Compute the ListNet ranking loss."""
    tau = max(float(tau), 1e-3)
    target = normalize_rank_targets(target_score) if normalize_targets else target_score
    target_prob = torch.softmax(target / tau, dim=1)
    pred_log_prob = torch.log_softmax(logits / tau, dim=1)
    return -(target_prob * pred_log_prob).sum(dim=1).mean()


def pairwise_topk_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float = 0.25) -> torch.Tensor:
    """Compute the ranking margin loss for selected users."""
    pos = logits.masked_fill(labels <= 0.5, float("inf")).min(dim=1).values
    neg = logits.masked_fill(labels > 0.5, float("-inf")).max(dim=1).values
    valid = torch.isfinite(pos) & torch.isfinite(neg)
    if not torch.any(valid):
        return logits.new_zeros(())
    return torch.relu(float(margin) - pos[valid] + neg[valid]).mean()


def _forward_train(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    if hasattr(model, "forward_with_aux"):
        out = model.forward_with_aux(x)
        return out["logits"], out.get("future_rate_hat")
    return model(x), None

def train_one_model(
    model_name: str,
    data_dir: Path | str,
    ckpt_dir: Path | str,
    cfg: ExpConfig,
    device: Optional[str] = None,
) -> Path:
    if cfg.epochs < 1:
        raise ValueError("cfg.epochs must be >= 1")
    torch.set_num_threads(getattr(cfg, "torch_num_threads", 2))
    run_seed = _model_seed(cfg.seed, model_name)
    seed_everything(run_seed, getattr(cfg, "deterministic", True))
    device_t = _device(device)
    data_dir = Path(data_dir)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_set = SchedulingDataset(data_dir, split="train")
    val_set = SchedulingDataset(data_dir, split="val")
    if len(train_set) == 0 or len(val_set) == 0:
        raise ValueError("Training requires non-empty train and val splits.")

    loader_generator = torch.Generator()
    loader_generator.manual_seed(run_seed)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
        generator=loader_generator,
        pin_memory=device_t.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
        pin_memory=device_t.type == "cuda",
    )

    model = build_model(model_name, cfg).to(device_t)
    pos_weight = torch.tensor([(cfg.num_ues - cfg.schedule_users) / max(cfg.schedule_users, 1)], device=device_t)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_objective = -float("inf")
    best_path = ckpt_dir / f"{model_name.lower()}_best.pt"
    history: list[Dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch in tqdm(train_loader, desc=f"Train {model_name} epoch {epoch}/{cfg.epochs}", leave=False):
            x = batch["x"].to(device_t, non_blocking=True)
            y = batch["label"].to(device_t, non_blocking=True)
            target_score = batch.get("target_score", y).to(device_t, non_blocking=True)
            logits, future_rate_hat = _forward_train(model, x)
            bce_loss = criterion(logits, y)
            rank_loss = listwise_rank_loss(
                logits,
                target_score,
                getattr(cfg, "rank_loss_tau", 0.25),
                getattr(cfg, "normalize_rank_targets", True),
            )
            pair_loss = pairwise_topk_loss(logits, y, getattr(cfg, "pairwise_margin", 0.25))
            future_loss = logits.new_zeros(())
            if future_rate_hat is not None:
                future_loss = torch.nn.functional.smooth_l1_loss(
                    future_rate_hat,
                    batch["rate_future"].to(device_t, non_blocking=True),
                    beta=0.25,
                )
            set_loss = logits.new_zeros(())
            if model_name.lower().startswith("ssa"):
                sizes = tuple(int(v) for v in getattr(cfg, "train_schedule_sizes", (cfg.schedule_users,)))
                valid_sizes = [v for v in sizes if 1 <= v <= cfg.num_ues]
                train_k = valid_sizes[(epoch + total_items // max(x.size(0), 1)) % len(valid_sizes)] if valid_sizes else cfg.schedule_users
                set_loss = torch_soft_set_utility_loss(
                    logits,
                    batch["rate_future"].to(device_t, non_blocking=True),
                    batch["hist_rate"].to(device_t, non_blocking=True),
                    batch["pmi_future"].to(device_t, non_blocking=True),
                    train_k,
                    cfg.eps,
                    getattr(cfg, "paper_set_interference_beta", 0.40),
                    getattr(cfg, "paper_power_reference_users", 0) or cfg.schedule_users,
                    cfg.fairness_alpha,
                    getattr(cfg, "set_utility_temperature", 0.75),
                )
            loss = (
                getattr(cfg, "bce_loss_weight", 0.35) * bce_loss
                + getattr(cfg, "rank_loss_weight", 0.85) * rank_loss
                + getattr(cfg, "pairwise_loss_weight", 0.20) * pair_loss
                + getattr(cfg, "future_rate_loss_weight", 0.45) * future_loss
                + getattr(cfg, "set_utility_loss_weight", 0.18) * set_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * x.size(0)
            total_items += x.size(0)

        val_metrics = evaluate_model_in_memory(model, val_loader, cfg, device_t)
        train_loss = total_loss / max(total_items, 1)
        record: Dict[str, float] = {"epoch": float(epoch), "train_loss": train_loss, **val_metrics}
        history.append(record)
        print(
            f"[{model_name}] epoch={epoch:03d} loss={train_loss:.5f} "
            f"val_q={val_metrics['avg_q']:.4f} val_sum_rate={val_metrics['avg_sum_rate']:.4f}"
        )

        val_objective = (
            getattr(cfg, "checkpoint_sum_rate_weight", 0.80)
            * val_metrics["avg_sum_rate"] / float(max(cfg.schedule_users, 1))
            + getattr(cfg, "checkpoint_q_weight", 0.20) * val_metrics["avg_q"]
        )
        if val_objective > best_val_objective:
            best_val_objective = val_objective
            torch.save(
                {
                    "checkpoint_version": 2,
                    "model_name": model_name.lower(),
                    "state_dict": model.state_dict(),
                    "cfg": {k: str(v) if hasattr(v, "as_posix") else v for k, v in cfg.__dict__.items()},
                    "seed": run_seed,
                    "best_val_q": val_metrics["avg_q"],
                    "best_val_sum_rate": val_metrics["avg_sum_rate"],
                    "best_val_objective": best_val_objective,
                },
                best_path,
            )

    with open(ckpt_dir / f"{model_name.lower()}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[{model_name}] best checkpoint saved to {best_path}")
    return best_path


@torch.no_grad()
def evaluate_model_in_memory(model: nn.Module, loader: DataLoader, cfg: ExpConfig, device_t: torch.device) -> Dict[str, float]:
    model.eval()
    metric_keys = [
        "avg_sum_rate", "avg_effective_sum_rate", "avg_raw_sum_rate",
        "avg_q", "avg_raw_q", "avg_corr_penalty", "avg_pair_corr",
        "avg_selected_hist_rate",
    ]
    total = {key: 0.0 for key in metric_keys}
    n = 0
    for batch in loader:
        x = batch["x"].to(device_t, non_blocking=True)
        logits = model(x)
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
    return {key: value / max(n, 1) for key, value in total.items()}

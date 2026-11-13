from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ExpConfig, ensure_dirs
from src.plot_results import (
    evaluate_speed_impact_from_official_data,
    evaluate_speed_impact_synthetic,
    plot_fairness_tradeoff,
    plot_robustness_from_results,
    plot_scheduled_ue_performance,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--fig_dir", type=str, default=None)
    parser.add_argument("--speed_samples", type=int, default=800)
    parser.add_argument("--models", type=str, default="lstm,ssa")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = ExpConfig()
    ensure_dirs(cfg)
    data_dir = Path(args.data_dir) if args.data_dir else cfg.data_dir
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else cfg.ckpt_dir
    fig_dir = Path(args.fig_dir) if args.fig_dir else cfg.fig_dir
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    if not ckpt_dir.is_absolute():
        ckpt_dir = ROOT / ckpt_dir
    if not fig_dir.is_absolute():
        fig_dir = ROOT / fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    models = [name.strip() for name in args.models.split(",") if name.strip()]

    meta_path = data_dir / "meta.json"
    is_official = False
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            is_official = str(json.load(f).get("source", "")).startswith("official")

    if is_official:
        speed_df = evaluate_speed_impact_from_official_data(
            data_dir,
            ckpt_dir,
            fig_dir,
            cfg,
            models=models,
            device=args.device,
        )
    else:
        speed_df = evaluate_speed_impact_synthetic(
            data_dir,
            ckpt_dir,
            fig_dir,
            cfg,
            num_samples_per_speed=args.speed_samples,
            models=models,
            device=args.device,
        )

    plot_scheduled_ue_performance(
        data_dir,
        ckpt_dir,
        fig_dir,
        cfg,
        schedule_sizes=(4, 8, 12, 16, 20),
        models=models,
        device=args.device,
        include_oracle=True,
    )

    plot_fairness_tradeoff(
        data_dir, ckpt_dir, fig_dir, cfg, models=models, device=args.device
    )

    plot_robustness_from_results(speed_df, fig_dir)


if __name__ == "__main__":
    main()

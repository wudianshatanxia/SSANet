"""Run data preparation, training, evaluation, and plotting."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import ExpConfig, ensure_dirs
from src.stress_experiments import run_stress_experiments
from src.data_generator import generate_synthetic_dataset
from src.evaluate import evaluate_all
from src.emergency_experiments import run_all_emergency_workshop_experiments
from src.official_data import convert_official_mat
from src.official_like_generator import generate_official_like_mat_files
from src.plot_results import (
    evaluate_speed_impact_from_official_data,
    evaluate_speed_impact_synthetic,
    plot_fairness_tradeoff,
    plot_robustness_from_results,
    plot_scheduled_ue_performance,
)
from src.train import train_one_model

def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_mode", choices=["official", "synthetic"], default="official")
    parser.add_argument("--train_mat", type=str, default="data/official_raw/Training_Data.mat")
    parser.add_argument("--test_mat", type=str, default="data/official_raw/Test_Data.mat")
    parser.add_argument("--official_out", type=str, default="data/official_converted")
    parser.add_argument("--generate_official_like", action="store_true")
    parser.add_argument("--generated_train_N", type=int, default=16000)
    parser.add_argument("--generated_test_N", type=int, default=2000)
    parser.add_argument("--num_samples", type=int, default=16000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--models", type=str, default="lstm,ssa")
    parser.add_argument("--speed_samples", type=int, default=300)
    parser.add_argument("--schedule_users", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_subdir", type=str, default=None, help="Optional outputs/<name>/ isolation for repeated runs.")
    parser.add_argument("--use_future_input", action="store_true", help="Explicit oracle diagnostic only.")
    parser.add_argument("--no_future_input", action="store_true", help="Deprecated compatibility flag.")
    parser.add_argument("--run_stress", action="store_true")
    parser.add_argument(
        "--run_emergency_workshop",
        action="store_true",
        help="After the main figures, run mission-priority, TRP-outage, and latency experiments for the emergency-communications workshop.",
    )
    parser.add_argument("--emergency_quick", action="store_true", help="Use short emergency experiments for pipeline checking.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = ExpConfig()
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.seed = int(args.seed)
    if args.schedule_users is not None:
        cfg.schedule_users = int(args.schedule_users)
    if args.use_future_input and args.no_future_input:
        parser.error("--use_future_input and --no_future_input cannot be used together")
    cfg.official_use_future_as_input = bool(args.use_future_input)
    if args.output_subdir:
        root = cfg.project_root / "outputs" / args.output_subdir
        cfg.ckpt_dir = root / "checkpoints"
        cfg.result_dir = root / "results"
        cfg.fig_dir = root / "figures"
    ensure_dirs(cfg)
    models = [name.strip() for name in args.models.split(",") if name.strip()]

    if args.data_mode == "official":
        train_mat = _resolve(cfg.project_root, args.train_mat)
        test_mat = _resolve(cfg.project_root, args.test_mat)
        if args.generate_official_like:
            gen_dir = cfg.project_root / "data" / "generated_official_like"
            if args.overwrite or not (gen_dir / "Training_Data.mat").exists() or not (gen_dir / "Test_Data.mat").exists():
                generate_official_like_mat_files(
                    train_mat,
                    test_mat,
                    gen_dir,
                    train_N=args.generated_train_N,
                    test_N=args.generated_test_N,
                    seed=cfg.seed,
                )
            train_mat = gen_dir / "Training_Data.mat"
            test_mat = gen_dir / "Test_Data.mat"

        base = _resolve(cfg.project_root, args.official_out)
        train_dir = base / "train"
        test_dir = base / "test"
        convert_official_mat(
            train_mat,
            train_dir,
            cfg,
            use_future_as_input=cfg.official_use_future_as_input,
            overwrite=args.overwrite,
            split_policy="train",
        )
        convert_official_mat(
            test_mat,
            test_dir,
            cfg,
            use_future_as_input=cfg.official_use_future_as_input,
            overwrite=args.overwrite,
            split_policy="test_all",
        )
        train_data_dir = train_dir
        eval_data_dir = test_dir
    else:
        demo_data_dir = cfg.project_root / "data" / f"demo_{args.num_samples}"
        generate_synthetic_dataset(demo_data_dir, cfg, num_samples=args.num_samples, overwrite=args.overwrite)
        train_data_dir = demo_data_dir
        eval_data_dir = demo_data_dir

    for model_name in models:
        train_one_model(model_name, train_data_dir, cfg.ckpt_dir, cfg, device=args.device)

    metrics_csv = evaluate_all(
        eval_data_dir,
        cfg.ckpt_dir,
        cfg.result_dir,
        cfg,
        models=models,
        split="test",
        device=args.device,
    )
    if args.data_mode == "official":
        speed_df = evaluate_speed_impact_from_official_data(
            eval_data_dir,
            cfg.ckpt_dir,
            cfg.fig_dir,
            cfg,
            models=models,
            device=args.device,
        )
    else:
        speed_df = evaluate_speed_impact_synthetic(
            eval_data_dir,
            cfg.ckpt_dir,
            cfg.fig_dir,
            cfg,
            num_samples_per_speed=args.speed_samples,
            models=models,
            device=args.device,
        )
    plot_scheduled_ue_performance(
        eval_data_dir,
        cfg.ckpt_dir,
        cfg.fig_dir,
        cfg,
        schedule_sizes=(4, 8, 12, 16, 20),
        models=models,
        device=args.device,
        include_oracle=True,
    )

    plot_fairness_tradeoff(
        eval_data_dir,
        cfg.ckpt_dir,
        cfg.fig_dir,
        cfg,
        models=models,
        device=args.device,
    )

    plot_robustness_from_results(speed_df, cfg.fig_dir)

    if args.run_stress:
        run_stress_experiments(eval_data_dir, cfg.ckpt_dir, cfg.result_dir, cfg, models=models, split="test", device=args.device)

    if args.run_emergency_workshop:
        emergency_dir = (cfg.fig_dir.parent / "emergency_workshop")
        emergency_results = run_all_emergency_workshop_experiments(
            eval_data_dir,
            cfg.ckpt_dir,
            emergency_dir,
            cfg,
            models=models,
            device=args.device,
            quick=args.emergency_quick,
        )
        print("Emergency workshop outputs:")
        for key, path in emergency_results.items():
            print(f"  {key}: {path}")

    print(f"\nMetrics: {metrics_csv}")
    print(f"Figures: {cfg.fig_dir}")


if __name__ == "__main__":
    main()

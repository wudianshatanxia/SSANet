from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _parse_seeds(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if len(seeds) < 2:
        raise ValueError("Provide at least two seeds, e.g. --seeds 42,43,44")
    return seeds


def _aggregate(files: list[tuple[int, Path]], output: Path, group_cols: list[str]) -> None:
    frames = []
    for seed, path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    numeric = [
        col for col in raw.select_dtypes(include=[np.number]).columns
        if col != "seed" and col not in group_cols
    ]
    grouped = raw.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std", "count"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    for metric in numeric:
        std_col = f"{metric}_std"
        count_col = f"{metric}_count"
        grouped[f"{metric}_ci95"] = 1.96 * grouped[std_col] / np.sqrt(grouped[count_col].clip(lower=1))
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output, index=False)
    raw.to_csv(output.with_name(output.stem + "_raw.csv"), index=False)
    print(f"[multiseed] Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent seeds and aggregate mean/std/95% CI.")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--data_mode", choices=["official", "synthetic"], default="official")
    parser.add_argument("--train_mat", type=str, default="data/official_raw/Training_Data.mat")
    parser.add_argument("--test_mat", type=str, default="data/official_raw/Test_Data.mat")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--models", type=str, default="lstm,ssa")
    parser.add_argument("--num_samples", type=int, default=16000)
    parser.add_argument("--speed_samples", type=int, default=800)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    metric_files: list[tuple[int, Path]] = []
    speed_files: list[tuple[int, Path]] = []
    for seed in seeds:
        tag = f"seed_{seed}"
        cmd = [
            sys.executable,
            str(ROOT / "run_demo.py"),
            "--data_mode",
            args.data_mode,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--models",
            args.models,
            "--seed",
            str(seed),
            "--output_subdir",
            tag,
            "--overwrite",
        ]
        if args.data_mode == "official":
            cmd += [
                "--train_mat",
                args.train_mat,
                "--test_mat",
                args.test_mat,
                "--official_out",
                f"data/official_converted_{tag}",
            ]
        else:
            cmd += ["--num_samples", str(args.num_samples), "--speed_samples", str(args.speed_samples)]
        if args.device:
            cmd += ["--device", args.device]
        print("[multiseed]", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)
        base = ROOT / "outputs" / tag
        metric_files.append((seed, base / "results" / "metrics_test.csv"))
        speed_files.append((seed, base / "figures" / "fig5_mobility_values.csv"))

    _aggregate(metric_files, ROOT / "outputs" / "multiseed_metrics_summary.csv", ["method", "type"])
    _aggregate(speed_files, ROOT / "outputs" / "multiseed_speed_summary.csv", ["speed_kmh", "method"])


if __name__ == "__main__":
    main()

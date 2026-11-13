from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ExpConfig, ensure_dirs
from src.stress_experiments import run_stress_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prediction-error stress and inference-timing experiments.")
    parser.add_argument("--data_dir", type=str, default="data/official_converted/test")
    parser.add_argument("--models", type=str, default="lstm,ssa")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = ExpConfig()
    ensure_dirs(cfg)
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    run_stress_experiments(data_dir, cfg.ckpt_dir, cfg.result_dir, cfg, models=models, split=args.split, device=args.device)


if __name__ == "__main__":
    main()

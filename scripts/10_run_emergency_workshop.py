from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ExpConfig, ensure_dirs
from src.emergency_experiments import run_all_emergency_workshop_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GLOBECOM emergency-communications experiments.")
    parser.add_argument("--data_dir", type=str, default="data/official_converted/test")
    parser.add_argument("--ckpt_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--out_dir", type=str, default="outputs/emergency_workshop")
    parser.add_argument("--models", type=str, default="lstm,ssa")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help="Short smoke-test experiment.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = ExpConfig()
    cfg.seed = int(args.seed)
    ensure_dirs(cfg)
    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    if not ckpt_dir.is_absolute():
        ckpt_dir = ROOT / ckpt_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    results = run_all_emergency_workshop_experiments(
        data_dir,
        ckpt_dir,
        out_dir,
        cfg,
        models=models,
        device=args.device,
        quick=args.quick,
    )
    for key, path in results.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()

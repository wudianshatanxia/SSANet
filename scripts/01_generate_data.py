import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ExpConfig, ensure_dirs
from src.data_generator import generate_synthetic_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=None, help="Default uses config.num_samples=16000.")
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--fixed_speed", type=float, default=None)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

cfg = ExpConfig()
ensure_dirs(cfg)
out_dir = Path(args.out_dir) if args.out_dir else cfg.data_dir
generate_synthetic_dataset(out_dir, cfg, num_samples=args.num_samples, fixed_speed_kmh=args.fixed_speed, overwrite=args.overwrite)

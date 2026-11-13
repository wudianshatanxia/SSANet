import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ExpConfig, ensure_dirs
from src.evaluate import evaluate_all

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--models", type=str, default="lstm,ssa")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--device", type=str, default=None)
args = parser.parse_args()

cfg = ExpConfig()
ensure_dirs(cfg)
models = [m.strip() for m in args.models.split(",") if m.strip()]
data_dir = Path(args.data_dir) if args.data_dir else cfg.data_dir
evaluate_all(data_dir, cfg.ckpt_dir, cfg.result_dir, cfg, models=models, split=args.split, device=args.device)

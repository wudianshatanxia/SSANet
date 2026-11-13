import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ExpConfig, ensure_dirs
from src.official_data import convert_official_mat
from src.train import train_one_model

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--train_mat", type=str, default=None, help="Official-format .mat file to convert and use for training.")
parser.add_argument("--official_out", type=str, default="data/official_converted/train")
parser.add_argument("--models", type=str, default="lstm,ssa")
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--schedule_users", type=int, default=None)
parser.add_argument("--use_future_input", action="store_true", help="Oracle mode: use PMI_5ms/eigenvalue_5ms as prediction-side inputs.")
parser.add_argument("--no_store_pmi", action="store_true")
parser.add_argument("--chunk_size", type=int, default=None)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--device", type=str, default=None)
args = parser.parse_args()

if args.data_dir and args.train_mat:
    parser.error("--data_dir and --train_mat cannot be used together")

cfg = ExpConfig()
if args.epochs is not None:
    cfg.epochs = args.epochs
if args.batch_size is not None:
    cfg.batch_size = args.batch_size
if args.schedule_users is not None:
    cfg.schedule_users = args.schedule_users
cfg.official_use_future_as_input = bool(args.use_future_input)
ensure_dirs(cfg)

if args.train_mat:
    train_mat = Path(args.train_mat)
    if not train_mat.is_absolute():
        train_mat = ROOT / train_mat
    data_dir = Path(args.official_out)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    convert_official_mat(
        train_mat,
        data_dir,
        cfg,
        use_future_as_input=cfg.official_use_future_as_input,
        store_pmi_for_official_score=not args.no_store_pmi,
        schedule_users=args.schedule_users,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
        split_policy="train",
    )
else:
    data_dir = Path(args.data_dir) if args.data_dir else cfg.data_dir

for name in [m.strip() for m in args.models.split(",") if m.strip()]:
    train_one_model(name, data_dir, cfg.ckpt_dir, cfg, device=args.device)

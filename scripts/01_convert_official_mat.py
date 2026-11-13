import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ExpConfig, ensure_dirs
from src.official_data import convert_official_mat

parser = argparse.ArgumentParser(description="Convert official example .mat files into the unified project data layout.")
parser.add_argument("--train_mat", type=str, default="data/official_raw/Training_Data.mat")
parser.add_argument("--test_mat", type=str, default="data/official_raw/Test_Data.mat")
parser.add_argument("--out_dir", type=str, default="data/official_converted")
parser.add_argument("--skip_train", action="store_true", help="Do not convert the training .mat file.")
parser.add_argument("--skip_test", action="store_true", help="Do not convert the test .mat file.")
parser.add_argument("--schedule_users", type=int, default=None)
parser.add_argument("--use_future_input", action="store_true", help="Oracle mode: use PMI_5ms/eigenvalue_5ms as prediction-side inputs.")
parser.add_argument("--no_future_input", action="store_true", help="Deprecated compatibility flag; future inputs are disabled by default.")
parser.add_argument("--no_store_pmi", action="store_true")
parser.add_argument("--chunk_size", type=int, default=None)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

cfg = ExpConfig()
if args.schedule_users is not None:
    cfg.schedule_users = args.schedule_users
if args.use_future_input and args.no_future_input:
    parser.error("--use_future_input and --no_future_input cannot be used together")
cfg.official_use_future_as_input = bool(args.use_future_input)
ensure_dirs(cfg)
base = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
train_mat = ROOT / args.train_mat if not Path(args.train_mat).is_absolute() else Path(args.train_mat)
test_mat = ROOT / args.test_mat if not Path(args.test_mat).is_absolute() else Path(args.test_mat)

if args.skip_train and args.skip_test:
    parser.error("At least one of train/test conversion must be enabled.")

if not args.skip_train:
    convert_official_mat(
        train_mat,
        base / "train",
        cfg,
        use_future_as_input=cfg.official_use_future_as_input,
        store_pmi_for_official_score=not args.no_store_pmi,
        schedule_users=args.schedule_users,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
        split_policy="train",
    )
if not args.skip_test:
    convert_official_mat(
        test_mat,
        base / "test",
        cfg,
        use_future_as_input=cfg.official_use_future_as_input,
        store_pmi_for_official_score=not args.no_store_pmi,
        schedule_users=args.schedule_users,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
        split_policy="test_all",
    )
print(f"Converted official data to {base}")

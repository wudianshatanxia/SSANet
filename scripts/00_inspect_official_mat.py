import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.official_data import inspect_official_mat

parser = argparse.ArgumentParser(description="Inspect official Training_Data.mat/Test_Data.mat keys and shapes.")
parser.add_argument("--mat", type=str, default="data/official_raw/Training_Data.mat")
args = parser.parse_args()

info = inspect_official_mat(ROOT / args.mat if not Path(args.mat).is_absolute() else args.mat)
print(json.dumps(info, indent=2, ensure_ascii=False))

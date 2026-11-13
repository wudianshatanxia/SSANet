import argparse
from pathlib import Path
import sys

import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ExpConfig
from src.dataset import SchedulingDataset
from src.official_score import official_score_from_indices, selected_indices_from_scores

parser = argparse.ArgumentParser(description="Score a schedule or score matrix using the official-example scorer.")
parser.add_argument("--data_dir", type=str, default="data/official_converted/test")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--scores_npy", type=str, default=None, help="Model scores matrix; if omitted, uses PF Prediction scores.")
parser.add_argument("--indices_npy", type=str, default=None, help="Pre-selected indices matrix. Takes priority over scores_npy.")
parser.add_argument("--topk", type=int, default=None)
parser.add_argument("--oracle_future_scores", action="store_true", help="Use true future rate for the default PF score; otherwise use rate_pred/current prediction.")
parser.add_argument("--out_csv", type=str, default="outputs/results/official_score_summary.csv")
args = parser.parse_args()

cfg = ExpConfig()
if args.topk is not None:
    cfg.schedule_users = args.topk

data_dir = ROOT / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
ds = SchedulingDataset(data_dir, split=args.split)

if args.indices_npy:
    indices_path = ROOT / args.indices_npy if not Path(args.indices_npy).is_absolute() else Path(args.indices_npy)
    indices = np.load(indices_path)
else:
    if args.scores_npy:
        scores_path = ROOT / args.scores_npy if not Path(args.scores_npy).is_absolute() else Path(args.scores_npy)
        scores = np.load(scores_path)
    else:
        rate_now = np.asarray(ds.rate_now[ds.indices], dtype=np.float32)
        if args.oracle_future_scores:
            rate_pred = np.asarray(ds.rate_future[ds.indices], dtype=np.float32)
        elif getattr(ds, "rate_pred", None) is not None:
            rate_pred = np.asarray(ds.rate_pred[ds.indices], dtype=np.float32)
        else:
            rate_pred = np.asarray(ds.features[ds.indices, :, cfg.idx_future_rate], dtype=np.float32)
        hist = np.asarray(ds.hist_rate[ds.indices], dtype=np.float32)
        scores = (cfg.eta * rate_now + (1.0 - cfg.eta) * rate_pred) / np.power(hist + cfg.eps, cfg.fairness_alpha)
    indices = selected_indices_from_scores(scores, cfg.schedule_users)

pmi0 = np.load(data_dir / "pmi_0ms.npy", mmap_mode="r")[ds.indices]
pmi5 = np.load(data_dir / "pmi_5ms.npy", mmap_mode="r")[ds.indices]
r5 = np.load(data_dir / "rate_future.npy", mmap_mode="r")[ds.indices]
hist = np.load(data_dir / "hist_rate.npy", mmap_mode="r")[ds.indices]
res = official_score_from_indices(indices, np.asarray(pmi0), np.asarray(pmi5), np.asarray(r5), np.asarray(hist), noise=cfg.official_score_noise)
summary = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
print(json.dumps(summary, indent=2))
out_csv = ROOT / args.out_csv if not Path(args.out_csv).is_absolute() else Path(args.out_csv)
out_csv.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame([summary]).to_csv(out_csv, index=False)
print(f"Saved summary to {out_csv}")

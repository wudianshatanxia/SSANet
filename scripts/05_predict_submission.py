from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.io import savemat
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ExpConfig, ensure_dirs
from src.checkpointing import load_model_from_checkpoint, validate_dataset_meta
from src.dataset import SchedulingDataset
from src.metrics import numpy_topk_mask
from src.official_score import official_score_from_indices, selected_indices_from_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained model and export scheduled UE indices/masks.")
    parser.add_argument("--data_dir", type=str, default="data/official_converted/test")
    parser.add_argument("--model", type=str, default="ssa")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--topk", type=int, default=None, help="Default cfg.schedule_users; use 60 for official scorer.")
    parser.add_argument("--out_prefix", type=str, default="outputs/submission/submission")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--score_official", action="store_true")
    args = parser.parse_args()

    cfg = ExpConfig()
    ensure_dirs(cfg)
    if args.topk is not None:
        cfg.schedule_users = int(args.topk)

    data_dir = ROOT / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    out_prefix = ROOT / args.out_prefix if not Path(args.out_prefix).is_absolute() else Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.ckpt) if args.ckpt else cfg.ckpt_dir / f"{args.model.lower()}_best.pt"
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = SchedulingDataset(data_dir, split=args.split)
    model, model_cfg, _ = load_model_from_checkpoint(args.model, ckpt_path, cfg, device)
    validate_dataset_meta(model_cfg, ds.meta)
    model.eval()

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    all_scores = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            all_scores.append(model(x).cpu().numpy())
    if not all_scores:
        raise ValueError(f"Split '{args.split}' is empty")

    scores = np.concatenate(all_scores, axis=0)
    indices = selected_indices_from_scores(scores, cfg.schedule_users)
    mask = numpy_topk_mask(scores, cfg.schedule_users)

    np.save(str(out_prefix) + "_scores.npy", scores)
    np.save(str(out_prefix) + "_indices.npy", indices)
    np.save(str(out_prefix) + "_mask.npy", mask)
    savemat(str(out_prefix) + "_mask.mat", {"schedule_mask": mask, "schedule_indices": indices + 1})
    pd.DataFrame(indices + 1).to_csv(str(out_prefix) + "_indices.csv", index=False, header=False)
    pd.DataFrame(mask.astype(int)).to_csv(str(out_prefix) + "_mask.csv", index=False, header=False)
    print(f"Saved submission files with prefix: {out_prefix}")

    if args.score_official:
        required = [data_dir / "pmi_0ms.npy", data_dir / "pmi_5ms.npy", data_dir / "rate_future.npy", data_dir / "hist_rate.npy"]
        if all(path.exists() for path in required):
            pmi0 = np.load(data_dir / "pmi_0ms.npy", mmap_mode="r")[ds.indices]
            pmi5 = np.load(data_dir / "pmi_5ms.npy", mmap_mode="r")[ds.indices]
            r5 = np.load(data_dir / "rate_future.npy", mmap_mode="r")[ds.indices]
            hist = np.load(data_dir / "hist_rate.npy", mmap_mode="r")[ds.indices]
            official = official_score_from_indices(
                indices,
                np.asarray(pmi0),
                np.asarray(pmi5),
                np.asarray(r5),
                np.asarray(hist),
                noise=cfg.official_score_noise,
            )
            print("Official-example score:", {k: v for k, v in official.items() if not isinstance(v, np.ndarray)})
        else:
            print("Official score arrays not found. Re-run conversion with PMI storage enabled.")


if __name__ == "__main__":
    main()

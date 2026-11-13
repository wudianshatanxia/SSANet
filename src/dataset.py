from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class SchedulingDataset(Dataset):
    """Load memory-mapped scheduling samples."""

    def __init__(self, data_dir: Path | str, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split
        with open(self.data_dir / "meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(self.data_dir / "splits.json", "r", encoding="utf-8") as f:
            splits: Dict[str, List[int]] = json.load(f)
        if split not in splits:
            raise ValueError(f"Unknown split={split}. Available: {list(splits)}")
        self.indices = np.array(splits[split], dtype=np.int64)
        self.features = np.load(self.data_dir / "features.npy", mmap_mode="r")
        self.labels = np.load(self.data_dir / "labels.npy", mmap_mode="r")
        target_path = self.data_dir / "target_scores.npy"
        self.target_scores = np.load(target_path, mmap_mode="r") if target_path.exists() else None
        self.rate_now = np.load(self.data_dir / "rate_now.npy", mmap_mode="r")
        self.rate_future = np.load(self.data_dir / "rate_future.npy", mmap_mode="r")
        rate_pred_path = self.data_dir / "rate_pred.npy"
        self.rate_pred = np.load(rate_pred_path, mmap_mode="r") if rate_pred_path.exists() else None
        self.hist_rate = np.load(self.data_dir / "hist_rate.npy", mmap_mode="r")
        self.speed = np.load(self.data_dir / "speed_kmh.npy", mmap_mode="r")
        pmi5_path = self.data_dir / "pmi_5ms.npy"
        self.pmi_future = np.load(pmi5_path, mmap_mode="r") if pmi5_path.exists() else None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        return {
            "x": torch.from_numpy(np.array(self.features[real_idx], dtype=np.float32)),
            "label": torch.from_numpy(np.array(self.labels[real_idx], dtype=np.float32)),
            "target_score": torch.from_numpy(
                np.array(self.target_scores[real_idx], dtype=np.float32)
                if self.target_scores is not None
                else np.array(self.labels[real_idx], dtype=np.float32)
            ),
            "rate_now": torch.from_numpy(np.array(self.rate_now[real_idx], dtype=np.float32)),
            "rate_future": torch.from_numpy(np.array(self.rate_future[real_idx], dtype=np.float32)),
            "rate_pred": torch.from_numpy(np.array(self.rate_pred[real_idx], dtype=np.float32)) if self.rate_pred is not None else torch.from_numpy(np.array(self.features[real_idx, :, self.meta.get("pmi_real_dim", 384) + 1 + self.meta.get("pmi_real_dim", 384)], dtype=np.float32)),
            "hist_rate": torch.from_numpy(np.array(self.hist_rate[real_idx], dtype=np.float32)),
            "speed_kmh": torch.tensor(float(self.speed[real_idx]), dtype=torch.float32),
            "pmi_future": torch.from_numpy(
                np.concatenate(
                    [
                        np.array(self.pmi_future[real_idx, ..., 0], dtype=np.float32),
                        np.array(self.pmi_future[real_idx, ..., 1], dtype=np.float32),
                    ],
                    axis=-1,
                )
                if self.pmi_future is not None
                else np.array(
                    self.features[
                        real_idx,
                        :,
                        self.meta.get("pmi_real_dim", 384) + 1 : self.meta.get("pmi_real_dim", 384) + 1 + self.meta.get("pmi_real_dim", 384),
                    ],
                    dtype=np.float32,
                )
            ),
        }

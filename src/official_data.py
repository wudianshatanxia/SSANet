from __future__ import annotations

"""Inspect and convert official-format MAT datasets."""

import json
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
from tqdm import tqdm

from config import ExpConfig
from src.baselines import set_sinr_greedy_scores
from src.metrics import compute_q
from src.mobility import augment_mobility_chunk

REQUIRED_KEYS = [
    "N",
    "PMI_0ms",
    "PMI_5ms",
    "eigenvalue_0ms",
    "eigenvalue_5ms",
    "rate_old",
]


def inspect_official_mat(mat_path: Path | str) -> Dict[str, object]:
    mat_path = Path(mat_path)
    info: Dict[str, object] = {"path": str(mat_path), "datasets": {}}
    with h5py.File(mat_path, "r") as f:
        for k in f.keys():
            obj = f[k]
            if isinstance(obj, h5py.Dataset):
                info["datasets"][k] = {"shape": tuple(obj.shape), "dtype": str(obj.dtype)}
        missing = [k for k in REQUIRED_KEYS if k not in f]
        info["missing_required_keys"] = missing
        if "N" in f:
            info["N_value"] = int(np.array(f["N"]).reshape(-1)[0])
    return info


def _read_n(f: h5py.File) -> int:
    return int(np.array(f["N"]).reshape(-1)[0])


def _read_matrix_120_by_n(f: h5py.File, key: str, start: int, end: int) -> np.ndarray:
    return np.array(f[key][:, start:end], dtype=np.float32).T


def _read_pmi(f: h5py.File, key: str, start: int, end: int) -> np.ndarray:
    arr = np.array(f[key][:, :, :, start:end], dtype=np.float32)
    return np.transpose(arr, (3, 2, 1, 0))


def _pmi_to_flat_realimag(pmi: np.ndarray) -> np.ndarray:
    return np.concatenate([pmi[..., 0], pmi[..., 1]], axis=-1).astype(np.float32)


def _eigen_to_rate(eigenvalue: np.ndarray, cfg: ExpConfig) -> np.ndarray:
    """Convert channel eigenvalues to scheduling rates."""
    scale = float(getattr(cfg, "official_eigen_to_rate_scale", 1000.0))
    return np.log2(1.0 + scale * np.maximum(eigenvalue.astype(np.float32), 0.0)).astype(np.float32)


def _make_splits(n: int, cfg: ExpConfig, seed: Optional[int] = None, split_policy: str = "train") -> Dict[str, list[int]]:
    idx = np.arange(n)
    if split_policy == "test_all":
        return {"train": [], "val": [], "test": idx.tolist()}
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    rng.shuffle(idx)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    if n >= 3:
        n_train = min(max(n_train, 1), n - 2)
        n_val = min(max(n_val, 1), n - n_train - 1)
    return {
        "train": idx[:n_train].tolist(),
        "val": idx[n_train : n_train + n_val].tolist(),
        "test": idx[n_train + n_val :].tolist(),
    }


def convert_official_mat(
    mat_path: Path | str,
    out_dir: Path | str,
    cfg: ExpConfig,
    *,
    split_seed: Optional[int] = None,
    use_future_as_input: Optional[bool] = None,
    store_pmi_for_official_score: bool = True,
    schedule_users: Optional[int] = None,
    chunk_size: Optional[int] = None,
    overwrite: bool = False,
    split_policy: str = "train",
) -> Path:
    """Convert a MAT file to the unified NumPy dataset layout."""
    mat_path = Path(mat_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not overwrite:
        print(f"[official-data] Dataset already exists at {out_dir}. Use --overwrite to regenerate.")
        return out_dir

    use_future = cfg.official_use_future_as_input if use_future_as_input is None else bool(use_future_as_input)
    topk = int(schedule_users if schedule_users is not None else cfg.schedule_users)
    chunk = int(chunk_size or cfg.chunk_size)

    with h5py.File(mat_path, "r") as f:
        missing = [k for k in REQUIRED_KEYS if k not in f]
        if missing:
            raise KeyError(f"Missing keys in {mat_path}: {missing}")
        n = _read_n(f)
        K = int(f["eigenvalue_0ms"].shape[0])
        C = int(f["PMI_0ms"].shape[1])
        if K != cfg.num_ues or C != cfg.pmi_complex_dim:
            print(f"[official-data] Warning: mat K={K}, C={C}; cfg K={cfg.num_ues}, C={cfg.pmi_complex_dim}.")

        shape_feat = (n, cfg.num_ues, cfg.feature_dim)
        features = np.lib.format.open_memmap(out_dir / "features.npy", mode="w+", dtype=np.float32, shape=shape_feat)
        labels = np.lib.format.open_memmap(out_dir / "labels.npy", mode="w+", dtype=np.float32, shape=(n, cfg.num_ues))
        target_scores = np.lib.format.open_memmap(out_dir / "target_scores.npy", mode="w+", dtype=np.float32, shape=(n, cfg.num_ues))
        rate_now_all = np.lib.format.open_memmap(out_dir / "rate_now.npy", mode="w+", dtype=np.float32, shape=(n, cfg.num_ues))
        rate_pred_all = np.lib.format.open_memmap(out_dir / "rate_pred.npy", mode="w+", dtype=np.float32, shape=(n, cfg.num_ues))
        rate_future_all = np.lib.format.open_memmap(out_dir / "rate_future.npy", mode="w+", dtype=np.float32, shape=(n, cfg.num_ues))
        hist_rate_all = np.lib.format.open_memmap(out_dir / "hist_rate.npy", mode="w+", dtype=np.float32, shape=(n, cfg.num_ues))
        speed_all = np.lib.format.open_memmap(out_dir / "speed_kmh.npy", mode="w+", dtype=np.float32, shape=(n,))

        if store_pmi_for_official_score:
            pmi0_all = np.lib.format.open_memmap(out_dir / "pmi_0ms.npy", mode="w+", dtype=np.float32, shape=(n, cfg.num_ues, cfg.pmi_complex_dim, 2))
            pmi5_all = np.lib.format.open_memmap(out_dir / "pmi_5ms.npy", mode="w+", dtype=np.float32, shape=(n, cfg.num_ues, cfg.pmi_complex_dim, 2))
        else:
            pmi0_all = pmi5_all = None

        for start in tqdm(range(0, n, chunk), desc=f"Converting {mat_path.name}"):
            end = min(start + chunk, n)
            b = end - start
            pmi0 = _read_pmi(f, "PMI_0ms", start, end)
            pmi5 = _read_pmi(f, "PMI_5ms", start, end)
            eig0 = _read_matrix_120_by_n(f, "eigenvalue_0ms", start, end)
            eig5 = _read_matrix_120_by_n(f, "eigenvalue_5ms", start, end)
            r0 = _eigen_to_rate(eig0, cfg)
            r5 = _eigen_to_rate(eig5, cfg)
            hist = _read_matrix_120_by_n(f, "rate_old", start, end).astype(np.float32)

            do_mobility_aug = (
                split_policy == "train"
                and bool(getattr(cfg, "train_mobility_augmentation", True))
                and not use_future
            )
            if do_mobility_aug:
                rng = np.random.default_rng(cfg.seed + start + 7717)
                speed_choices = np.asarray(cfg.speeds_kmh, dtype=np.float32)
                weights = np.asarray(
                    getattr(cfg, "train_speed_sampling_weights", np.ones_like(speed_choices)),
                    dtype=np.float64,
                )
                if weights.shape != speed_choices.shape or np.any(weights < 0) or weights.sum() <= 0:
                    raise ValueError("train_speed_sampling_weights must match speeds_kmh and be non-negative")
                weights = weights / weights.sum()
                sample_speeds = rng.choice(speed_choices, size=b, replace=True, p=weights).astype(np.float32)
                aug = augment_mobility_chunk(pmi0, pmi5, r0, r5, sample_speeds, cfg, rng)
                pmi_true = aug["pmi_true_pair"]
                pmi_pred_flat = aug["pmi_pred_flat"]
                label_future = aug["rate_true"]
                rate_pred = aug["rate_pred"]
                speed_norm = aug["speed_norm"]
            elif use_future:
                pmi_true = pmi5
                pmi_pred_flat = _pmi_to_flat_realimag(pmi5)
                label_future = r5
                rate_pred = r5.copy()
                speed_norm = np.zeros((b,), dtype=np.float32)
            else:
                pmi_true = pmi5
                rng = np.random.default_rng(cfg.seed + start + 2027)
                sigma = float(getattr(cfg, "official_rate_pred_noise", 0.08))
                noise = rng.normal(-0.5 * sigma * sigma, sigma, size=r0.shape).astype(np.float32)
                rate_pred = np.clip(r0 * np.exp(noise), 0.0, None).astype(np.float32)
                pmi_pred = pmi0.astype(np.float32).copy()
                pmi_pred += rng.normal(0.0, float(getattr(cfg, "official_pmi_pred_noise", 0.004)), size=pmi_pred.shape).astype(np.float32)
                norm = np.sqrt(np.maximum((pmi_pred**2).sum(axis=(2, 3), keepdims=True), cfg.eps))
                pmi_pred = pmi_pred / norm
                pmi_pred_flat = _pmi_to_flat_realimag(pmi_pred)
                label_future = r5 if bool(getattr(cfg, "official_label_uses_true_future", True)) else rate_pred
                speed_norm = np.zeros((b,), dtype=np.float32)

            x = np.zeros((b, cfg.num_ues, cfg.feature_dim), dtype=np.float32)
            x[:, :, cfg.idx_now_pmi_start : cfg.idx_now_pmi_start + cfg.pmi_real_dim] = _pmi_to_flat_realimag(pmi0)
            x[:, :, cfg.idx_now_rate] = r0
            x[:, :, cfg.idx_future_pmi_start : cfg.idx_future_pmi_start + cfg.pmi_real_dim] = pmi_pred_flat
            x[:, :, cfg.idx_future_rate] = rate_pred
            x[:, :, cfg.idx_hist_rate] = hist
            x[:, :, cfg.idx_speed] = speed_norm[:, None] if getattr(cfg, "mobility_use_speed_feature", True) else 0.0

            teacher_pmi_flat = _pmi_to_flat_realimag(pmi_true)
            greedy_scores = set_sinr_greedy_scores(
                r0,
                label_future,
                hist,
                teacher_pmi_flat,
                cfg.eta,
                topk,
                eps=cfg.eps,
                fairness_alpha=cfg.fairness_alpha,
                interference_beta=getattr(cfg, "paper_set_interference_beta", 0.40),
                power_reference_users=(getattr(cfg, "paper_power_reference_users", 0) or cfg.schedule_users),
            )
            y = (greedy_scores > 0).astype(np.float32)
            q = compute_q(r0, label_future, hist, cfg.eta, cfg.eps, cfg.fairness_alpha)
            qn = (q - q.mean(axis=1, keepdims=True)) / (q.std(axis=1, keepdims=True) + cfg.eps)
            target_score = greedy_scores.astype(np.float32) + 0.05 * np.tanh(qn).astype(np.float32)

            features[start:end] = x
            labels[start:end] = y
            target_scores[start:end] = target_score.astype(np.float32)
            rate_now_all[start:end] = r0
            rate_pred_all[start:end] = rate_pred
            rate_future_all[start:end] = label_future
            hist_rate_all[start:end] = hist
            speed_all[start:end] = speed_norm * float(max(cfg.speeds_kmh))
            if pmi0_all is not None:
                pmi0_all[start:end] = pmi0
                pmi5_all[start:end] = pmi_true

    splits = _make_splits(n, cfg, split_seed, split_policy=split_policy)
    with open(out_dir / "splits.json", "w", encoding="utf-8") as sf:
        json.dump(splits, sf, indent=2)

    meta = {
        "source": "official_mat",
        "mat_path": str(mat_path),
        "num_samples": n,
        "num_ues": cfg.num_ues,
        "num_trps": cfg.num_trps,
        "antennas_per_trp": cfg.antennas_per_trp,
        "pmi_complex_dim": cfg.pmi_complex_dim,
        "pmi_real_dim": cfg.pmi_real_dim,
        "feature_dim": cfg.feature_dim,
        "schedule_users_for_labels": topk,
        "eta": cfg.eta,
        "fairness_alpha": cfg.fairness_alpha,
        "scheduling_delay_ms": cfg.scheduling_delay_ms,
        "use_future_as_input": use_future,
        "train_mobility_augmentation": bool(split_policy == "train" and getattr(cfg, "train_mobility_augmentation", True) and not use_future),
        "split_policy": split_policy,
        "feature_layout": {
            "0:384": "PMI_0ms real then imaginary parts",
            "384": "eigenvalue_0ms/current channel feature value",
            "385:769": "current-only predicted PMI by default; true PMI_5ms only in explicit oracle mode",
            "769": "current-only predicted rate by default; mapped eigenvalue_5ms only in explicit oracle mode",
            "770": "rate_old/historical scheduled rate",
            "771": "UE speed; official example has no speed field, set to zero",
        },
    }
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2, ensure_ascii=False)
    print(f"[official-data] Converted {mat_path} -> {out_dir}")
    return out_dir

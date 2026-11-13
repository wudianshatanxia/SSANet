"""Generate official-format MAT datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import h5py
import numpy as np
from tqdm import tqdm

REQUIRED_KEYS = [
    "N",
    "PMI_0ms",
    "PMI_5ms",
    "eigenvalue_0ms",
    "eigenvalue_5ms",
    "rate_old",
]


@dataclass
class OfficialLikeGenerationConfig:
    train_N: int = 16000
    test_N: int = 2000
    K: int = 120
    C: int = 192
    seed: int = 42
    pmi_jitter: float = 0.015
    eigen_jitter: float = 0.015
    rate_jitter: float = 0.06
    clip_quantile_low: float = 0.001
    clip_quantile_high: float = 0.999
    dtype: str = "float64"
    use_ref_test: bool = True
    task_aligned: bool = True
    num_clusters: int = 12
    cluster_strength: float = 0.86
    temporal_corr: float = 0.88
    future_pattern_strength: float = 1.05


def _read_required_h5(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    out: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        missing = [k for k in REQUIRED_KEYS if k not in f]
        if missing:
            raise KeyError(f"{path} is missing keys: {missing}")
        for k in REQUIRED_KEYS:
            out[k] = f[k][()]
    return out


def _validate_official_shape(data: Dict[str, np.ndarray], path: Path) -> None:
    N = int(np.asarray(data["N"]).reshape(-1)[0])
    pmi_shape = data["PMI_0ms"].shape
    if len(pmi_shape) != 4:
        raise ValueError(f"PMI_0ms in {path} should be 4-D, got {pmi_shape}")
    if pmi_shape[0] != 2 or pmi_shape[1] != 192 or pmi_shape[2] != 120 or pmi_shape[3] != N:
        raise ValueError(f"Unexpected PMI_0ms shape in {path}: {pmi_shape}; expected (2,192,120,N).")
    if data["PMI_5ms"].shape != pmi_shape:
        raise ValueError(f"PMI_5ms shape {data['PMI_5ms'].shape} != {pmi_shape}")
    for key in ["eigenvalue_0ms", "eigenvalue_5ms", "rate_old"]:
        if data[key].shape != (120, N):
            raise ValueError(f"{key} shape {data[key].shape}; expected {(120, N)}")


def load_reference_pool(paths: Iterable[Path]) -> Dict[str, np.ndarray]:
    """Build an empirical sample pool from reference MAT files."""
    pools = {"pmi0": [], "pmi5": [], "eig0": [], "eig5": [], "rate": []}
    for path in paths:
        data = _read_required_h5(path)
        _validate_official_shape(data, path)
        pmi0 = np.transpose(data["PMI_0ms"], (3, 2, 0, 1)).reshape(-1, 2, 192)
        pmi5 = np.transpose(data["PMI_5ms"], (3, 2, 0, 1)).reshape(-1, 2, 192)
        eig0 = np.transpose(data["eigenvalue_0ms"], (1, 0)).reshape(-1)
        eig5 = np.transpose(data["eigenvalue_5ms"], (1, 0)).reshape(-1)
        rate = np.transpose(data["rate_old"], (1, 0)).reshape(-1)
        pools["pmi0"].append(pmi0)
        pools["pmi5"].append(pmi5)
        pools["eig0"].append(eig0)
        pools["eig5"].append(eig5)
        pools["rate"].append(rate)
    return {k: np.concatenate(v, axis=0) for k, v in pools.items()}


def _as_real_vector(pmi: np.ndarray) -> np.ndarray:
    return np.transpose(pmi, (0, 2, 1)).reshape(pmi.shape[0], -1)


def _from_real_vector(x: np.ndarray, C: int) -> np.ndarray:
    return np.transpose(x.reshape(x.shape[0], C, 2), (0, 2, 1))


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _random_unit_like(rng: np.random.Generator, shape: Tuple[int, int]) -> np.ndarray:
    return _normalize_rows(rng.normal(0.0, 1.0, size=shape))


def _add_unit_jitter(rng: np.random.Generator, base: np.ndarray, jitter: float) -> np.ndarray:
    if jitter <= 0:
        return _normalize_rows(base.copy())
    noise = _random_unit_like(rng, base.shape)
    mixed = (1.0 - jitter) * base + jitter * noise
    return _normalize_rows(mixed)


def _clip_by_ref(x: np.ndarray, ref: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    lo, hi = np.quantile(ref, [q_low, q_high])
    return np.clip(x, lo, hi)


def _generate_one_task_aligned(pool: Dict[str, np.ndarray], N: int, cfg: OfficialLikeGenerationConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Generate channel samples with correlated scheduling features."""
    K, C = cfg.K, cfg.C
    G = int(cfg.num_clusters)
    records = N * K
    pool_size = pool["eig0"].shape[0]

    eig0_ref, eig5_ref, rate_ref = pool["eig0"], pool["eig5"], pool["rate"]
    idx_scalar = rng.integers(0, pool_size, size=records)
    eig0 = eig0_ref[idx_scalar] * rng.lognormal(mean=0.0, sigma=cfg.eigen_jitter, size=records)
    rate = rate_ref[idx_scalar] * rng.lognormal(mean=0.0, sigma=cfg.rate_jitter, size=records)

    proto_idx = rng.integers(0, pool["pmi0"].shape[0], size=N * G)
    proto = _as_real_vector(pool["pmi0"][proto_idx]).reshape(N, G, C * 2)
    proto = _normalize_rows(proto.reshape(-1, C * 2)).reshape(N, G, C * 2)
    cluster_id = rng.integers(0, G, size=(N, K))
    proto_user = proto[np.arange(N)[:, None], cluster_id].reshape(records, C * 2)
    user_noise = _random_unit_like(rng, (records, C * 2))
    s = float(np.clip(cfg.cluster_strength, 0.0, 0.98))
    pmi0_vec = _normalize_rows(s * proto_user + np.sqrt(max(1.0 - s * s, 1e-6)) * user_noise)

    temporal_noise = _random_unit_like(rng, (records, C * 2))
    rho = float(np.clip(cfg.temporal_corr, 0.0, 0.99))
    pmi5_vec = _normalize_rows(rho * pmi0_vec + np.sqrt(max(1.0 - rho * rho, 1e-6)) * temporal_noise)

    task_rng = np.random.default_rng(int(cfg.seed) + 91_337)
    quality_w1 = task_rng.normal(0.0, 1.0, size=(C * 2,)).astype(np.float64)
    quality_w2 = task_rng.normal(0.0, 1.0, size=(C * 2,)).astype(np.float64)
    quality_w1 /= np.linalg.norm(quality_w1) + 1e-12
    quality_w2 /= np.linalg.norm(quality_w2) + 1e-12
    proto_q1 = np.einsum("ngd,d->ng", proto, quality_w1)
    proto_q2 = np.einsum("ngd,d->ng", proto, quality_w2)
    cluster_quality = 0.85 * proto_q1 + 0.35 * np.tanh(2.0 * proto_q2)
    latent_cluster = cluster_quality[np.arange(N)[:, None], cluster_id].reshape(records)
    latent_pmi = pmi0_vec @ quality_w1
    latent = 0.90 * latent_cluster + 0.55 * latent_pmi + rng.normal(0.0, 0.12, size=records)

    load = np.zeros((N, K), dtype=np.float64)
    for b in range(N):
        counts = np.bincount(cluster_id[b], minlength=G).astype(np.float64)
        load[b] = counts[cluster_id[b]] - 1.0
    load = load.reshape(records)
    load_penalty = 1.0 / (1.0 + 0.18 * load / max(K / G, 1.0))

    drift_sigma = float(cfg.future_pattern_strength)
    drift = np.exp(0.35 * drift_sigma * latent - 0.5 * (0.35 * drift_sigma) ** 2) * load_penalty
    eig5 = eig0 * drift * rng.lognormal(mean=0.0, sigma=cfg.eigen_jitter, size=records)

    eig0 = _clip_by_ref(eig0, eig0_ref, cfg.clip_quantile_low, cfg.clip_quantile_high)
    eig5 = _clip_by_ref(eig5, eig5_ref, cfg.clip_quantile_low, cfg.clip_quantile_high)
    rate = _clip_by_ref(rate, rate_ref, cfg.clip_quantile_low, cfg.clip_quantile_high)

    pmi0 = _from_real_vector(pmi0_vec, C)
    pmi5 = _from_real_vector(pmi5_vec, C)
    pmi0_official = np.transpose(pmi0.reshape(N, K, 2, C), (2, 3, 1, 0))
    pmi5_official = np.transpose(pmi5.reshape(N, K, 2, C), (2, 3, 1, 0))
    eig0_official = eig0.reshape(N, K).T
    eig5_official = eig5.reshape(N, K).T
    rate_official = rate.reshape(N, K).T

    dtype = np.float64 if cfg.dtype == "float64" else np.float32
    return {
        "N": np.array([[float(N)]], dtype=np.float64),
        "PMI_0ms": pmi0_official.astype(dtype),
        "PMI_5ms": pmi5_official.astype(dtype),
        "eigenvalue_0ms": eig0_official.astype(dtype),
        "eigenvalue_5ms": eig5_official.astype(dtype),
        "rate_old": rate_official.astype(dtype),
    }

def _generate_one(pool: Dict[str, np.ndarray], N: int, cfg: OfficialLikeGenerationConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    if bool(getattr(cfg, "task_aligned", True)):
        return _generate_one_task_aligned(pool, N, cfg, rng)
    K, C = cfg.K, cfg.C
    records = N * K
    idx = rng.integers(0, pool["eig0"].shape[0], size=records)

    pmi0_base = _as_real_vector(pool["pmi0"][idx])
    pmi5_base = _as_real_vector(pool["pmi5"][idx])
    pmi0_vec = _add_unit_jitter(rng, pmi0_base, cfg.pmi_jitter)
    pmi5_vec = _add_unit_jitter(rng, pmi5_base, cfg.pmi_jitter)

    pmi0 = _from_real_vector(pmi0_vec, C)
    pmi5 = _from_real_vector(pmi5_vec, C)

    eig0_ref, eig5_ref, rate_ref = pool["eig0"], pool["eig5"], pool["rate"]
    eig0 = eig0_ref[idx] * rng.lognormal(mean=0.0, sigma=cfg.eigen_jitter, size=records)
    ratio = eig5_ref[idx] / (eig0_ref[idx] + 1e-12)
    eig5 = eig0 * ratio * rng.lognormal(mean=0.0, sigma=cfg.eigen_jitter / 2.0, size=records)
    rate = rate_ref[idx] * rng.lognormal(mean=0.0, sigma=cfg.rate_jitter, size=records)

    eig0 = _clip_by_ref(eig0, eig0_ref, cfg.clip_quantile_low, cfg.clip_quantile_high)
    eig5 = _clip_by_ref(eig5, eig5_ref, cfg.clip_quantile_low, cfg.clip_quantile_high)
    rate = _clip_by_ref(rate, rate_ref, cfg.clip_quantile_low, cfg.clip_quantile_high)

    pmi0_official = np.transpose(pmi0.reshape(N, K, 2, C), (2, 3, 1, 0))
    pmi5_official = np.transpose(pmi5.reshape(N, K, 2, C), (2, 3, 1, 0))
    eig0_official = eig0.reshape(N, K).T
    eig5_official = eig5.reshape(N, K).T
    rate_official = rate.reshape(N, K).T

    dtype = np.float64 if cfg.dtype == "float64" else np.float32
    return {
        "N": np.array([[float(N)]], dtype=np.float64),
        "PMI_0ms": pmi0_official.astype(dtype),
        "PMI_5ms": pmi5_official.astype(dtype),
        "eigenvalue_0ms": eig0_official.astype(dtype),
        "eigenvalue_5ms": eig5_official.astype(dtype),
        "rate_old": rate_official.astype(dtype),
    }


def save_h5_mat(path: Path, data: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k in REQUIRED_KEYS:
            f.create_dataset(k, data=data[k], compression="gzip", compression_opts=4)


def _create_h5_output(path: Path, N: int, cfg: OfficialLikeGenerationConfig) -> Dict[str, h5py.Dataset]:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.float64 if cfg.dtype == "float64" else np.float32
    f = h5py.File(path, "w")
    sample_chunk = max(1, min(8, N))
    matrix_chunk = max(1, min(256, N))
    try:
        dsets = {
            "N": f.create_dataset("N", data=np.array([[float(N)]], dtype=np.float64)),
            "PMI_0ms": f.create_dataset(
                "PMI_0ms",
                shape=(2, cfg.C, cfg.K, N),
                dtype=dtype,
                chunks=(2, cfg.C, cfg.K, sample_chunk),
                compression="gzip",
                compression_opts=4,
            ),
            "PMI_5ms": f.create_dataset(
                "PMI_5ms",
                shape=(2, cfg.C, cfg.K, N),
                dtype=dtype,
                chunks=(2, cfg.C, cfg.K, sample_chunk),
                compression="gzip",
                compression_opts=4,
            ),
            "eigenvalue_0ms": f.create_dataset(
                "eigenvalue_0ms",
                shape=(cfg.K, N),
                dtype=dtype,
                chunks=(cfg.K, matrix_chunk),
                compression="gzip",
                compression_opts=4,
            ),
            "eigenvalue_5ms": f.create_dataset(
                "eigenvalue_5ms",
                shape=(cfg.K, N),
                dtype=dtype,
                chunks=(cfg.K, matrix_chunk),
                compression="gzip",
                compression_opts=4,
            ),
            "rate_old": f.create_dataset(
                "rate_old",
                shape=(cfg.K, N),
                dtype=dtype,
                chunks=(cfg.K, matrix_chunk),
                compression="gzip",
                compression_opts=4,
            ),
        }
    except Exception:
        f.close()
        raise
    dsets["_file"] = f
    return dsets


def _new_stats(shape: Tuple[int, ...]) -> Dict[str, object]:
    return {
        "shape": list(shape),
        "count": 0,
        "sum": 0.0,
        "sumsq": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
    }


def _update_stats(stats: Dict[str, object], x: np.ndarray) -> None:
    arr = np.asarray(x, dtype=np.float64)
    stats["count"] = int(stats["count"]) + arr.size
    stats["sum"] = float(stats["sum"]) + float(np.sum(arr))
    stats["sumsq"] = float(stats["sumsq"]) + float(np.sum(arr * arr))
    stats["min"] = min(float(stats["min"]), float(np.min(arr)))
    stats["max"] = max(float(stats["max"]), float(np.max(arr)))


def _finalize_stats(stats: Dict[str, object]) -> Dict[str, object]:
    count = max(int(stats["count"]), 1)
    mean = float(stats["sum"]) / count
    var = max(float(stats["sumsq"]) / count - mean * mean, 0.0)
    return {
        "shape": stats["shape"],
        "mean": mean,
        "std": float(np.sqrt(var)),
        "min": float(stats["min"]),
        "max": float(stats["max"]),
    }


def generate_and_save_h5_mat_chunked(
    path: Path,
    pool: Dict[str, np.ndarray],
    N: int,
    cfg: OfficialLikeGenerationConfig,
    rng: np.random.Generator,
    *,
    chunk_samples: int = 64,
) -> Dict[str, Dict[str, object]]:
    """Write generated samples to an HDF5 MAT file in chunks."""
    path = Path(path)
    if N < 1:
        raise ValueError("N must be >= 1")
    chunk_samples = max(1, int(chunk_samples))
    dsets = _create_h5_output(path, N, cfg)
    h5_file = dsets.pop("_file")
    stats = {
        "PMI_0ms": _new_stats((2, cfg.C, cfg.K, N)),
        "PMI_5ms": _new_stats((2, cfg.C, cfg.K, N)),
        "eigenvalue_0ms": _new_stats((cfg.K, N)),
        "eigenvalue_5ms": _new_stats((cfg.K, N)),
        "rate_old": _new_stats((cfg.K, N)),
    }
    try:
        for start in tqdm(range(0, N, chunk_samples), desc=f"Generating {path.name}"):
            end = min(start + chunk_samples, N)
            chunk = _generate_one(pool, end - start, cfg, rng)
            dsets["PMI_0ms"][:, :, :, start:end] = chunk["PMI_0ms"]
            dsets["PMI_5ms"][:, :, :, start:end] = chunk["PMI_5ms"]
            dsets["eigenvalue_0ms"][:, start:end] = chunk["eigenvalue_0ms"]
            dsets["eigenvalue_5ms"][:, start:end] = chunk["eigenvalue_5ms"]
            dsets["rate_old"][:, start:end] = chunk["rate_old"]
            for key in stats:
                _update_stats(stats[key], chunk[key])
    finally:
        h5_file.close()
    return {k: _finalize_stats(v) for k, v in stats.items()}


def summarize(data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for key in ["PMI_0ms", "PMI_5ms", "eigenvalue_0ms", "eigenvalue_5ms", "rate_old"]:
        x = np.asarray(data[key]).reshape(-1)
        summary[key] = {
            "shape": list(data[key].shape),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "p01": float(np.quantile(x, 0.01)),
            "p50": float(np.quantile(x, 0.50)),
            "p99": float(np.quantile(x, 0.99)),
            "max": float(np.max(x)),
        }
    return summary


def _reference_pool_from_config(
    ref_train: Path | str,
    ref_test: Path | str,
    no_ref_test: bool,
) -> tuple[list[Path], Dict[str, np.ndarray]]:
    ref_train = Path(ref_train)
    ref_test = Path(ref_test)
    refs = [ref_train]
    if not no_ref_test and ref_test.exists():
        refs.append(ref_test)
    return refs, load_reference_pool(refs)


def generate_official_like_mat_file(
    ref_train: Path | str = Path("data/official_raw/Training_Data.mat"),
    ref_test: Path | str = Path("data/official_raw/Test_Data.mat"),
    out_mat: Path | str = Path("data/generated_official_like/official_like_16000.mat"),
    *,
    N: int = 16000,
    seed: int = 42,
    pmi_jitter: float = 0.015,
    eigen_jitter: float = 0.015,
    rate_jitter: float = 0.06,
    dtype: str = "float64",
    no_ref_test: bool = False,
    chunk_samples: int = 64,
) -> Path:
    """Generate one official-format MAT dataset."""
    out_mat = Path(out_mat)
    cfg = OfficialLikeGenerationConfig(
        train_N=int(N),
        test_N=0,
        seed=int(seed),
        pmi_jitter=float(pmi_jitter),
        eigen_jitter=float(eigen_jitter),
        rate_jitter=float(rate_jitter),
        dtype=dtype,
        use_ref_test=not no_ref_test,
    )
    refs, pool = _reference_pool_from_config(ref_train, ref_test, no_ref_test)
    summary = generate_and_save_h5_mat_chunked(
        out_mat,
        pool,
        cfg.train_N,
        cfg,
        np.random.default_rng(cfg.seed),
        chunk_samples=chunk_samples,
    )

    report = {
        "config": asdict(cfg),
        "reference_files": [str(p) for p in refs],
        "summary": summary,
    }
    with open(out_mat.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[official-like] Saved generated .mat file to: {out_mat}")
    print(f"[official-like] N={cfg.train_N}")
    for k in REQUIRED_KEYS:
        shape = (1, 1) if k == "N" else summary[k]["shape"]
        print(f"  {k:16s} {shape}")
    return out_mat


def generate_official_like_mat_files(
    ref_train: Path | str = Path("data/official_raw/Training_Data.mat"),
    ref_test: Path | str = Path("data/official_raw/Test_Data.mat"),
    out_dir: Path | str = Path("data/generated_official_like"),
    *,
    train_N: int = 16000,
    test_N: int = 2000,
    seed: int = 42,
    pmi_jitter: float = 0.015,
    eigen_jitter: float = 0.015,
    rate_jitter: float = 0.06,
    dtype: str = "float64",
    no_ref_test: bool = False,
    chunk_samples: int = 64,
) -> Path:
    """Generate training and test MAT datasets."""
    out_dir = Path(out_dir)
    cfg = OfficialLikeGenerationConfig(
        train_N=int(train_N),
        test_N=int(test_N),
        seed=int(seed),
        pmi_jitter=float(pmi_jitter),
        eigen_jitter=float(eigen_jitter),
        rate_jitter=float(rate_jitter),
        dtype=dtype,
        use_ref_test=not no_ref_test,
    )
    refs, pool = _reference_pool_from_config(ref_train, ref_test, no_ref_test)

    train_summary = generate_and_save_h5_mat_chunked(
        out_dir / "Training_Data.mat",
        pool,
        cfg.train_N,
        cfg,
        np.random.default_rng(cfg.seed),
        chunk_samples=chunk_samples,
    )
    test_summary = generate_and_save_h5_mat_chunked(
        out_dir / "Test_Data.mat",
        pool,
        cfg.test_N,
        cfg,
        np.random.default_rng(cfg.seed + 1000003),
        chunk_samples=chunk_samples,
    )

    report = {
        "config": asdict(cfg),
        "reference_files": [str(p) for p in refs],
        "train_summary": train_summary,
        "test_summary": test_summary,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "generation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[official-like] Saved generated .mat files to: {out_dir}")
    print(f"[official-like] Training_Data.mat N={cfg.train_N}; Test_Data.mat N={cfg.test_N}")
    for k in REQUIRED_KEYS:
        shape = (1, 1) if k == "N" else train_summary[k]["shape"]
        print(f"  {k:16s} {shape}")
    return out_dir

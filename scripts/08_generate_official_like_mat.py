from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.official_like_generator import generate_official_like_mat_file, generate_official_like_mat_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate official-like .mat data under data/.")
    parser.add_argument("--ref_train", type=Path, default=Path("data/official_raw/Training_Data.mat"))
    parser.add_argument("--ref_test", type=Path, default=Path("data/official_raw/Test_Data.mat"))
    parser.add_argument(
        "--out_mat",
        type=Path,
        default=None,
        help="Generate one .mat dataset file, for example data/generated_official_like/official_like_16000.mat.",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("data/generated_official_like"))
    parser.add_argument("--N", type=int, default=16000, help="Sample count when --out_mat is used.")
    parser.add_argument("--train_N", type=int, default=16000)
    parser.add_argument("--test_N", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pmi_jitter", type=float, default=0.015)
    parser.add_argument("--eigen_jitter", type=float, default=0.015)
    parser.add_argument("--rate_jitter", type=float, default=0.06)
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--chunk_samples", type=int, default=64, help="Number of samples generated before each HDF5 write.")
    parser.add_argument("--no_ref_test", action="store_true", help="Only use ref_train as distribution reference.")
    args = parser.parse_args()

    ref_train = ROOT / args.ref_train if not args.ref_train.is_absolute() else args.ref_train
    ref_test = ROOT / args.ref_test if not args.ref_test.is_absolute() else args.ref_test
    if args.out_mat is not None:
        out_mat = ROOT / args.out_mat if not args.out_mat.is_absolute() else args.out_mat
        generate_official_like_mat_file(
            ref_train=ref_train,
            ref_test=ref_test,
            out_mat=out_mat,
            N=args.N,
            seed=args.seed,
            pmi_jitter=args.pmi_jitter,
            eigen_jitter=args.eigen_jitter,
            rate_jitter=args.rate_jitter,
            dtype=args.dtype,
            no_ref_test=args.no_ref_test,
            chunk_samples=args.chunk_samples,
        )
    else:
        generate_official_like_mat_files(
            ref_train=ref_train,
            ref_test=ref_test,
            out_dir=ROOT / args.out_dir if not args.out_dir.is_absolute() else args.out_dir,
            train_N=args.train_N,
            test_N=args.test_N,
            seed=args.seed,
            pmi_jitter=args.pmi_jitter,
            eigen_jitter=args.eigen_jitter,
            rate_jitter=args.rate_jitter,
            dtype=args.dtype,
            no_ref_test=args.no_ref_test,
            chunk_samples=args.chunk_samples,
        )


if __name__ == "__main__":
    main()

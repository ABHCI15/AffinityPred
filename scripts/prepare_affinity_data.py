#!/usr/bin/env python3
"""CLI for generating processed datasets used by TubercuProbe."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Dict

DATASET_FACTORIES: Dict[str, Dict[str, Any]] = {
    "chembl": {
        "module": "data_processing_utils.prep_data_chembl",
        "class": "AffinityDataProcessor",
        "defaults": {
            "input_col": "Smiles",
            "label_col": "pCHEMBL",
            "seq_col": "Sequence",
            "device": "cuda",
            "embedding_cache_path": "datasets/protein_chembl_embeddings.pkl",
        },
    },
    "cysdb": {
        "module": "data_processing_utils.prep_data_cysdb",
        "class": "AffinityDataProcessor",
        "defaults": {
            "input_col": "SMILES",
            "label_col": "Activity",
            "seq_col": "Sequence",
            "device": "cuda",
            "embedding_cache_path": "datasets/protein_cysdb_embeddings.pkl",
        },
    },
}


def _build_processor(dataset: str, overrides: Dict[str, Any]):
    cfg = DATASET_FACTORIES[dataset]
    module = importlib.import_module(cfg["module"])
    processor_cls = getattr(module, cfg["class"])
    init_kwargs = cfg["defaults"].copy()
    init_kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return processor_cls(**init_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create processed train/valid/test splits with cached graphs and protein embeddings."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_FACTORIES),
        default="chembl",
        help="Which preset to use for column defaults and caches.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw affinity table (CSV/TSV).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed_data"),
        help="Directory to write processed pickle artifacts.",
    )
    parser.add_argument("--input-col", type=str, help="Override SMILES column name.")
    parser.add_argument("--label-col", type=str, help="Override label column name.")
    parser.add_argument("--seq-col", type=str, help="Override protein sequence column name.")
    parser.add_argument(
        "--embedding-cache-path",
        type=Path,
        help="Path used to cache computed protein embeddings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("cpu", "cuda"),
        help="Device for ESM embedding model (default depends on preset).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples in the held-out test split (default: 0.2).",
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.1,
        help="Fraction of samples in validation relative to full dataset (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=5,
        help="Random seed for deterministic splits (default: 5).",
    )
    parser.add_argument(
        "--stratify-col",
        type=str,
        help="Optional column name to stratify by when splitting data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "input_col": args.input_col,
        "label_col": args.label_col,
        "seq_col": args.seq_col,
        "device": args.device,
        "embedding_cache_path": str(args.embedding_cache_path) if args.embedding_cache_path else None,
    }
    processor = _build_processor(args.dataset, overrides)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing dataset '{args.dataset}' from {args.input} ...")
    train_df, valid_df, test_df = processor.prepare_dataset(
        str(args.input),
        test_size=args.test_size,
        valid_size=args.valid_size,
        random_state=args.seed,
        stratify_col=args.stratify_col,
        output_dir=str(args.output_dir),
    )

    print("Saved processed splits to", args.output_dir.resolve())
    print(f"Train size: {len(train_df)} | Valid size: {len(valid_df)} | Test size: {len(test_df)}")


if __name__ == "__main__":
    main()

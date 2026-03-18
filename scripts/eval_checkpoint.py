#!/usr/bin/env python3
"""Evaluate a saved best.pt checkpoint on the test set.

Use when a training job times out after saving best.pt but before test evaluation.

Usage:
    python scripts/eval_checkpoint.py \
        --model drugban \
        --split random \
        --negative uniform_random \
        --data_dir exports/ \
        --output_dir results/baselines/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_baseline as baseline


def _resolve_run_directory(output_dir: Path, args) -> tuple[str, Path]:
    """Locate the checkpoint directory, supporting legacy pre-seed naming."""
    canonical_run_name = baseline._build_run_name(
        args.model,
        args.dataset,
        args.split,
        args.negative,
        args.seed,
    )
    canonical_dir = output_dir / canonical_run_name
    if canonical_dir.exists():
        return canonical_run_name, canonical_dir

    legacy_names = [
        f"{args.model}_{args.split}_{args.negative}",
        f"{args.model}_{args.split}_{args.negative}_seed{args.seed}",
    ]
    for legacy_name in legacy_names:
        legacy_dir = output_dir / legacy_name
        if legacy_dir.exists():
            logger.warning(
                "Using legacy run directory %s for logical run %s",
                legacy_name,
                canonical_run_name,
            )
            return legacy_name, legacy_dir

    return canonical_run_name, canonical_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoint on test set.")
    parser.add_argument("--model", required=True, choices=["deepdta", "graphdta", "drugban"])
    parser.add_argument("--split", required=True, choices=list(baseline._SPLIT_COL_MAP))
    parser.add_argument("--negative", required=True,
                        choices=["negbiodb", "uniform_random", "degree_matched"])
    parser.add_argument("--dataset", default="balanced", choices=["balanced", "realistic"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=Path, default=ROOT / "exports")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results" / "baselines")
    args = parser.parse_args(argv)

    if args.split == "ddb" and args.dataset != "balanced":
        logger.error("DDB split is only supported for dataset=balanced.")
        return 1

    filename = baseline._resolve_dataset_file(args.dataset, args.split, args.negative)
    if filename is None:
        logger.error(
            "Dataset combination not supported: dataset=%s, split=%s, negative=%s.",
            args.dataset,
            args.split,
            args.negative,
        )
        return 1
    parquet_path = args.data_dir / filename
    if not parquet_path.exists():
        logger.error("Dataset file not found: %s", parquet_path)
        return 1

    split_col = baseline._SPLIT_COL_MAP[args.split]

    # Run name and output dir (must match train_baseline.py naming)
    run_name, out_dir = _resolve_run_directory(args.output_dir, args)

    checkpoint = out_dir / "best.pt"
    if not checkpoint.exists():
        logger.error("No checkpoint found at %s", checkpoint)
        return 1

    results_path = out_dir / "results.json"
    if results_path.exists():
        logger.info("results.json already exists at %s — skipping.", results_path)
        return 0

    logger.info("Evaluating %s on test set from %s", run_name, parquet_path.name)

    # Device
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    except ImportError:
        logger.error("torch not found.")
        return 1
    logger.info("Device: %s", device)

    # Graph cache
    graph_cache = None
    if args.model in ("graphdta", "drugban"):
        cache_path = args.data_dir / "graph_cache.pt"
        graph_cache = baseline._prepare_graph_cache(parquet_path, cache_path)

    # Only need test split for eval
    test_ds = baseline.DTIDataset(parquet_path, split_col, "test", args.model, graph_cache)
    train_ds = baseline.DTIDataset(parquet_path, split_col, "train", args.model, graph_cache)
    val_ds = baseline.DTIDataset(parquet_path, split_col, "val", args.model, graph_cache)

    if len(test_ds) == 0:
        logger.error("Empty test split. Check split_col=%s", split_col)
        return 1
    logger.info("Test set: %d samples", len(test_ds))

    batch_size = args.batch_size
    if args.model == "drugban" and batch_size > 128:
        batch_size = 128
    test_loader = baseline.make_dataloader(test_ds, batch_size, shuffle=False, device=device)

    # Build model and evaluate
    model = baseline.build_model(args.model).to(device)
    test_metrics = baseline.evaluate(model, test_loader, checkpoint, device)

    # Load best_val_log_auc from training log if available
    training_log = out_dir / "training_log.csv"
    best_val = float("nan")
    if training_log.exists():
        import csv
        with open(training_log) as f:
            rows = list(csv.DictReader(f))
        if rows:
            best_val = max(float(r["val_log_auc"]) for r in rows if r["val_log_auc"])

    results = {
        "run_name": run_name,
        "model": args.model,
        "split": args.split,
        "negative": args.negative,
        "dataset": args.dataset,
        "seed": args.seed,
        "best_val_log_auc": best_val,
        "test_metrics": test_metrics,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }
    baseline.write_results_json(results_path, results)

    logger.info("Test metrics:")
    for k, v in test_metrics.items():
        logger.info("  %-15s = %.4f", k, v)
    logger.info("Results saved → %s", results_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

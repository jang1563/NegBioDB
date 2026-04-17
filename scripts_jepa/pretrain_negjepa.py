#!/usr/bin/env python
"""Negative-JEPA pretraining entry point.

Usage:
  # Full pretraining (HPC):
  python scripts_jepa/pretrain_negjepa.py --config configs/negbiojepa/pretrain_3domain.yaml

  # Smoke test (local, no parquet files required):
  python scripts_jepa/pretrain_negjepa.py \
      --config configs/negbiojepa/pretrain_3domain.yaml \
      --smoke-test

  # Override config values on the fly:
  python scripts_jepa/pretrain_negjepa.py \
      --config configs/negbiojepa/pretrain_3domain.yaml \
      --epochs 5 --batch-size 64
"""

import argparse
import os
import sys

# Ensure src/ is on path when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.utils.data

from negbiojepa.config import JEPAConfig
from negbiojepa.dataset import (
    NegJEPADataset,
    MultiDomainDataset,
    jepa_collate_fn,
)
from negbiojepa.encoders import build_encoder_pair
from negbiojepa.predictor import JEPAPredictor
from negbiojepa.trainer import NegJEPATrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Negative-JEPA pretraining")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 10 steps on synthetic data (no parquet needed)")
    # Override knobs
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = JEPAConfig.from_yaml(args.config)

    # Apply CLI overrides
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.smoke_test:
        overrides["data_root"] = "synthetic"
        overrides["epochs"] = 1
        overrides["log_every"] = 5
        overrides["save_every"] = 1
        overrides["output_dir"] = "/tmp/negjepa_smoke_test"
        overrides["num_workers"] = 0  # avoid worker spawn overhead for tiny smoke batches
    if overrides:
        cfg = cfg.replace(**overrides)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {cfg}")

    # Build datasets
    print(f"Loading domains: {cfg.domains}")
    datasets: dict = {}
    for domain in cfg.domains:
        ds = NegJEPADataset.from_data_root(
            cfg.data_root, domain, cfg.tabular_max_features,
            max_samples=cfg.max_samples_per_domain,
        )
        print(f"  {domain}: {len(ds)} samples")
        datasets[domain] = ds

    combined = MultiDomainDataset(datasets)
    loader = torch.utils.data.DataLoader(
        combined,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=jepa_collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"DataLoader: {len(loader)} batches per epoch")

    # Build model
    context_enc, target_enc = build_encoder_pair(cfg)
    predictor = JEPAPredictor.from_config(cfg)
    n_params = sum(p.numel() for p in context_enc.parameters()) + \
               sum(p.numel() for p in predictor.parameters())
    print(f"Model params: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Build trainer (optionally resume)
    if args.resume:
        print(f"Resuming from: {args.resume}")
        trainer = NegJEPATrainer.load(
            args.resume, context_enc, target_enc, predictor, device
        )
    else:
        trainer = NegJEPATrainer(cfg, context_enc, target_enc, predictor, device)

    # Save config alongside output
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.to_yaml(os.path.join(cfg.output_dir, "config.yaml"))

    # Optionally log to W&B
    try:
        import wandb
        wandb.init(
            project="negbiojepa",
            config=vars(cfg),
            name=os.path.basename(cfg.output_dir),
        )
        print("W&B logging enabled")
    except ImportError:
        pass

    if args.smoke_test:
        print("=== Smoke test: running 10 steps ===")
        total_steps = 10
        losses = []
        for i, batch in enumerate(loader):
            metrics = trainer._train_step(batch, total_steps)
            losses.append(metrics["loss"])
            print(f"  Step {i+1}: loss={metrics['loss']:.4f}")
            if i + 1 >= 10:
                break
        import math
        assert all(not math.isnan(l) for l in losses), "Loss is NaN during smoke test"
        print(f"Smoke test PASSED. Final loss: {losses[-1]:.4f}")
    else:
        trainer.fit(loader)
        print(f"Pretraining complete. Best checkpoint: {cfg.output_dir}/best.pt")


if __name__ == "__main__":
    main()

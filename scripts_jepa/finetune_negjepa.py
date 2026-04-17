#!/usr/bin/env python
"""Negative-JEPA fine-tuning entry point.

Loads a pretrained JEPA encoder, adds a linear classification head, and
fine-tunes on a specific domain + split. Saves results as JSON.

Usage:
  python scripts_jepa/finetune_negjepa.py \
      --pretrained /athena/.../negjepa_pretrain_3d_sigreg/best.pt \
      --domain dti \
      --split cold_compound \
      --config configs/negbiojepa/finetune.yaml

HPC batch:
  sbatch --export=ALL,\\
    PRETRAINED=/athena/.../best.pt,\\
    DOMAIN=dti,SPLIT=cold_compound \\
    slurm/run_negjepa_finetune.slurm

Valid splits (domain-dependent):
  random, cold_compound, cold_target, cold_both, scaffold, temporal,
  degree_balanced, cold_protein, cold_gene, cold_cell_line
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.utils.data

from negbiojepa.config import JEPAConfig
from negbiojepa.dataset import NegJEPADataset, jepa_collate_fn
from negbiojepa.encoders import UnifiedEncoder
from negbiojepa.trainer import NegJEPAFinetuner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Negative-JEPA fine-tuning")
    parser.add_argument("--pretrained", required=True, help="Path to pretrained checkpoint")
    parser.add_argument("--domain", required=True,
                        help="Domain to fine-tune on (dti, ppi, ge, ct, vp, dc, cp, md)")
    parser.add_argument("--split", default="random",
                        help="Split strategy name, e.g. random, cold_compound (default: random)")
    parser.add_argument("--config", required=True, help="Path to finetune YAML config")
    parser.add_argument("--split-col", type=str, default=None,
                        help="Explicit parquet column for split. Default: auto-detect "
                        "('split' or 'split_{--split}')")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output_dir from config")
    parser.add_argument("--freeze-encoder", action="store_true", default=None)
    parser.add_argument("--unfreeze-encoder", dest="freeze_encoder", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = JEPAConfig.from_yaml(args.config)

    overrides = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.freeze_encoder is not None:
        overrides["freeze_encoder"] = args.freeze_encoder
    if overrides:
        cfg = cfg.replace(**overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Domain: {args.domain} | Split: {args.split}")
    print(f"Pretrained: {args.pretrained}")

    # Load pretrained encoder
    ckpt = torch.load(args.pretrained, map_location=device)
    # cfg stored in checkpoint may have different domains — use finetune cfg for arch
    encoder = UnifiedEncoder(cfg)
    encoder.load_state_dict(ckpt["context_enc"])
    print(f"Loaded context encoder from step {ckpt.get('global_step', '?')}")

    # Resolve split column: prefer explicit --split-col, else derive from --split
    split_col = args.split_col or f"split_{args.split}"

    # Build data loaders
    def make_loader(split_name: str, shuffle: bool) -> torch.utils.data.DataLoader:
        ds = NegJEPADataset.from_data_root(
            cfg.data_root, args.domain, cfg.tabular_max_features,
            split=split_name, split_col=split_col,
        )
        print(f"  {split_name}: {len(ds)} samples")
        return torch.utils.data.DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            collate_fn=jepa_collate_fn,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    # Parquet split columns (e.g. split_random, split_cold_gene_v1) hold
    # the values "train" / "val" / "test". args.split only selects which
    # column to use (via split_col resolution).
    train_loader = make_loader("train", shuffle=True)
    val_loader   = make_loader("val", shuffle=False)
    test_loader  = make_loader("test", shuffle=False)

    # Fine-tune
    finetuner = NegJEPAFinetuner(cfg, encoder, domain=args.domain, device=device)
    print(f"Encoder {'frozen' if cfg.freeze_encoder else 'unfrozen (full fine-tune)'}")

    test_metrics = finetuner.fit(train_loader, val_loader, test_loader)

    print(f"\nTest results ({args.domain}, {args.split}):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    out_name = f"ft_{args.domain}_{args.split}.json"
    out_path = os.path.join(cfg.output_dir, out_name)
    finetuner.save_results(
        {"split": args.split, "pretrained": args.pretrained, **test_metrics},
        out_path,
    )


if __name__ == "__main__":
    main()

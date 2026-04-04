#!/usr/bin/env python
"""Phase 3 ablation: DPO training using L3/L4 preference pairs.

Tests the "GRPO is secretly DPO" hypothesis (arXiv:2510.00977) by comparing
DPO-trained models against GRPO-trained models on the same data.

Usage:
    accelerate launch scripts_rl/06b_train_dpo.py \
        --base-model Qwen/Qwen3-8B \
        --dataset results/negbiorl/phase2_training_data/dpo_pairs.jsonl \
        --output-dir results/negbiorl/phase3_checkpoints/qwen3_dpo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="DPO training with preference pairs")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--sft-adapter", type=Path, default=None, help="Path to SFT LoRA adapter (optional)")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2560)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    args = parser.parse_args()

    config = {}
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Lazy imports
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    # Load DPO pairs
    print(f"Loading dataset: {args.dataset}")
    records = []
    with open(args.dataset) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    dataset = Dataset.from_list(records)
    print(f"  {len(dataset)} preference pairs")

    # Model
    model_id = args.base_model
    if args.sft_adapter:
        print(f"Loading SFT adapter from {args.sft_adapter}")
        model_id = str(args.sft_adapter)

    # LoRA
    lora_config = LoraConfig(
        r=config.get("lora_r", args.lora_r),
        lora_alpha=config.get("lora_alpha", args.lora_alpha),
        target_modules=config.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        task_type="CAUSAL_LM",
    )

    # DPO config
    dpo_config = DPOConfig(
        output_dir=str(args.output_dir),
        beta=config.get("beta", args.beta),
        learning_rate=config.get("learning_rate", args.learning_rate),
        num_train_epochs=config.get("epochs", args.epochs),
        per_device_train_batch_size=config.get("batch_size", args.batch_size),
        max_length=config.get("max_length", args.max_length),
        max_prompt_length=config.get("max_prompt_length", args.max_prompt_length),
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=config.get("report_to", "wandb"),
    )

    print(f"DPO config: beta={dpo_config.beta}, lr={dpo_config.learning_rate}, "
          f"epochs={dpo_config.num_train_epochs}")

    trainer = DPOTrainer(
        model=model_id,
        args=dpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    print("Starting DPO training...")
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    print(f"Model saved to {args.output_dir / 'final'}")


if __name__ == "__main__":
    main()

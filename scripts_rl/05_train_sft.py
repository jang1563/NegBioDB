#!/usr/bin/env python
"""Phase 3: SFT training with LoRA on NegBioDB negative results data.

Usage:
    python scripts_rl/05_train_sft.py --base-model Qwen/Qwen3-8B
    accelerate launch scripts_rl/05_train_sft.py --base-model Qwen/Qwen3-8B --config configs/negbiorl/sft_qwen3_8b.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="SFT training with LoRA")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization (for large models)")
    args = parser.parse_args()

    # Load config overrides if provided
    config = {}
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Lazy imports — only on HPC with rltrain deps installed
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    load_4bit = args.load_in_4bit or config.get("load_in_4bit", False)

    print(f"Loading model: {args.base_model} (4bit={load_4bit})")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    load_kwargs = {
        "torch_dtype": "bfloat16",
        "trust_remote_code": True,
    }
    if load_4bit:
        import torch
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    records = []
    with open(args.dataset) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    dataset = Dataset.from_list(records)
    print(f"  {len(dataset)} records")

    # LoRA config
    lora_config = LoraConfig(
        r=config.get("lora_r", args.lora_r),
        lora_alpha=config.get("lora_alpha", args.lora_alpha),
        target_modules=config.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        task_type="CAUSAL_LM",
    )

    # SFT config
    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        learning_rate=config.get("learning_rate", args.learning_rate),
        num_train_epochs=config.get("epochs", args.epochs),
        per_device_train_batch_size=config.get("batch_size", args.batch_size),
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=config.get("report_to", "wandb"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Starting SFT training...")
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    print(f"Model saved to {args.output_dir / 'final'}")


if __name__ == "__main__":
    main()

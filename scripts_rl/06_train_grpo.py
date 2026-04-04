#!/usr/bin/env python
"""Phase 3: GRPO training with experimental rewards.

This is the CORE training script. Uses trl GRPOTrainer with:
- Multi-reward functions (L1 correctness, L4 correctness, evidence bonus)
- vLLM colocate mode for efficient generation
- LoRA for parameter-efficient training
- scale_rewards="batch" for cross-domain normalization

Usage:
    accelerate launch scripts_rl/06_train_grpo.py \
        --base-model Qwen/Qwen3-8B \
        --dataset results/negbiorl/phase2_training_data/grpo_dataset.jsonl \
        --output-dir results/negbiorl/phase3_checkpoints/qwen3_grpo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="GRPO training with experimental rewards")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--sft-adapter", type=Path, default=None, help="Path to SFT LoRA adapter (optional)")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-generations", type=int, default=8, help="Group size G")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--use-vllm", action="store_true", default=True)
    parser.add_argument("--no-vllm", dest="use_vllm", action="store_false")
    args = parser.parse_args()

    config = {}
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Lazy imports
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from negbiorl.rewards import (
        DEFAULT_REWARD_FUNCS,
        DEFAULT_REWARD_WEIGHTS,
    )

    # Load GRPO dataset
    print(f"Loading dataset: {args.dataset}")
    records = []
    with open(args.dataset) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    dataset = Dataset.from_list(records)
    print(f"  {len(dataset)} prompts")

    # Model — either base or SFT-adapted
    model_id = args.base_model
    if args.sft_adapter:
        print(f"Loading SFT adapter from {args.sft_adapter}")
        # When using SFT adapter, pass it as the model and GRPOTrainer handles merging
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

    # GRPO config
    num_gen = config.get("num_generations", args.num_generations)
    grpo_config = GRPOConfig(
        output_dir=str(args.output_dir),
        num_generations=num_gen,
        use_vllm=config.get("use_vllm", args.use_vllm),
        vllm_mode="colocate",
        scale_rewards="batch",
        learning_rate=config.get("learning_rate", args.learning_rate),
        num_train_epochs=config.get("epochs", args.epochs),
        per_device_train_batch_size=config.get("batch_size", args.batch_size),
        max_prompt_length=config.get("max_prompt_length", args.max_prompt_length),
        max_completion_length=config.get("max_completion_length", args.max_completion_length),
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=config.get("report_to", "wandb"),
    )

    # Reward functions — use trl multi-reward API
    reward_funcs = config.get("reward_funcs", DEFAULT_REWARD_FUNCS)
    reward_weights = config.get("reward_weights", DEFAULT_REWARD_WEIGHTS)

    print(f"GRPO config: G={num_gen}, lr={grpo_config.learning_rate}, "
          f"epochs={grpo_config.num_train_epochs}, vLLM={grpo_config.use_vllm}")
    print(f"Rewards: {len(reward_funcs)} functions, weights={reward_weights}")

    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    print("Starting GRPO training...")
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    print(f"Model saved to {args.output_dir / 'final'}")


if __name__ == "__main__":
    main()

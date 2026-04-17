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
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--use-vllm", action="store_true", default=False)
    parser.add_argument("--no-vllm", dest="use_vllm", action="store_false")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization (for large models like Gemma4-31B)")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume training from")
    args = parser.parse_args()

    config = {}
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Lazy imports
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from negbiorl.adapter_utils import load_tokenizer, prepare_merged_adapter
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
        print(f"Merging SFT adapter from {args.sft_adapter}")
        _, merged_dir = prepare_merged_adapter(args.sft_adapter, args.base_model)
        print(f"  Using merged model at {merged_dir}")
        model_id = str(merged_dir)

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
    use_vllm = config.get("use_vllm", args.use_vllm)
    grpo_config = GRPOConfig(
        output_dir=str(args.output_dir),
        num_generations=num_gen,
        use_vllm=use_vllm,
        vllm_mode="colocate",  # only used when use_vllm=True
        scale_rewards="batch",
        learning_rate=config.get("learning_rate", args.learning_rate),
        num_train_epochs=config.get("epochs", args.epochs),
        per_device_train_batch_size=config.get("batch_size", args.batch_size),
        generation_batch_size=config.get("generation_batch_size", num_gen),
        max_completion_length=config.get("max_completion_length", args.max_completion_length),
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=config.get("report_to", "wandb"),
    )

    # Reward functions — use trl multi-reward API
    # Note: reward_weights belongs in GRPOConfig, not GRPOTrainer
    reward_weights = config.get("reward_weights", DEFAULT_REWARD_WEIGHTS)
    grpo_config.reward_weights = reward_weights

    load_4bit = args.load_in_4bit or config.get("load_in_4bit", False)

    print(f"GRPO config: G={num_gen}, lr={grpo_config.learning_rate}, "
          f"epochs={grpo_config.num_train_epochs}, vLLM={grpo_config.use_vllm}, "
          f"4bit={load_4bit}")
    print(f"Rewards: {len(DEFAULT_REWARD_FUNCS)} functions, weights={reward_weights}")

    tokenizer = load_tokenizer(model_id)

    if load_4bit:
        # Pre-load model with 4-bit quantization (e.g. Gemma4-31B).
        # GRPOTrainer accepts a model object instead of a string ID.
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        print(f"Loading model in 4-bit: {model_id}")
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        trainer = GRPOTrainer(
            model=model_obj,
            reward_funcs=DEFAULT_REWARD_FUNCS,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )
    else:
        trainer = GRPOTrainer(
            model=model_id,
            reward_funcs=DEFAULT_REWARD_FUNCS,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )

    print("Starting GRPO training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / "final"))
    print(f"Model saved to {args.output_dir / 'final'}")


if __name__ == "__main__":
    main()

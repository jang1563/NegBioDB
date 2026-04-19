#!/usr/bin/env python
"""Generate predictions from a fine-tuned model using vLLM or HuggingFace.

Loads LoRA adapter, runs inference on L1/L4 exports, saves predictions.

Usage:
    # vLLM backend (default, fast)
    python scripts_rl/10_generate_predictions.py \
        --adapter results/negbiorl/phase3_checkpoints/grpo_G8_12345/final \
        --base-model Qwen/Qwen3-8B \
        --domains dti ct ppi ge vp

    # HF backend (for 4-bit models like Gemma4-31B)
    python scripts_rl/10_generate_predictions.py \
        --adapter results/negbiorl/phase3_checkpoints/grpo_gemma4/final \
        --base-model unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
        --backend hf --load-in-4bit \
        --domains dti ct ppi ge vp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from negbiorl.data_registry import (
    ALL_DOMAINS,
    PROJECT_ROOT,
    get_domain,
    get_export_path,
    get_gold_answer_field,
    load_jsonl,
)
from negbiorl.sft_data import _format_prompt


def _build_prompt_data(args):
    """Build prompts, gold answers, and question IDs for all domain/task combos."""
    tasks_data = []
    for domain in args.domains:
        for task in args.tasks:
            try:
                export_path = get_export_path(domain, task)
            except (ValueError, FileNotFoundError):
                continue
            if not export_path.exists():
                print(f"  SKIP {domain}/{task}: no export")
                continue

            records = load_jsonl(export_path)
            test_records = [r for r in records if r.get("split") == args.split]
            if not test_records:
                print(f"  SKIP {domain}/{task}: no {args.split} split records")
                continue

            gold_field = get_gold_answer_field(domain)
            prompts, gold_answers, question_ids = [], [], []
            for rec in test_records:
                sys_prompt, user_prompt = _format_prompt(domain, task, rec)
                prompts.append(f"{sys_prompt}\n\nUser: {user_prompt}\n\nAssistant:")
                gold_answers.append(rec.get(gold_field, ""))
                question_ids.append(rec.get("question_id", ""))

            tasks_data.append({
                "domain": domain,
                "task": task,
                "prompts": prompts,
                "gold_answers": gold_answers,
                "question_ids": question_ids,
            })
    return tasks_data


def _save_predictions(out_path, question_ids, gold_answers, predictions):
    """Save predictions to JSONL."""
    with open(out_path, "w") as f:
        for qid, gold, pred in zip(question_ids, gold_answers, predictions):
            f.write(json.dumps({
                "question_id": qid,
                "prediction": pred,
                "gold_answer": gold,
            }) + "\n")
    print(f"    → {out_path}")


def _run_vllm(args, tasks_data):
    """Generate predictions using vLLM backend."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"Loading base model (vLLM): {args.base_model}")
    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_lora_rank=64,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    lora_request = LoRARequest("negbiorl", 1, str(args.adapter))
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    for td in tasks_data:
        print(f"  Generating {td['domain']}/{td['task']}: {len(td['prompts'])} prompts...")
        outputs = llm.generate(td["prompts"], sampling_params, lora_request=lora_request)
        predictions = [out.outputs[0].text.strip() for out in outputs]
        out_path = args.output_dir / f"{td['domain']}_{td['task']}_predictions.jsonl"
        _save_predictions(out_path, td["question_ids"], td["gold_answers"], predictions)


def _run_hf(args, tasks_data):
    """Generate predictions using HuggingFace transformers backend.

    Supports 4-bit quantization via BitsAndBytesConfig for large models
    like Gemma4-31B where vLLM doesn't support bitsandbytes.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    print(f"Loading base model (HF, 4bit={args.load_in_4bit}): {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.adapter:
        from peft import PeftModel
        print(f"Loading adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, str(args.adapter))

    model.eval()

    for td in tasks_data:
        print(f"  Generating {td['domain']}/{td['task']}: {len(td['prompts'])} prompts...")
        predictions = []
        # Process in batches to avoid OOM
        batch_size = args.hf_batch_size
        for i in range(0, len(td["prompts"]), batch_size):
            batch_prompts = td["prompts"][i:i + batch_size]
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_model_len,
            ).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature if args.temperature > 0 else 1.0,
                    do_sample=args.temperature > 0,
                )
            # Decode only the generated tokens (skip input)
            for j, output in enumerate(outputs):
                input_len = inputs["input_ids"][j].shape[0]
                generated = output[input_len:]
                pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
                predictions.append(pred)

        out_path = args.output_dir / f"{td['domain']}_{td['task']}_predictions.jsonl"
        _save_predictions(out_path, td["question_ids"], td["gold_answers"], predictions)


def main():
    parser = argparse.ArgumentParser(description="Generate predictions from fine-tuned model")
    parser.add_argument("--adapter", type=Path, default=None,
                        help="Path to LoRA adapter (omit for baseline inference)")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--domains", nargs="+", default=ALL_DOMAINS)
    parser.add_argument("--tasks", nargs="+", default=["l1", "l4"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm",
                        help="Inference backend (vllm=fast, hf=supports 4-bit)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization (HF backend only)")
    parser.add_argument("--hf-batch-size", type=int, default=4,
                        help="Batch size for HF generate() (default 4)")
    # vLLM-specific
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80,
                        help="vLLM GPU memory utilization (default 0.80)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Max context length (default 4096)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks_data = _build_prompt_data(args)

    if not tasks_data:
        print("No tasks to generate. Check --domains and export paths.")
        return

    if args.load_in_4bit and args.backend == "vllm":
        parser.error("--load-in-4bit is only supported with --backend hf (vLLM does not support bitsandbytes)")

    if args.backend == "hf":
        _run_hf(args, tasks_data)
    else:
        if args.adapter is None:
            parser.error("--adapter is required for vLLM backend")
        _run_vllm(args, tasks_data)

    print("Done.")


if __name__ == "__main__":
    main()

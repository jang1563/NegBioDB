#!/usr/bin/env python
"""Generate predictions from a fine-tuned model using vLLM.

Loads LoRA adapter, runs inference on L1/L4 exports, saves predictions.

Usage:
    python scripts_rl/10_generate_predictions.py \
        --adapter results/negbiorl/phase3_checkpoints/grpo_G8_12345/final \
        --base-model Qwen/Qwen3-8B \
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


def main():
    parser = argparse.ArgumentParser(description="Generate predictions from fine-tuned model")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--domains", nargs="+", default=ALL_DOMAINS)
    parser.add_argument("--tasks", nargs="+", default=["l1", "l4"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80,
                        help="vLLM GPU memory utilization (default 0.80; reduce if OOM)")
    args = parser.parse_args()

    # Lazy imports for HPC
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"Loading base model: {args.base_model}")
    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_lora_rank=64,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    lora_request = LoRARequest("negbiorl", 1, str(args.adapter))
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

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

            # Build prompts
            gold_field = get_gold_answer_field(domain)
            prompts = []
            gold_answers = []
            question_ids = []
            for rec in test_records:
                sys_prompt, user_prompt = _format_prompt(domain, task, rec)
                # Format as chat for vLLM
                prompts.append(f"{sys_prompt}\n\nUser: {user_prompt}\n\nAssistant:")
                gold_answers.append(rec.get(gold_field, ""))
                question_ids.append(rec.get("question_id", ""))

            print(f"  Generating {domain}/{task}: {len(prompts)} prompts...")
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

            # Save predictions
            out_path = args.output_dir / f"{domain}_{task}_predictions.jsonl"
            with open(out_path, "w") as f:
                for qid, gold, output in zip(question_ids, gold_answers, outputs):
                    pred = output.outputs[0].text.strip()
                    f.write(json.dumps({
                        "question_id": qid,
                        "prediction": pred,
                        "gold_answer": gold,
                    }) + "\n")
            print(f"    → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()

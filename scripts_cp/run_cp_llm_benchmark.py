#!/usr/bin/env python3
"""Run CP LLM benchmark inference, with two-stage judging for CP-L3."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from negbiodb.llm_client import LLMClient
from negbiodb_cp.llm_eval import compute_all_cp_llm_metrics, parse_cp_l3_judge_scores
from negbiodb_cp.llm_prompts import format_cp_l3_judge_prompt, format_cp_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "exports" / "cp_llm"
OUTPUT_BASE = PROJECT_ROOT / "results" / "cp_llm"

TASK_FILES = {
    "cp-l1": "cp_l1_dataset.jsonl",
    "cp-l2": "cp_l2_dataset.jsonl",
    "cp-l3": "cp_l3_dataset.jsonl",
    "cp-l4": "cp_l4_dataset.jsonl",
}

TASK_MAX_TOKENS = {
    "cp-l1": 256,
    "cp-l2": 1024,
    "cp-l3": 768,
    "cp-l4": 256,
}


def load_dataset(task: str, data_dir: Path) -> list[dict]:
    path = data_dir / TASK_FILES[task]
    records = []
    with open(path) as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def load_task_metadata(task: str, data_dir: Path) -> dict:
    meta_path = data_dir / f"{task.replace('-', '_')}_metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def get_fewshot_examples(records: list[dict], fewshot_set: int, n_per_class: int = 2) -> list[dict]:
    import random

    fewshot = [record for record in records if record.get("split") == "fewshot"]
    by_answer: dict[str, list[dict]] = {}
    for record in fewshot:
        by_answer.setdefault(record.get("gold_answer", "default"), []).append(record)

    rng = random.Random(42 + fewshot_set)
    examples = []
    for _answer, group in by_answer.items():
        pool = list(group)
        rng.shuffle(pool)
        start = fewshot_set * n_per_class
        examples.extend(pool[start:start + n_per_class])
    rng.shuffle(examples)
    return examples


def sanitize_model_name(model: str) -> str:
    return model.split("/")[-1].replace(".", "-").lower()


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _load_prediction_map(pred_path: Path) -> dict[str, str]:
    predictions_by_id: dict[str, str] = {}
    if not pred_path.exists():
        return predictions_by_id
    with open(pred_path) as handle:
        for line in handle:
            record = json.loads(line)
            predictions_by_id[record["question_id"]] = record["prediction"]
    return predictions_by_id


def _load_judge_map(judge_path: Path) -> dict[str, dict]:
    judged_by_id: dict[str, dict] = {}
    if not judge_path.exists():
        return judged_by_id
    with open(judge_path) as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("scores") is not None:
                judged_by_id[record["question_id"]] = record
    return judged_by_id


def _build_client(
    *,
    provider: str,
    model: str,
    api_base: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
) -> LLMClient:
    return LLMClient(
        provider=provider,
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CP LLM benchmark")
    parser.add_argument("--task", required=True, choices=list(TASK_FILES))
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", required=True, choices=["vllm", "gemini", "openai", "anthropic"])
    parser.add_argument("--config", default="zero-shot", choices=["zero-shot", "3-shot"])
    parser.add_argument("--fewshot-set", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_BASE)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--judge-provider", default=None, choices=["vllm", "gemini", "openai", "anthropic"])
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-api-base", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument("--allow-proxy-smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.max_tokens is None:
        args.max_tokens = TASK_MAX_TOKENS.get(args.task, 1024)

    task_meta = load_task_metadata(args.task, args.data_dir)
    if not task_meta.get("production_ready", True) and not args.allow_proxy_smoke:
        logger.error(
            "CP LLM benchmark is blocked for plate_proxy datasets. "
            "Re-run with --allow-proxy-smoke only for plumbing smoke validation."
        )
        return 1

    model_name = sanitize_model_name(args.model)
    run_name = f"{args.task}_{model_name}_{args.config}_fs{args.fewshot_set}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    records = load_dataset(args.task, args.data_dir)
    test_records = [record for record in records if record.get("split") == "test"]
    fewshot_examples = None
    if args.config == "3-shot":
        fewshot_examples = get_fewshot_examples(records, args.fewshot_set)

    client = _build_client(
        provider=args.provider,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    pred_path = run_dir / "predictions.jsonl"
    predictions_by_id = _load_prediction_map(pred_path)

    remaining = [
        (idx, record) for idx, record in enumerate(test_records)
        if record.get("question_id", f"Q{idx}") not in predictions_by_id
    ]

    start_time = time.time()
    with open(pred_path, "a") as handle:
        for idx, record in remaining:
            system, user = format_cp_prompt(args.task, record, args.config, fewshot_examples)
            try:
                response = client.generate(user, system)
            except Exception as exc:
                response = f"ERROR: {exc}"
            question_id = record.get("question_id", f"Q{idx}")
            predictions_by_id[question_id] = response
            handle.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "prediction": response,
                        "gold_answer": record.get("gold_answer"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()

    elapsed = time.time() - start_time

    metrics_input = [
        predictions_by_id.get(record.get("question_id", f"Q{idx}"), "")
        for idx, record in enumerate(test_records)
    ]
    judge_meta = None

    if args.task == "cp-l3":
        judge_provider = args.judge_provider or args.provider
        judge_model = args.judge_model or args.model
        judge_api_base = args.judge_api_base if args.judge_api_base is not None else args.api_base
        judge_api_key = args.judge_api_key if args.judge_api_key is not None else args.api_key
        judge_client = _build_client(
            provider=judge_provider,
            model=judge_model,
            api_base=judge_api_base,
            api_key=judge_api_key,
            temperature=0.0,
            max_tokens=args.judge_max_tokens,
        )

        judge_path = run_dir / "judge_scores.jsonl"
        judged_by_id = _load_judge_map(judge_path)
        remaining_for_judge = [
            (idx, record) for idx, record in enumerate(test_records)
            if record.get("question_id", f"Q{idx}") not in judged_by_id
        ]
        judge_start = time.time()
        with open(judge_path, "a") as handle:
            for idx, record in remaining_for_judge:
                question_id = record.get("question_id", f"Q{idx}")
                prediction = predictions_by_id.get(question_id, "")
                system, user = format_cp_l3_judge_prompt(record, prediction)
                try:
                    judge_response = judge_client.generate(user, system)
                except Exception as exc:
                    judge_response = f"ERROR: {exc}"
                scores = parse_cp_l3_judge_scores(judge_response)
                judged_by_id[question_id] = {
                    "question_id": question_id,
                    "judge_response": judge_response,
                    "scores": scores,
                }
                handle.write(
                    json.dumps(
                        {
                            "question_id": question_id,
                            "prediction": prediction,
                            "judge_response": judge_response,
                            "scores": scores,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()

        judge_elapsed = time.time() - judge_start
        metrics_input = [
            judged_by_id.get(record.get("question_id", f"Q{idx}"), {}).get("judge_response", "")
            for idx, record in enumerate(test_records)
        ]
        judge_meta = {
            "judge_provider": judge_provider,
            "judge_model": judge_model,
            "judge_api_base": judge_api_base,
            "judge_max_tokens": args.judge_max_tokens,
            "elapsed_seconds": judge_elapsed,
            "n_judged": sum(1 for record in judged_by_id.values() if record.get("scores") is not None),
        }

    metrics = compute_all_cp_llm_metrics(args.task, metrics_input, test_records)

    with open(run_dir / "results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, default=_json_safe)
    with open(run_dir / "run_meta.json", "w") as handle:
        json.dump(
            {
                "task": args.task,
                "model": args.model,
                "provider": args.provider,
                "config": args.config,
                "fewshot_set": args.fewshot_set,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "n_test": len(test_records),
                "n_predictions": len(predictions_by_id),
                "elapsed_seconds": elapsed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_name": run_name,
                "task_metadata": task_meta,
                "judge": judge_meta,
            },
            handle,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

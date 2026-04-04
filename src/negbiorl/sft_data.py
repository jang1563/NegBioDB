"""SFT and GRPO dataset construction from NegBioDB LLM benchmark exports.

Converts domain exports into trl-compatible dataset formats:
- SFT: {"messages": [system, user, assistant]}
- GRPO: {"prompt": [system, user], "gold_answer": ..., "task": ..., "domain": ..., "tier": ...}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from negbiorl.data_registry import (
    TRAIN_DOMAINS,
    get_domain,
    get_export_path,
    get_gold_answer_field,
    get_prefixed_task,
    load_jsonl,
)


# ---------------------------------------------------------------------------
# Prompt formatting — reuses NegBioDB domain-specific formatters
# ---------------------------------------------------------------------------

def _format_prompt(domain: str, task: str, record: dict) -> tuple[str, str]:
    """Format a prompt using the domain's existing formatter.

    Returns (system_prompt, user_prompt).
    Each domain has its own function name (format_prompt, format_ct_prompt, etc.).
    """
    import importlib
    reg = get_domain(domain)
    mod = importlib.import_module(reg["prompt_module"])
    func = getattr(mod, reg["format_prompt"])
    prefixed_task = get_prefixed_task(domain, task)
    return func(prefixed_task, record, config="zero-shot")


# ---------------------------------------------------------------------------
# SFT dataset construction
# ---------------------------------------------------------------------------

def build_sft_record(
    domain: str,
    task: str,
    record: dict,
    gold_response: str,
) -> dict[str, Any]:
    """Build a single SFT training record in trl messages format.

    Args:
        domain: Domain key (dti, ct, ppi, ge)
        task: Task level (l1, l3, l4)
        record: Raw export record from JSONL
        gold_response: The target assistant response

    Returns:
        {"messages": [system_msg, user_msg, assistant_msg],
         "domain": str, "task": str}
    """
    system_prompt, user_prompt = _format_prompt(domain, task, record)
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": gold_response},
        ],
        "domain": domain,
        "task": task,
    }


def _make_l1_gold_response(record: dict, domain: str) -> str:
    """Construct a gold L1 response: letter + brief explanation."""
    gold_field = get_gold_answer_field(domain)
    answer = record.get(gold_field, "")
    # For SFT we provide the letter plus minimal reasoning
    return f"{answer}"


def _make_l4_gold_response(record: dict, domain: str) -> str:
    """Construct a gold L4 response: tested/untested + evidence from context."""
    gold_field = get_gold_answer_field(domain)
    answer = record.get(gold_field, "")
    return f"{answer}"


def _filter_by_split(records: list[dict], split: str) -> list[dict]:
    """Filter records by split specification.

    Special handling: NegBioDB exports have test/val/fewshot splits (no 'train').
    - "train" → use fewshot + val (non-test) records for training
    - "test" / "val" / "fewshot" → exact match
    """
    if split == "train":
        return [r for r in records if r.get("split") in ("fewshot", "val")]
    return [r for r in records if r.get("split") == split]


def build_sft_dataset(
    domains: list[str] | None = None,
    tasks: list[str] | None = None,
    split: str = "train",
    max_per_domain_task: int | None = None,
) -> list[dict]:
    """Build SFT dataset from benchmark exports across domains.

    Args:
        domains: Which domains to include (default: TRAIN_DOMAINS)
        tasks: Which task levels (default: ["l1", "l4"])
        split: Which data split to use ("train" uses fewshot+val)
        max_per_domain_task: Cap per domain×task (for balancing)

    Returns:
        List of SFT records in trl messages format
    """
    domains = domains or TRAIN_DOMAINS
    tasks = tasks or ["l1", "l4"]

    records = []
    for domain in domains:
        for task in tasks:
            try:
                export_path = get_export_path(domain, task)
            except (ValueError, FileNotFoundError):
                continue
            if not export_path.exists():
                continue

            raw = load_jsonl(export_path)
            split_records = _filter_by_split(raw, split)

            if task == "l1":
                gold_fn = _make_l1_gold_response
            elif task == "l4":
                gold_fn = _make_l4_gold_response
            else:
                continue  # L2/L3 handled separately

            if max_per_domain_task and len(split_records) > max_per_domain_task:
                split_records = split_records[:max_per_domain_task]

            for rec in split_records:
                gold_response = gold_fn(rec, domain)
                sft_rec = build_sft_record(domain, task, rec, gold_response)
                records.append(sft_rec)

    return records


# ---------------------------------------------------------------------------
# GRPO dataset construction (prompts only — model generates completions)
# ---------------------------------------------------------------------------

def build_grpo_record(domain: str, task: str, record: dict) -> dict[str, Any]:
    """Build a single GRPO training record (prompt + metadata, no completion).

    Returns:
        {"prompt": [system_msg, user_msg],
         "gold_answer": str, "task": str, "domain": str,
         "tier": str|None, "difficulty": str|None}
    """
    system_prompt, user_prompt = _format_prompt(domain, task, record)
    gold_field = get_gold_answer_field(domain)

    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "gold_answer": record.get(gold_field, ""),
        "task": task,
        "domain": domain,
        "tier": record.get("confidence_tier") or record.get("metadata", {}).get("confidence_tier"),
        "difficulty": record.get("difficulty"),
    }


def build_grpo_dataset(
    domains: list[str] | None = None,
    tasks: list[str] | None = None,
    split: str = "train",
    max_per_domain_task: int | None = None,
) -> list[dict]:
    """Build GRPO dataset (prompts only) from benchmark exports.

    Args:
        domains: Which domains to include (default: TRAIN_DOMAINS)
        tasks: Which task levels (default: ["l1", "l4"])
        split: Which data split to use
        max_per_domain_task: Cap per domain×task

    Returns:
        List of GRPO records (prompt + metadata for reward functions)
    """
    domains = domains or TRAIN_DOMAINS
    tasks = tasks or ["l1", "l4"]

    records = []
    for domain in domains:
        for task in tasks:
            try:
                export_path = get_export_path(domain, task)
            except (ValueError, FileNotFoundError):
                continue
            if not export_path.exists():
                continue

            raw = load_jsonl(export_path)
            split_records = _filter_by_split(raw, split)

            if max_per_domain_task and len(split_records) > max_per_domain_task:
                split_records = split_records[:max_per_domain_task]

            for rec in split_records:
                grpo_rec = build_grpo_record(domain, task, rec)
                records.append(grpo_rec)

    return records


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def save_dataset(records: list[dict], path: Path) -> int:
    """Save dataset as JSONL, returning record count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)

"""Prompt formatters for the Cell Painting LLM benchmark."""

from __future__ import annotations

import json

CP_L1_QUESTION = (
    "{context_text}\n\n"
    "Answer with one letter only: A, B, C, or D."
)

CP_L2_QUESTION = (
    "{context_text}\n\n"
    "Return a JSON object with keys: compound_identifier, dose, dose_unit, "
    "cell_line, batch_id, dmso_distance_summary, reproducibility_summary, "
    "qc_summary, outcome_label."
)

CP_L3_QUESTION = (
    "{context_text}\n\n"
    "Provide a short evidence-grounded explanation. Use only the assay evidence in the prompt."
)

CP_L3_JUDGE_TEMPLATE = (
    "You are grading a Cell Painting explanation for evidence-grounded assay reasoning.\n"
    "Use the assay summary and the reference explanation only as grading context.\n"
    "Do not reward invented mechanisms.\n\n"
    "Scoring rubric:\n"
    "- evidence_grounding: Did the response stay anchored to the provided assay evidence?\n"
    "- assay_reasoning: Did it correctly interpret DMSO distance, reproducibility, and QC/viability?\n"
    "- specificity: Was the explanation concrete rather than generic?\n"
    "- non_speculation: Did it avoid unsupported mechanistic claims?\n\n"
    "Return JSON only with integer scores from 1 to 5 for keys:\n"
    "evidence_grounding, assay_reasoning, specificity, non_speculation.\n\n"
    "Assay summary:\n"
    "{context_text}\n\n"
    "Stored outcome label: {gold_category}\n"
    "Reference explanation:\n"
    "{gold_reasoning}\n\n"
    "Model response to grade:\n"
    "{response_text}"
)

CP_L4_QUESTION = (
    "{context_text}\n\n"
    "Answer on the first line with either tested or untested. "
    "If you answer tested, use the second line to cite the Cell Painting evidence."
)


def _format_examples(examples: list[dict], mode: str) -> str:
    chunks = []
    for ex in examples:
        metadata = ex.get("metadata", {})
        if mode == "l1":
            chunks.append(f"{ex['context_text']}\nAnswer: {ex['gold_answer']}")
        elif mode == "l2":
            chunks.append(
                f"{ex['context_text']}\n\nExtracted:\n"
                f"{json.dumps(ex.get('gold_extraction') or metadata.get('gold_extraction', {}), indent=2)}"
            )
        elif mode == "l3":
            chunks.append(
                f"{ex['context_text']}\n\nExplanation:\n"
                f"{ex.get('gold_reasoning') or metadata.get('gold_reasoning', '')}"
            )
        else:
            chunks.append(f"{ex['context_text']}\nAnswer: {ex['gold_answer']}")
    return "\n\n---\n\n".join(chunks)


def format_cp_prompt(
    task: str,
    record: dict,
    config: str,
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Return (system, user) prompts for a CP task."""
    system = (
        "You are a careful biomedical reasoning assistant. "
        "Ground answers in the provided Cell Painting assay evidence only."
    )
    task = task.lower()
    fewshot_examples = fewshot_examples or []

    if task == "cp-l1":
        user = CP_L1_QUESTION.format(context_text=record["context_text"])
        if config == "3-shot" and fewshot_examples:
            user = (
                "Examples:\n"
                f"{_format_examples(fewshot_examples, 'l1')}\n\n"
                f"Now solve the next example.\n\n{user}"
            )
    elif task == "cp-l2":
        user = CP_L2_QUESTION.format(context_text=record["context_text"])
        if config == "3-shot" and fewshot_examples:
            user = (
                "Examples:\n"
                f"{_format_examples(fewshot_examples, 'l2')}\n\n"
                f"Now extract the next report.\n\n{user}"
            )
    elif task == "cp-l3":
        user = CP_L3_QUESTION.format(context_text=record["context_text"])
        if config == "3-shot" and fewshot_examples:
            user = (
                "Examples:\n"
                f"{_format_examples(fewshot_examples, 'l3')}\n\n"
                f"Now explain the next perturbation.\n\n{user}"
            )
    elif task == "cp-l4":
        user = CP_L4_QUESTION.format(context_text=record["context_text"])
        if config == "3-shot" and fewshot_examples:
            user = (
                "Examples:\n"
                f"{_format_examples(fewshot_examples, 'l4')}\n\n"
                f"Now answer the next tuple.\n\n{user}"
            )
    else:
        raise ValueError(f"Unknown CP task: {task}")

    return system, user


def format_cp_l3_judge_prompt(record: dict, response_text: str) -> tuple[str, str]:
    """Return (system, user) prompts for CP-L3 judge scoring."""
    metadata = record.get("metadata", {})
    gold_reasoning = record.get("gold_reasoning") or metadata.get("gold_reasoning") or "N/A"
    system = (
        "You are a strict Cell Painting benchmark judge. "
        "Score only what is supported by the assay evidence and return JSON only."
    )
    user = CP_L3_JUDGE_TEMPLATE.format(
        context_text=record.get("context_text", ""),
        gold_category=record.get("gold_category", "unknown"),
        gold_reasoning=gold_reasoning,
        response_text=response_text,
    )
    return system, user

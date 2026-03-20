#!/usr/bin/env python3
"""Export all LLM benchmark prompt templates to a reproducibility appendix.

Reads templates from src/negbiodb/llm_prompts.py and the judge rubric from
src/negbiodb/llm_eval.py, then writes docs/appendix_prompts.md.

Usage:
    python scripts/export_prompt_appendix.py
"""

from pathlib import Path

from negbiodb.llm_eval import L3_JUDGE_PROMPT
from negbiodb.llm_prompts import (
    L1_ANSWER_FORMAT,
    L1_FEW_SHOT,
    L1_ZERO_SHOT,
    L2_FEW_SHOT,
    L2_ZERO_SHOT,
    L3_FEW_SHOT,
    L3_ZERO_SHOT,
    L4_ANSWER_FORMAT,
    L4_FEW_SHOT,
    L4_ZERO_SHOT,
    SYSTEM_PROMPT,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "docs" / "appendix_prompts.md"


def code_block(text: str, lang: str = "") -> str:
    """Wrap text in a fenced code block."""
    return f"```{lang}\n{text}\n```"


def main():
    sections = []

    sections.append("# Appendix A: LLM Benchmark Prompt Templates\n")
    sections.append(
        "This appendix documents all prompt templates used in the NegBioDB "
        "LLM benchmark (tasks L1--L4). Templates are reproduced verbatim "
        "from `src/negbiodb/llm_prompts.py` and `src/negbiodb/llm_eval.py`.\n"
    )

    # A.1 System Prompt
    sections.append("## A.1 System Prompt (Shared Across All Tasks)\n")
    sections.append(code_block(SYSTEM_PROMPT))
    sections.append("")

    # A.2 L1
    sections.append("## A.2 L1: Activity Classification (Multiple Choice)\n")
    sections.append("### A.2.1 Zero-Shot Template\n")
    sections.append(code_block(L1_ZERO_SHOT))
    sections.append("")
    sections.append("### A.2.2 Few-Shot Template\n")
    sections.append(code_block(L1_FEW_SHOT))
    sections.append("")
    sections.append("### A.2.3 Answer Format Instruction\n")
    sections.append(code_block(L1_ANSWER_FORMAT))
    sections.append(
        "\nThe answer format instruction is appended after both zero-shot "
        "and few-shot templates.\n"
    )

    # A.3 L2
    sections.append("## A.3 L2: Structured Extraction\n")
    sections.append("### A.3.1 Zero-Shot Template\n")
    sections.append(code_block(L2_ZERO_SHOT))
    sections.append("")
    sections.append("### A.3.2 Few-Shot Template\n")
    sections.append(code_block(L2_FEW_SHOT))
    sections.append(
        "\nFew-shot examples include the abstract text and the corresponding "
        "gold extraction in JSON format, separated by `---` delimiters.\n"
    )

    # A.4 L3
    sections.append("## A.4 L3: Scientific Reasoning\n")
    sections.append("### A.4.1 Zero-Shot Template\n")
    sections.append(code_block(L3_ZERO_SHOT))
    sections.append("")
    sections.append("### A.4.2 Few-Shot Template\n")
    sections.append(code_block(L3_FEW_SHOT))
    sections.append("")
    sections.append("### A.4.3 LLM-as-Judge Rubric\n")
    sections.append(
        "Responses are evaluated by a judge model (Gemini 2.5 Flash-Lite) "
        "using the following rubric:\n"
    )
    sections.append(code_block(L3_JUDGE_PROMPT))
    sections.append(
        "\nThe judge returns scores as JSON with four dimensions "
        "(accuracy, reasoning, completeness, specificity), each rated 1--5.\n"
    )

    # A.5 L4
    sections.append("## A.5 L4: Tested vs Untested Discrimination\n")
    sections.append("### A.5.1 Zero-Shot Template\n")
    sections.append(code_block(L4_ZERO_SHOT))
    sections.append("")
    sections.append("### A.5.2 Few-Shot Template\n")
    sections.append(code_block(L4_FEW_SHOT))
    sections.append("")
    sections.append("### A.5.3 Answer Format Instruction\n")
    sections.append(code_block(L4_ANSWER_FORMAT))
    sections.append(
        "\nThe answer format instruction is appended after both zero-shot "
        "and few-shot templates.\n"
    )

    # A.6 Model Configuration
    sections.append("## A.6 Model Configuration\n")
    sections.append("| Parameter | Value |")
    sections.append("|-----------|-------|")
    sections.append("| Temperature | 0.0 (deterministic) |")
    sections.append("| Max output tokens | 1024 (L1/L4), 2048 (L2/L3) |")
    sections.append("| Few-shot sets | 3 independent sets (fs0, fs1, fs2) |")
    sections.append("| Retry policy | Exponential backoff, max 8 retries |")
    sections.append("")
    sections.append("### Models\n")
    sections.append("| Model | Provider | Inference |")
    sections.append("|-------|----------|-----------|")
    sections.append("| Llama-3.3-70B-Instruct-AWQ | vLLM | Local (A100 GPU) |")
    sections.append("| Qwen2.5-32B-Instruct-AWQ | vLLM | Local (A100 GPU) |")
    sections.append("| Mistral-7B-Instruct-v0.3 | vLLM | Local (A100 GPU) |")
    sections.append("| GPT-4o-mini | OpenAI API | Cloud |")
    sections.append("| Gemini 2.5 Flash | Google Gemini API | Cloud |")
    sections.append("| Gemini 2.5 Flash-Lite | Google Gemini API | Cloud |")
    sections.append("")
    sections.append(
        "Gemini 2.5 Flash uses `thinkingConfig: {thinkingBudget: 0}` to "
        "disable internal reasoning tokens and ensure the full output budget "
        "is available for the response.\n"
    )

    # Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(sections)
    OUTPUT.write_text(text)
    print(f"Written: {OUTPUT}")
    print(f"  Lines: {len(text.splitlines())}")

    # Verify completeness: check that each template's content appears in output
    checks = {
        "SYSTEM_PROMPT": SYSTEM_PROMPT[:40],
        "L1_ZERO_SHOT": "{context}",
        "L1_FEW_SHOT": "examples of drug-target interaction classification",
        "L1_ANSWER_FORMAT": "Respond with ONLY the letter",
        "L2_ZERO_SHOT": "Extract all negative drug-target interaction",
        "L2_FEW_SHOT": "extract from this abstract",
        "L3_ZERO_SHOT": "Structural compatibility",
        "L3_FEW_SHOT": "examples of scientific reasoning",
        "L3_JUDGE_PROMPT": "Rate the following scientific explanation",
        "L4_ZERO_SHOT": "{context}",
        "L4_FEW_SHOT": "tested/untested compound-target pair",
        "L4_ANSWER_FORMAT": "tested' or 'untested'",
    }
    missing = [name for name, snippet in checks.items() if snippet not in text]
    if missing:
        print(f"  WARNING: Missing templates: {missing}")
    else:
        print("  All 12 templates included.")


if __name__ == "__main__":
    main()

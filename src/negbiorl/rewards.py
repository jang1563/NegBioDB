"""GRPO reward functions for trl GRPOTrainer.

Compatible with trl>=1.0.0 GRPOTrainer multi-reward API:
- Each function takes (completions, **kwargs) where kwargs come from dataset columns
- Returns list[float | None]; None = not applicable (skipped for that reward)
- Multiple reward functions combined via reward_weights

Usage with GRPOTrainer:
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[l1_reward_func, l4_reward_func, evidence_reward_func],
        reward_weights=[1.0, 1.0, 0.2],
        ...
    )
"""

from __future__ import annotations

import re
from functools import lru_cache

from negbiorl.data_registry import get_domain, get_l1_parser, parse_l4_unified


# ---------------------------------------------------------------------------
# Parser cache — avoids repeated importlib calls during training
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _cached_l1_parser(domain: str):
    return get_l1_parser(domain)


def _extract_text(comp) -> str:
    """Safely extract text from a trl completion (list of dicts or str)."""
    if isinstance(comp, list) and comp:
        return comp[0].get("content", "")
    if isinstance(comp, str):
        return comp
    return ""


# ---------------------------------------------------------------------------
# L1 Reward: Verifiable MCQ correctness
# ---------------------------------------------------------------------------

def l1_reward_func(
    completions: list[list[dict[str, str]]],
    gold_answer: list[str],
    task: list[str],
    domain: list[str],
    **kwargs,
) -> list[float | None]:
    """Binary reward for L1 multiple-choice correctness.

    Returns 1.0 if parsed answer matches gold, 0.0 if wrong, None for non-L1.
    """
    rewards: list[float | None] = []
    for comp, gold, t, d in zip(completions, gold_answer, task, domain):
        if t != "l1":
            rewards.append(None)
            continue
        # Extract text from chat completion format
        text = _extract_text(comp)
        parser = _cached_l1_parser(d)
        parsed = parser(text)
        if parsed is None:
            rewards.append(0.0)  # unparseable = wrong
        else:
            rewards.append(1.0 if parsed.upper() == gold.upper() else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# L4 Reward: Verifiable tested/untested discrimination
# ---------------------------------------------------------------------------

def l4_reward_func(
    completions: list[list[dict[str, str]]],
    gold_answer: list[str],
    task: list[str],
    domain: list[str],
    **kwargs,
) -> list[float | None]:
    """Binary reward for L4 tested/untested classification.

    Returns 1.0 if parsed answer matches gold, 0.0 if wrong, None for non-L4.
    """
    rewards: list[float | None] = []
    for comp, gold, t, d in zip(completions, gold_answer, task, domain):
        if t != "l4":
            rewards.append(None)
            continue
        text = _extract_text(comp)
        answer, _evidence = parse_l4_unified(text, d) if text else (None, None)
        if answer is None:
            rewards.append(0.0)
        else:
            rewards.append(1.0 if answer.lower() == gold.lower() else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Evidence Reward: Bonus for citing real experimental evidence
# ---------------------------------------------------------------------------

def evidence_reward_func(
    completions: list[list[dict[str, str]]],
    task: list[str],
    domain: list[str],
    **kwargs,
) -> list[float | None]:
    """Bonus reward for citing domain-specific experimental evidence.

    Only applied to L4 tasks (where evidence citation is meaningful).
    Returns 0.0–1.0 based on keyword density, None for non-L4.
    """
    rewards: list[float | None] = []
    for comp, t, d in zip(completions, task, domain):
        if t != "l4":
            rewards.append(None)
            continue
        text = _extract_text(comp)
        reg = get_domain(d)
        keywords = reg["evidence_keywords"]
        # Count distinct keyword matches (case-insensitive)
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        # Normalize: 0 keywords = 0.0, 3+ keywords = 1.0
        score = min(matches / 3.0, 1.0)
        rewards.append(score)
    return rewards


# ---------------------------------------------------------------------------
# Format Reward: Bonus for following expected response format
# ---------------------------------------------------------------------------

_L1_LETTER_RE = re.compile(r"\b([A-E])\b")
_L4_TESTED_RE = re.compile(r"\b(tested|untested)\b", re.IGNORECASE)


def format_reward_func(
    completions: list[list[dict[str, str]]],
    task: list[str],
    **kwargs,
) -> list[float | None]:
    """Small reward for following the expected response format.

    L1: starts with or contains a clear letter answer
    L4: contains "tested" or "untested" clearly
    """
    rewards: list[float | None] = []
    for comp, t in zip(completions, task):
        text = _extract_text(comp)
        if t == "l1":
            # Reward if response starts with or clearly contains a letter choice
            match = _L1_LETTER_RE.search(text[:50])
            rewards.append(0.5 if match else 0.0)
        elif t == "l4":
            match = _L4_TESTED_RE.search(text[:100])
            rewards.append(0.5 if match else 0.0)
        else:
            rewards.append(None)
    return rewards


# ---------------------------------------------------------------------------
# Convenience: default reward configuration
# ---------------------------------------------------------------------------

DEFAULT_REWARD_FUNCS = [l1_reward_func, l4_reward_func, evidence_reward_func, format_reward_func]
DEFAULT_REWARD_WEIGHTS = [1.0, 1.0, 0.2, 0.1]

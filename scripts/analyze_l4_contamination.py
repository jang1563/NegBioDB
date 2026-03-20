#!/usr/bin/env python3
"""Analyze L4 temporal accuracy for contamination detection.

Reads existing results.json files from results/llm/l4_*/ and reports
accuracy_pre_2023 vs accuracy_post_2024 gap per model/config.

A gap > 0.15 flags potential training data contamination.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "llm"
THRESHOLD = 0.15


def main():
    rows = []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("l4_"):
            continue
        results_file = run_dir / "results.json"
        if not results_file.exists():
            continue

        with open(results_file) as f:
            m = json.load(f)

        name = run_dir.name
        pre = m.get("accuracy_pre_2023")
        post = m.get("accuracy_post_2024")
        gap = round(pre - post, 4) if pre is not None and post is not None else None
        flag = gap > THRESHOLD if gap is not None else None

        rows.append({
            "run": name,
            "accuracy": m.get("accuracy"),
            "mcc": m.get("mcc"),
            "pre_2023": pre,
            "post_2024": post,
            "gap": gap,
            "flag": flag,
        })

    if not rows:
        print("No L4 results found.")
        return

    # Print table
    print(f"{'Run':<55} {'Acc':>5} {'MCC':>6} {'Pre23':>6} {'Post24':>6} {'Gap':>6} {'Flag'}")
    print("-" * 100)
    for r in rows:
        flag_str = "YES" if r["flag"] else ("NO" if r["flag"] is not None else "N/A")
        print(
            f"{r['run']:<55} "
            f"{r['accuracy']:>5.3f} "
            f"{r['mcc']:>6.3f} "
            f"{r['pre_2023']:>6.3f} "
            f"{r['post_2024']:>6.3f} "
            f"{r['gap']:>6.3f} "
            f"{flag_str}"
        )

    # Summary by model
    from collections import defaultdict
    model_gaps = defaultdict(list)
    for r in rows:
        # Extract model from run name
        parts = r["run"][3:]  # strip "l4_"
        if parts.endswith("_zero-shot"):
            model = parts[:-10]
        elif parts.endswith("_3-shot"):
            model = parts[:-7]
        else:
            model = parts
        model = model.rsplit("_fs", 1)[0]
        if r["gap"] is not None:
            model_gaps[model].append(r["gap"])

    print("\n--- Summary by Model ---")
    print(f"{'Model':<40} {'Mean Gap':>8} {'Flag'}")
    print("-" * 55)
    for model, gaps in sorted(model_gaps.items()):
        mean_gap = sum(gaps) / len(gaps)
        flag = "YES" if mean_gap > THRESHOLD else "NO"
        print(f"{model:<40} {mean_gap:>8.4f} {flag}")

    print(f"\nThreshold: {THRESHOLD}")
    flagged = sum(1 for r in rows if r["flag"])
    print(f"Flagged runs: {flagged}/{len(rows)}")


if __name__ == "__main__":
    main()

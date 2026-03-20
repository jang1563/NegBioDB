#!/usr/bin/env python3
"""Finalize L2 gold dataset from human-corrected annotations.

Reads the human-corrected l2_preannotated.jsonl (with manual corrections
applied), validates schema, assigns question IDs and splits, then writes
the final l2_gold.jsonl.

Usage:
    python scripts/finalize_l2_gold.py --input exports/llm_benchmarks/l2_corrected.jsonl
    python scripts/finalize_l2_gold.py --validate  # validate existing gold file

Input format (l2_corrected.jsonl):
    Each record should have fields corrected by human annotator:
    - abstract_text: str
    - negative_results: list[dict] with compound, target, activity_type, outcome
    - total_inactive_count: int
    - positive_results_mentioned: bool
    - search_category: str (explicit/hedged/implicit)
    - split: str (fewshot/val/test) — assigned during annotation
    - include: bool (True to include, False to exclude)

Output:
    exports/llm_benchmarks/l2_gold.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "exports" / "llm_benchmarks"
DEFAULT_INPUT = DATA_DIR / "l2_corrected.jsonl"
OUTPUT_FILE = DATA_DIR / "l2_gold.jsonl"

REQUIRED_FIELDS = [
    "abstract_text",
    "negative_results",
    "total_inactive_count",
    "positive_results_mentioned",
]
RESULT_REQUIRED_FIELDS = ["compound", "target", "activity_type", "outcome"]
VALID_SPLITS = {"fewshot", "val", "test"}
VALID_CATEGORIES = {"explicit", "hedged", "implicit"}


def validate_record(rec: dict, idx: int) -> list[str]:
    """Validate a single record. Returns list of error messages."""
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in rec:
            errors.append(f"[{idx}] Missing required field: {field}")

    # Validate negative_results
    nr = rec.get("negative_results", [])
    if not isinstance(nr, list):
        errors.append(f"[{idx}] negative_results must be a list")
    else:
        for j, result in enumerate(nr):
            for rf in RESULT_REQUIRED_FIELDS:
                if rf not in result:
                    errors.append(
                        f"[{idx}] negative_results[{j}] missing: {rf}"
                    )

    # Validate split
    split = rec.get("split")
    if split and split not in VALID_SPLITS:
        errors.append(f"[{idx}] Invalid split: {split} (expected {VALID_SPLITS})")

    # Validate search_category
    cat = rec.get("search_category")
    if cat and cat not in VALID_CATEGORIES:
        errors.append(f"[{idx}] Invalid category: {cat}")

    return errors


def validate_gold_file(path: Path) -> bool:
    """Validate an existing gold file."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))

    all_errors = []
    for i, rec in enumerate(records):
        all_errors.extend(validate_record(rec, i))

    # Check question_id uniqueness
    qids = [r.get("question_id") for r in records]
    dupes = [qid for qid, count in Counter(qids).items() if count > 1]
    if dupes:
        all_errors.append(f"Duplicate question_ids: {dupes}")

    # Stats
    split_counts = Counter(r.get("split") for r in records)
    cat_counts = Counter(r.get("search_category") for r in records)
    nr_counts = [len(r.get("negative_results", [])) for r in records]

    print(f"=== Gold file validation: {path.name} ===")
    print(f"  Total records: {len(records)}")
    print(f"  Split distribution: {dict(split_counts)}")
    print(f"  Category distribution: {dict(cat_counts)}")
    print(f"  Avg negative results/record: {sum(nr_counts)/len(nr_counts):.1f}")
    print(f"  Min/Max results: {min(nr_counts)}/{max(nr_counts)}")

    if all_errors:
        print(f"\n  ERRORS ({len(all_errors)}):")
        for err in all_errors[:20]:
            print(f"    {err}")
        return False
    else:
        print("\n  VALID: No errors found")
        return True


def main():
    parser = argparse.ArgumentParser(description="Finalize L2 gold dataset")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Human-corrected input file",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate existing gold file only",
    )
    args = parser.parse_args()

    if args.validate:
        if not OUTPUT_FILE.exists():
            print(f"Gold file not found: {OUTPUT_FILE}")
            sys.exit(1)
        valid = validate_gold_file(OUTPUT_FILE)
        sys.exit(0 if valid else 1)

    # Read corrected annotations
    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        print("Run preannotate_l2.py first, then manually correct the output.")
        sys.exit(1)

    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {args.input.name}")

    # Filter: only include=True (or if field missing, include all)
    included = [r for r in records if r.get("include", True)]
    excluded = len(records) - len(included)
    if excluded:
        print(f"  Excluded {excluded} records (include=False)")
    print(f"  Included: {len(included)}")

    # Validate
    all_errors = []
    for i, rec in enumerate(included):
        all_errors.extend(validate_record(rec, i))

    if all_errors:
        print(f"\nValidation errors ({len(all_errors)}):")
        for err in all_errors:
            print(f"  {err}")
        print("\nFix errors before finalizing.")
        sys.exit(1)

    # Assign question IDs (L2-0001 through L2-NNNN)
    gold_records = []
    for i, rec in enumerate(included):
        gold_rec = {
            "question_id": f"L2-{i + 1:04d}",
            "abstract_text": rec["abstract_text"],
            "negative_results": rec["negative_results"],
            "total_inactive_count": rec["total_inactive_count"],
            "positive_results_mentioned": rec["positive_results_mentioned"],
            "search_category": rec.get("search_category", "explicit"),
            "split": rec.get("split", "test"),
        }
        # Preserve optional metadata
        for key in ["pmid", "title", "year"]:
            if key in rec:
                gold_rec[key] = rec[key]
        gold_records.append(gold_rec)

    # Write gold file
    with open(OUTPUT_FILE, "w") as f:
        for rec in gold_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nWritten: {OUTPUT_FILE}")

    # Validate the output
    validate_gold_file(OUTPUT_FILE)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate HuggingFace dataset card and Croissant JSON-LD metadata.

Reads parquet/JSONL export files, extracts schema and statistics,
and generates:
  - exports/README.md  (HuggingFace dataset card)
  - exports/croissant.json  (MLCommons Croissant 1.0 metadata)

Usage:
    python scripts/generate_dataset_card.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = PROJECT_ROOT / "exports"

# Try importing pyarrow for parquet schema extraction
try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


def get_parquet_info(path: Path) -> dict:
    """Extract row count and schema from a parquet file."""
    if not HAS_PYARROW or not path.exists():
        return {"rows": "N/A", "columns": []}
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    return {
        "rows": pf.metadata.num_rows,
        "columns": [
            {"name": field.name, "type": str(field.type)}
            for field in schema
        ],
    }


def count_jsonl(path: Path) -> int:
    """Count lines in a JSONL file."""
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def format_size(path: Path) -> str:
    """Human-readable file size."""
    if not path.exists():
        return "N/A"
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def generate_readme() -> str:
    """Generate HuggingFace dataset card content."""
    sections = []

    # YAML frontmatter
    sections.append("""---
license: cc-by-sa-4.0
language:
- en
tags:
- drug-target-interaction
- clinical-trials
- protein-protein-interaction
- negative-results
- benchmark
- bioinformatics
task_categories:
- text-classification
- question-answering
- text-generation
size_categories:
- 10M<n<100M
---
""")

    sections.append("# NegBioDB: A Negative Results Database for Biomedical Sciences\n")
    sections.append(
        "NegBioDB is a large-scale database of experimentally confirmed negative "
        "results across three biomedical domains, paired with dual ML/LLM benchmarks "
        "for evaluating how well computational methods handle negative evidence.\n"
    )

    # Overview
    sections.append("## Overview\n")
    sections.append("| Domain | Negative Results | Entities | Benchmark Tasks |")
    sections.append("|--------|-----------------|----------|-----------------|")

    # DTI stats
    dti_pairs = EXPORTS_DIR / "negbiodb_dti_pairs.parquet"
    dti_info = get_parquet_info(dti_pairs)
    sections.append(
        f"| DTI (Drug-Target Interaction) | {dti_info.get('rows', '30.5M'):,} pairs "
        f"| 919K compounds, 3.7K targets | ML (M1) + LLM (L1-L4) |"
    )

    # CT stats
    ct_pairs = EXPORTS_DIR / "ct" / "negbiodb_ct_pairs.parquet"
    ct_info = get_parquet_info(ct_pairs)
    sections.append(
        f"| CT (Clinical Trial Failure) | {ct_info.get('rows', '132K'):,} results "
        f"| 216K trials, 176K interventions | ML (M1-M2) + LLM (L1-L4) |"
    )

    # PPI stats
    ppi_pairs = EXPORTS_DIR / "ppi" / "negbiodb_ppi_pairs.parquet"
    ppi_info = get_parquet_info(ppi_pairs)
    sections.append(
        f"| PPI (Protein-Protein Interaction) | {ppi_info.get('rows', '2.2M'):,} pairs "
        f"| 18.4K proteins | ML (M1) |"
    )
    sections.append("")

    # Data Sources
    sections.append("## Data Sources\n")
    sections.append("### DTI Domain")
    sections.append("| Source | License | Contribution |")
    sections.append("|--------|---------|-------------|")
    sections.append("| ChEMBL 34 | CC BY-SA 3.0 | Curated bioactivity data |")
    sections.append("| PubChem BioAssay | Public domain | HTS screening results |")
    sections.append("| BindingDB | CC BY-SA 3.0 | Binding measurements |")
    sections.append("| DAVIS | CC BY 4.0 | Kinase selectivity panel |")
    sections.append("")

    sections.append("### CT Domain")
    sections.append("| Source | License | Contribution |")
    sections.append("|--------|---------|-------------|")
    sections.append("| AACT (ClinicalTrials.gov) | Public domain | Trial metadata |")
    sections.append("| Open Targets | Apache 2.0 | Drug-target mappings |")
    sections.append("| CTO | MIT | Clinical trial outcomes |")
    sections.append("| Shi & Du 2024 | CC BY 4.0 | Safety/efficacy data |")
    sections.append("")

    sections.append("### PPI Domain")
    sections.append("| Source | License | Contribution |")
    sections.append("|--------|---------|-------------|")
    sections.append("| IntAct | CC BY 4.0 | Curated non-interactions |")
    sections.append("| HuRI | CC BY 4.0 | Y2H systematic negatives |")
    sections.append("| hu.MAP 3.0 | CC BY 4.0 | Complex-derived negatives |")
    sections.append("| STRING v12.0 | CC BY 4.0 | Zero-score protein pairs |")
    sections.append("")

    # File Structure
    sections.append("## File Structure\n")
    sections.append("### DTI Files")
    sections.append("| File | Size | Rows | Description |")
    sections.append("|------|------|------|-------------|")

    dti_files = [
        ("negbiodb_dti_pairs.parquet", "All negative DTI pairs with metadata"),
        ("negbiodb_m1_balanced.parquet", "M1 balanced dataset (1:1 pos:neg)"),
        ("negbiodb_m1_realistic.parquet", "M1 realistic dataset (1:10 pos:neg)"),
        ("negbiodb_m1_uniform_random.parquet", "Control: uniform random negatives"),
        ("negbiodb_m1_degree_matched.parquet", "Control: degree-matched negatives"),
        ("negbiodb_m1_balanced_ddb.parquet", "M1 balanced with degree-balanced split"),
    ]
    for fname, desc in dti_files:
        fpath = EXPORTS_DIR / fname
        info = get_parquet_info(fpath)
        sections.append(
            f"| `{fname}` | {format_size(fpath)} | {info.get('rows', 'N/A'):,} | {desc} |"
        )
    sections.append("")

    sections.append("### CT Files")
    sections.append("| File | Size | Rows | Description |")
    sections.append("|------|------|------|-------------|")
    ct_files = [
        ("ct/negbiodb_ct_pairs.parquet", "All CT failure pairs"),
        ("ct/negbiodb_ct_m1_balanced.parquet", "CT-M1 balanced (success vs failure)"),
        ("ct/negbiodb_ct_m1_realistic.parquet", "CT-M1 realistic ratio"),
        ("ct/negbiodb_ct_m2.parquet", "CT-M2 multiclass failure category"),
    ]
    for fname, desc in ct_files:
        fpath = EXPORTS_DIR / fname
        info = get_parquet_info(fpath)
        rows = info.get("rows", "N/A")
        rows_str = f"{rows:,}" if isinstance(rows, int) else rows
        sections.append(f"| `{fname}` | {format_size(fpath)} | {rows_str} | {desc} |")
    sections.append("")

    sections.append("### PPI Files")
    sections.append("| File | Size | Rows | Description |")
    sections.append("|------|------|------|-------------|")
    ppi_files = [
        ("ppi/negbiodb_ppi_pairs.parquet", "All negative PPI pairs"),
        ("ppi/ppi_m1_balanced.parquet", "PPI-M1 balanced (1:1 pos:neg)"),
        ("ppi/ppi_m1_realistic.parquet", "PPI-M1 realistic (1:10 ratio)"),
    ]
    for fname, desc in ppi_files:
        fpath = EXPORTS_DIR / fname
        info = get_parquet_info(fpath)
        rows = info.get("rows", "N/A")
        rows_str = f"{rows:,}" if isinstance(rows, int) else rows
        sections.append(f"| `{fname}` | {format_size(fpath)} | {rows_str} | {desc} |")
    sections.append("")

    sections.append("### LLM Benchmark Files (DTI)")
    sections.append("| File | Items | Description |")
    sections.append("|------|-------|-------------|")
    llm_files = [
        ("llm_benchmarks/l1_mcq.jsonl", "L1: 4-class activity MCQ"),
        ("llm_benchmarks/l3_reasoning_pilot.jsonl", "L3: Scientific reasoning (pilot)"),
        ("llm_benchmarks/l4_tested_untested.jsonl", "L4: Tested vs untested discrimination"),
    ]
    for fname, desc in llm_files:
        fpath = EXPORTS_DIR / fname
        n = count_jsonl(fpath)
        sections.append(f"| `{fname}` | {n:,} | {desc} |")
    sections.append("")

    # Benchmark Tasks
    sections.append("## Benchmark Tasks\n")
    sections.append("### ML Benchmarks")
    sections.append("| Task | Domain | Type | Splits |")
    sections.append("|------|--------|------|--------|")
    sections.append("| M1 | DTI | Binary (active/inactive) | random, cold_compound, cold_target, degree_balanced |")
    sections.append("| CT-M1 | CT | Binary (success/failure) | random, cold_drug, cold_condition, temporal, scaffold, cold_both |")
    sections.append("| CT-M2 | CT | 7-way failure category | Same as CT-M1 |")
    sections.append("| PPI-M1 | PPI | Binary (interact/non-interact) | random, cold_protein, cold_both, degree_balanced |")
    sections.append("")

    sections.append("### LLM Benchmarks (DTI)")
    sections.append("| Task | Type | Size | Description |")
    sections.append("|------|------|------|-------------|")
    sections.append("| L1 | MCQ classification | 1,600 | 4-class activity level |")
    sections.append("| L2 | Structured extraction | ~100 | Extract results from abstracts |")
    sections.append("| L3 | Reasoning | 50 | Explain compound-target inactivity |")
    sections.append("| L4 | Discrimination | 400 | Tested vs untested pair |")
    sections.append("")

    # License
    sections.append("## License\n")
    sections.append(
        "This dataset is released under **CC BY-SA 4.0**, due to the viral "
        "clause of ChEMBL's CC BY-SA 3.0 license. See the LICENSE file for details.\n"
    )

    # Citation
    sections.append("## Citation\n")
    sections.append("```bibtex")
    sections.append("@dataset{negbiodb2026,")
    sections.append("  title={NegBioDB: A Negative Results Database for Biomedical Sciences},")
    sections.append("  author={Jang, Jungwon},")
    sections.append("  year={2026},")
    sections.append("  url={https://github.com/jang1563/NegBioDB}")
    sections.append("}")
    sections.append("```")

    return "\n".join(sections)


def generate_croissant() -> dict:
    """Generate MLCommons Croissant JSON-LD metadata."""
    croissant = {
        "@context": {
            "@vocab": "https://schema.org/",
            "sc": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "rai": "http://mlcommons.org/croissant/RAI/",
        },
        "@type": "sc:Dataset",
        "name": "NegBioDB",
        "description": (
            "A large-scale database of experimentally confirmed negative results "
            "across three biomedical domains (DTI, Clinical Trials, PPI), "
            "with dual ML/LLM benchmarks."
        ),
        "license": "https://creativecommons.org/licenses/by-sa/4.0/",
        "url": "https://github.com/jang1563/NegBioDB",
        "version": "1.0.0",
        "datePublished": "2026",
        "creator": {
            "@type": "sc:Person",
            "name": "Jungwon Jang",
        },
        "distribution": [],
        "recordSet": [],
    }

    # File objects (distribution)
    file_defs = [
        {
            "name": "dti_pairs",
            "contentUrl": "negbiodb_dti_pairs.parquet",
            "encodingFormat": "application/x-parquet",
            "description": "All negative DTI pairs with source, tier, and activity data",
        },
        {
            "name": "dti_m1_balanced",
            "contentUrl": "negbiodb_m1_balanced.parquet",
            "encodingFormat": "application/x-parquet",
            "description": "DTI M1 balanced benchmark dataset (1:1 positive:negative)",
        },
        {
            "name": "ct_pairs",
            "contentUrl": "ct/negbiodb_ct_pairs.parquet",
            "encodingFormat": "application/x-parquet",
            "description": "All clinical trial failure pairs",
        },
        {
            "name": "ppi_pairs",
            "contentUrl": "ppi/negbiodb_ppi_pairs.parquet",
            "encodingFormat": "application/x-parquet",
            "description": "All negative PPI pairs",
        },
        {
            "name": "llm_l1",
            "contentUrl": "llm_benchmarks/l1_mcq.jsonl",
            "encodingFormat": "application/jsonl",
            "description": "L1 MCQ classification benchmark",
        },
        {
            "name": "llm_l4",
            "contentUrl": "llm_benchmarks/l4_tested_untested.jsonl",
            "encodingFormat": "application/jsonl",
            "description": "L4 tested/untested discrimination benchmark",
        },
    ]

    for fd in file_defs:
        croissant["distribution"].append({
            "@type": "cr:FileObject",
            "name": fd["name"],
            "contentUrl": fd["contentUrl"],
            "encodingFormat": fd["encodingFormat"],
            "description": fd["description"],
        })

    # Record sets (key columns)
    record_defs = [
        {
            "name": "dti_pairs_record",
            "source": "dti_pairs",
            "fields": [
                {"name": "inchikey_connectivity", "dataType": "sc:Text"},
                {"name": "uniprot_id", "dataType": "sc:Text"},
                {"name": "activity_type", "dataType": "sc:Text"},
                {"name": "pchembl_value", "dataType": "sc:Float"},
                {"name": "source", "dataType": "sc:Text"},
                {"name": "tier", "dataType": "sc:Text"},
            ],
        },
        {
            "name": "ct_pairs_record",
            "source": "ct_pairs",
            "fields": [
                {"name": "nct_id", "dataType": "sc:Text"},
                {"name": "intervention_name", "dataType": "sc:Text"},
                {"name": "failure_category", "dataType": "sc:Text"},
                {"name": "tier", "dataType": "sc:Text"},
                {"name": "highest_phase_reached", "dataType": "sc:Text"},
            ],
        },
        {
            "name": "ppi_pairs_record",
            "source": "ppi_pairs",
            "fields": [
                {"name": "protein_a", "dataType": "sc:Text"},
                {"name": "protein_b", "dataType": "sc:Text"},
                {"name": "source", "dataType": "sc:Text"},
                {"name": "tier", "dataType": "sc:Text"},
            ],
        },
    ]

    for rd in record_defs:
        croissant["recordSet"].append({
            "@type": "cr:RecordSet",
            "name": rd["name"],
            "source": rd["source"],
            "field": [
                {
                    "@type": "cr:Field",
                    "name": f["name"],
                    "dataType": f["dataType"],
                    "description": f["name"].replace("_", " "),
                }
                for f in rd["fields"]
            ],
        })

    return croissant


def main():
    # Generate README
    readme_text = generate_readme()
    readme_path = EXPORTS_DIR / "README.md"
    readme_path.write_text(readme_text)
    print(f"Written: {readme_path}")
    print(f"  Lines: {len(readme_text.splitlines())}")

    # Generate Croissant
    croissant = generate_croissant()
    croissant_path = EXPORTS_DIR / "croissant.json"
    with open(croissant_path, "w") as f:
        json.dump(croissant, f, indent=2)
    print(f"Written: {croissant_path}")

    # Verify: check all referenced files exist
    missing = []
    for dist in croissant["distribution"]:
        fpath = EXPORTS_DIR / dist["contentUrl"]
        if not fpath.exists():
            missing.append(dist["contentUrl"])
    if missing:
        print(f"\n  WARNING: {len(missing)} referenced files not found:")
        for m in missing:
            print(f"    {m}")
    else:
        print("\n  All referenced files exist.")


if __name__ == "__main__":
    main()

# NegBioDB

**A Negative-Results Database and Dual ML/LLM Benchmark for Biomedical Sciences**

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗-Datasets-yellow.svg)](https://huggingface.co/datasets/jang1563/NegBioDB)

An estimated 90% of biomedical experiments produce null or inconclusive findings, yet the overwhelming majority remain unpublished. **NegBioDB** systematically aggregates experimentally confirmed *negative* results across five biomedical domains and pairs them with a dual-track benchmark — traditional ML prediction and modern LLM reasoning — that quantifies how publication bias propagates into AI systems.

---

## Contents

- [Overview](#overview)
- [Domains at a glance](#domains-at-a-glance)
- [Key findings](#key-findings)
- [Quick start](#quick-start)
- [Per-domain pipelines](#per-domain-pipelines)
- [Benchmark protocol](#benchmark-protocol)
- [Datasets and downloads](#datasets-and-downloads)
- [Repository layout](#repository-layout)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

## Overview

NegBioDB unifies negative experimental evidence from 17 primary sources into five domain-specific SQLite databases under a common schema (entities → studies → results → tiered confidence). Every record carries source provenance, confidence tier, and the splits required to reproduce held-out evaluations.

**Highlights**

- Five biomedical domains: **DTI** (drug–target), **CT** (clinical trial failure), **PPI** (protein–protein), **GE** (gene essentiality / DepMap), **VP** (variant pathogenicity).
- ~63 million confirmed negative results aggregated from 17 primary sources.
- Dual benchmark: an **ML track** (4–7 splits per domain incl. cold-entity, temporal, degree-balanced) and an **LLM track** (four reasoning levels L1–L4).
- 380 ML runs + 401 LLM runs reported, evaluated under a shared rubric (LogAUC / BEDROC / EF / AUROC / AUPRC / MCC for ML; field-level + judge-graded scores for LLM).
- Reproducible from raw downloads through evaluation: config-driven ETL, deterministic splits, SLURM/HPC support.

---

## Domains at a glance

| Domain | Negative results | Key entities | Sources | DB size | ML runs | LLM runs |
|--------|------------------|--------------|---------|---------|--------:|---------:|
| **DTI** | 30,459,583 | 919K compounds, 3.7K targets | ChEMBL, PubChem, BindingDB, DAVIS | ~21 GB | 24 / 24 | 81 / 81 |
| **CT**  | 132,925     | 177K interventions, 56K conditions | AACT, CTO, Open Targets, Shi & Du | ~500 MB | 108 / 108 | 80 / 80 |
| **PPI** | 2,229,670   | 18.4K proteins | IntAct, HuRI, hu.MAP 3.0, STRING | 849 MB | 54 / 54 | 80 / 80 |
| **GE**  | 28,759,256  | 19,554 genes, 2,132 cell lines | DepMap CRISPR + RNAi | ~16 GB | 42 / 42 | 80 / 80 |
| **VP**  | 2,442,718   | 2.43M variants, 18.4K genes, 10K diseases | ClinVar, gnomAD, ClinGen, CADD/REVEL/AlphaMissense | ~1.5 GB | 72 / 72 | 20 / 20 |
| **Total** | **~64.0M** | | **17 sources** | **~39.5 GB** | **300** | **341** |

PPI export rows after split filtering: 2,220,786. VP M1 balanced export: 1,255,150 rows.

---

## Key findings

### ML track — the choice of negatives shapes every metric

- **DTI.** Degree-matched negatives inflate LogAUC by +0.112 on average versus NegBioDB negatives, while cold-target splits collapse LogAUC to 0.15–0.33. AUROC remains misleadingly high (0.76–0.89), masking the failure.
- **CT.** Confirmed-failure negatives are trivially separable for binary tasks (AUROC ≈ 1.0). The 7-way failure-mode classification (CT-M2) remains hard (best macro-F1 = 0.51).
- **PPI.** PIPR cold-both AUROC drops below random (0.409); MLP-on-features stays robust (0.950) thanks to hand-crafted descriptors — a textbook representation-vs-generalization trade-off.
- **GE.** Cold-gene splits expose generalization gaps invisible under random splits; degree-balanced negatives modestly improve ranking metrics.
- **VP.** Random splits saturate (XGBoost AUROC 0.995 / MCC 0.932); cold-disease splits expose a calibration failure where AUROC stays high but MCC = 0.0.

| DTI model | Random (NegBioDB) | Random (degree-matched) | Cold-target |
|-----------|------------------:|------------------------:|------------:|
| DeepDTA   | 0.833 | **0.919** | 0.325 |
| GraphDTA  | 0.843 | **0.967** | 0.241 |
| DrugBAN   | 0.830 | **0.955** | 0.151 |

### LLM track — L4 (tested vs. untested) is where models actually differ

L4 forces the model to decide whether a (compound, target) — or analogous pair — has been *experimentally tested as negative* or simply *never tested*. It is the most discriminating level and the most diagnostic of memorization vs. reasoning.

| Domain | L4 MCC range | Interpretation | Memorization signal |
|--------|--------------|----------------|---------------------|
| DTI | ≤ 0.18  | Near random | Not detected |
| GE  | ≤ 0.22  | Near random | Not detected |
| PPI | 0.33–0.44 | Moderate | **Yes** — pre-2015 pairs identified at 59–79%; post-2020 pairs at 7–25% |
| CT  | 0.48–0.56 | Meaningful  | Not detected |
| VP  | Single-class test (data limit) | n/a | n/a |

Across domains, **L3** (open-ended scientific reasoning, judge-graded) shows zero-shot ≫ few-shot for most models — providing exemplars *degrades* reasoning quality, a robust pattern across PPI / GE / DC / CP / VP.

Full per-model tables: [`experiment_results.md`](experiment_results.md).

---

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/jang1563/NegBioDB.git
cd NegBioDB
make setup          # create venv, install dependencies
make db             # initialize SQLite database (DTI core)

# Download + populate one domain end-to-end (DTI shown):
make download       # ChEMBL, PubChem, BindingDB, DAVIS
make load-all
uv run python scripts/export_ml_dataset.py
```

Other domains follow the same `download → load → export` pattern (see below).
Tests:

```bash
PYTHONPATH=src uv run pytest tests/ -v
```

---

## Per-domain pipelines

### DTI — Drug–Target Interaction

```bash
make download                                     # 4 raw sources
make load-all                                     # standardize + ETL
uv run python scripts/export_ml_dataset.py        # Parquet exports, 5 splits
uv run python scripts/build_l1_dataset.py         # LLM L1 dataset (others: l2/l3/l4)
uv run python scripts/run_llm_benchmark.py --model gemini --level l1 --config zeroshot
```

### CT — Clinical Trial Failure

```bash
# AACT URL changes monthly; check https://aact.ctti-clinicaltrials.org/snapshots
uv run python scripts_ct/download_aact.py --url <AACT_URL>
uv run python scripts_ct/download_cto.py
uv run python scripts_ct/download_opentargets.py
uv run python scripts_ct/download_shi_du.py

uv run python scripts_ct/load_aact.py
uv run python scripts_ct/classify_failures.py    # 3-tier failure classification
uv run python scripts_ct/resolve_drugs.py         # 4-step drug-name resolution
uv run python scripts_ct/load_outcomes.py         # p-values, SAE enrichment
uv run python scripts_ct/export_ct_ml_dataset.py
```

### PPI — Protein–Protein Interaction

```bash
uv run python scripts_ppi/download_intact.py
uv run python scripts_ppi/download_huri.py
uv run python scripts_ppi/download_humap.py
uv run python scripts_ppi/download_string.py

uv run python scripts_ppi/load_intact.py          # IntAct PSI-MI TAB 2.7
uv run python scripts_ppi/load_huri.py            # Y2H systematic negatives
uv run python scripts_ppi/load_humap.py           # ML-derived non-complex pairs
uv run python scripts_ppi/load_string.py          # zero-evidence pairs
uv run python scripts_ppi/fetch_sequences.py      # UniProt sequences
uv run python scripts_ppi/export_ppi_ml_dataset.py
```

### GE — Gene Essentiality (DepMap)

```bash
uv run python scripts_depmap/download_depmap.py   # CRISPR (Chronos) + RNAi (DEMETER2)
uv run python scripts_depmap/load_depmap.py
uv run python scripts_depmap/load_rnai.py
uv run python scripts_depmap/fetch_gene_descriptions.py
uv run python scripts_depmap/export_ge_ml_dataset.py
```

### VP — Variant Pathogenicity

```bash
PYTHONPATH=src uv run python scripts_vp/download_clinvar.py
PYTHONPATH=src uv run python scripts_vp/download_gnomad.py
PYTHONPATH=src uv run python scripts_vp/download_clingen.py

PYTHONPATH=src uv run python scripts_vp/load_clinvar.py
PYTHONPATH=src uv run python scripts_vp/load_gnomad.py \
    --constraint data/vp/gnomad/gnomad.v4.1.constraint_metrics.tsv
PYTHONPATH=src uv run python scripts_vp/load_clingen.py \
    --csv data/vp/clingen_gene_disease_validity.csv

# Optional functional scores (CADD, REVEL, AlphaMissense): see slurm/run_vp_*.slurm
```

ML training, LLM inference, and result-collection commands for every domain are listed in the [Reproducibility](#reproducibility) section below.

---

## Benchmark protocol

### ML track

| Task ID | Domain | Type | Splits |
|---------|--------|------|--------|
| **M1**  | DTI | Binary (active / inactive) | random, cold_compound, cold_target, degree_balanced |
| **CT-M1** | CT | Binary (success / failure) | random, cold_drug, cold_condition, temporal, scaffold, cold_both |
| **CT-M2** | CT | 7-way failure category | same as CT-M1 |
| **PPI-M1** | PPI | Binary (interact / non-interact) | random, cold_protein, cold_both, degree_balanced |
| **GE-M1** | GE | Binary (essential / non-essential) | random, cold_gene, cold_cell_line, cold_both, degree_balanced |
| **VP-M1** | VP | Binary (pathogenic / benign) | random, cold_gene, cold_disease, temporal |

Metrics: **LogAUC[0.001,0.1]** (primary, early-enrichment), **BEDROC (α=20)**, **EF@1% / EF@5%**, **AUPRC**, **MCC**, **AUROC**.

### LLM track — four reasoning levels

| Level | Question | Evaluation |
|-------|----------|------------|
| **L1** | Multiple-choice: which of these is *not* a known interaction? | accuracy, MCC |
| **L2** | Structured extraction: parse a result statement into a typed schema | field-level F1, schema compliance |
| **L3** | Open-ended scientific reasoning: explain a negative finding | LLM-judge rubric (1–5 across 4–6 axes) |
| **L4** | Tested-vs-untested discrimination on real (entity, entity) pairs | MCC; contamination-flag analysis |

Configurations: zero-shot and 3-shot for every level, every model.
Models evaluated: GPT-4o-mini, Claude Haiku 4.5, Gemini 2.5 Flash, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct.

Prompt templates and judge rubrics: [`docs/appendix_prompts.md`](docs/appendix_prompts.md). Full methodology: [`docs/methodology_notes.md`](docs/methodology_notes.md).

---

## Datasets and downloads

Pre-built ML and LLM datasets are mirrored on the Hugging Face Hub:
👉 **[huggingface.co/datasets/jang1563/NegBioDB](https://huggingface.co/datasets/jang1563/NegBioDB)**

### DTI exports

| File | Description |
|------|-------------|
| `negbiodb_dti_pairs.parquet` | 1.7M compound–target pairs with 5 split columns |
| `negbiodb_m1_balanced.parquet` | M1 balanced (1:1), 1.73M rows |
| `negbiodb_m1_realistic.parquet` | M1 realistic (1:10), 9.49M rows |
| `negbiodb_m1_balanced_ddb.parquet` | Degree-balanced split |
| `negbiodb_m1_uniform_random.parquet` | Control: uniform-random negatives |
| `negbiodb_m1_degree_matched.parquet` | Control: degree-matched negatives |
| `chembl_positives_pchembl6.parquet` | 863K ChEMBL actives (pChEMBL ≥ 6) |

### CT exports (`exports/ct/`)

| File | Rows | Description |
|------|------|-------------|
| `negbiodb_ct_pairs.parquet` | 102,850 | All failure pairs, 6 splits |
| `negbiodb_ct_m1_balanced.parquet` | 11,222 | Binary, 1:1 |
| `negbiodb_ct_m1_realistic.parquet` | 36,957 | Binary, ~1:6 |
| `negbiodb_ct_m1_smiles_only.parquet` | 3,878 | SMILES-resolved subset |
| `negbiodb_ct_m2.parquet` | 112,298 | 7-way failure category |

### PPI exports (`exports/ppi/`)

| File | Rows | Description |
|------|------|-------------|
| `negbiodb_ppi_pairs.parquet` | 2,220,786 | All negative pairs, 4 splits |
| `ppi_m1_balanced.parquet` | 123,456 | M1 (1:1) |
| `ppi_m1_realistic.parquet` | 679,008 | M1 (1:10) |
| `ppi_m1_balanced_ddb.parquet` | — | Degree-balanced split |
| `ppi_m1_uniform_random.parquet` | — | Control |
| `ppi_m1_degree_matched.parquet` | — | Control |

### GE exports (`exports/ge/`)

| File | Description |
|------|-------------|
| `negbiodb_ge_pairs.parquet` | 22.5M gene–cell-line pairs, 5 split columns (~770 MB) |
| `ge_gene_aggregates.parquet` | Per-gene aggregated essentiality features |

### VP exports (`exports/vp_ml/`)

| File | Rows | Description |
|------|------|-------------|
| `vp_m1_balanced.parquet` | 1,255,150 | M1 balanced (gold/silver positives, 1:1) |
| `vp_m1_realistic.parquet` | 2,442,718 | M1 realistic (full negative set) |

---

## Repository layout

```
NegBioDB/
├── src/
│   ├── negbiodb/            # DTI core library (db, ETL, splits, ML, LLM client)
│   ├── negbiodb_ct/         # CT domain (failure classification, drug resolver, M2)
│   ├── negbiodb_ppi/        # PPI domain (IntAct/HuRI/hu.MAP/STRING, sequence models)
│   ├── negbiodb_depmap/     # GE domain (DepMap CRISPR + RNAi)
│   └── negbiodb_vp/         # VP domain (ClinVar, gnomAD, ClinGen, functional scores)
├── scripts/                 # DTI CLI entry points
├── scripts_ct/              # CT CLI entry points
├── scripts_ppi/             # PPI CLI entry points
├── scripts_depmap/          # GE CLI entry points
├── scripts_vp/              # VP CLI entry points
├── slurm/                   # SLURM job scripts (path-agnostic)
├── migrations*/             # SQL schema migrations per domain
├── tests/                   # ~1,000 unit / integration tests
├── docs/                    # Methodology notes & prompt appendices
├── paper/                   # LaTeX source
├── exports/                 # Parquet exports (mirrored on Hugging Face)
├── results/                 # Run-level metrics (per-seed, per-config)
├── experiment_results.md    # Aggregated ML/LLM result tables
├── PROJECT_OVERVIEW.md      # Long-form project overview
└── ROADMAP.md               # Execution roadmap
```

---

## Reproducibility

### ML training (per domain)

```bash
# DTI
uv run python scripts/train_baseline.py --model deepdta --split random \
    --negative negbiodb --dataset balanced
bash slurm/submit_all.sh

# CT
uv run python scripts_ct/train_ct_baseline.py --model xgboost --task m1 \
    --split random --negative negbiodb
bash slurm/submit_ct_all.sh

# PPI
uv run python scripts_ppi/train_baseline.py --model siamese_cnn --split random \
    --negative negbiodb --dataset balanced
bash slurm/submit_ppi_all.sh

# GE
uv run python scripts_depmap/train_ge_baseline.py --model xgboost --split random \
    --negative negbiodb
bash slurm/submit_ge_ml_all.sh

# VP
uv run python scripts_vp/train_vp_baseline.py --model xgboost --split random \
    --negative negbiodb
bash slurm/submit_vp_ml_all.sh
```

### LLM benchmark (per domain)

```bash
# Build datasets (DTI shown; per-domain analogues in scripts_ct/, _ppi/, _depmap/, _vp/)
uv run python scripts/build_l1_dataset.py
uv run python scripts/build_l2_dataset.py
uv run python scripts/build_l3_dataset.py
uv run python scripts/build_l4_dataset.py

# Run inference
uv run python scripts/run_llm_benchmark.py --model gemini --level l1 --config zeroshot

# Collect aggregated results
uv run python scripts/collect_llm_results.py
uv run python scripts_ct/collect_ct_llm_results.py
uv run python scripts_ppi/collect_ppi_llm_results.py
uv run python scripts_depmap/collect_ge_results.py --llm
uv run python scripts_vp/collect_vp_llm_results.py
```

### Tests

```bash
PYTHONPATH=src uv run pytest tests/ -v                                # all
PYTHONPATH=src uv run pytest tests/ -v -m "not integration"           # skip network
PYTHONPATH=src uv run pytest tests/test_ct_*.py -v                    # one domain
```

### Determinism

- Every split is generated from a fixed seed (`42`, `43`, `44` for ML).
- ETL is config-driven (`config.yaml`); raw downloads are checksum-verified.
- LLM inference logs prompt hashes alongside model output for verifiable replay.

---

## Citation

```bibtex
@misc{negbiodb2026,
  title  = {NegBioDB: A Negative-Results Database and Dual ML/LLM Benchmark
            for Biomedical Sciences},
  author = {Kim, JangKeun},
  year   = {2026},
  url    = {https://github.com/jang1563/NegBioDB}
}
```

---

## License

**CC BY-SA 4.0** — see [LICENSE](LICENSE).

This license is required by the viral clause of ChEMBL's CC BY-SA 3.0. All redistributed source data retain their original licenses; per-source attribution is provided in [`docs/methodology_notes.md`](docs/methodology_notes.md).

### Source licenses

| Domain | Sources |
|--------|---------|
| DTI | ChEMBL v36 (CC BY-SA 3.0), PubChem BioAssay (Public Domain), BindingDB (CC BY 3.0), DAVIS (Public) |
| CT  | AACT / ClinicalTrials.gov (Public Domain), CTO (MIT), Open Targets (Apache 2.0), Shi & Du 2024 (CC BY 4.0) |
| PPI | IntAct (CC BY 4.0), HuRI (CC BY 4.0), hu.MAP 3.0 (MIT), STRING v12.0 (CC BY 4.0) |
| GE  | DepMap CRISPR (CC BY 4.0), DepMap RNAi (CC BY 4.0) |
| VP  | ClinVar (Public Domain), gnomAD (CC0), ClinGen (CC0), CADD (free non-commercial), REVEL (free), AlphaMissense (CC BY-NC-SA 4.0 — non-commercial) |

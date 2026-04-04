# NegBioDB

**Negative Results Database & Dual ML/LLM Benchmark for Biomedical Sciences**

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Approximately 90% of scientific experiments produce null or inconclusive results, yet the vast majority remain unpublished. NegBioDB systematically collects experimentally confirmed negative results across five biomedical domains and provides dual-track ML + LLM benchmarks to quantify the impact of this publication bias on AI models.

## Key Features

- **Five domains**: Drug-Target Interaction (DTI), Clinical Trial Failure (CT), Protein-Protein Interaction (PPI), Gene Essentiality (GE/DepMap), Variant Pathogenicity (VP)
- **~63.3M negative results** across 5 SQLite databases (30.5M DTI + 133K CT + 2.2M PPI + 28.8M GE + 1.68M VP core)
- **Dual benchmark**: ML track (traditional prediction) + LLM track (biomedical NLP tasks)
- **242 ML runs** + **321 LLM runs** completed across all domains
- **Multiple split strategies**: random, cold-entity, temporal, scaffold, degree-balanced
- **Reproducible pipeline**: SQLite databases, config-driven ETL, SLURM/HPC support
- **Standardized evaluation**: 7 ML metrics (LogAUC, BEDROC, EF, AUROC, AUPRC, MCC) + LLM rubrics

## Database Statistics

| Domain | Negative Results | Key Entities | Sources | DB Size |
|--------|-----------------|--------------|---------|---------|
| **DTI** | 30,459,583 | 919K compounds, 3.7K targets | ChEMBL, PubChem, BindingDB, DAVIS | ~21 GB |
| **CT** | 132,925 | 177K interventions, 56K conditions | AACT, CTO, Open Targets, Shi & Du | ~500 MB |
| **PPI** | 2,229,670 | 18.4K proteins | IntAct, HuRI, hu.MAP, STRING | 849 MB |
| **GE** | 28,759,256 | 19,554 genes, 2,132 cell lines | DepMap (CRISPR, RNAi) | ~16 GB |
| **VP** | 1,677,904 | 1.49M variants, 18.4K genes, 10K diseases | ClinVar, gnomAD, ClinGen | ~1.5 GB |
| **Total** | **~63.3M** | | **17 sources** | **~39.5 GB** |

*PPI DB total: 2,229,670; export rows after split filtering: 2,220,786.*

## Project Status

| Domain | ETL | ML Benchmark | LLM Benchmark | Status |
|--------|-----|-------------|---------------|--------|
| DTI | 4 sources | 24/24 runs | 81/81 runs | Complete |
| CT | 4 sources | 108/108 runs | 80/80 runs | Complete |
| PPI | 4 sources | 54/54 runs | 80/80 runs | Complete |
| GE | 2 sources | 14/14 runs (seed 42) | 64/80 runs* | Seed 42 ML complete, LLM 4/5 models |
| VP | 3 local sources loaded; HPC extras pending | Code complete, runs pending | Code complete, runs pending | Core ETL complete; gnomAD sites + scores + HPC model runs pending |

*Llama 3.1-8B results pending HPC GPU availability; seeds 43/44 in progress.

---

## Key Findings

### ML: Negative Source Matters

**DTI** — Degree-matched negatives inflate LogAUC by +0.112 on average. Cold-target splits cause catastrophic failure (LogAUC 0.15–0.33), while AUROC misleadingly stays 0.76–0.89.

| DTI Model | Random (NegBioDB) | Random (Degree-Matched) | Cold-Target |
|-----------|------------------|------------------------|-------------|
| DeepDTA | 0.833 | **0.919** | 0.325 |
| GraphDTA | 0.843 | **0.967** | 0.241 |
| DrugBAN | 0.830 | **0.955** | 0.151 |

**PPI** — PIPR cold_both AUROC drops to 0.409 (below random); MLPFeatures remains robust at 0.950 due to hand-crafted features.

**CT** — NegBioDB negatives are trivially separable (AUROC ~1.0); M2 7-way classification is challenging (best macro-F1 = 0.51).

**GE** — Cold-gene splits reveal severe generalization gaps; degree-balanced negatives modestly improve ranking metrics over random negatives.

### LLM: L4 Discrimination Reveals Domain Differences

| Domain | L4 MCC Range | Interpretation | Contamination |
|--------|-------------|----------------|---------------|
| DTI | ≤ 0.18 | Near random | Not detected |
| PPI | 0.33–0.44 | Moderate | **Yes** (temporal gap) |
| CT | 0.48–0.56 | Meaningful | Not detected |
| GE | Pending full run | — | — |

PPI L4 reveals **temporal contamination**: pre-2015 interaction data is identified at 59–79% accuracy, while post-2020 data drops to 7–25%. LLMs rely on memorized training data, not biological reasoning.

---

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/jang1563/NegBioDB.git
cd NegBioDB
make setup          # Create venv and install dependencies
make db             # Initialize SQLite database
```

## Data Pipeline

### DTI Domain

```bash
make download       # Download all 4 sources (ChEMBL, PubChem, BindingDB, DAVIS)
make load-all       # Run all ETL loaders
uv run python scripts/export_ml_dataset.py   # Export ML datasets
```

### CT Domain

```bash
# Download sources (AACT URL changes monthly)
uv run python scripts_ct/download_aact.py --url <AACT_URL>
uv run python scripts_ct/download_cto.py
uv run python scripts_ct/download_opentargets.py
uv run python scripts_ct/download_shi_du.py

# Load and process
uv run python scripts_ct/load_aact.py
uv run python scripts_ct/classify_failures.py
uv run python scripts_ct/resolve_drugs.py
uv run python scripts_ct/load_outcomes.py
uv run python scripts_ct/export_ct_ml_dataset.py
```

### PPI Domain

```bash
# Download sources
uv run python scripts_ppi/download_intact.py
uv run python scripts_ppi/download_huri.py
uv run python scripts_ppi/download_humap.py
uv run python scripts_ppi/download_string.py

# Load and process
uv run python scripts_ppi/load_intact.py
uv run python scripts_ppi/load_huri.py
uv run python scripts_ppi/load_humap.py
uv run python scripts_ppi/load_string.py
uv run python scripts_ppi/fetch_sequences.py
uv run python scripts_ppi/export_ppi_ml_dataset.py
```

### GE Domain (DepMap)

```bash
# Download DepMap CRISPR and RNAi screens
uv run python scripts_depmap/download_depmap.py

# Load and process
uv run python scripts_depmap/load_depmap.py
uv run python scripts_depmap/load_rnai.py
uv run python scripts_depmap/fetch_gene_descriptions.py
uv run python scripts_depmap/export_ge_ml_dataset.py
```

### VP Domain (Variant Pathogenicity)

```bash
# Local sources
PYTHONPATH=src uv run python scripts_vp/download_clinvar.py
PYTHONPATH=src uv run python scripts_vp/download_gnomad.py
PYTHONPATH=src uv run python scripts_vp/download_clingen.py

# Local ETL
PYTHONPATH=src uv run python scripts_vp/load_clinvar.py
PYTHONPATH=src uv run python scripts_vp/load_gnomad.py --constraint data/vp/gnomad/gnomad.v4.1.constraint_metrics.tsv
PYTHONPATH=src uv run python scripts_vp/load_clingen.py --csv data/vp/clingen_gene_disease_validity.csv

# HPC-backed follow-up
# See research/21_vp_hpc_runbook.md for gnomAD sites extraction and score extraction
```

## ML Experiments

```bash
# DTI training (local or SLURM)
uv run python scripts/train_baseline.py --model deepdta --split random --negative negbiodb --dataset balanced
bash slurm/submit_all.sh

# CT training
uv run python scripts_ct/train_ct_baseline.py --model xgboost --task m1 --split random --negative negbiodb
bash slurm/submit_ct_all.sh

# PPI training
uv run python scripts_ppi/train_baseline.py --model siamese_cnn --split random --negative negbiodb --dataset balanced
bash slurm/submit_ppi_all.sh

# GE training
uv run python scripts_depmap/train_ge_baseline.py --model xgboost --split random --negative negbiodb
bash slurm/submit_ge_ml_all.sh

# Results collection (all domains support --aggregate-seeds)
uv run python scripts/collect_results.py --dataset balanced --aggregate-seeds
uv run python scripts_ct/collect_ct_results.py --aggregate-seeds
uv run python scripts_ppi/collect_results.py --dataset balanced --aggregate-seeds
uv run python scripts_depmap/collect_ge_results.py --aggregate-seeds
```

## LLM Benchmark

```bash
# Build LLM datasets (example: DTI)
uv run python scripts/build_l1_dataset.py
uv run python scripts/build_l2_dataset.py
uv run python scripts/build_l3_dataset.py
uv run python scripts/build_l4_dataset.py

# Run LLM inference
uv run python scripts/run_llm_benchmark.py --model gemini --level l1 --config zeroshot

# GE-specific LLM datasets and inference
uv run python scripts_depmap/build_ge_l1_dataset.py
uv run python scripts_depmap/run_ge_llm_benchmark.py --model gemini --level l1 --config zeroshot

# Collect results
uv run python scripts/collect_llm_results.py
uv run python scripts_ct/collect_ct_llm_results.py
uv run python scripts_ppi/collect_ppi_llm_results.py
uv run python scripts_depmap/collect_ge_results.py --llm
```

## Testing

```bash
# All tests across the codebase
PYTHONPATH=src uv run pytest tests/ -v

# By domain
PYTHONPATH=src uv run pytest tests/test_db.py tests/test_etl_*.py tests/test_export.py -v       # DTI
PYTHONPATH=src uv run pytest tests/test_ct_*.py tests/test_etl_aact.py -v                       # CT
PYTHONPATH=src uv run pytest tests/test_ppi_*.py tests/test_etl_intact.py -v                    # PPI
PYTHONPATH=src uv run pytest tests/test_ge_*.py tests/test_etl_depmap.py -v                     # GE

# Skip network-dependent tests
PYTHONPATH=src uv run pytest tests/ -v -m "not integration"
```

## Project Structure

```
NegBioDB/
├── src/
│   ├── negbiodb/              # DTI core library
│   │   ├── db.py              # Database creation & migrations
│   │   ├── download.py        # Download utilities (resume, checksum)
│   │   ├── standardize.py     # Compound/target standardization (RDKit)
│   │   ├── etl_davis.py       # DAVIS ETL pipeline
│   │   ├── etl_chembl.py      # ChEMBL ETL pipeline
│   │   ├── etl_pubchem.py     # PubChem ETL (streaming, 29M rows)
│   │   ├── etl_bindingdb.py   # BindingDB ETL pipeline
│   │   ├── export.py          # ML dataset export (Parquet, 5 splits)
│   │   ├── metrics.py         # ML evaluation metrics (7 metrics)
│   │   ├── llm_client.py      # LLM API client (vLLM, Gemini, OpenAI, Anthropic)
│   │   ├── llm_prompts.py     # LLM prompt templates (L1-L4)
│   │   ├── llm_eval.py        # LLM evaluation functions
│   │   └── models/            # ML baseline models
│   │       ├── deepdta.py     # DeepDTA (sequence CNN)
│   │       ├── graphdta.py    # GraphDTA (graph neural network)
│   │       └── drugban.py     # DrugBAN (bilinear attention)
│   ├── negbiodb_ct/           # Clinical Trial domain
│   │   ├── ct_db.py           # CT database & migrations
│   │   ├── etl_aact.py        # AACT ETL (13 tables)
│   │   ├── etl_classify.py    # 3-tier failure classification
│   │   ├── drug_resolver.py   # 4-step drug name resolution
│   │   ├── etl_outcomes.py    # Outcome enrichment (p-values, SAE)
│   │   ├── ct_export.py       # ML export (M1/M2, 6 splits)
│   │   ├── ct_features.py     # Feature encoding (1044/1066-dim)
│   │   ├── ct_models.py       # CT_MLP, CT_GNN_Tab models
│   │   ├── llm_prompts.py     # CT LLM prompts (L1-L4)
│   │   ├── llm_eval.py        # CT LLM evaluation
│   │   └── llm_dataset.py     # CT LLM dataset construction
│   ├── negbiodb_ppi/          # PPI domain
│   │   ├── ppi_db.py          # PPI database & migrations
│   │   ├── etl_intact.py      # IntAct PSI-MI TAB 2.7
│   │   ├── etl_huri.py        # HuRI Y2H screen negatives
│   │   ├── etl_humap.py       # hu.MAP ML-derived negatives
│   │   ├── etl_string.py      # STRING zero-score pairs
│   │   ├── protein_mapper.py  # UniProt validation, ENSG mapping
│   │   ├── export.py          # ML export (4 splits, controls)
│   │   ├── llm_prompts.py     # PPI LLM prompts (L1-L4)
│   │   ├── llm_eval.py        # PPI LLM evaluation
│   │   ├── llm_dataset.py     # PPI LLM dataset construction
│   │   └── models/            # PPI ML models
│   │       ├── siamese_cnn.py # Shared CNN encoder
│   │       ├── pipr.py        # Cross-attention PPI model
│   │       └── mlp_features.py # Hand-crafted feature MLP
│   └── negbiodb_depmap/       # Gene Essentiality (DepMap) domain
│       ├── depmap_db.py       # GE database & migrations
│       ├── etl_depmap.py      # DepMap CRISPR ETL
│       ├── etl_rnai.py        # RNAi screen ETL
│       ├── etl_prism.py       # PRISM drug screen ETL (optional)
│       ├── export.py          # ML export (5 splits, 770 MB parquet)
│       ├── ge_features.py     # Gene/cell-line feature encoding
│       ├── llm_prompts.py     # GE LLM prompts (L1-L4)
│       ├── llm_eval.py        # GE LLM evaluation
│       └── llm_dataset.py     # GE LLM dataset construction
├── scripts/                   # DTI CLI entry points
├── scripts_ct/                # CT CLI entry points
├── scripts_ppi/               # PPI CLI entry points
├── scripts_depmap/            # GE CLI entry points
├── slurm/                     # SLURM job scripts (HPC-ready, path-agnostic)
├── migrations/                # DTI SQL schema migrations
├── migrations_ct/             # CT SQL schema migrations
├── migrations_ppi/            # PPI SQL schema migrations
├── migrations_depmap/         # GE SQL schema migrations
├── tests/                     # Test suite across 5 domains
├── docs/                      # Methodology notes and prompt appendices
├── paper/                     # LaTeX source (NeurIPS 2026 submission)
├── data/                      # SQLite databases (not in repo, ~38 GB)
├── exports/                   # ML/LLM export files (Parquet, not in repo)
├── results/                   # Experiment results (not in repo)
├── config.yaml                # Pipeline configuration
├── Makefile                   # Build/pipeline commands
├── pyproject.toml             # Python project metadata
├── experiment_results.md      # ML/LLM result tables
├── PROJECT_OVERVIEW.md        # Detailed project overview
└── ROADMAP.md                 # Execution roadmap
```

## Exported Datasets

### DTI (`exports/`)

| File | Description |
|------|-------------|
| `negbiodb_dti_pairs.parquet` | 1.7M compound-target pairs with 5 split columns |
| `negbiodb_m1_balanced.parquet` | M1: 1.73M rows (1:1 active:inactive) |
| `negbiodb_m1_realistic.parquet` | M1: 9.49M rows (1:10 ratio) |
| `negbiodb_m1_balanced_ddb.parquet` | Exp 4: degree-balanced split |
| `negbiodb_m1_uniform_random.parquet` | Exp 1: uniform random negatives |
| `negbiodb_m1_degree_matched.parquet` | Exp 1: degree-matched negatives |
| `chembl_positives_pchembl6.parquet` | 863K ChEMBL actives (pChEMBL >= 6) |
| `compound_names.parquet` | 144K compound names (for LLM tasks) |

### CT (`exports/ct/`)

| File | Description |
|------|-------------|
| `negbiodb_ct_pairs.parquet` | 102,850 failure pairs, 6 splits, all tiers |
| `negbiodb_ct_m1_balanced.parquet` | Binary: 11,222 rows (5,611 pos + 5,611 neg) |
| `negbiodb_ct_m1_realistic.parquet` | Binary: 36,957 rows (1:~6 ratio) |
| `negbiodb_ct_m1_smiles_only.parquet` | Binary: 3,878 rows (SMILES-resolved only) |
| `negbiodb_ct_m2.parquet` | 7-way category: 112,298 rows (non-copper) |

### PPI (`exports/ppi/`)

| File | Description |
|------|-------------|
| `negbiodb_ppi_pairs.parquet` | 2,220,786 negative pairs with split columns |
| `ppi_m1_balanced.parquet` | M1: 123,456 rows (1:1 pos:neg) |
| `ppi_m1_realistic.parquet` | M1: 679,008 rows (1:10 ratio) |
| `ppi_m1_balanced_ddb.parquet` | Exp 4: degree-balanced split |
| `ppi_m1_uniform_random.parquet` | Exp 1: uniform random negatives |
| `ppi_m1_degree_matched.parquet` | Exp 1: degree-matched negatives |

### GE (`exports/ge/`)

| File | Description |
|------|-------------|
| `negbiodb_ge_pairs.parquet` | 770 MB; 22.5M gene-cell-line pairs with 5 split columns |
| `ge_m1_random.parquet` | Random split (train/val/test) |
| `ge_m1_cold_gene.parquet` | Cold-gene generalization split |
| `ge_m1_cold_cell_line.parquet` | Cold-cell-line generalization split |
| `ge_m1_cold_both.parquet` | Cold-both (hardest) split |
| `ge_m1_degree_balanced.parquet` | Degree-balanced negative control |

## Data Sources

### DTI

| Source | Records | License |
|--------|---------|---------|
| [ChEMBL v36](https://www.ebi.ac.uk/chembl/) | 371K | CC BY-SA 3.0 |
| [PubChem BioAssay](https://pubchem.ncbi.nlm.nih.gov/) | 29.6M | Public Domain |
| [BindingDB](https://www.bindingdb.org/) | 404K | CC BY |
| [DAVIS](https://github.com/dingyan20/Davis-Dataset-for-DTA-Prediction) | 20K | Public |

### CT

| Source | Records | License |
|--------|---------|---------|
| [AACT (ClinicalTrials.gov)](https://aact.ctti-clinicaltrials.org/) | 216,987 trials | Public Domain |
| [CTO](https://github.com/fairnessforensics/CTO) | 20,627 | MIT |
| [Open Targets](https://www.opentargets.org/) | 32,782 targets | Apache 2.0 |
| [Shi & Du 2024](https://doi.org/10.1038/s41597-024-03399-2) | 119K + 803K rows | CC BY 4.0 |

### PPI

| Source | Records | License |
|--------|---------|---------|
| [IntAct](https://www.ebi.ac.uk/intact/) | 779 pairs | CC BY 4.0 |
| [HuRI](http://www.interactome-atlas.org/) | 500,000 pairs | CC BY 4.0 |
| [hu.MAP 3.0](https://humap3.proteincomplexes.org/) | 1,228,891 pairs | MIT |
| [STRING v12.0](https://string-db.org/) | 500,000 pairs | CC BY 4.0 |

### GE

| Source | Records | License |
|--------|---------|---------|
| [DepMap CRISPR (Chronos)](https://depmap.org/) | 28.7M gene-cell pairs | CC BY 4.0 |
| [DepMap RNAi (DEMETER2)](https://depmap.org/) | Integrated | CC BY 4.0 |

## ML Evaluation Metrics

| Metric | Role |
|--------|------|
| **LogAUC[0.001,0.1]** | Primary ranking metric (early enrichment) |
| **BEDROC (alpha=20)** | Early enrichment (exponentially weighted) |
| **EF@1%, EF@5%** | Enrichment factor at top 1%/5% |
| **AUPRC** | Secondary ranking metric |
| **MCC** | Balanced classification |
| **AUROC** | Backward compatibility |

## Citation

If you use NegBioDB in your research, please cite:

```bibtex
@misc{negbiodb2026,
  title={NegBioDB: A Negative Results Database and Dual ML/LLM Benchmark for Biomedical Sciences},
  author={Kim, JangKeun},
  year={2026},
  url={https://github.com/jang1563/NegBioDB}
}
```

## License

**CC BY-SA 4.0** — see [LICENSE](LICENSE) for details.

This license is required by the viral clause in ChEMBL's CC BY-SA 3.0 license.

# NegBioDB

**Negative Results Database & Dual ML/LLM Benchmark for Drug-Target Interactions**

Approximately 90% of scientific experiments produce null or inconclusive results, yet the vast majority remain unpublished. NegBioDB systematically collects experimentally confirmed negative results for DTI (Drug-Target Interaction) and provides dual-track benchmarks for ML and LLM evaluation.

## Key Features

- **Curated negative DTI data** from ChEMBL, PubChem, BindingDB, and DAVIS (30M+ records, 919K compounds, 3.7K targets)
- **Biology-first, science-extensible** architecture (DTI first, expandable to other domains)
- **Dual benchmark**: ML track (traditional DTI prediction) + LLM track (6 biomedical NLP tasks)
- **Standardized inactivity threshold**: 10 µM (pChEMBL < 5), with borderline exclusion zone
- **6 split strategies**: random, cold-compound, cold-target, temporal, scaffold, degree-balanced
- **3 ML baseline models**: DeepDTA, GraphDTA, DrugBAN (with SLURM/HPC support)
- **7 evaluation metrics**: LogAUC[0.001,0.1], BEDROC(α=20), EF@1%/5%, AUROC, AUPRC, MCC
- **Reproducible pipeline**: SQLite database, config-driven ETL, automated ML export

## Current Database Statistics

| Metric | Count |
|--------|-------|
| `negative_results` | 30,459,583 |
| `compounds` | 919,062 |
| `targets` | 3,711 |
| `target_variants` | 48 (mutations only) |
| `compound_target_pairs` | 24,965,618 |
| DB file size | 13.22 GB |

**Source breakdown:** PubChem 29.6M · BindingDB 404K · ChEMBL 371K · DAVIS 20K

## Project Status

| Component | Status |
|-----------|--------|
| Schema & database | ✅ Done |
| Data download & ETL (4 sources) | ✅ Done |
| Compound/target standardization & dedup | ✅ Done |
| ML export & splits (6 strategies) | ✅ Done |
| M1 benchmark datasets (balanced + realistic) | ✅ Done |
| ML evaluation metrics (7 metrics) | ✅ Done |
| ML baseline training (18/18 runs on Cayuga HPC) | ✅ Done |
| LLM benchmark infrastructure (L1–L4 datasets, prompts, eval) | ✅ Done |
| LLM benchmark execution (Cayuga) | ⏳ Pending |
| Paper writing | Planned |

**Target:** NeurIPS 2026 Datasets & Benchmarks Track (~May 15, 2026)

## Key ML Results (18/18 runs complete)

**Exp 1 — Negative type matters (random split, LogAUC):**

| Model | NegBioDB | Uniform Random | Degree Matched |
|-------|---------|---------------|----------------|
| DeepDTA | 0.833 | 0.824 | **0.919** |
| GraphDTA | 0.843 | 0.888 | **0.967** |
| DrugBAN | 0.830 | 0.825 | **0.955** |

Degree-matched negatives inflate performance by +0.112 on average — **benchmark inflation confirmed**.

**Split effect — cold-target exposes model failure:**

| Model | Random | Cold Compound | Cold Target |
|-------|--------|--------------|-------------|
| DeepDTA | 0.833 | 0.792 | 0.325 |
| GraphDTA | 0.843 | 0.823 | 0.241 |
| DrugBAN | 0.830 | 0.828 | 0.151 |

Cold-target LogAUC drops catastrophically while AUROC remains 0.76–0.89 — **AUROC is misleading**.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/jang1563/NegBioDB.git
cd NegBioDB
make setup          # Create venv and install dependencies
make db             # Initialize SQLite database
```

## Data Pipeline

```bash
make download       # Download all 4 sources (ChEMBL, PubChem, BindingDB, DAVIS)
make load-davis     # ETL: load DAVIS into database
make load-chembl    # ETL: load ChEMBL inactive records
make load-pubchem   # ETL: load PubChem confirmatory inactive records
make load-bindingdb # ETL: load BindingDB inactive records
make load-all       # Run all ETL loaders
```

Individual downloads:
```bash
make download-chembl
make download-pubchem
make download-bindingdb
make download-davis
```

## ML Export & Experiments

```bash
# Export ML dataset (Parquet + CSV, 6 split strategies)
uv run python scripts/export_ml_dataset.py

# Prepare experiment data (Exp 1 random negatives, Exp 4 DDB split)
uv run python scripts/prepare_exp_data.py

# Train ML baselines (local)
uv run python scripts/train_baseline.py --model deepdta --split random --negative negbiodb --dataset balanced

# Note:
# - Exp 1 random-control negatives (`uniform_random`, `degree_matched`) are supported only for `--dataset balanced`
# - Exp 4 / DDB runs use `--split ddb --negative negbiodb --dataset balanced`
# - The DDB parquet is a full-task degree-balanced split recomputed on merged M1 balanced data, so positives and negatives are reassigned together

# Submit to HPC (Cornell Cayuga SLURM)
bash slurm/submit_all.sh
SEEDS="42 43 44" bash slurm/submit_all.sh
MODELS="deepdta graphdta" SPLITS="random ddb" NEGATIVES="negbiodb" SEEDS="42 43 44" bash slurm/submit_all.sh

# Submit from local machine via Cayuga SSH ControlMaster
ssh -fN cayuga-login1
bash slurm/remote_submit_cayuga.sh
LOG_GLOB="negbio_deepdta_balanced_random_negbiodb_seed42" bash slurm/remote_monitor_cayuga.sh

# Collect results for a controlled subset
uv run python scripts/collect_results.py --dataset balanced --seed 42
uv run python scripts/collect_results.py --dataset balanced --model deepdta --split random --negative negbiodb
uv run python scripts/collect_results.py --dataset balanced --aggregate-seeds
# Writes raw tables plus table1_aggregated.csv / table1_aggregated.md with mean +/- std over seeds
# By default, stale `--split ddb` runs older than `exports/negbiodb_m1_balanced_ddb.parquet` are excluded
```

## Testing

```bash
make test                                    # All tests (329 total)
uv run pytest tests/ -v -m "not integration" # Skip network-dependent tests
```

## Project Structure

```
NegBioDB/
├── src/negbiodb/         # Core library
│   ├── db.py             # Database creation & migration runner
│   ├── download.py       # Download utilities (resume, checksum)
│   ├── standardize.py    # Compound/target standardization (RDKit)
│   ├── etl_davis.py      # DAVIS ETL pipeline
│   ├── etl_chembl.py     # ChEMBL ETL pipeline
│   ├── etl_pubchem.py    # PubChem ETL pipeline (streaming, 29M rows)
│   ├── etl_bindingdb.py  # BindingDB ETL pipeline
│   ├── export.py         # ML dataset export (Parquet/CSV, 6 splits)
│   ├── metrics.py        # ML evaluation metrics (7 metrics)
│   ├── llm_client.py     # LLM API client (vLLM, Gemini)
│   ├── llm_prompts.py    # LLM prompt templates (L1–L4 tasks)
│   ├── llm_eval.py       # LLM evaluation functions
│   └── models/           # ML baseline models
│       ├── deepdta.py    # DeepDTA (sequence CNN)
│       ├── graphdta.py   # GraphDTA (graph neural network)
│       └── drugban.py    # DrugBAN (bilinear attention)
├── scripts/              # CLI entry points
│   ├── download_*.py     # Download scripts for each source
│   ├── load_*.py         # Source loading scripts
│   ├── export_ml_dataset.py  # ML dataset export
│   ├── train_baseline.py     # ML training harness
│   ├── prepare_exp_data.py   # Experiment data preparation (Exp 1, 4)
│   ├── collect_results.py    # ML results collection & aggregation
│   ├── build_l{1,2,3,4}_dataset.py  # LLM benchmark dataset builders
│   ├── build_compound_names.py      # Compound name cache builder
│   ├── run_llm_benchmark.py         # LLM benchmark runner
│   ├── collect_llm_results.py       # LLM results aggregation (Table 2)
│   └── eval_checkpoint.py           # Checkpoint evaluation for timed-out jobs
├── slurm/                # SLURM job scripts (Cornell Cayuga HPC)
│   ├── train_baseline.slurm    # ML training job template
│   ├── submit_all.sh           # ML batch submission
│   ├── run_llm_local.slurm     # LLM local model job template
│   ├── run_llm_gemini.slurm    # LLM Gemini API job template
│   ├── submit_llm_all.sh       # LLM batch submission
│   ├── setup_env.sh / setup_llm_env.sh  # Environment setup
│   └── start_vllm_server.sh    # vLLM server launcher
├── results/baselines/    # ML baseline results (18 runs, rsynced from Cayuga)
├── migrations/           # SQL schema migrations
├── tests/                # Test suite (329 tests)
├── exports/              # ML/LLM export files (Parquet, leakage report)
├── research/             # Research documents (01-12)
├── config.yaml           # Pipeline configuration (thresholds, paths, URLs)
├── ROADMAP.md            # Execution roadmap (v10)
├── PROJECT_OVERVIEW.md   # Project overview & document index
├── Makefile              # Build/pipeline commands
└── pyproject.toml        # Python project metadata
```

## Exported Datasets (`exports/`)

| File | Description |
|------|-------------|
| `negbiodb_dti_pairs.parquet` | All 25M compound-target pairs with 6 split columns |
| `negbiodb_m1_balanced.parquet` | M1 task: 1.73M rows (1:1 active:inactive) |
| `negbiodb_m1_realistic.parquet` | M1 task: 9.49M rows (1:10 active:inactive) |
| `negbiodb_m1_balanced_ddb.parquet` | Exp 4: degree-balanced full-task split |
| `negbiodb_m1_uniform_random.parquet` | Exp 1 control: uniform random negatives |
| `negbiodb_m1_degree_matched.parquet` | Exp 1 control: degree-matched random negatives |
| `chembl_positives_pchembl6.parquet` | 863K ChEMBL actives (pChEMBL ≥ 6) |
| `compound_names.parquet` | 144K compound names from ChEMBL (for LLM tasks) |

## ML Evaluation Metrics

| Metric | Role |
|--------|------|
| **LogAUC[0.001,0.1]** | Primary ranking metric (early enrichment) |
| **BEDROC (α=20)** | Early enrichment (exponentially weighted) |
| **EF@1%, EF@5%** | Enrichment factor at top 1%/5% |
| **AUPRC** | Secondary ranking metric |
| **MCC** | Balanced classification |
| **AUROC** | Backward compatibility |

## Data Sources

| Source | Type | Records | License |
|--------|------|---------|---------|
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | Literature-curated bioactivity | 371K | CC BY-SA 3.0 |
| [PubChem BioAssay](https://pubchem.ncbi.nlm.nih.gov/) | HTS & confirmatory assays | 29.6M | Public Domain |
| [BindingDB](https://www.bindingdb.org/) | Binding affinity measurements | 404K | CC BY |
| [DAVIS](https://github.com/dingyan20/Davis-Dataset-for-DTA-Prediction) | Kinase-inhibitor Kd panel | 20K | Public |

## License

**CC BY-SA 4.0** — see [LICENSE](LICENSE) for details.

This license is required by the viral clause in ChEMBL's CC BY-SA 3.0 license.

## Target Publication

NeurIPS 2026 Datasets & Benchmarks Track

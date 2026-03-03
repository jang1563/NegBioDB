# NegBioDB

**Negative Results Database & Dual ML/LLM Benchmark for Drug-Target Interactions**

Approximately 90% of scientific experiments produce null or inconclusive results, yet the vast majority remain unpublished. NegBioDB systematically collects experimentally confirmed negative results for DTI (Drug-Target Interaction) and provides dual-track benchmarks for ML and LLM evaluation.

## Key Features

- **Curated negative DTI data** from ChEMBL, PubChem, BindingDB, and DAVIS
- **Biology-first, science-extensible** architecture (DTI first, expandable to other domains)
- **Dual benchmark**: ML track (traditional DTI prediction) + LLM track (6 biomedical NLP tasks)
- **Standardized inactivity threshold**: 10 uM (pChEMBL < 5), with borderline exclusion zone
- **Reproducible pipeline**: SQLite database, config-driven ETL, automated ML export splits

## Project Status

**Early development** — data download and ETL pipelines under construction.

| Component | Status |
|-----------|--------|
| Schema & database | Done |
| Data download (4 sources) | Done |
| ETL: DAVIS | Done |
| ETL: ChEMBL, PubChem, BindingDB | Done (initial implementation) |
| Deduplication & pair creation | Planned |
| ML export & splits | Planned |
| LLM benchmark tasks | Planned |

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

## Testing

```bash
make test                                    # All tests
uv run pytest tests/ -v -m "not integration" # Skip network-dependent tests
```

## Project Structure

```
NegBioDB/
├── src/negbiodb/         # Core library
│   ├── db.py             # Database creation & migration runner
│   ├── download.py       # Download utilities (resume, checksum)
│   ├── etl_davis.py      # DAVIS ETL pipeline
│   ├── etl_chembl.py     # ChEMBL ETL pipeline
│   ├── etl_pubchem.py    # PubChem ETL pipeline
│   └── etl_bindingdb.py  # BindingDB ETL pipeline
├── scripts/              # CLI entry points
│   ├── download_*.py     # Download scripts for each source
│   └── load_*.py         # Source loading scripts
├── migrations/           # SQL schema migrations
│   ├── 001_initial_schema.sql
│   └── 002_target_variants.sql
├── tests/                # Test suite
├── research/             # Research documents (01-12)
├── config.yaml           # Pipeline configuration (thresholds, paths, URLs)
├── ROADMAP.md            # Execution roadmap
├── PROJECT_OVERVIEW.md   # Project overview & document index
├── Makefile              # Build/pipeline commands
└── pyproject.toml        # Python project metadata
```

## Data Sources

| Source | Type | License |
|--------|------|---------|
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | Literature-curated bioactivity | CC BY-SA 3.0 |
| [PubChem BioAssay](https://pubchem.ncbi.nlm.nih.gov/) | HTS & confirmatory assays | Public Domain |
| [BindingDB](https://www.bindingdb.org/) | Binding affinity measurements | CC BY |
| [DAVIS](https://github.com/dingyan20/Davis-Dataset-for-DTA-Prediction) | Kinase-inhibitor Kd panel | Public |

## License

**CC BY-SA 4.0** — see [LICENSE](LICENSE) for details.

This license is required by the viral clause in ChEMBL's CC BY-SA 3.0 license.

## Target Publication

NeurIPS 2026 Datasets & Benchmarks Track

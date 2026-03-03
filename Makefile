# NegBioDB Pipeline Makefile

.PHONY: setup db test clean

# === Week 1: Scaffolding ===

setup:
	uv venv
	uv sync --all-extras
	mkdir -p data exports

db: setup
	uv run python -c "from negbiodb.db import create_database; create_database()"

test: setup
	uv run pytest tests/ -v

clean:
	rm -f data/negbiodb.db
	rm -rf exports/* __pycache__ .pytest_cache
	find . -name "*.pyc" -delete 2>/dev/null || true

# === Week 2: Data Download ===

.PHONY: download download-pubchem download-chembl download-bindingdb download-davis

download-pubchem: setup
	uv run python scripts/download_pubchem.py

download-chembl: setup
	uv run python scripts/download_chembl.py

download-bindingdb: setup
	uv run python scripts/download_bindingdb.py

download-davis: setup
	uv run python scripts/download_davis.py

download: download-pubchem download-chembl download-bindingdb download-davis

# === ETL: Load Sources ===

.PHONY: load-davis

load-davis: db download-davis
	uv run python scripts/load_davis.py

# === Week 3+ Pipeline (placeholders) ===
# load-pubchem: db download-pubchem
# load-chembl: db download-chembl
# load-bindingdb: db download-bindingdb
# pairs: load-davis load-pubchem load-chembl load-bindingdb
# splits: pairs
# export: splits
# all: export

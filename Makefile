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

.PHONY: load-davis load-chembl load-pubchem load-bindingdb load-all

load-davis: db download-davis
	uv run python scripts/load_davis.py

load-chembl: db download-chembl
	uv run python scripts/load_chembl.py

load-pubchem: db download-pubchem
	uv run python scripts/load_pubchem.py

load-bindingdb: db download-bindingdb
	uv run python scripts/load_bindingdb.py

load-all: load-davis load-chembl load-pubchem load-bindingdb

# ============================================================
# Clinical Trial Failure Domain
# ============================================================

.PHONY: ct-db ct-download ct-load-aact ct-classify ct-resolve ct-outcomes ct-all ct-clean

ct-db: setup
	uv run python -c "from negbiodb_ct.ct_db import create_ct_database; create_ct_database()"

ct-download-aact: setup
	uv run python scripts_ct/download_aact.py

ct-download-opentargets: setup
	uv run python scripts_ct/download_opentargets.py

ct-download-cto: setup
	uv run python scripts_ct/download_cto.py

ct-download-shi-du: setup
	uv run python scripts_ct/download_shi_du.py

ct-download: ct-download-opentargets ct-download-cto ct-download-shi-du
	@echo "NOTE: AACT requires --url flag. Run: make ct-download-aact"

ct-load-aact: ct-db
	uv run python scripts_ct/load_aact.py

ct-classify: ct-load-aact
	uv run python scripts_ct/classify_failures.py

ct-resolve: ct-load-aact
	uv run python scripts_ct/resolve_drugs.py

ct-outcomes: ct-classify
	uv run python scripts_ct/load_outcomes.py

ct-all: ct-db ct-load-aact ct-classify ct-resolve ct-outcomes
	@echo "CT pipeline complete."

ct-clean:
	rm -f data/negbiodb_ct.db
	@echo "CT database removed."

ct-test: setup
	uv run pytest tests/test_ct_db.py tests/test_etl_aact.py tests/test_etl_classify.py tests/test_drug_resolver.py tests/test_etl_outcomes.py -v

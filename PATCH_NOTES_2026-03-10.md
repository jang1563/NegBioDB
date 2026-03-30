# Patch Notes: Experiment Pipeline Fixes

Date: 2026-03-10

## Scope

This patch fixes experiment orchestration bugs found during review across:

- `src/negbiodb/export.py`
- `scripts/train_baseline.py`
- `scripts/prepare_exp_data.py`
- `scripts/collect_results.py`
- `tests/test_pipeline_scripts.py`

## Bugs Fixed

### 1. Split export row fanout with multiple split versions

Problem:
- `export_negative_dataset()` joined split columns by `split_strategy`.
- If `random_v2` or another later split definition existed for the same strategy,
  the export query could duplicate rows.

Fix:
- Export now resolves a single latest `split_id` per strategy before joining
  `split_assignments`.
- Cold split integrity checks were aligned to use the latest split definition too.
- Split selection now prefers explicit version semantics (`version` column and `_vN`
  suffixes) rather than raw insertion order.

Impact:
- Prevents silent corruption of `negbiodb_dti_pairs.parquet` when new split versions are added.

### 2. Incomplete graph cache silently degrading GraphDTA/DrugBAN

Problem:
- Existing `graph_cache.pt` files could be incomplete.
- Missing SMILES fell through to placeholder graphs during batching.
- When no cache existed, train fold could save a train-only cache that val/test later reused.

Fix:
- Added `_prepare_graph_cache()` in `train_baseline.py`.
- The cache is now built or backfilled against the full dataset parquet before folds are created.
- `DTIDataset._load_graphs()` also backfills missing SMILES if a partial cache path is still passed in.

Impact:
- Prevents valid molecules from silently training/evaluating as dummy graphs.

### 3. Training run outputs overwriting each other

Problem:
- Output directories were keyed only by `model/split/negative`.
- Different datasets or seeds could overwrite prior runs.

Fix:
- Run names now include `dataset` and `seed`.
- Added `_build_run_name()` helper to keep naming deterministic and testable.

Impact:
- Balanced vs. realistic and multi-seed runs can now coexist safely.

### 3b. Unsupported realistic Exp 1 controls now fail fast

Problem:
- `uniform_random` and `degree_matched` controls did not have realistic dataset variants.
- The CLI still accepted `--dataset realistic` for those negatives, which mislabeled balanced-control runs as realistic.

Fix:
- Removed those unsupported mappings from `train_baseline.py`.
- The CLI now rejects realistic random-control combinations explicitly.

Impact:
- Prevents mislabeled Exp 1 results from being written to disk and summarized later.

### 4. Invalid DDB CLI combinations

Problem:
- `ddb` was exposed as a negative source even though it is a split mode.
- Commands like `--split random --negative ddb` were accepted.

Fix:
- Removed `ddb` from `--negative` choices.
- Added explicit validation so DDB is only valid for `split=ddb`, `negative=negbiodb`,
  and `dataset=balanced`.

Impact:
- Reduces mislabeled experiment runs and ambiguous result directories.

### 5. `prepare_exp_data.py --skip-exp4` still required the pairs parquet

Problem:
- Input validation always required `exports/negbiodb_dti_pairs.parquet`,
  even when Exp 4 was being skipped.

Fix:
- Conditionalized required inputs so the large pairs parquet is only required when Exp 4 runs.
- Added defensive deduplication and `many_to_one` merge validation in `prepare_exp4_ddb()`.

Impact:
- Exp 1 preparation can now run independently.

### 6. Result collection mixed datasets and hid seed information

Problem:
- `collect_results.py` dropped `dataset` and `seed` from Table 1 output.
- Exp 1 summary selected the first matching row instead of grouping safely.

Fix:
- Table output now preserves `dataset` and `seed`.
- Exp 1 summary is now grouped by dataset and averages only across seeds present in all compared negative conditions.

Impact:
- Prevents balanced/realistic runs from being conflated in downstream summaries.

### 6b. Result collection now supports explicit dataset/seed filters

Problem:
- `collect_results.py` previously aggregated every run under `results/baselines/`.
- As more experiments accumulate, paper tables can accidentally mix exploratory and final runs.

Fix:
- Added `--dataset` and repeatable `--seed` filters to `collect_results.py`.
- Added repeatable `--model`, `--split`, and `--negative` filters too.
- The command now fails fast if filters remove all rows.
- Added optional `--aggregate-seeds` output that writes `table1_aggregated.csv`
  with mean/std over seeds.
- Added `table1_aggregated.md` with human-readable `mean +/- std` formatting.

Impact:
- Makes paper/report generation reproducible from an explicit run subset.
- Adds a paper-friendly seed-aggregated summary without removing access to raw runs.

### 7. SLURM submission metadata aligned with current run naming

Problem:
- SLURM job names and log file names did not include dataset or seed.
- Submission wrappers relied on implicit defaults instead of exporting `DATASET` and `SEED` explicitly.

Fix:
- Updated SLURM wrapper scripts to include dataset and seed in job/log naming.
- Submission wrappers now export `DATASET` and `SEED` explicitly to the training job.
- Training job logging now prints seed together with model/split/negative/dataset.

Impact:
- Makes cluster logs line up with the run directories and result-collection filters.

### 7b. SLURM submission scripts now support seed sweeps

Problem:
- Multi-seed experiments required manual resubmission or manual editing of shell scripts.

Fix:
- Added `SEEDS="42 43 44"` style support to SLURM submission wrappers.
- Added optional `MODELS`, `SPLITS`, and `NEGATIVES` filters to `submit_all.sh`.
- Default behavior remains `SEEDS=42`, preserving the prior single-seed workflow.

Impact:
- Makes it straightforward to launch reproducible seed sweeps that match `collect_results.py --seed ...`.
- Makes selective experiment submission possible without editing shell scripts.

### 7c. Added Cayuga SSH helper wrappers for remote submission and monitoring

Problem:
- The documented Cayuga workflow relies on SSH ControlMaster and non-interactive remote commands.
- Reconstructing the exact remote submit/monitor commands by hand is error-prone.

Fix:
- Added `slurm/remote_submit_cayuga.sh` to call `submit_all.sh` remotely via `ssh ${HPC_LOGIN:-cayuga-login1}`.
- Added `slurm/remote_monitor_cayuga.sh` to inspect `squeue` and recent log files remotely.
- Made SLURM binary paths overridable via `SBATCH_BIN` / `SQUEUE_BIN`.

Impact:
- Makes the repository align directly with the Cayuga SSH workflow documented in the lab notes.

## Regression Tests Added

Added `tests/test_pipeline_scripts.py` covering:

- export uses latest split version without row duplication
- `prepare_exp_data.py --skip-exp4` no longer requires the pairs parquet
- run names include dataset and seed
- invalid DDB dataset resolution is rejected
- unsupported realistic random-control runs are rejected
- `train_baseline.py` rejects `split=ddb` with `dataset=realistic`
- `train_baseline.py` writes results into a dataset+seed-qualified run directory
- graph cache backfills missing SMILES
- result collection preserves dataset/seed and groups Exp 1 summaries by dataset using matched seeds only
- result collection can filter by dataset and seed before building tables
- result collection can also filter by model, split, and negative source
- result collection can optionally write seed-aggregated tables
- result collection writes both aggregated CSV and aggregated Markdown summaries
- SLURM job/log names now include dataset and seed, matching run output naming
- SLURM submission wrappers support multi-seed sweeps via the `SEEDS` environment variable
- `submit_all.sh` can filter submitted experiments by model, split, and negative source
- Added remote Cayuga SSH helper scripts for submission and monitoring

## Verification

Commands run:

```bash
uv run pytest tests/test_export.py -q
uv run pytest tests/test_pipeline_scripts.py -q
```

Observed results:

- `tests/test_export.py`: 52 passed
- `tests/test_pipeline_scripts.py`: 17 passed, 1 skipped

Skip reason:
- the graph cache backfill test is skipped when `torch` is unavailable in the environment

## Follow-up

Recommended next step:

- add one end-to-end script test that exercises `train_baseline.py` CLI with heavy training
  functions mocked out, to lock down argument validation and results writing behavior together

## Lessons Learned (2026-03-12)

### 1. Exp 4 must be defined as a full-task split, not a negative-only transform

What we learned:
- Treating DDB as "only reassign negatives while positives keep `split_random`" weakens the claim.
- That setup mixes split bias with class-specific handling and no longer measures the benchmark-level effect of node-degree bias.

Decision:
- Exp 4 is now defined as a **full-task degree-balanced split on merged M1 balanced data**.
- Positives and negatives must be reassigned together under the same `split_degree_balanced` policy.

Operational rule:
- Any future regeneration of `exports/negbiodb_m1_balanced_ddb.parquet` must be done from the merged M1 benchmark, not by patching only the negative subset.

### 2. Eval-only rewrites can make stale checkpoints look fresh

What we learned:
- Using `results.json` modification time alone is not a reliable freshness signal.
- `eval_checkpoint.py` can rewrite results for an old checkpoint and make a stale run appear newly generated.

Decision:
- Freshness checks must use training artifacts first (`best.pt`, `last.pt`, `training_log.csv`) and only fall back to `results.json` if no training artifacts exist.

Operational rule:
- Any future result filtering or release packaging should treat checkpoint/log timestamps as the source of truth for run freshness.

### 3. DDB benchmark regeneration invalidates prior DDB model results

What we learned:
- Once `exports/negbiodb_m1_balanced_ddb.parquet` changes, all prior `*_ddb_*` model results become semantically stale even if files still parse.
- Silent reuse of those results is more dangerous than having missing rows.

Decision:
- `collect_results.py` now excludes stale DDB runs by default when they are older than the current DDB parquet.

Operational rule:
- After any future DDB regeneration, immediately retrain the DDB model trio before producing paper tables.

### 4. Remote HPC execution can drift from local fixes

What we learned:
- The Cayuga working directory is effectively a deployed copy, not a git-controlled checkout.
- Local fixes do not automatically exist on the remote side, so "submitted successfully" does not imply "submitted the corrected code."

Decision:
- Before remote submission, explicitly sync changed code or regenerate artifacts in place on Cayuga.
- For large derived files, remote regeneration is usually safer than bulk transfer.

Operational rule:
- Prefer:
  1. sync small code/script changes
  2. regenerate derived artifacts on Cayuga
  3. then submit jobs

### 5. Submission wrappers need shell syntax coverage, not just static review

What we learned:
- `slurm/submit_all.sh` still contained loop-closing bugs after functional edits.
- These did not surface until the exact submission path was exercised on the remote cluster.

Decision:
- Shell wrappers should be syntax-checked with `bash -n` whenever edited.

Operational rule:
- For any future SLURM wrapper change:
  - run `bash -n <script>`
  - then run one narrow submission command before treating the wrapper as good

## Future Reference

### Step 4 current ground truth

- Exp 4 = `split=ddb`, `negative=negbiodb`, `dataset=balanced`
- DDB parquet = `exports/negbiodb_m1_balanced_ddb.parquet`
- DDB parquet is generated from merged M1 balanced data with positives and negatives reassigned together
- Stale DDB results are excluded by default in `scripts/collect_results.py`
- Override flag if absolutely needed: `--allow-stale-ddb`

### Safe workflow for future DDB reruns

1. Sync changed code to Cayuga.
2. Regenerate DDB parquet on Cayuga:
   - `source ${CONDA_PREFIX:-/path/to/conda}/miniconda3/etc/profile.d/conda.sh`
   - `conda activate negbiodb-ml`
   - `cd ${SCRATCH_DIR:-/path/to/scratch}/negbiodb`
   - `python scripts/prepare_exp_data.py --skip-exp1`
3. Submit only DDB jobs:
   - `SEEDS="42" MODELS="deepdta graphdta drugban" SPLITS="ddb" NEGATIVES="negbiodb" DATASETS="balanced" bash slurm/submit_all.sh`
4. Verify queue state:
   - `/opt/ohpc/pub/software/slurm/24.05.2/bin/squeue -u ${USER}`
5. Re-collect tables after those runs finish.

### Current known-good DDB submission example

Successful Cayuga submission on 2026-03-12:

- `negbio_deepdta_balanced_ddb_negbiodb_seed42` → job `2702356`
- `negbio_graphdta_balanced_ddb_negbiodb_seed42` → job `2702357`
- `negbio_drugban_balanced_ddb_negbiodb_seed42` → job `2702358`

### Files to inspect first next time

- `scripts/prepare_exp_data.py`
- `src/negbiodb/export.py`
- `scripts/collect_results.py`
- `scripts/train_baseline.py`
- `scripts/eval_checkpoint.py`
- `slurm/submit_all.sh`
- `tests/test_pipeline_scripts.py`

# NegBioDB Experiment Results

*Last updated: 2026-03-24*

## Cross-Domain Summary

| Domain | Negatives | ML Status | LLM L1 (acc) | LLM L4 (MCC) |
|--------|-----------|-----------|--------------|--------------|
| DTI (Drug-Target Interaction) | 30,459,583 | 21/24 complete | Gemini 1.000 (3-shot) | ≤ 0.18 (near random) |
| CT (Clinical Trial Failure) | 132,925 | 108/108 complete | Gemini 0.68 | Gemini 0.56 |
| PPI (Protein-Protein Interaction) | 2,220,786 | 54/54 complete | ~1.000 (3-shot artifact) | Llama 0.441 |
| GE (Gene Essentiality / DepMap) | 28,759,256 | Seed 42 done | Partial (Llama pending) | Partial |

**Key insight:** LLM discrimination ability (L4 MCC) increases with task complexity: DTI (~0.05–0.18) < PPI (~0.33–0.44) < CT (~0.48–0.56). All domains show evidence of training data contamination in PPI/CT.

---

## DTI Domain (Drug-Target Interaction)

**Status as of 2026-03-24:** ML 21/24 complete (3 uniform_random E1 jobs pending), LLM 81/81 complete.

### Database
- 30,459,583 negative results, 5 splits (random/cold_compound/cold_target/scaffold/temporal)
- 1,725,446 export rows (862,723 pos + 862,723 neg)
- Source tiers: gold=818,611 / silver=198 / bronze=28,845,632 (PubChem dynamic tiers)

### ML Results Summary
- 3 models: DeepDTA, GraphDTA, DrugBAN × 5 splits × 2 negative types (NegBioDB + uniform_random)
- NegBioDB negatives: Random split near-perfect AUROC (trivially separable)
- Control negatives (uniform_random): Harder splits show meaningful degradation
- 3 pending jobs: uniform_random E1 for DeepDTA/GraphDTA/DrugBAN (SLURM 2716037-2716039)

### LLM Results (81/81 complete)
| Task | Best Model | Metric | Value |
|------|-----------|--------|-------|
| L1 (MCQ) | Gemini | Accuracy (3-shot) | 1.000 |
| L4 (Discrim) | Best model | MCC | ≤ 0.18 (all near random) |

**Key finding:** DTI LLMs show near-random discrimination (L4 MCC ≤ 0.18). The binding/non-binding decision is too nuanced for zero-shot or few-shot LLMs. This is the hardest domain.

---

## CT Domain (Clinical Trial Failure)

**Status as of 2026-03-20:** ML 108/108 complete, LLM 80/80 complete, L3 judge complete.

### Database
- 132,925 failure results from 216,987 trials
- Tiers: gold 23,570 / silver 28,505 / bronze 60,223 / copper 20,627
- 8 failure categories: safety > efficacy > enrollment > strategic > regulatory > design > other (PK=0)

### ML Results (CT-M1: Binary Failure Prediction)
Aggregated over 3 seeds (source: `results/ct_table_m1_aggregated.csv`):

| Model | Split | Negatives | AUROC | MCC |
|-------|-------|-----------|-------|-----|
| XGBoost | random | NegBioDB | **1.000** | 1.000 |
| GNN | random | NegBioDB | 1.000 | 1.000 |
| MLP | random | NegBioDB | 1.000 | 0.992 |
| XGBoost | random | degree_matched | 0.844 | 0.553 |
| MLP | random | degree_matched | 0.801 | 0.454 |
| GNN | random | degree_matched | 0.758 | 0.440 |
| XGBoost | cold_condition | NegBioDB | 1.000 | 1.000 |
| XGBoost | cold_drug | NegBioDB | 1.000 | 0.999 |

**Key finding:** CT-M1 is trivially separable on NegBioDB negatives (AUROC=1.0 random/cold splits). Control negatives (degree_matched) reveal meaningful discrimination AUROC ~0.76–0.84.

### ML Results (CT-M2: 7-way Failure Category)
Aggregated over 3 seeds (source: `results/ct_table_m2_aggregated.csv`):

| Model | Split | Macro-F1 | Weighted-F1 | MCC |
|-------|-------|----------|-------------|-----|
| XGBoost | random | **0.510** | 0.751 | 0.637 |
| XGBoost | degree_balanced | 0.521 | 0.758 | 0.645 |
| XGBoost | cold_condition | 0.338 | 0.686 | 0.570 |
| XGBoost | cold_drug | 0.414 | 0.683 | 0.555 |
| XGBoost | scaffold | 0.193 | 0.567 | 0.374 |
| XGBoost | temporal | 0.193 | 0.602 | 0.454 |
| GNN | random | 0.468 | 0.672 | 0.526 |
| MLP | random | 0.358 | 0.619 | 0.432 |

**Key finding:** XGBoost best for M2. Scaffold/temporal splits are hardest (macro-F1 ~0.19). Degree-balanced helps.

### LLM Results (80/80 complete)
| Task | Gemini | GPT-4o-mini | Haiku | Qwen-7B | Llama-8B |
|------|--------|-------------|-------|---------|---------|
| L1 (5-way MCQ, acc) | **0.68** | 0.64 | 0.66 | 0.65 | 0.63 |
| L2 (extraction, field_f1) | 0.75 | 0.73 | 0.48 | **0.81** | 0.77 |
| L3 (reasoning, /5.0) | — | — | — | — | — |
| L4 (discrim, MCC) | **0.56** | 0.49 | 0.50 | 0.48 | 0.50 |

**Key finding:** CT LLMs show meaningful discrimination (MCC ~0.5). L3 ceiling effect — GPT-4o-mini judge gave 4.4–5.0/5 (too lenient). L2 Qwen/Llama outperform API models.

---

## PPI Domain (Protein-Protein Interaction)

**Status as of 2026-03-23:** ML 54/54 complete (seeds 42/43/44), LLM 80/80 complete, L3 judged.

### Database
- 2,220,786 negative results (IntAct 779 / HuRI 500K / hu.MAP 1.23M / STRING 500K)
- 61,728 positive pairs (HuRI Y2H), 18,412 proteins

### ML Results (PPI-M1: Binary Non-interaction Prediction)
Aggregated over 3 seeds (source: `results/ppi/table1_aggregated.md`):

| Model | Split | Negatives | AUROC | MCC | LogAUC |
|-------|-------|-----------|-------|-----|--------|
| SiameseCNN | random | NegBioDB | 0.963 ± 0.000 | 0.794 ± 0.012 | 0.517 ± 0.018 |
| PIPR | random | NegBioDB | 0.964 ± 0.001 | 0.812 ± 0.006 | 0.519 ± 0.009 |
| MLPFeatures | random | NegBioDB | 0.962 ± 0.001 | 0.788 ± 0.003 | 0.567 ± 0.005 |
| SiameseCNN | cold_protein | NegBioDB | 0.873 ± 0.002 | 0.568 ± 0.019 | 0.314 ± 0.014 |
| PIPR | cold_protein | NegBioDB | 0.859 ± 0.008 | 0.565 ± 0.019 | 0.288 ± 0.010 |
| **MLPFeatures** | **cold_protein** | **NegBioDB** | **0.931 ± 0.001** | **0.706 ± 0.005** | **0.476 ± 0.005** |
| SiameseCNN | cold_both | NegBioDB | 0.585 ± 0.040 | 0.070 ± 0.004 | 0.037 ± 0.010 |
| PIPR | cold_both | NegBioDB | 0.409 ± 0.077 | −0.018 ± 0.044 | 0.031 ± 0.019 |
| **MLPFeatures** | **cold_both** | **NegBioDB** | **0.950 ± 0.021** | **0.749 ± 0.043** | **0.595 ± 0.051** |
| SiameseCNN | random | uniform_random | 0.965 ± 0.001 | 0.806 ± 0.007 | 0.552 ± 0.002 |
| PIPR | random | uniform_random | 0.966 ± 0.000 | 0.810 ± 0.002 | 0.565 ± 0.005 |

**Key findings:**
- MLPFeatures dominates cold splits: AUROC 0.95 on cold_both (vs PIPR collapse to 0.41)
- PIPR catastrophic failure on cold_both: AUROC 0.41 (below random!)
- Control negatives (uniform_random, degree_matched) inflate LogAUC by +0.04–0.05

### LLM Results (80/80 complete, post-hoc fixes applied 2026-03-23)
| Task | Gemini | GPT-4o-mini | Haiku | Qwen-7B | Llama-8B |
|------|--------|-------------|-------|---------|---------|
| L1 (4-way MCQ, acc) zero-shot | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| L1 (4-way MCQ, acc) 3-shot | **1.000** | 0.997 | 0.998 | 0.998 | 0.997 |
| L2 (method_accuracy) 3-shot | **1.000** | ~0.94 | ~0.08 | ~0.94 | ~0.08 |
| L3 (reasoning, /5.0) zero-shot | 4.4–4.7 | 4.4–4.7 | 4.3–4.7 | 4.3–4.7 | 4.3–4.7 |
| L3 (reasoning, /5.0) 3-shot | 3.1–3.7 | 3.1–3.7 | 3.1–3.7 | 3.1–3.7 | 3.1–3.7 |
| L4 (discrim, MCC) zero-shot | 0.43 | **0.430** | 0.38 | 0.36 | **0.441** |

**Key findings:**
- L1: 3-shot near-perfect is an artifact (in-context examples reveal pattern format)
- L2: Gemini/GPT/Qwen correctly identify interaction methods, Haiku/Llama fail
- L3: zero-shot > 3-shot (gold reasoning examples degrade performance); structural reasoning collapses to ~1.2/5 in 3-shot
- L4: All models show contamination (acc_pre_2015 >> acc_post_2020)

---

## GE Domain (Gene Essentiality / DepMap)

**Status as of 2026-03-24:** ETL+ML complete, LLM 4/5 models done (Llama pending), L3 judged.

### Database
- 28,759,256 negative results (CRISPR 19.7M + RNAi 9.1M)
- Final tiers: Gold 753,878 / Silver 18,608,686 / Bronze 9,396,692
- 22,549,910 aggregated pairs (19,554 genes × 2,132 cell lines)

### ML Results (Seed 42 complete, seeds 43/44 submitted)
5 splits: random/cold_gene/cold_cell_line/cold_both/degree_balanced
Models: XGBoost, MLPFeatures
*Results pending — seeds 43/44 on scu-cpu queue, aggregated table not yet generated.*

### LLM Results (4/5 models, partial — Llama pending)
| Task | Note | Status |
|------|------|--------|
| L1 (4-way MCQ, 1,200 items) | Haiku/Gemini/GPT/Qwen done | Llama pending |
| L2 (extraction, 500 items) | fs0 reruns needed (Haiku/Gemini/GPT) | Qwen rerun queued (job 2716071) |
| L3 (reasoning, 200 items) | Judged: 160/160 parsed, zero-shot >> 3-shot (4.5 vs 2.5) | Llama pending |
| L4 (discrimination, 475 items) | Haiku/Gemini/GPT/Qwen done | Llama pending |

**Post-hoc fixes applied (2026-03-24):**
- L2: Gold JSONL schema patched (`essentiality_findings` → `genes`)
- L3 judge: Markdown fence stripping fix (Gemini returns ```json...```)
- Collector: `removesuffix("_mean")` fix

**Expected key finding:** GE L4 MCC likely intermediate between PPI and DTI given DepMap is a public widely-studied dataset with high training data overlap.

---

## Methodology Notes

### ML Evaluation Protocol
- **Metrics:** AUROC (primary), LogAUC (early enrichment), AUPRC, MCC, BEDROC
- **Seeds:** 3 seeds (42, 43, 44) for statistical robustness (except GE seed 42 only so far)
- **Negative types:** NegBioDB (structured negatives) vs control negatives (uniform_random, degree_matched)
- **Splits:** Random → Cold (one entity unseen) → Cold-Both (both entities unseen, hardest)

### LLM Evaluation Protocol
- **Models:** Claude Haiku-4.5, Gemini 2.5-Flash, GPT-4o-mini, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct
- **Configs:** Zero-shot × 1 + 3-shot × 3 fewshot sets = 4 configs per (model × task)
- **L3 judge:** Gemini 2.5-Flash LLM-as-judge, 4 dimensions × 5-point scale
- **L4 contamination test:** Older vs newer entity pairs (pre-cutoff vs post-cutoff)

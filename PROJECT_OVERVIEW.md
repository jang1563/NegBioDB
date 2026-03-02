# NegBioDB: Negative Results Database for Drug-Target Interactions

> Biology-first, Science-extensible negative results database and benchmark for AI/ML

## Project Vision

Approximately 90% of scientific experiments produce null or inconclusive results, yet the vast majority remain unpublished. This systematic gap fundamentally distorts AI/ML model training and evaluation.

**Goal:** Starting with Drug-Target Interactions (DTI), systematically collect and structure experimentally confirmed negative results, and build benchmarks for AI/ML training and evaluation.

## Why This Matters

1. **Publication Bias**: 85% of published papers report only positive results (as of 2007)
2. **AI Model Bias**: Models trained without negative data produce excessive false positives
3. **Economic Waste**: Duplicated experiments, failed drug discovery pipelines (billions of dollars)
4. **Proven Impact**: Models trained with negative data are more accurate (Organic Letters 2023, bioRxiv 2024)

## Scope & Strategy

```
Biology-first, Science-extensible Architecture
┌─────────────────────────────────────┐
│  Common Layer                        │
│  - Hypothesis structure              │
│  - Experimental metadata             │
│  - Outcome classification            │
│  - Confidence / Statistical power    │
│  - Author annotation                 │
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────────┐
    ▼          ▼              ▼
┌────────┐ ┌────────┐  ┌──────────┐
│Biology │ │Chem    │  │Materials │  ← Phase 2+
│(DTI)   │ │Domain  │  │Domain    │
└────────┘ └────────┘  └──────────┘
```

**Expansion Path:** DTI → Gene Function → Clinical Trial → Chemistry → Materials Science

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | Biology-first | Most severe problem, highest commercial value, largest AI evaluation gap |
| Starting Domain | Drug-Target Interaction | Data accessibility + existing infrastructure (ChEMBL) + pharma demand |
| Architecture | Extensible (common + domain layers) | Future expansion to Chemistry, Materials |

## Project Documents

| Document | Description |
|----------|-------------|
| [research/01_dti_negative_data_landscape.md](research/01_dti_negative_data_landscape.md) | Current DTI negative data sources landscape |
| [research/02_benchmark_analysis.md](research/02_benchmark_analysis.md) | Analysis of existing DTI benchmarks and their negative data handling |
| [research/03_data_collection_methodology.md](research/03_data_collection_methodology.md) | Data collection, curation, and structuring methodologies |
| [research/04_publication_commercial_strategy.md](research/04_publication_commercial_strategy.md) | Academic publication and commercialization strategy |
| [ROADMAP.md](ROADMAP.md) | Execution roadmap |

## Timeline
- Project initiated: 2026-03-02
- Last updated: 2026-03-02

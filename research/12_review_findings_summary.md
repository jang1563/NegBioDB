# NegBioDB Plan/Implementation Document Review Summary (2026-03-02)

## Review Scope
- `PROJECT_OVERVIEW.md`
- `ROADMAP.md`
- `research/05_technical_deep_dive.md`
- `research/06_paper_narrative.md`
- `research/08_expert_review_and_feasibility.md`
- `research/09_schema_and_ml_export_design.md`
- `research/10_expert_panel_review.md`
- `research/11_full_plan_review.md`

## Key Conclusion
- The strategic direction is strong, but there are still schema/pipeline consistency issues that can cause implementation failure.
- In particular, migration versioning, large-scale streaming behavior, and positive/negative data integration rules must be fixed before implementation.

## Major Issues (Ordered by Severity)

1. `CRITICAL` Migration idempotency flaw
- Issue: File-based migration versions (`001`, `002`) and DB-recorded versions (`1.0`, `1.1`) use different formats, so already-applied migrations may be reapplied.
- Impact: Migration conflicts and possible schema corruption during reruns/redeployments.
- Evidence:
  - `research/09_schema_and_ml_export_design.md:701`
  - `research/09_schema_and_ml_export_design.md:737`
  - `research/09_schema_and_ml_export_design.md:745`
  - `research/09_schema_and_ml_export_design.md:334`
  - `research/09_schema_and_ml_export_design.md:719`

2. `CRITICAL` Contradiction between "streaming required" and example implementation
- Issue: The streaming example accumulates filtered chunks in memory and concatenates at the end.
- Impact: OOM risk remains for large PubChem processing.
- Evidence:
  - `research/09_schema_and_ml_export_design.md:1221`
  - `research/09_schema_and_ml_export_design.md:1233`
  - `research/09_schema_and_ml_export_design.md:1252`
  - `research/09_schema_and_ml_export_design.md:1260`

3. `CRITICAL` M1 binary DTI requirement conflicts with export implementation
- Issue: The plan requires positive-data pairing, but export SQL hardcodes `Y=0` (negative-only).
- Impact: Exp 1/Exp 4 and M1 binary task cannot be reproduced as specified.
- Evidence:
  - `ROADMAP.md:68`
  - `ROADMAP.md:200`
  - `research/09_schema_and_ml_export_design.md:521`

4. `HIGH` DAVIS inactive threshold inconsistency
- Issue: Internal threshold statements conflict logically (`min Kd=10,000 nM` vs `pKd < 5`).
- Impact: DAVIS negative set may be built incorrectly or severely distorted.
- Evidence:
  - `research/05_technical_deep_dive.md:126`
  - `research/05_technical_deep_dive.md:128`
  - `ROADMAP.md:176`
  - `ROADMAP.md:196`

5. `HIGH` Duplicate-prevention index is weak with NULL handling
- Issue: UNIQUE index includes nullable `assay_id`, which can allow duplicate rows in NULL cases under SQLite semantics.
- Impact: Source-level duplication and distorted pair aggregation.
- Evidence:
  - `research/09_schema_and_ml_export_design.md:165`
  - `research/09_schema_and_ml_export_design.md:223`
  - `research/09_schema_and_ml_export_design.md:224`

6. `HIGH` Standard export column definition does not match actual SQL select
- Issue: Documented standard columns (`compound_id`, `target_id`, `result_type`, `publication_year`, etc.) are partially missing in the SQL example.
- Impact: Interface mismatch between downstream code and documentation.
- Evidence:
  - `research/09_schema_and_ml_export_design.md:459`
  - `research/09_schema_and_ml_export_design.md:471`
  - `research/09_schema_and_ml_export_design.md:473`
  - `research/09_schema_and_ml_export_design.md:514`

7. `MODERATE` Citation verification status is inconsistent across documents
- Issue: One document marks citation verification as completed, while others still mark them as `[UNVERIFIED]`.
- Impact: Confusion in evidence reliability and progress tracking.
- Evidence:
  - `research/10_expert_panel_review.md:185`
  - `research/06_paper_narrative.md:350`
  - `research/06_paper_narrative.md:351`
  - `ROADMAP.md:26`

8. `MODERATE` Required submission metadata still has placeholders
- Issue: Author/contact/checksum/citation fields are still placeholders.
- Impact: Metadata quality risk near submission.
- Evidence:
  - `research/09_schema_and_ml_export_design.md:838`
  - `research/09_schema_and_ml_export_design.md:864`
  - `research/09_schema_and_ml_export_design.md:1067`
  - `research/09_schema_and_ml_export_design.md:1189`

9. `LOW` Checklist references wrong section
- Issue: The referenced section for standardization implementation does not match where the actual content exists.
- Impact: Implementer confusion.
- Evidence:
  - `research/09_schema_and_ml_export_design.md:1316`
  - `research/05_technical_deep_dive.md:48`
  - `research/05_technical_deep_dive.md:256`

## Additional Clarifications Needed
1. Decide and enforce one `schema_migrations.version` policy (`001` style vs `1.0` style).
2. Fix a single DAVIS inactive criterion (`Kd >= 10,000 nM` or `pKd <= 5`) and synchronize all documents.

## Note
- This document is a review summary only and does not include code modifications.

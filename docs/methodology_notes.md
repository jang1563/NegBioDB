# Methodology Notes for Paper

These notes address known limitations and design decisions that should be
documented in the paper methodology section.

## Temporal Split Limitation (C3)

The temporal split (pre-2020 train / 2020-2021 val / 2022+ test) yields a
highly imbalanced distribution (train ~99.7%, val ~0.14%, test ~0.14%)
reflecting the historical concentration of bioactivity data before 2020. We
retain this split for completeness as a chronological validation, while noting
that cold-compound and cold-target splits provide more robust generalization
assessment.

## L1 Context Design (C4)

L1 provides contextual assay data (activity types and values) alongside the
question, testing the model's ability to interpret bioactivity data rather
than factual recall. This is intentional: L4 tests factual recall without
context, while L1 evaluates data interpretation capability. The context text
includes activity measurements that inform the correct answer, simulating a
scientist reviewing assay results.

## Contamination Threshold (M12)

We flag potential data contamination when pre-2023 accuracy exceeds post-2024
accuracy by > 15 percentage points. This threshold balances sensitivity to
temporal bias against random fluctuation in small subsets. Models showing
higher performance on older data may have encountered these compound-target
pairs during pre-training.

## Scaffold Split Coverage (m1)

The scaffold split assigns compounds to folds based on Murcko scaffold
grouping. The number of unique scaffolds and their pair distribution should
be reported. If the dataset contains fewer than ~100 unique scaffolds, this
limitation should be noted as it may reduce the generalization challenge of
the scaffold split relative to the cold-compound split.

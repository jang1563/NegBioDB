"""Failure classification pipeline for NegBioDB-CT.

Three-tier detection:
  Tier 1: Terminated trials + NLP on why_stopped text → bronze
  Tier 2: Completed trials + p_value > 0.05 on primary → silver/gold
  Tier 3: CTO binary failure labels (gap-fill) → copper

Open Targets curated stop reason labels (Apache 2.0) are used to train
a TF-IDF + LinearSVC classifier for Tier 1. Cross-validation is run to
verify accuracy >= 60% before use; otherwise falls back to keyword rules.

Failure categories: safety, efficacy, pharmacokinetic, enrollment,
strategic, regulatory, design, other.
"""

import logging
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from negbiodb_ct.ct_db import (
    get_connection,
    DEFAULT_CT_DB_PATH,
)
from negbiodb.download import load_config

logger = logging.getLogger(__name__)

BATCH_SIZE = 500

# ============================================================
# Open Targets → NegBioDB-CT category mapping
# OT has ~17 categories; we collapse to 8
# ============================================================

OT_CATEGORY_MAP = {
    # Safety
    "safety_sideeffects": "safety",
    # Efficacy
    "negative": "efficacy",
    "interim_analysis": "efficacy",
    # Enrollment
    "insufficient_enrollment": "enrollment",
    # Strategic
    "business_administrative": "strategic",
    "another_study": "strategic",
    # Regulatory
    "regulatory": "regulatory",
    # Design
    "study_design": "design",
    "insufficient_data": "design",
    # Other
    "no_context": "other",
    "logistics_resources": "other",
    "ethical_reason": "other",
    "covid19": "other",
    "study_staff_moved": "other",
    "invalid_reason": "other",
    # Positive outcomes → None (skipped)
    "success": None,
    "endpoint_met": None,
}

# Keywords for rule-based fallback classification
KEYWORD_RULES = [
    ("safety", re.compile(
        r"safe|toxic|adverse|death|fatal|SAE|SUSAR|side.?effect|hepato|cardio|liver",
        re.IGNORECASE)),
    ("efficacy", re.compile(
        r"efficac|futili|lack.?of.?effect|ineffect|no.?benefit|negative|"
        r"primary.?endpoint.?not|did.?not.?meet|failed.?to.?demonstrate",
        re.IGNORECASE)),
    ("pharmacokinetic", re.compile(
        r"pharmacokinetic|PK|bioavailab|ADME|absorption|metabolism|"
        r"drug.?drug.?interaction|DDI",
        re.IGNORECASE)),
    ("enrollment", re.compile(
        r"enroll|recruit|accrual|low.?participation|insufficient.?patient|"
        r"slow.?enroll|unable.?to.?enroll",
        re.IGNORECASE)),
    ("strategic", re.compile(
        r"strategic|business|commercial|sponsor|portfolio|company|financial|"
        r"merger|acquisition|decision|funding|priority",
        re.IGNORECASE)),
    ("regulatory", re.compile(
        r"regulat|FDA|EMA|approval|IND|clinical.?hold|CRL",
        re.IGNORECASE)),
    ("design", re.compile(
        r"design|protocol|amendment|endpoint.?change|sample.?size|"
        r"inadequate.?power|statistical",
        re.IGNORECASE)),
]


# ============================================================
# LOAD TRAINING DATA
# ============================================================


def load_opentargets_labels(parquet_path: Path) -> pd.DataFrame:
    """Load Open Targets curated stop reason labels.

    Uses the 17 boolean one-hot columns (not label_descriptions, which is
    a numpy array and unreliable for string matching). Maps column names
    through OT_CATEGORY_MAP to our 8 failure categories.

    Multi-label rows are resolved via resolve_multi_label() precedence.
    Rows with only positive labels (Endpoint_Met, Success) are dropped.

    Returns DataFrame with columns: text, label.
    """
    df = pd.read_parquet(parquet_path)
    logger.info("Open Targets raw: %d rows, columns: %s", len(df), list(df.columns))

    # Find text column
    text_col = None
    for col in df.columns:
        if col.lower() in ("text", "reason", "why_stopped"):
            text_col = col
            break
    if text_col is None:
        raise ValueError(
            f"Cannot identify text column in OT data: {list(df.columns)}"
        )

    # Find boolean label columns
    bool_cols = [c for c in df.columns if df[c].dtype == "bool"]
    if not bool_cols:
        raise ValueError(
            f"No boolean label columns found in OT data: {list(df.columns)}"
        )
    logger.info("Using text_col=%s, %d boolean label columns", text_col, len(bool_cols))

    # Map each row's True boolean columns → failure categories
    labels = []
    for _, row in df.iterrows():
        active = [c for c in bool_cols if row[c]]
        mapped = []
        for col_name in active:
            cat = OT_CATEGORY_MAP.get(col_name.lower())
            if cat is not None:
                mapped.append(cat)
        if mapped:
            labels.append(resolve_multi_label(mapped))
        else:
            labels.append(None)

    result = pd.DataFrame({"text": df[text_col].values, "label": labels})

    before = len(result)
    result = result[result["label"].notna()].copy()
    dropped = before - len(result)
    if dropped:
        logger.info("Dropped %d rows (positive/unmapped OT labels)", dropped)

    logger.info(
        "Open Targets training data: %d rows, %d categories",
        len(result), result["label"].nunique(),
    )
    return result[["text", "label"]]


def load_cto_outcomes(parquet_path: Path) -> pd.DataFrame:
    """Load CTO binary outcome labels.

    Returns DataFrame with columns: nct_id, outcome (0=failure, 1=success).
    """
    df = pd.read_parquet(parquet_path)
    logger.info("CTO raw: %d rows, columns: %s", len(df), list(df.columns))

    nct_col = None
    outcome_col = None
    for col in df.columns:
        cl = col.lower()
        if nct_col is None and (cl == "nct_id" or cl == "nctid" or "trial_id" in cl):
            nct_col = col
        elif outcome_col is None and ("outcome" in cl or "label" in cl or "success" in cl):
            outcome_col = col

    if nct_col is None or outcome_col is None:
        raise ValueError(
            f"Cannot identify nct_id/outcome columns in CTO data: {list(df.columns)}"
        )

    result = df[[nct_col, outcome_col]].copy()
    result.columns = ["nct_id", "outcome"]
    result["nct_id"] = result["nct_id"].astype(str).str.strip()

    if result["outcome"].dtype == "object":
        result["outcome"] = result["outcome"].map(
            {"failure": 0, "success": 1, "0": 0, "1": 1, "fail": 0, "pass": 1}
        )

    result = result.dropna(subset=["outcome"])
    result["outcome"] = result["outcome"].astype(int)

    logger.info(
        "CTO outcomes: %d total, %d failures, %d successes",
        len(result), (result["outcome"] == 0).sum(), (result["outcome"] == 1).sum(),
    )
    return result


# ============================================================
# NLP CLASSIFIER
# ============================================================


def train_failure_classifier(
    training_data: pd.DataFrame,
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
    cv_folds: int = 5,
) -> tuple:
    """Train TF-IDF + LinearSVC failure classifier.

    Returns (vectorizer, classifier, cv_accuracy).
    Returns (None, None, cv_accuracy) if accuracy < 60%.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    texts = training_data["text"].fillna("").astype(str).tolist()
    labels = training_data["label"].tolist()

    logger.info(
        "Training classifier: %d samples, %d classes",
        len(texts), len(set(labels)),
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
    )
    classifier = LinearSVC(max_iter=5000, class_weight="balanced")
    pipeline = Pipeline([("tfidf", vectorizer), ("svc", classifier)])

    scores = cross_val_score(pipeline, texts, labels, cv=cv_folds, scoring="accuracy")
    cv_accuracy = float(np.mean(scores))
    logger.info("CV accuracy: %.3f (+/- %.3f)", cv_accuracy, float(np.std(scores)))

    if cv_accuracy < 0.60:
        logger.warning(
            "CV accuracy %.3f below 60%% threshold. Falling back to keyword rules.",
            cv_accuracy,
        )
        return None, None, cv_accuracy

    pipeline.fit(texts, labels)
    return pipeline["tfidf"], pipeline["svc"], cv_accuracy


def classify_text_nlp(text: str, vectorizer, classifier) -> str | None:
    """Classify a single why_stopped text using trained NLP model."""
    if not text or not text.strip():
        return None
    X = vectorizer.transform([text])
    return classifier.predict(X)[0]


def classify_text_keywords(text: str) -> str | None:
    """Rule-based fallback classification using keyword patterns."""
    if not text or not text.strip():
        return None
    for category, pattern in KEYWORD_RULES:
        if pattern.search(text):
            return category
    return "other"


# ============================================================
# MULTI-LABEL RESOLUTION
# ============================================================


def resolve_multi_label(
    categories: list[str],
    precedence: list[str] | None = None,
) -> str:
    """Pick single category using precedence ordering.

    Default: safety > efficacy > pharmacokinetic > enrollment >
    strategic > regulatory > design > other.
    """
    if precedence is None:
        precedence = [
            "safety", "efficacy", "pharmacokinetic", "enrollment",
            "strategic", "regulatory", "design", "other",
        ]
    cat_set = set(categories)
    for cat in precedence:
        if cat in cat_set:
            return cat
    return "other"


# ============================================================
# TERMINATION TYPE ASSIGNMENT
# ============================================================

_CLINICAL_PAT = re.compile(
    r"efficac|futili|safe|toxic|adverse|endpoint|fail|no.?benefit|"
    r"lack.?of|pharmacokinetic|negative.?result|death|dsmb",
    re.IGNORECASE,
)
_ADMIN_PAT = re.compile(
    r"business|strategic|sponsor|funding|portfolio|company|"
    r"administrative|logistic|resource|decision.?to",
    re.IGNORECASE,
)
_EXTERNAL_PAT = re.compile(
    r"covid|pandemic|earthquake|hurricane|natural.?disaster|war",
    re.IGNORECASE,
)


def assign_termination_types(conn: sqlite3.Connection) -> dict[str, int]:
    """Classify terminated trials into termination_type categories.

    Rules:
    - why_stopped contains clinical keywords → clinical_failure
    - why_stopped contains admin keywords → administrative
    - why_stopped IS NULL and status = Terminated → unknown
    """
    stats: dict[str, int] = {
        "clinical_failure": 0, "administrative": 0,
        "external_event": 0, "unknown": 0,
    }

    cursor = conn.execute(
        "SELECT trial_id, why_stopped FROM clinical_trials "
        "WHERE overall_status = 'Terminated' AND termination_type IS NULL"
    )

    updates: list[tuple[str, int]] = []
    for trial_id, why_stopped in cursor.fetchall():
        if why_stopped and why_stopped.strip():
            text = why_stopped
            if _CLINICAL_PAT.search(text):
                term_type = "clinical_failure"
            elif _EXTERNAL_PAT.search(text):
                term_type = "external_event"
            elif _ADMIN_PAT.search(text):
                term_type = "administrative"
            else:
                term_type = "clinical_failure"
        else:
            term_type = "unknown"

        updates.append((term_type, trial_id))
        stats[term_type] += 1

    for i in range(0, len(updates), BATCH_SIZE):
        conn.executemany(
            "UPDATE clinical_trials SET termination_type = ? WHERE trial_id = ?",
            updates[i:i + BATCH_SIZE],
        )
    conn.commit()

    logger.info("Termination type assignment: %s", stats)
    return stats


# ============================================================
# TIER 1: TERMINATED TRIALS + NLP
# ============================================================


def classify_terminated_trials(
    conn: sqlite3.Connection,
    vectorizer,
    classifier,
    use_keywords: bool = False,
) -> list[dict]:
    """Tier 1: Classify terminated trials with why_stopped text.

    Only processes trials where termination_type = 'clinical_failure'.
    Returns list of failure result dicts ready for insertion.
    """
    cursor = conn.execute("""
        SELECT DISTINCT
            ct.trial_id, ct.source_trial_id, ct.why_stopped,
            ct.trial_phase, ct.has_results,
            ti.intervention_id, tc.condition_id
        FROM clinical_trials ct
        JOIN trial_interventions ti ON ct.trial_id = ti.trial_id
        JOIN trial_conditions tc ON ct.trial_id = tc.trial_id
        WHERE ct.termination_type = 'clinical_failure'
          AND ct.why_stopped IS NOT NULL
          AND ct.why_stopped != ''
    """)

    rows = cursor.fetchall()
    logger.info("Tier 1 candidates: %d triples", len(rows))

    results: list[dict] = []
    for trial_id, nct_id, why_stopped, phase, has_results, interv_id, cond_id in tqdm(
        rows, desc="Tier 1: NLP classify"
    ):
        if use_keywords or vectorizer is None:
            category = classify_text_keywords(why_stopped)
        else:
            category = classify_text_nlp(why_stopped, vectorizer, classifier)

        if category is None:
            category = "other"

        interpretation = "definitive_negative"
        if category == "safety":
            interpretation = "safety_stopped"
        elif not has_results:
            interpretation = "inconclusive_underpowered"

        results.append({
            "intervention_id": interv_id,
            "condition_id": cond_id,
            "trial_id": trial_id,
            "failure_category": category,
            "failure_detail": why_stopped,
            "confidence_tier": "bronze",
            "highest_phase_reached": phase,
            "source_db": "clinicaltrials_gov",
            "source_record_id": f"terminated:{nct_id}",
            "extraction_method": "nlp_classified",
            "result_interpretation": interpretation,
        })

    logger.info("Tier 1 results: %d", len(results))
    return results


# ============================================================
# TIER 2: ENDPOINT FAILURES (p > 0.05)
# ============================================================


def detect_endpoint_failures(
    conn: sqlite3.Connection,
    data_dir: Path,
) -> list[dict]:
    """Tier 2: Detect failures from AACT outcome_analyses (p > 0.05).

    Identifies trials where primary endpoint p-value > 0.05.
    Returns list of failure result dicts ready for insertion.
    """
    from negbiodb_ct.etl_aact import load_aact_table

    try:
        oa = load_aact_table(data_dir, "outcome_analyses", usecols=[
            "nct_id", "p_value", "p_value_description", "param_type",
            "method", "ci_lower_limit", "ci_upper_limit",
        ])
    except FileNotFoundError:
        logger.warning("outcome_analyses.txt not found, skipping Tier 2")
        return []

    oa = oa.dropna(subset=["p_value"])
    oa["p_value"] = pd.to_numeric(oa["p_value"], errors="coerce")
    oa = oa.dropna(subset=["p_value"])

    # Filter to valid p-values (0 ≤ p ≤ 1); values > 1 are typically
    # test statistics mis-entered as p-values in the AACT source data.
    before_range = len(oa)
    oa = oa[(oa["p_value"] >= 0) & (oa["p_value"] <= 1)].copy()
    dropped_range = before_range - len(oa)
    if dropped_range:
        logger.info("Dropped %d rows with p_value outside [0, 1]", dropped_range)

    failures = oa[oa["p_value"] > 0.05].copy()
    logger.info("Outcome analyses with p > 0.05: %d", len(failures))

    if len(failures) == 0:
        return []

    nct_ids = tuple(failures["nct_id"].unique().tolist())
    if not nct_ids:
        return []

    placeholders = ",".join(["?"] * len(nct_ids))
    cursor = conn.execute(
        f"SELECT source_trial_id, trial_id, trial_phase, has_results "
        f"FROM clinical_trials WHERE source_trial_id IN ({placeholders})",
        nct_ids,
    )
    trial_info = {
        row[0]: {"trial_id": row[1], "phase": row[2], "has_results": row[3]}
        for row in cursor.fetchall()
    }

    results: list[dict] = []
    seen: set[tuple] = set()

    for _, row in tqdm(failures.iterrows(), total=len(failures),
                       desc="Tier 2: endpoint failures"):
        nct_id = str(row["nct_id"])
        info = trial_info.get(nct_id)
        if not info:
            continue

        trial_id = info["trial_id"]

        interv_ids = [r[0] for r in conn.execute(
            "SELECT intervention_id FROM trial_interventions WHERE trial_id = ?",
            (trial_id,),
        ).fetchall()]
        cond_ids = [r[0] for r in conn.execute(
            "SELECT condition_id FROM trial_conditions WHERE trial_id = ?",
            (trial_id,),
        ).fetchall()]

        for interv_id in interv_ids:
            for cond_id in cond_ids:
                key = (interv_id, cond_id, trial_id)
                if key in seen:
                    continue
                seen.add(key)

                phase = info.get("phase")
                if phase in ("phase_3", "phase_4") and info.get("has_results"):
                    tier = "gold"
                elif phase in ("phase_2", "phase_2_3", "phase_3"):
                    tier = "silver"
                else:
                    tier = "bronze"

                p_val = float(row["p_value"])
                ci_lower = (
                    float(row["ci_lower_limit"])
                    if pd.notna(row.get("ci_lower_limit")) else None
                )
                ci_upper = (
                    float(row["ci_upper_limit"])
                    if pd.notna(row.get("ci_upper_limit")) else None
                )

                results.append({
                    "intervention_id": interv_id,
                    "condition_id": cond_id,
                    "trial_id": trial_id,
                    "failure_category": "efficacy",
                    "failure_detail": (
                        f"p={p_val:.4f} ({row.get('method', 'unknown method')})"
                    ),
                    "confidence_tier": tier,
                    "p_value_primary": p_val,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "primary_endpoint_met": 0,
                    "highest_phase_reached": phase,
                    "source_db": "clinicaltrials_gov",
                    "source_record_id": f"outcome:{nct_id}",
                    "extraction_method": "database_direct",
                    "result_interpretation": "definitive_negative",
                })

    logger.info("Tier 2 results: %d", len(results))
    return results


# ============================================================
# TIER 3: CTO GAP-FILL
# ============================================================


def enrich_with_cto(
    conn: sqlite3.Connection,
    cto_df: pd.DataFrame,
) -> list[dict]:
    """Tier 3: Add copper-tier failure results from CTO binary labels.

    Only adds results for trials NOT already covered by Tier 1/2.
    """
    covered_ncts: set[str] = set()
    cursor = conn.execute(
        "SELECT DISTINCT ct.source_trial_id "
        "FROM trial_failure_results tfr "
        "JOIN clinical_trials ct ON tfr.trial_id = ct.trial_id"
    )
    for row in cursor.fetchall():
        covered_ncts.add(row[0])

    logger.info("Already covered NCT IDs: %d", len(covered_ncts))

    cto_failures = cto_df[cto_df["outcome"] == 0].copy()
    cto_failures = cto_failures[~cto_failures["nct_id"].isin(covered_ncts)]
    logger.info("CTO gap-fill candidates: %d", len(cto_failures))

    if len(cto_failures) == 0:
        return []

    nct_list = cto_failures["nct_id"].unique().tolist()

    # Batch IN clause to avoid SQLite variable limit
    trial_info: dict[str, dict] = {}
    for i in range(0, len(nct_list), BATCH_SIZE):
        batch = nct_list[i:i + BATCH_SIZE]
        placeholders = ",".join(["?"] * len(batch))
        cursor = conn.execute(
            f"SELECT source_trial_id, trial_id, trial_phase "
            f"FROM clinical_trials WHERE source_trial_id IN ({placeholders})",
            batch,
        )
        for row in cursor.fetchall():
            trial_info[row[0]] = {"trial_id": row[1], "phase": row[2]}

    # Pre-fetch interventions and conditions in bulk (avoid N+1 queries)
    trial_ids = [info["trial_id"] for info in trial_info.values()]
    trial_to_intervs: dict[int, list[int]] = {}
    trial_to_conds: dict[int, list[int]] = {}

    for i in range(0, len(trial_ids), BATCH_SIZE):
        batch = trial_ids[i:i + BATCH_SIZE]
        placeholders = ",".join(["?"] * len(batch))

        for row in conn.execute(
            f"SELECT trial_id, intervention_id FROM trial_interventions "
            f"WHERE trial_id IN ({placeholders})", batch,
        ).fetchall():
            trial_to_intervs.setdefault(row[0], []).append(row[1])

        for row in conn.execute(
            f"SELECT trial_id, condition_id FROM trial_conditions "
            f"WHERE trial_id IN ({placeholders})", batch,
        ).fetchall():
            trial_to_conds.setdefault(row[0], []).append(row[1])

    results: list[dict] = []
    seen: set[tuple] = set()

    for _, row in tqdm(cto_failures.iterrows(), total=len(cto_failures),
                       desc="Tier 3: CTO gap-fill"):
        nct_id = row["nct_id"]
        info = trial_info.get(nct_id)
        if not info:
            continue

        trial_id = info["trial_id"]
        interv_ids = trial_to_intervs.get(trial_id, [])
        cond_ids = trial_to_conds.get(trial_id, [])

        for interv_id in interv_ids:
            for cond_id in cond_ids:
                key = (interv_id, cond_id, trial_id)
                if key in seen:
                    continue
                seen.add(key)

                results.append({
                    "intervention_id": interv_id,
                    "condition_id": cond_id,
                    "trial_id": trial_id,
                    "failure_category": "other",
                    "failure_detail": "CTO binary failure label",
                    "confidence_tier": "copper",
                    "highest_phase_reached": info.get("phase"),
                    "source_db": "cto",
                    "source_record_id": f"cto:{nct_id}",
                    "extraction_method": "text_mining",
                    "result_interpretation": None,
                })

    logger.info("Tier 3 results: %d", len(results))
    return results


# ============================================================
# INSERT RESULTS
# ============================================================


def insert_failure_results(
    conn: sqlite3.Connection,
    results: list[dict],
) -> int:
    """Insert trial_failure_results rows (INSERT OR IGNORE).

    Returns number of rows actually inserted.
    """
    if not results:
        return 0

    cols = [
        "intervention_id", "condition_id", "trial_id",
        "failure_category", "failure_detail", "confidence_tier",
        "p_value_primary", "ci_lower", "ci_upper",
        "primary_endpoint_met", "highest_phase_reached",
        "source_db", "source_record_id", "extraction_method",
        "result_interpretation",
    ]
    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)
    sql = (
        f"INSERT OR IGNORE INTO trial_failure_results ({col_names}) "
        f"VALUES ({placeholders})"
    )

    count_before = conn.execute(
        "SELECT COUNT(*) FROM trial_failure_results"
    ).fetchone()[0]

    for i in tqdm(range(0, len(results), BATCH_SIZE),
                  desc="Insert failure results"):
        batch = results[i:i + BATCH_SIZE]
        rows = [tuple(r.get(c) for c in cols) for r in batch]
        conn.executemany(sql, rows)

    conn.commit()

    count_after = conn.execute(
        "SELECT COUNT(*) FROM trial_failure_results"
    ).fetchone()[0]

    inserted = count_after - count_before
    logger.info("Inserted %d failure results (of %d attempted)", inserted, len(results))
    return inserted


# ============================================================
# ORCHESTRATOR
# ============================================================


def run_classification_pipeline(
    db_path: Path = DEFAULT_CT_DB_PATH,
    data_dir: Path | None = None,
) -> dict:
    """Run the full failure classification pipeline.

    Steps:
      1. Assign termination types to terminated trials
      2. Load Open Targets labels and train NLP classifier
      3. Tier 1: Classify terminated trials via NLP
      4. Tier 2: Detect endpoint failures via p-values
      5. Tier 3: Gap-fill from CTO binary labels
      6. Insert all results

    Returns dict with classification stats.
    """
    if data_dir is None:
        cfg = load_config()
        data_dir = Path(cfg["ct_domain"]["downloads"]["aact"]["dest_dir"])

    cfg = load_config()
    ct_cfg = cfg["ct_domain"]
    classifier_cfg = ct_cfg.get("classifier", {})

    conn = get_connection(db_path)
    stats: dict = {}

    try:
        # Step 1: Assign termination types
        logger.info("=== Step 1: Assign termination types ===")
        term_stats = assign_termination_types(conn)
        stats["termination_types"] = term_stats

        # Step 2: Train NLP classifier
        logger.info("=== Step 2: Train NLP classifier ===")
        ot_path = Path(ct_cfg["downloads"]["opentargets"]["dest"])
        vectorizer = None
        classifier = None
        cv_accuracy = 0.0
        use_keywords = True

        if ot_path.exists():
            training_data = load_opentargets_labels(ot_path)
            if len(training_data) >= 100:
                vectorizer, classifier, cv_accuracy = train_failure_classifier(
                    training_data,
                    max_features=classifier_cfg.get("max_features", 10000),
                    ngram_range=tuple(classifier_cfg.get("ngram_range", [1, 2])),
                    cv_folds=classifier_cfg.get("cv_folds", 5),
                )
                use_keywords = vectorizer is None
            else:
                logger.warning(
                    "Only %d OT training samples, using keyword rules",
                    len(training_data),
                )
        else:
            logger.warning(
                "Open Targets data not found at %s, using keyword rules", ot_path
            )

        stats["cv_accuracy"] = cv_accuracy
        stats["use_keywords"] = use_keywords

        # Step 3: Tier 1 — terminated trials
        logger.info("=== Step 3: Tier 1 — Terminated trials ===")
        tier1_results = classify_terminated_trials(
            conn, vectorizer, classifier, use_keywords=use_keywords
        )
        n_tier1 = insert_failure_results(conn, tier1_results)
        stats["tier1_inserted"] = n_tier1

        # Step 4: Tier 2 — endpoint failures
        logger.info("=== Step 4: Tier 2 — Endpoint failures ===")
        tier2_results = detect_endpoint_failures(conn, data_dir)
        n_tier2 = insert_failure_results(conn, tier2_results)
        stats["tier2_inserted"] = n_tier2

        # Step 5: Tier 3 — CTO gap-fill
        logger.info("=== Step 5: Tier 3 — CTO gap-fill ===")
        cto_path = Path(ct_cfg["downloads"]["cto"]["dest"])
        n_tier3 = 0
        if cto_path.exists():
            cto_df = load_cto_outcomes(cto_path)
            tier3_results = enrich_with_cto(conn, cto_df)
            n_tier3 = insert_failure_results(conn, tier3_results)
        else:
            logger.warning("CTO data not found at %s, skipping Tier 3", cto_path)
        stats["tier3_inserted"] = n_tier3

        # Summary
        total = conn.execute(
            "SELECT COUNT(*) FROM trial_failure_results"
        ).fetchone()[0]
        stats["total_failure_results"] = total

        cursor = conn.execute(
            "SELECT failure_category, COUNT(*) "
            "FROM trial_failure_results GROUP BY failure_category "
            "ORDER BY COUNT(*) DESC"
        )
        stats["category_distribution"] = {
            row[0]: row[1] for row in cursor.fetchall()
        }

        cursor = conn.execute(
            "SELECT confidence_tier, COUNT(*) "
            "FROM trial_failure_results GROUP BY confidence_tier "
            "ORDER BY COUNT(*) DESC"
        )
        stats["tier_distribution"] = {
            row[0]: row[1] for row in cursor.fetchall()
        }

    finally:
        conn.close()

    logger.info("=== Classification Pipeline Complete ===")
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)

    return stats

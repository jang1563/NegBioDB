"""Outcome enrichment pipeline for NegBioDB-CT.

Extracts structured p-values and effect sizes from two sources:
  1. AACT outcome_analyses (already downloaded, pipe-delimited)
  2. Shi & Du 2024 (119K efficacy + 803K safety records, CC0)

Updates existing trial_failure_results with quantitative data.
Upgrades confidence tiers when quantitative evidence is added.

Tier upgrade logic:
  - Bronze + quantitative outcome → Silver
  - Silver + Phase III + linked PubMed → Gold
"""

import logging
import sqlite3
from pathlib import Path

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
# AACT OUTCOME ANALYSES
# ============================================================


def load_aact_outcomes(data_dir: Path) -> pd.DataFrame:
    """Load and filter AACT outcome_analyses to usable p-values.

    Returns DataFrame with: nct_id, p_value, method, ci_lower, ci_upper,
    param_type, effect_size_type (from param_type).
    """
    from negbiodb_ct.etl_aact import load_aact_table

    try:
        oa = load_aact_table(data_dir, "outcome_analyses", usecols=[
            "nct_id", "p_value", "p_value_description",
            "param_type", "param_value", "method",
            "ci_lower_limit", "ci_upper_limit",
        ])
    except FileNotFoundError:
        logger.warning("outcome_analyses.txt not found")
        return pd.DataFrame()

    # Convert and filter
    oa["p_value"] = pd.to_numeric(oa["p_value"], errors="coerce")
    oa = oa.dropna(subset=["p_value"])

    # Filter to valid p-values [0, 1]
    oa = oa[(oa["p_value"] >= 0) & (oa["p_value"] <= 1)].copy()

    # Convert effect size columns
    oa["param_value"] = pd.to_numeric(oa["param_value"], errors="coerce")
    oa["ci_lower_limit"] = pd.to_numeric(oa["ci_lower_limit"], errors="coerce")
    oa["ci_upper_limit"] = pd.to_numeric(oa["ci_upper_limit"], errors="coerce")

    logger.info("AACT outcome analyses: %d rows with valid p-values", len(oa))
    return oa


def load_shi_du_efficacy(csv_path: Path) -> pd.DataFrame:
    """Load Shi & Du 2024 efficacy results.

    Returns DataFrame with: nct_id, p_value, effect_size, effect_size_type,
    ci_lower, ci_upper, endpoint_met.
    """
    if not csv_path.exists():
        logger.warning("Shi & Du efficacy file not found: %s", csv_path)
        return pd.DataFrame()

    df = pd.read_csv(csv_path, low_memory=False)
    logger.info("Shi & Du efficacy raw: %d rows, columns: %s", len(df), list(df.columns))

    # Find relevant columns (names may vary)
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "nct" in cl:
            col_map["nct_id"] = col
        elif "p_value" in cl or "pvalue" in cl:
            col_map["p_value"] = col
        elif "effect" in cl and "size" in cl:
            col_map["effect_size"] = col
        elif "ci_lower" in cl or "lower" in cl:
            col_map["ci_lower"] = col
        elif "ci_upper" in cl or "upper" in cl:
            col_map["ci_upper"] = col

    if "nct_id" not in col_map:
        logger.warning("Cannot find NCT ID column in Shi & Du efficacy")
        return pd.DataFrame()

    result = pd.DataFrame()
    result["nct_id"] = df[col_map["nct_id"]].astype(str).str.strip()

    if "p_value" in col_map:
        result["p_value"] = pd.to_numeric(df[col_map["p_value"]], errors="coerce")
    if "effect_size" in col_map:
        result["effect_size"] = pd.to_numeric(df[col_map["effect_size"]], errors="coerce")
    if "ci_lower" in col_map:
        result["ci_lower"] = pd.to_numeric(df[col_map["ci_lower"]], errors="coerce")
    if "ci_upper" in col_map:
        result["ci_upper"] = pd.to_numeric(df[col_map["ci_upper"]], errors="coerce")

    result = result.dropna(subset=["nct_id"])
    logger.info("Shi & Du efficacy: %d rows after parsing", len(result))
    return result


def load_shi_du_safety(csv_path: Path) -> pd.DataFrame:
    """Load Shi & Du 2024 safety results (arm-level adverse events).

    Aggregates per trial: total serious adverse events (affected count)
    across all arms. Returns DataFrame with: nct_id, sae_total.
    """
    if not csv_path.exists():
        logger.warning("Shi & Du safety file not found: %s", csv_path)
        return pd.DataFrame()

    df = pd.read_csv(csv_path, low_memory=False)
    logger.info("Shi & Du safety raw: %d rows, columns: %s", len(df), list(df.columns))

    # Detect NCT ID column
    nct_col = None
    for col in df.columns:
        if "nct" in col.lower():
            nct_col = col
            break
    if nct_col is None:
        logger.warning("Cannot find NCT ID column in Shi & Du safety")
        return pd.DataFrame()

    # Filter to serious adverse events only
    sae_col = None
    for col in df.columns:
        if "serious" in col.lower() or col.lower() == "serious/other":
            sae_col = col
            break

    if sae_col is not None:
        serious_mask = df[sae_col].astype(str).str.lower().str.contains("serious")
        df = df[serious_mask].copy()
        logger.info("Shi & Du safety: %d serious AE rows", len(df))

    # Find affected/events count column
    count_col = None
    for col in df.columns:
        cl = col.lower()
        if cl == "affected" or cl == "events":
            count_col = col
            break

    if count_col is None:
        logger.warning("Cannot find count column in Shi & Du safety")
        return pd.DataFrame()

    df["_nct"] = df[nct_col].astype(str).str.strip()
    df["_count"] = pd.to_numeric(df[count_col], errors="coerce").fillna(0)

    # Aggregate: sum of affected counts per trial
    agg = df.groupby("_nct")["_count"].sum().reset_index()
    agg.columns = ["nct_id", "sae_total"]
    agg = agg[agg["sae_total"] > 0]

    logger.info("Shi & Du safety: %d trials with SAE data", len(agg))
    return agg


# ============================================================
# UPDATE EXISTING RESULTS
# ============================================================


def enrich_results_with_aact(
    conn: sqlite3.Connection,
    aact_outcomes: pd.DataFrame,
) -> int:
    """Update trial_failure_results with AACT outcome data.

    Matches on trial_id (via nct_id). Updates p_value_primary,
    ci_lower, ci_upper, primary_endpoint_met, effect_size.

    Returns number of rows updated.
    """
    if aact_outcomes.empty:
        return 0

    # Get nct_id -> trial_id mapping
    cursor = conn.execute(
        "SELECT source_trial_id, trial_id FROM clinical_trials "
        "WHERE source_db = 'clinicaltrials_gov'"
    )
    nct_to_trial: dict[str, int] = {row[0]: row[1] for row in cursor.fetchall()}

    # Group outcomes by nct_id, take the one with smallest p-value per trial
    best_per_nct: dict[str, dict] = {}
    for _, row in aact_outcomes.iterrows():
        nct = str(row["nct_id"])
        p = row["p_value"]
        if nct not in best_per_nct or p < best_per_nct[nct]["p_value"]:
            best_per_nct[nct] = {
                "p_value": p,
                "ci_lower": row.get("ci_lower_limit"),
                "ci_upper": row.get("ci_upper_limit"),
                "effect_size": row.get("param_value"),
                "effect_size_type": row.get("param_type"),
                "method": row.get("method"),
            }

    count = 0
    for nct, data in tqdm(best_per_nct.items(), desc="Enrich from AACT"):
        trial_id = nct_to_trial.get(nct)
        if trial_id is None:
            continue

        p_val = data["p_value"]
        endpoint_met = 1 if p_val <= 0.05 else 0

        sets = ["p_value_primary = ?", "primary_endpoint_met = ?"]
        vals = [p_val, endpoint_met]

        if pd.notna(data.get("ci_lower")):
            sets.append("ci_lower = ?")
            vals.append(float(data["ci_lower"]))
        if pd.notna(data.get("ci_upper")):
            sets.append("ci_upper = ?")
            vals.append(float(data["ci_upper"]))
        if pd.notna(data.get("effect_size")):
            sets.append("effect_size = ?")
            vals.append(float(data["effect_size"]))
        if pd.notna(data.get("effect_size_type")):
            sets.append("effect_size_type = ?")
            vals.append(str(data["effect_size_type"]))

        vals.append(trial_id)
        conn.execute(
            f"UPDATE trial_failure_results SET {', '.join(sets)} "
            f"WHERE trial_id = ? AND p_value_primary IS NULL",
            vals,
        )
        count += conn.execute("SELECT changes()").fetchone()[0]

    conn.commit()
    logger.info("AACT outcomes: updated %d failure results", count)
    return count


def enrich_results_with_shi_du(
    conn: sqlite3.Connection,
    safety_df: pd.DataFrame,
) -> int:
    """Update trial_failure_results with Shi & Du aggregated SAE counts.

    Sets serious_adverse_events from summed affected counts.
    Returns number of rows updated.
    """
    if safety_df.empty:
        return 0

    cursor = conn.execute(
        "SELECT source_trial_id, trial_id FROM clinical_trials "
        "WHERE source_db = 'clinicaltrials_gov'"
    )
    nct_to_trial: dict[str, int] = {row[0]: row[1] for row in cursor.fetchall()}

    count = 0
    for _, row in tqdm(safety_df.iterrows(), total=len(safety_df),
                       desc="Enrich from Shi & Du safety"):
        nct = row["nct_id"]
        trial_id = nct_to_trial.get(nct)
        if trial_id is None:
            continue

        sae_total = int(row["sae_total"])
        conn.execute(
            "UPDATE trial_failure_results SET serious_adverse_events = ? "
            "WHERE trial_id = ?",
            (sae_total, trial_id),
        )
        count += conn.execute("SELECT changes()").fetchone()[0]

    conn.commit()
    logger.info("Shi & Du safety: updated %d failure results", count)
    return count


# ============================================================
# TIER UPGRADES
# ============================================================


def upgrade_confidence_tiers(conn: sqlite3.Connection) -> dict[str, int]:
    """Upgrade confidence tiers based on quantitative evidence.

    Bronze + p_value → Silver
    Silver + Phase III + PubMed → Gold
    """
    stats: dict[str, int] = {}

    # Bronze → Silver: has quantitative p-value
    cursor = conn.execute(
        "UPDATE trial_failure_results SET confidence_tier = 'silver' "
        "WHERE confidence_tier = 'bronze' "
        "  AND p_value_primary IS NOT NULL"
    )
    stats["bronze_to_silver"] = conn.execute("SELECT changes()").fetchone()[0]

    # Silver → Gold: Phase III + has_results + linked PubMed
    cursor = conn.execute("""
        UPDATE trial_failure_results SET confidence_tier = 'gold'
        WHERE confidence_tier = 'silver'
          AND highest_phase_reached IN ('phase_3', 'phase_4')
          AND trial_id IN (
              SELECT ct.trial_id FROM clinical_trials ct
              WHERE ct.has_results = 1
          )
          AND trial_id IN (
              SELECT tp.trial_id FROM trial_publications tp
          )
    """)
    stats["silver_to_gold"] = conn.execute("SELECT changes()").fetchone()[0]

    conn.commit()
    logger.info("Tier upgrades: %s", stats)
    return stats


# ============================================================
# ORCHESTRATOR
# ============================================================


def run_outcome_enrichment(
    db_path: Path = DEFAULT_CT_DB_PATH,
    data_dir: Path | None = None,
) -> dict:
    """Run the outcome enrichment pipeline.

    Steps:
      1. Load AACT outcome_analyses
      2. Load Shi & Du efficacy + safety data
      3. Enrich existing failure results with AACT outcomes
      4. Enrich with Shi & Du safety data
      5. Upgrade confidence tiers

    Returns dict with enrichment statistics.
    """
    cfg = load_config()
    ct_cfg = cfg["ct_domain"]

    if data_dir is None:
        data_dir = Path(ct_cfg["downloads"]["aact"]["dest_dir"])

    conn = get_connection(db_path)
    stats: dict = {}

    try:
        # Step 1: AACT outcomes
        logger.info("=== Step 1: Load AACT outcomes ===")
        aact_outcomes = load_aact_outcomes(data_dir)
        stats["aact_outcome_rows"] = len(aact_outcomes)

        # Step 2: Shi & Du data
        logger.info("=== Step 2: Load Shi & Du data ===")
        shi_du_cfg = ct_cfg["downloads"]["shi_du"]

        efficacy_path = Path(shi_du_cfg["efficacy_dest"])
        efficacy_df = load_shi_du_efficacy(efficacy_path)
        stats["shi_du_efficacy_rows"] = len(efficacy_df)

        safety_path = Path(shi_du_cfg["safety_dest"])
        safety_df = load_shi_du_safety(safety_path)
        stats["shi_du_safety_rows"] = len(safety_df)

        # Step 3: Enrich from AACT
        logger.info("=== Step 3: Enrich from AACT outcomes ===")
        n_aact = enrich_results_with_aact(conn, aact_outcomes)
        stats["aact_enriched"] = n_aact

        # Step 4: Enrich from Shi & Du
        logger.info("=== Step 4: Enrich from Shi & Du safety ===")
        n_shidu = enrich_results_with_shi_du(conn, safety_df)
        stats["shi_du_enriched"] = n_shidu

        # Step 5: Tier upgrades
        logger.info("=== Step 5: Upgrade confidence tiers ===")
        tier_stats = upgrade_confidence_tiers(conn)
        stats["tier_upgrades"] = tier_stats

        # Summary
        cursor = conn.execute(
            "SELECT COUNT(*) FROM trial_failure_results "
            "WHERE p_value_primary IS NOT NULL"
        )
        stats["results_with_pvalue"] = cursor.fetchone()[0]

        cursor = conn.execute(
            "SELECT confidence_tier, COUNT(*) "
            "FROM trial_failure_results GROUP BY confidence_tier"
        )
        stats["tier_distribution"] = {row[0]: row[1] for row in cursor.fetchall()}

    finally:
        conn.close()

    logger.info("=== Outcome Enrichment Complete ===")
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)

    return stats

"""Comprehensive data quality analysis of the NegBioDB-CT database.

Produces JSON + Markdown reports covering 16 analytical queries:
table counts, distributions, top entities, data completeness, and quality flags.

Usage:
    python scripts_ct/analyze_ct_data.py [--db DB_PATH] [--output-dir DIR]
"""

import argparse
import json
import sqlite3
from pathlib import Path

from negbiodb_ct.ct_db import DEFAULT_CT_DB_PATH


def _fetchall(conn: sqlite3.Connection, sql: str) -> list[dict]:
    cur = conn.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def run_analysis(db_path: Path) -> dict:
    conn = sqlite3.connect(str(db_path))
    results = {}

    # Q1: Table row counts
    tables = [
        "clinical_trials", "interventions", "conditions",
        "trial_failure_results", "trial_interventions", "trial_conditions",
        "trial_publications", "intervention_targets",
        "intervention_condition_pairs", "combination_components",
    ]
    counts = {}
    for t in tables:
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
            counts[t] = row[0]
        except sqlite3.OperationalError:
            counts[t] = None
    results["table_counts"] = counts

    # Q2: Failure category distribution
    results["failure_category"] = _fetchall(conn, """
        SELECT failure_category, COUNT(*) AS n,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM trial_failure_results), 1) AS pct
        FROM trial_failure_results
        GROUP BY failure_category ORDER BY n DESC
    """)

    # Q3: Confidence tier distribution
    results["confidence_tier"] = _fetchall(conn, """
        SELECT confidence_tier, COUNT(*) AS n,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM trial_failure_results), 1) AS pct
        FROM trial_failure_results
        GROUP BY confidence_tier
        ORDER BY CASE confidence_tier
            WHEN 'gold' THEN 1 WHEN 'silver' THEN 2
            WHEN 'bronze' THEN 3 WHEN 'copper' THEN 4 END
    """)

    # Q4: Trial phase distribution
    results["trial_phase"] = _fetchall(conn, """
        SELECT ct.trial_phase, COUNT(DISTINCT tfr.result_id) AS n_results,
               COUNT(DISTINCT tfr.trial_id) AS n_trials
        FROM trial_failure_results tfr
        JOIN clinical_trials ct ON tfr.trial_id = ct.trial_id
        GROUP BY ct.trial_phase ORDER BY n_results DESC
    """)

    # Q5: Temporal distribution (by start_date year)
    results["temporal_start"] = _fetchall(conn, """
        SELECT SUBSTR(ct.start_date, 1, 4) AS year,
               COUNT(DISTINCT tfr.result_id) AS n_results
        FROM trial_failure_results tfr
        JOIN clinical_trials ct ON tfr.trial_id = ct.trial_id
        WHERE ct.start_date IS NOT NULL AND CAST(SUBSTR(ct.start_date, 1, 4) AS INTEGER) BETWEEN 1990 AND 2026
        GROUP BY year ORDER BY year
    """)

    # Q5b: Temporal by completion_date
    results["temporal_completion"] = _fetchall(conn, """
        SELECT SUBSTR(ct.completion_date, 1, 4) AS year,
               COUNT(DISTINCT tfr.result_id) AS n_results
        FROM trial_failure_results tfr
        JOIN clinical_trials ct ON tfr.trial_id = ct.trial_id
        WHERE ct.completion_date IS NOT NULL AND CAST(SUBSTR(ct.completion_date, 1, 4) AS INTEGER) BETWEEN 1990 AND 2026
        GROUP BY year ORDER BY year
    """)

    # Q6: Top 20 conditions by failure count
    results["top_conditions"] = _fetchall(conn, """
        SELECT c.condition_name, COUNT(*) AS n_failures,
               COUNT(DISTINCT tfr.intervention_id) AS n_drugs,
               COUNT(DISTINCT tfr.trial_id) AS n_trials
        FROM trial_failure_results tfr
        JOIN conditions c ON tfr.condition_id = c.condition_id
        GROUP BY c.condition_id ORDER BY n_failures DESC LIMIT 20
    """)

    # Q7: Top 20 interventions by failure count
    results["top_interventions"] = _fetchall(conn, """
        SELECT i.intervention_name, i.intervention_type, i.chembl_id,
               COUNT(*) AS n_failures,
               COUNT(DISTINCT tfr.condition_id) AS n_conditions
        FROM trial_failure_results tfr
        JOIN interventions i ON tfr.intervention_id = i.intervention_id
        GROUP BY i.intervention_id ORDER BY n_failures DESC LIMIT 20
    """)

    # Q8: Trial status distribution
    results["trial_status"] = _fetchall(conn, """
        SELECT overall_status, COUNT(*) AS n
        FROM clinical_trials GROUP BY overall_status ORDER BY n DESC
    """)

    # Q9: Termination type distribution
    results["termination_type"] = _fetchall(conn, """
        SELECT termination_type, COUNT(*) AS n,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) AS pct
        FROM clinical_trials
        WHERE overall_status = 'Terminated'
        GROUP BY termination_type ORDER BY n DESC
    """)

    # Q10: Data completeness (tier-level)
    results["data_completeness"] = _fetchall(conn, """
        SELECT confidence_tier,
          COUNT(*) AS total,
          SUM(CASE WHEN p_value_primary IS NOT NULL THEN 1 ELSE 0 END) AS has_pvalue,
          SUM(CASE WHEN effect_size IS NOT NULL THEN 1 ELSE 0 END) AS has_effect_size,
          SUM(CASE WHEN serious_adverse_events IS NOT NULL THEN 1 ELSE 0 END) AS has_sae,
          SUM(CASE WHEN ci_lower IS NOT NULL AND ci_upper IS NOT NULL THEN 1 ELSE 0 END) AS has_ci,
          SUM(CASE WHEN primary_endpoint_met IS NOT NULL THEN 1 ELSE 0 END) AS has_endpoint_met,
          SUM(CASE WHEN result_interpretation IS NOT NULL THEN 1 ELSE 0 END) AS has_interpretation
        FROM trial_failure_results GROUP BY confidence_tier
        ORDER BY CASE confidence_tier
            WHEN 'gold' THEN 1 WHEN 'silver' THEN 2
            WHEN 'bronze' THEN 3 WHEN 'copper' THEN 4 END
    """)

    # Q11: Pair statistics
    try:
        results["pair_stats"] = _fetchall(conn, """
            SELECT COUNT(*) AS total_pairs,
                   ROUND(AVG(num_trials), 2) AS avg_trials,
                   MAX(num_trials) AS max_trials,
                   SUM(CASE WHEN num_trials >= 2 THEN 1 ELSE 0 END) AS multi_trial_pairs,
                   ROUND(AVG(intervention_degree), 1) AS avg_drug_degree,
                   ROUND(AVG(condition_degree), 1) AS avg_condition_degree
            FROM intervention_condition_pairs
        """)
    except sqlite3.OperationalError:
        results["pair_stats"] = [{"note": "table empty or missing"}]

    # Q12: Drug resolution coverage (by intervention type)
    results["drug_resolution"] = _fetchall(conn, """
        SELECT intervention_type,
          COUNT(*) AS total,
          SUM(CASE WHEN chembl_id IS NOT NULL THEN 1 ELSE 0 END) AS has_chembl,
          SUM(CASE WHEN canonical_smiles IS NOT NULL THEN 1 ELSE 0 END) AS has_smiles,
          SUM(CASE WHEN inchikey IS NOT NULL THEN 1 ELSE 0 END) AS has_inchikey,
          SUM(CASE WHEN pubchem_cid IS NOT NULL THEN 1 ELSE 0 END) AS has_pubchem,
          SUM(CASE WHEN molecular_type IS NOT NULL THEN 1 ELSE 0 END) AS has_mol_type
        FROM interventions
        GROUP BY intervention_type ORDER BY total DESC
    """)

    # Q13: Tier × category cross-tab
    results["tier_category_cross"] = _fetchall(conn, """
        SELECT confidence_tier, failure_category, COUNT(*) AS n
        FROM trial_failure_results
        GROUP BY confidence_tier, failure_category
        ORDER BY confidence_tier, n DESC
    """)

    # Q14: Extraction method distribution
    results["extraction_method"] = _fetchall(conn, """
        SELECT extraction_method, COUNT(*) AS n,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM trial_failure_results), 1) AS pct
        FROM trial_failure_results GROUP BY extraction_method ORDER BY n DESC
    """)

    # Q15: Result interpretation distribution
    results["result_interpretation"] = _fetchall(conn, """
        SELECT COALESCE(result_interpretation, 'NULL') AS interpretation,
               COUNT(*) AS n,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM trial_failure_results), 1) AS pct
        FROM trial_failure_results GROUP BY result_interpretation ORDER BY n DESC
    """)

    # Q16: Data quality flags
    quality_flags = {}
    # Bad dates
    quality_flags["bad_start_dates"] = conn.execute("""
        SELECT COUNT(*) FROM clinical_trials
        WHERE start_date IS NOT NULL
          AND CAST(SUBSTR(start_date, 1, 4) AS INTEGER) > 2026
    """).fetchone()[0]
    quality_flags["bad_completion_dates"] = conn.execute("""
        SELECT COUNT(*) FROM clinical_trials
        WHERE completion_date IS NOT NULL
          AND CAST(SUBSTR(completion_date, 1, 4) AS INTEGER) > 2026
    """).fetchone()[0]
    # NULL category
    quality_flags["null_failure_category"] = conn.execute("""
        SELECT COUNT(*) FROM trial_failure_results WHERE failure_category IS NULL
    """).fetchone()[0]
    # Orphan results (no matching trial)
    quality_flags["orphan_results"] = conn.execute("""
        SELECT COUNT(*) FROM trial_failure_results tfr
        LEFT JOIN clinical_trials ct ON tfr.trial_id = ct.trial_id
        WHERE ct.trial_id IS NULL
    """).fetchone()[0]
    # NULL intervention/condition in results
    quality_flags["null_intervention"] = conn.execute("""
        SELECT COUNT(*) FROM trial_failure_results WHERE intervention_id IS NULL
    """).fetchone()[0]
    quality_flags["null_condition"] = conn.execute("""
        SELECT COUNT(*) FROM trial_failure_results WHERE condition_id IS NULL
    """).fetchone()[0]
    results["quality_flags"] = quality_flags

    conn.close()
    return results


def format_markdown(results: dict) -> str:
    lines = ["# NegBioDB-CT Data Quality Report\n"]

    # Table counts
    lines.append("## 1. Table Row Counts\n")
    lines.append("| Table | Rows |")
    lines.append("|-------|------|")
    for t, n in results["table_counts"].items():
        lines.append(f"| {t} | {n:,}" if n is not None else f"| {t} | N/A")
    lines.append("")

    # Failure category
    lines.append("## 2. Failure Category Distribution\n")
    lines.append("| Category | Count | % |")
    lines.append("|----------|-------|---|")
    for r in results["failure_category"]:
        lines.append(f"| {r['failure_category']} | {r['n']:,} | {r['pct']}% |")
    lines.append("")

    # Confidence tier
    lines.append("## 3. Confidence Tier Distribution\n")
    lines.append("| Tier | Count | % |")
    lines.append("|------|-------|---|")
    for r in results["confidence_tier"]:
        lines.append(f"| {r['confidence_tier']} | {r['n']:,} | {r['pct']}% |")
    lines.append("")

    # Trial phase
    lines.append("## 4. Failure by Trial Phase\n")
    lines.append("| Phase | Results | Trials |")
    lines.append("|-------|---------|--------|")
    for r in results["trial_phase"]:
        lines.append(f"| {r['trial_phase'] or 'NULL'} | {r['n_results']:,} | {r['n_trials']:,} |")
    lines.append("")

    # Temporal (start)
    lines.append("## 5. Temporal Distribution (by start year)\n")
    lines.append("| Year | Results |")
    lines.append("|------|---------|")
    for r in results["temporal_start"]:
        lines.append(f"| {r['year']} | {r['n_results']:,} |")
    lines.append("")

    # Top 20 conditions
    lines.append("## 6. Top 20 Conditions by Failure Count\n")
    lines.append("| Condition | Failures | Drugs | Trials |")
    lines.append("|-----------|----------|-------|--------|")
    for r in results["top_conditions"]:
        lines.append(f"| {r['condition_name'][:50]} | {r['n_failures']:,} | {r['n_drugs']:,} | {r['n_trials']:,} |")
    lines.append("")

    # Top 20 interventions
    lines.append("## 7. Top 20 Interventions by Failure Count\n")
    lines.append("| Intervention | Type | ChEMBL | Failures | Conditions |")
    lines.append("|-------------|------|--------|----------|------------|")
    for r in results["top_interventions"]:
        lines.append(f"| {r['intervention_name'][:40]} | {r['intervention_type']} | {r['chembl_id'] or '-'} | {r['n_failures']:,} | {r['n_conditions']:,} |")
    lines.append("")

    # Trial status
    lines.append("## 8. Trial Status Distribution\n")
    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    for r in results["trial_status"]:
        lines.append(f"| {r['overall_status']} | {r['n']:,} |")
    lines.append("")

    # Termination type
    lines.append("## 9. Termination Type Distribution\n")
    lines.append("| Type | Count | % |")
    lines.append("|------|-------|---|")
    for r in results["termination_type"]:
        lines.append(f"| {r['termination_type'] or 'NULL'} | {r['n']:,} | {r['pct']}% |")
    lines.append("")

    # Data completeness
    lines.append("## 10. Data Completeness by Tier\n")
    lines.append("| Tier | Total | p-value | Effect Size | SAE | CI | Endpoint Met | Interpretation |")
    lines.append("|------|-------|---------|-------------|-----|----|--------------|--------------------|")
    for r in results["data_completeness"]:
        lines.append(
            f"| {r['confidence_tier']} | {r['total']:,} | {r['has_pvalue']:,} | "
            f"{r['has_effect_size']:,} | {r['has_sae']:,} | {r['has_ci']:,} | "
            f"{r['has_endpoint_met']:,} | {r['has_interpretation']:,} |"
        )
    lines.append("")

    # Pair stats
    lines.append("## 11. Intervention-Condition Pair Statistics\n")
    for r in results["pair_stats"]:
        for k, v in r.items():
            lines.append(f"- **{k}:** {v}")
    lines.append("")

    # Drug resolution coverage
    lines.append("## 12. Drug Resolution Coverage by Type\n")
    lines.append("| Type | Total | ChEMBL | SMILES | InChIKey | PubChem | MolType |")
    lines.append("|------|-------|--------|--------|----------|---------|---------|")
    for r in results["drug_resolution"]:
        lines.append(
            f"| {r['intervention_type']} | {r['total']:,} | {r['has_chembl']:,} | "
            f"{r['has_smiles']:,} | {r['has_inchikey']:,} | {r['has_pubchem']:,} | "
            f"{r['has_mol_type']:,} |"
        )
    lines.append("")

    # Tier × category cross-tab
    lines.append("## 13. Tier × Category Cross-Tab\n")
    lines.append("| Tier | Category | Count |")
    lines.append("|------|----------|-------|")
    for r in results["tier_category_cross"]:
        lines.append(
            f"| {r['confidence_tier']} | {r['failure_category']} | {r['n']:,} |"
        )
    lines.append("")

    # Extraction method
    lines.append("## 14. Extraction Method Distribution\n")
    lines.append("| Method | Count | % |")
    lines.append("|--------|-------|---|")
    for r in results["extraction_method"]:
        lines.append(f"| {r['extraction_method']} | {r['n']:,} | {r['pct']}% |")
    lines.append("")

    # Result interpretation
    lines.append("## 15. Result Interpretation Distribution\n")
    lines.append("| Interpretation | Count | % |")
    lines.append("|----------------|-------|---|")
    for r in results["result_interpretation"]:
        lines.append(f"| {r['interpretation']} | {r['n']:,} | {r['pct']}% |")
    lines.append("")

    # Quality flags
    lines.append("## 16. Data Quality Flags\n")
    for k, v in results["quality_flags"].items():
        status = "OK" if v == 0 else f"**{v:,}**"
        lines.append(f"- {k}: {status}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="CT data quality analysis")
    parser.add_argument("--db", type=str, default=str(DEFAULT_CT_DB_PATH))
    parser.add_argument("--output-dir", type=str, default="results/ct")
    args = parser.parse_args()

    db_path = Path(args.db)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing {db_path}...")
    results = run_analysis(db_path)

    json_path = output_dir / "ct_data_quality.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON report: {json_path}")

    md_path = output_dir / "ct_data_quality.md"
    md = format_markdown(results)
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown report: {md_path}")

    # Print summary
    tc = results["table_counts"]
    print(f"\n=== Summary ===")
    print(f"  Trials: {tc.get('clinical_trials', 0):,}")
    print(f"  Failure results: {tc.get('trial_failure_results', 0):,}")
    print(f"  Interventions: {tc.get('interventions', 0):,}")
    print(f"  Conditions: {tc.get('conditions', 0):,}")
    print(f"  Pairs: {tc.get('intervention_condition_pairs', 0):,}")
    for r in results["confidence_tier"]:
        print(f"  {r['confidence_tier']}: {r['n']:,} ({r['pct']}%)")
    qf = results["quality_flags"]
    issues = {k: v for k, v in qf.items() if v > 0}
    if issues:
        print(f"  Quality issues: {issues}")
    else:
        print("  Quality issues: none")


if __name__ == "__main__":
    main()

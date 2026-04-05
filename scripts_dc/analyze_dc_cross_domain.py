#!/usr/bin/env python3
"""Cross-domain analysis for DC (Drug Combination Synergy).

Examines connections between DC domain and DTI, CT, GE domains:
  1. DC ↔ DTI: Target overlap hypothesis (shared targets → antagonism?)
  2. DC ↔ GE: Essentiality-driven combinations (essential gene targets → synergy?)
  3. DC ↔ CT: Clinical translation (in vitro antagonism → clinical failure?)
  4. Publication Bias Score (PBS): Compare LLM L4 MCC across domains

Usage:
    PYTHONPATH=src python scripts_dc/analyze_dc_cross_domain.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def analyze_dc_dti_overlap(dc_conn: sqlite3.Connection, dti_db: Path) -> dict:
    """DC ↔ DTI: Target overlap hypothesis.

    Are drug pairs that share the same target more likely to be antagonistic?
    """
    result = {}

    # Get DC pair stats grouped by target overlap
    df = pd.read_sql_query("""
        SELECT
            CASE WHEN num_shared_targets > 0 THEN 'shared' ELSE 'no_shared' END AS overlap_group,
            consensus_class,
            COUNT(*) AS n_pairs,
            AVG(antagonism_fraction) AS mean_antag_frac,
            AVG(median_zip) AS mean_zip
        FROM drug_drug_pairs
        WHERE consensus_class IS NOT NULL
        GROUP BY overlap_group, consensus_class
    """, dc_conn)

    if df.empty:
        return {"status": "no_data"}

    result["overlap_vs_class"] = df.to_dict("records")

    # Summary: shared-target pairs antagonism rate vs no-shared
    summary = pd.read_sql_query("""
        SELECT
            CASE WHEN num_shared_targets > 0 THEN 'shared' ELSE 'no_shared' END AS overlap_group,
            COUNT(*) AS n_pairs,
            AVG(antagonism_fraction) AS mean_antag_frac,
            AVG(synergy_fraction) AS mean_synergy_frac,
            SUM(CASE WHEN consensus_class = 'antagonistic' THEN 1 ELSE 0 END) AS n_antagonistic,
            SUM(CASE WHEN consensus_class = 'synergistic' THEN 1 ELSE 0 END) AS n_synergistic
        FROM drug_drug_pairs
        WHERE consensus_class IS NOT NULL
        GROUP BY overlap_group
    """, dc_conn)

    result["overlap_summary"] = summary.to_dict("records")

    # DTI compound overlap
    if dti_db.exists():
        dti_conn = sqlite3.connect(str(dti_db))
        try:
            dc_drugs = set(
                r[0] for r in dc_conn.execute(
                    "SELECT DISTINCT drug_name FROM compounds WHERE drug_name IS NOT NULL"
                )
            )
            dti_drugs = set(
                r[0] for r in dti_conn.execute(
                    "SELECT DISTINCT compound_name FROM compounds WHERE compound_name IS NOT NULL"
                )
            )
            overlap = dc_drugs & dti_drugs
            result["dti_overlap"] = {
                "dc_drugs": len(dc_drugs),
                "dti_drugs": len(dti_drugs),
                "overlap": len(overlap),
                "overlap_pct": len(overlap) / max(len(dc_drugs), 1) * 100,
            }
        except Exception as e:
            result["dti_overlap"] = {"error": str(e)}
        finally:
            dti_conn.close()

    return result


def analyze_dc_ge_overlap(dc_conn: sqlite3.Connection, ge_db: Path) -> dict:
    """DC ↔ GE: Are combinations targeting essential genes more synergistic?"""
    result = {}

    if not ge_db.exists():
        return {"status": "ge_db_not_found"}

    ge_conn = sqlite3.connect(str(ge_db))
    try:
        # Get essential genes from GE domain
        essential_genes = set(
            r[0].upper() for r in ge_conn.execute("""
                SELECT DISTINCT gene_symbol FROM genes
                WHERE gene_symbol IS NOT NULL
            """)
        )
        result["ge_genes"] = len(essential_genes)

        # Get DC drug targets
        dc_targets = pd.read_sql_query("""
            SELECT compound_id, drug_name, known_targets
            FROM compounds WHERE known_targets IS NOT NULL
        """, dc_conn)

        # Expand targets
        target_genes = set()
        for _, row in dc_targets.iterrows():
            for t in str(row["known_targets"]).split(";"):
                t = t.strip().upper()
                if t:
                    target_genes.add(t)

        overlap = target_genes & essential_genes
        result["target_gene_overlap"] = {
            "dc_target_genes": len(target_genes),
            "ge_genes": len(essential_genes),
            "overlap": len(overlap),
            "overlap_pct": len(overlap) / max(len(target_genes), 1) * 100,
        }

        # Are pairs targeting essential genes more synergistic?
        pairs = pd.read_sql_query("""
            SELECT ddp.pair_id, ddp.consensus_class, ddp.antagonism_fraction,
                   ddp.median_zip, ca.known_targets AS targets_a, cb.known_targets AS targets_b
            FROM drug_drug_pairs ddp
            JOIN compounds ca ON ddp.compound_a_id = ca.compound_id
            JOIN compounds cb ON ddp.compound_b_id = cb.compound_id
            WHERE ca.known_targets IS NOT NULL AND cb.known_targets IS NOT NULL
        """, dc_conn)

        def targets_essential(targets_str):
            if not targets_str:
                return False
            genes = {t.strip().upper() for t in targets_str.split(";")}
            return bool(genes & essential_genes)

        pairs["targets_essential_a"] = pairs["targets_a"].apply(targets_essential)
        pairs["targets_essential_b"] = pairs["targets_b"].apply(targets_essential)
        pairs["any_essential"] = pairs["targets_essential_a"] | pairs["targets_essential_b"]
        pairs["both_essential"] = pairs["targets_essential_a"] & pairs["targets_essential_b"]

        for group_col, group_name in [("any_essential", "any_essential"), ("both_essential", "both_essential")]:
            for val in [True, False]:
                subset = pairs[pairs[group_col] == val]
                if len(subset) > 0:
                    result[f"{group_name}_{val}"] = {
                        "n_pairs": len(subset),
                        "mean_zip": float(subset["median_zip"].mean()),
                        "antag_rate": float((subset["consensus_class"] == "antagonistic").mean()),
                        "synergy_rate": float((subset["consensus_class"] == "synergistic").mean()),
                    }

    except Exception as e:
        result["error"] = str(e)
    finally:
        ge_conn.close()

    return result


def analyze_pbs(llm_results_dir: Path) -> dict:
    """Publication Bias Score: Compare L4 MCC across domains."""
    result = {}

    # Collect L4 MCC from each domain's results
    domain_dirs = {
        "dti": PROJECT_ROOT / "results" / "baselines",
        "ct": PROJECT_ROOT / "results" / "ct_llm",
        "ppi": PROJECT_ROOT / "results" / "ppi_llm",
        "ge": PROJECT_ROOT / "results" / "ge_llm",
        "dc": llm_results_dir,
    }

    for domain, dir_path in domain_dirs.items():
        if not dir_path.exists():
            continue
        mccs = []
        for run_dir in dir_path.iterdir():
            if not run_dir.is_dir():
                continue
            if f"{domain}-l4_" not in run_dir.name and f"l4_" not in run_dir.name:
                continue
            results_file = run_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    metrics = json.load(f)
                mcc = metrics.get("mcc")
                if mcc is not None:
                    mccs.append(mcc)
        if mccs:
            result[domain] = {
                "mean_mcc": float(np.mean(mccs)),
                "std_mcc": float(np.std(mccs)),
                "n_runs": len(mccs),
            }

    return result


def analyze_dc_stats(dc_conn: sqlite3.Connection) -> dict:
    """Basic DC domain statistics."""
    stats = {}
    for table, label in [
        ("compounds", "n_compounds"),
        ("cell_lines", "n_cell_lines"),
        ("drug_drug_pairs", "n_pairs"),
        ("dc_synergy_results", "n_synergy_results"),
    ]:
        try:
            n = dc_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[label] = n
        except Exception:
            stats[label] = 0

    # Class distribution
    try:
        classes = pd.read_sql_query(
            "SELECT consensus_class, COUNT(*) AS n FROM drug_drug_pairs GROUP BY consensus_class",
            dc_conn,
        )
        stats["class_distribution"] = dict(zip(classes["consensus_class"], classes["n"]))
    except Exception:
        stats["class_distribution"] = {}

    # Tier distribution
    try:
        tiers = pd.read_sql_query(
            "SELECT best_confidence, COUNT(*) AS n FROM drug_drug_pairs GROUP BY best_confidence",
            dc_conn,
        )
        stats["tier_distribution"] = dict(zip(tiers["best_confidence"], tiers["n"]))
    except Exception:
        stats["tier_distribution"] = {}

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DC cross-domain analysis")
    parser.add_argument("--dc-db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_dc.db")
    parser.add_argument("--dti-db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb.db")
    parser.add_argument("--ge-db", type=Path, default=PROJECT_ROOT / "data" / "negbiodb_depmap.db")
    parser.add_argument("--llm-dir", type=Path, default=PROJECT_ROOT / "results" / "dc_llm")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "dc_cross_domain.json")
    args = parser.parse_args(argv)

    if not args.dc_db.exists():
        logger.warning("DC database not found at %s — run ETL first", args.dc_db)
        return 1

    dc_conn = sqlite3.connect(str(args.dc_db))
    try:
        results = {}

        logger.info("1/5: DC domain statistics")
        results["dc_stats"] = analyze_dc_stats(dc_conn)

        logger.info("2/5: DC ↔ DTI target overlap")
        results["dc_dti"] = analyze_dc_dti_overlap(dc_conn, args.dti_db)

        logger.info("3/5: DC ↔ GE essentiality overlap")
        results["dc_ge"] = analyze_dc_ge_overlap(dc_conn, args.ge_db)

        logger.info("4/5: Publication Bias Score (L4 MCC)")
        results["pbs"] = analyze_pbs(args.llm_dir)

        logger.info("5/5: Saving results")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("Cross-domain analysis saved to %s", args.output)

        # Print summary
        print("\n=== DC Cross-Domain Summary ===")
        stats = results["dc_stats"]
        print(f"  Compounds: {stats.get('n_compounds', 0)}")
        print(f"  Cell lines: {stats.get('n_cell_lines', 0)}")
        print(f"  Drug pairs: {stats.get('n_pairs', 0)}")
        print(f"  Synergy results: {stats.get('n_synergy_results', 0)}")

        if "overlap_summary" in results["dc_dti"]:
            print("\n  DC ↔ DTI: Target overlap vs antagonism")
            for row in results["dc_dti"]["overlap_summary"]:
                print(f"    {row['overlap_group']}: {row['n_pairs']} pairs, "
                      f"antag_frac={row['mean_antag_frac']:.2f}")

        if "target_gene_overlap" in results["dc_ge"]:
            tgo = results["dc_ge"]["target_gene_overlap"]
            print(f"\n  DC ↔ GE: {tgo['overlap']} genes overlap "
                  f"({tgo['overlap_pct']:.1f}% of DC targets)")

        if results["pbs"]:
            print("\n  PBS (L4 MCC by domain):")
            for domain, vals in sorted(results["pbs"].items()):
                print(f"    {domain}: {vals['mean_mcc']:.3f} ± {vals['std_mcc']:.3f} (n={vals['n_runs']})")

    finally:
        dc_conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

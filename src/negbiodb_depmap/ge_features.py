"""Omics feature computation for GE domain ML models.

Features per gene-cell_line pair (~75 dimensions):
  Gene-level (8 dims): essentiality profile stats, fraction essential,
    common essential flag, reference nonessential flag, RNAi concordance
  Cell line (variable): lineage one-hot, disease one-hot, mutation burden
  Gene × Cell line omics (3 dims): expression TPM, copy number, mutation indicator

Omics files (OmicsExpression, OmicsCN, OmicsSomatic) are NOT stored in SQLite
(too large). They are loaded directly from CSV during feature computation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_gene_features(conn) -> pd.DataFrame:
    """Compute gene-level aggregate features from gene_cell_pairs.

    Returns DataFrame indexed by gene_id with columns:
        mean_effect, std_effect, min_effect, max_effect,
        fraction_essential, is_common_essential, is_reference_nonessential,
        rnai_concordance_fraction
    """
    # Essentiality profile across cell lines
    query = """
    SELECT
        g.gene_id,
        AVG(p.mean_gene_effect) as mean_effect,
        CASE COUNT(p.pair_id) WHEN 1 THEN 0.0 ELSE
            SQRT(SUM((p.mean_gene_effect - sub.gmean) * (p.mean_gene_effect - sub.gmean))
                 / (COUNT(p.pair_id) - 1)) END as std_effect,
        MIN(p.min_gene_effect) as min_effect,
        MAX(p.max_gene_effect) as max_effect,
        g.is_common_essential,
        g.is_reference_nonessential,
        p.gene_degree
    FROM genes g
    JOIN gene_cell_pairs p ON g.gene_id = p.gene_id
    LEFT JOIN (
        SELECT gene_id, AVG(mean_gene_effect) as gmean
        FROM gene_cell_pairs GROUP BY gene_id
    ) sub ON g.gene_id = sub.gene_id
    GROUP BY g.gene_id
    """
    df = pd.read_sql_query(query, conn, index_col="gene_id")

    # RNAi concordance: fraction of this gene's cell lines with multi-source support
    rnai_query = """
    SELECT gene_id,
           CAST(SUM(CASE WHEN num_sources >= 2 THEN 1 ELSE 0 END) AS REAL) /
           NULLIF(COUNT(*), 0) as rnai_concordance_fraction
    FROM gene_cell_pairs
    GROUP BY gene_id
    """
    rnai_df = pd.read_sql_query(rnai_query, conn, index_col="gene_id")
    df = df.join(rnai_df, how="left")
    df["rnai_concordance_fraction"] = df["rnai_concordance_fraction"].fillna(0.0)

    logger.info("Computed gene features for %d genes", len(df))
    return df


def compute_cell_line_features(conn) -> pd.DataFrame:
    """Compute cell line features: lineage/disease one-hot + degree.

    Returns DataFrame indexed by cell_line_id.
    """
    query = """
    SELECT cell_line_id, lineage, primary_disease
    FROM cell_lines
    """
    df = pd.read_sql_query(query, conn, index_col="cell_line_id")

    # One-hot encode lineage
    lineage_dummies = pd.get_dummies(df["lineage"], prefix="lineage", dtype=float)
    disease_dummies = pd.get_dummies(df["primary_disease"], prefix="disease", dtype=float)

    result = pd.concat([lineage_dummies, disease_dummies], axis=1)
    result.index = df.index

    # Cell line degree from gene_cell_pairs
    degree_query = """
    SELECT cell_line_id, cell_line_degree
    FROM gene_cell_pairs
    GROUP BY cell_line_id
    """
    deg_df = pd.read_sql_query(degree_query, conn, index_col="cell_line_id")
    result = result.join(deg_df, how="left")
    result["cell_line_degree"] = result["cell_line_degree"].fillna(0)

    logger.info("Computed cell line features: %d cell lines, %d dims", len(result), result.shape[1])
    return result


def load_omics_features(
    expression_file: Path | None = None,
    cn_file: Path | None = None,
    mutation_file: Path | None = None,
    model_ids: list[str] | None = None,
    gene_symbols: list[str] | None = None,
) -> dict[tuple[str, str], np.ndarray]:
    """Load per-gene-per-cell-line omics features from CSV files.

    Each file is a wide matrix: rows = ModelID, columns = gene symbols.

    Args:
        expression_file: OmicsExpressionProteinCodingGenesTPMLogp1.csv
        cn_file: OmicsCNGene.csv
        mutation_file: OmicsSomaticMutationsMatrixDamaging.csv
        model_ids: Filter to these model IDs (reduces memory).
        gene_symbols: Filter to these gene symbols.

    Returns:
        Dict mapping (model_id, gene_symbol) → np.array([expression, cn, mutation]).
    """
    features: dict[tuple[str, str], list[float]] = {}

    def _load_matrix(filepath: Path, feat_idx: int) -> None:
        """Load a single omics matrix and populate features dict."""
        df = pd.read_csv(filepath, index_col=0)

        # Filter columns (gene symbols) — column headers may be "HUGO (EntrezID)"
        if gene_symbols is not None:
            # Try direct match first
            matching_cols = [c for c in df.columns if c in gene_symbols]
            # Also try stripping entrez suffix
            if not matching_cols:
                col_map = {}
                for c in df.columns:
                    parts = c.split(" (")
                    if parts[0] in gene_symbols:
                        col_map[c] = parts[0]
                if col_map:
                    df = df[list(col_map.keys())]
                    df.columns = [col_map[c] for c in df.columns]
            else:
                df = df[matching_cols]

        # Filter rows (model IDs)
        if model_ids is not None:
            available = [m for m in model_ids if m in df.index]
            df = df.loc[available]

        for model_id in df.index:
            mid = str(model_id).strip()
            for col in df.columns:
                gene = col.split(" (")[0] if " (" in col else col
                key = (mid, gene)
                if key not in features:
                    features[key] = [0.0, 0.0, 0.0]

                val = df.at[model_id, col]
                if not pd.isna(val):
                    features[key][feat_idx] = float(val)

    if expression_file and expression_file.exists():
        _load_matrix(expression_file, 0)
        logger.info("Loaded expression features from %s", expression_file)

    if cn_file and cn_file.exists():
        _load_matrix(cn_file, 1)
        logger.info("Loaded copy number features from %s", cn_file)

    if mutation_file and mutation_file.exists():
        _load_matrix(mutation_file, 2)
        logger.info("Loaded mutation features from %s", mutation_file)

    result = {k: np.array(v) for k, v in features.items()}
    logger.info("Loaded omics features for %d (model, gene) pairs", len(result))
    return result


def build_feature_matrix(
    conn,
    pairs_df: pd.DataFrame,
    omics_features: dict[tuple[str, str], np.ndarray] | None = None,
) -> np.ndarray:
    """Build combined feature matrix for ML training.

    Args:
        conn: SQLite connection.
        pairs_df: DataFrame with gene_id, cell_line_id, gene_symbol, model_id columns.
        omics_features: Optional dict from load_omics_features.

    Returns:
        Feature matrix (n_samples × n_features).
    """
    gene_feats = compute_gene_features(conn)
    cl_feats = compute_cell_line_features(conn)

    gene_cols = ["mean_effect", "std_effect", "min_effect", "max_effect",
                 "is_common_essential", "is_reference_nonessential",
                 "rnai_concordance_fraction", "gene_degree"]
    cl_cols = list(cl_feats.columns)

    rows = []
    for _, pair in pairs_df.iterrows():
        gid = pair["gene_id"]
        clid = pair["cell_line_id"]

        # Gene features
        if gid in gene_feats.index:
            g = gene_feats.loc[gid, gene_cols].values.astype(float)
        else:
            g = np.zeros(len(gene_cols))

        # Cell line features
        if clid in cl_feats.index:
            c = cl_feats.loc[clid, cl_cols].values.astype(float)
        else:
            c = np.zeros(len(cl_cols))

        # Omics features
        if omics_features is not None:
            gene_symbol = pair.get("gene_symbol", "")
            model_id = pair.get("model_id", "")
            omics = omics_features.get((model_id, gene_symbol), np.zeros(3))
        else:
            omics = np.zeros(3)

        row = np.concatenate([g, c, omics])
        rows.append(row)

    X = np.stack(rows)
    n_nan = np.isnan(X).sum()
    if n_nan > 0:
        logger.warning("Feature matrix has %d NaN values, replacing with 0", n_nan)
        X = np.nan_to_num(X, nan=0.0)

    logger.info("Feature matrix: %d samples × %d features", X.shape[0], X.shape[1])
    return X

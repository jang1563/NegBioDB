"""Cross-domain transfer matrix — train on domain A, evaluate on domain B.

Produces a 5×5 matrix (4 training domains + 1 multi-domain → 5 eval domains).
The VP column tests zero-shot transfer (VP never in training set).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from negbiorl.data_registry import ALL_DOMAINS, TRAIN_DOMAINS, TRANSFER_TEST_DOMAIN


def build_transfer_matrix(
    results: dict[str, dict[str, float]],
    metric: str = "mcc",
) -> dict[str, Any]:
    """Build cross-domain transfer matrix from evaluation results.

    Args:
        results: {train_domain: {eval_domain: {metric: value, ...}}}
            train_domain keys: "dti", "ct", "ppi", "ge", "multi"
            eval_domain keys: "dti", "ct", "ppi", "ge", "vp"
        metric: Which metric to use for the matrix (default: "mcc")

    Returns:
        {"matrix": 2D dict, "row_means": dict, "col_means": dict,
         "diagonal": dict, "vp_transfer": dict, "best_source": dict}
    """
    train_sources = list(results.keys())
    eval_targets = ALL_DOMAINS

    # Build matrix
    matrix: dict[str, dict[str, float]] = {}
    for src in train_sources:
        matrix[src] = {}
        for tgt in eval_targets:
            if tgt in results.get(src, {}):
                val = results[src][tgt]
                matrix[src][tgt] = val[metric] if isinstance(val, dict) else val
            else:
                matrix[src][tgt] = float("nan")

    # Row means (average transfer from one source)
    row_means = {}
    for src in train_sources:
        vals = [v for v in matrix[src].values() if not np.isnan(v)]
        row_means[src] = float(np.mean(vals)) if vals else float("nan")

    # Column means (average transfer to one target)
    col_means = {}
    for tgt in eval_targets:
        vals = [matrix[src][tgt] for src in train_sources if not np.isnan(matrix[src].get(tgt, float("nan")))]
        col_means[tgt] = float(np.mean(vals)) if vals else float("nan")

    # Diagonal (in-domain performance, excluding multi and vp)
    diagonal = {}
    for d in TRAIN_DOMAINS:
        if d in matrix and d in matrix[d]:
            diagonal[d] = matrix[d][d]

    # VP transfer column (zero-shot generalization)
    vp_transfer = {}
    for src in train_sources:
        if TRANSFER_TEST_DOMAIN in matrix.get(src, {}):
            vp_transfer[src] = matrix[src][TRANSFER_TEST_DOMAIN]

    # Best source for each target
    best_source = {}
    for tgt in eval_targets:
        best_src, best_val = None, float("-inf")
        for src in train_sources:
            val = matrix.get(src, {}).get(tgt, float("nan"))
            if not np.isnan(val) and val > best_val:
                best_val = val
                best_src = src
        best_source[tgt] = {"source": best_src, metric: best_val}

    return {
        "matrix": matrix,
        "metric": metric,
        "row_means": row_means,
        "col_means": col_means,
        "diagonal": diagonal,
        "vp_transfer": vp_transfer,
        "best_source": best_source,
        "train_sources": train_sources,
        "eval_targets": eval_targets,
    }


def compute_transfer_gain(
    transfer_matrix: dict[str, Any],
    baseline_scores: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Compute transfer gain over vanilla baseline for each cell.

    Args:
        transfer_matrix: Output of build_transfer_matrix
        baseline_scores: {eval_domain: baseline_metric_value}

    Returns:
        {train_source: {eval_target: gain}}
    """
    matrix = transfer_matrix["matrix"]
    gains = {}
    for src in transfer_matrix["train_sources"]:
        gains[src] = {}
        for tgt in transfer_matrix["eval_targets"]:
            trained_val = matrix.get(src, {}).get(tgt, float("nan"))
            baseline_val = baseline_scores.get(tgt, float("nan"))
            if np.isnan(trained_val) or np.isnan(baseline_val):
                gains[src][tgt] = float("nan")
            else:
                gains[src][tgt] = trained_val - baseline_val
    return gains


def format_transfer_matrix_latex(
    transfer_matrix: dict[str, Any],
    gains: dict[str, dict[str, float]] | None = None,
) -> str:
    """Format transfer matrix as LaTeX table for the paper."""
    metric = transfer_matrix["metric"]
    sources = transfer_matrix["train_sources"]
    targets = transfer_matrix["eval_targets"]
    matrix = transfer_matrix["matrix"]

    lines = [
        r"\begin{tabular}{l" + "c" * len(targets) + "}",
        r"\toprule",
        r"Train $\backslash$ Eval & " + " & ".join(t.upper() for t in targets) + r" \\",
        r"\midrule",
    ]

    for src in sources:
        cells = []
        for tgt in targets:
            val = matrix.get(src, {}).get(tgt, float("nan"))
            if np.isnan(val):
                cells.append("---")
            else:
                cell = f"{val:.3f}"
                # Bold diagonal (in-domain)
                if src == tgt:
                    cell = r"\textbf{" + cell + "}"
                # Add gain annotation if available
                if gains and src in gains and tgt in gains[src]:
                    g = gains[src][tgt]
                    if not np.isnan(g) and abs(g) >= 0.01:
                        sign = "+" if g > 0 else ""
                        cell += r"\scriptsize{" + f"({sign}{g:.2f})" + "}"
                cells.append(cell)

        label = src.upper() if src != "multi" else "Multi"
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    return "\n".join(lines)

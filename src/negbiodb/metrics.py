"""ML evaluation metrics for NegBioDB DTI benchmark.

Provides 7 metrics for evaluating DTI prediction models:

Custom implementations:
  - log_auc: LogAUC[0.001, 0.1] — primary ranking metric
  - bedroc: BEDROC(alpha=20) — early enrichment
  - enrichment_factor: EF@k% — top-ranked performance

sklearn wrappers:
  - auroc: AUROC — backward compatibility
  - auprc: AUPRC — secondary ranking metric
  - mcc: MCC — balanced classification

Convenience functions:
  - compute_all_metrics: all 7 metrics in one call
  - evaluate_splits: per-fold evaluation
  - summarize_runs: mean ± std across runs
  - save_results: persist to JSON/CSV

All individual metric functions follow the sklearn convention:
    metric(y_true, y_score) -> float

References:
    LogAUC: Mysinger & Shoichet, JCIM 2010, 50, 1561-1573.
    BEDROC: Truchon & Bayly, JCIM 2007, 47, 488-508.
"""

from __future__ import annotations

import warnings
from math import ceil
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

__all__ = [
    "auroc", "auprc", "mcc",
    "log_auc", "bedroc", "enrichment_factor",
    "compute_all_metrics", "evaluate_splits", "summarize_runs", "save_results",
]


# ------------------------------------------------------------------
# Input validation
# ------------------------------------------------------------------

def _validate_inputs(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce inputs to numpy arrays.

    Args:
        y_true: True binary labels {0, 1}.
        y_score: Prediction scores (higher = more likely active).

    Returns:
        Tuple of (y_true, y_score) as 1-D float64 arrays.

    Raises:
        ValueError: If inputs have mismatched lengths, fewer than 2 samples,
            or y_true contains values other than 0 and 1.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    if len(y_true) != len(y_score):
        raise ValueError(
            f"y_true and y_score must have the same length, "
            f"got {len(y_true)} and {len(y_score)}"
        )
    if len(y_true) < 2:
        raise ValueError(
            f"Need at least 2 samples, got {len(y_true)}"
        )

    unique_labels = set(np.unique(y_true))
    if not unique_labels.issubset({0.0, 1.0}):
        raise ValueError(
            f"y_true must contain only 0 and 1, got unique values {unique_labels}"
        )

    if np.any(~np.isfinite(y_score)):
        raise ValueError("y_score contains NaN or Inf values")

    return y_true, y_score


def _check_binary_classes(y_true: np.ndarray, metric_name: str) -> bool:
    """Check that both classes are present. Warn and return False if not."""
    if len(np.unique(y_true)) < 2:
        warnings.warn(
            f"{metric_name}: only one class present in y_true, returning NaN",
            stacklevel=3,
        )
        return False
    return True


# ------------------------------------------------------------------
# sklearn wrappers
# ------------------------------------------------------------------

def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the ROC Curve (AUROC).

    Included for backward compatibility with DAVIS/TDC benchmarks.
    NOT used for model ranking (insensitive to class imbalance).

    Args:
        y_true: True binary labels {0, 1}.
        y_score: Prediction scores (higher = more likely active).

    Returns:
        AUROC in [0, 1]. Returns NaN if only one class is present.
    """
    y_true, y_score = _validate_inputs(y_true, y_score)
    if not _check_binary_classes(y_true, "auroc"):
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the Precision-Recall Curve (AUPRC).

    Secondary ranking metric. More informative than AUROC for
    imbalanced datasets (e.g., realistic 1:10 ratio).

    Args:
        y_true: True binary labels {0, 1}.
        y_score: Prediction scores (higher = more likely active).

    Returns:
        AUPRC in [0, 1]. Random baseline = fraction of positives.
        Returns NaN if only one class is present.
    """
    y_true, y_score = _validate_inputs(y_true, y_score)
    if not _check_binary_classes(y_true, "auprc"):
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def mcc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute Matthews Correlation Coefficient (MCC).

    Binarizes y_score at the given threshold, then computes MCC.
    Best single metric for imbalanced binary classification.

    Args:
        y_true: True binary labels {0, 1}.
        y_score: Prediction scores (higher = more likely active).
        threshold: Score threshold for binarization. Default 0.5.

    Returns:
        MCC in [-1, 1]. 0 = random, 1 = perfect. Returns NaN if only
        one class present after binarization.
    """
    y_true, y_score = _validate_inputs(y_true, y_score)
    if not _check_binary_classes(y_true, "mcc"):
        return float("nan")

    y_pred = (y_score >= threshold).astype(int)

    # MCC is undefined if predictions are all one class
    if len(np.unique(y_pred)) < 2:
        warnings.warn(
            "mcc: all predictions are the same class after binarization "
            f"(threshold={threshold}), returning NaN",
            stacklevel=2,
        )
        return float("nan")

    return float(matthews_corrcoef(y_true, y_pred))


# ------------------------------------------------------------------
# LogAUC
# ------------------------------------------------------------------

def log_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fpr_range: tuple[float, float] = (0.001, 0.1),
) -> float:
    """Compute LogAUC in a specified FPR range.

    Area under the ROC curve with log10-scaled FPR axis, restricted to
    [fpr_min, fpr_max]. Normalized to [0, 1] where 1 is perfect.

    Uses sklearn's roc_curve for FPR/TPR computation, then integrates
    TPR vs log10(FPR) using the trapezoidal rule with boundary
    interpolation.

    Args:
        y_true: True binary labels {0, 1}.
        y_score: Prediction scores (higher = more likely active).
        fpr_range: (lower_bound, upper_bound) for FPR restriction.
            Default (0.001, 0.1).

    Returns:
        LogAUC score in [0, 1]. Random baseline ~0.0215 for default range at
        5% prevalence; varies with active fraction. Returns NaN if only one
        class is present.

    References:
        Mysinger & Shoichet, JCIM 2010, 50, 1561-1573.
    """
    y_true, y_score = _validate_inputs(y_true, y_score)
    if not _check_binary_classes(y_true, "log_auc"):
        return float("nan")

    fpr_min, fpr_max = fpr_range
    if fpr_min <= 0:
        raise ValueError(f"fpr_range lower bound must be > 0, got {fpr_min}")
    if fpr_max > 1.0:
        raise ValueError(f"fpr_range upper bound must be <= 1.0, got {fpr_max}")
    if fpr_min >= fpr_max:
        raise ValueError(
            f"fpr_range lower must be < upper, got ({fpr_min}, {fpr_max})"
        )

    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Interpolate TPR at exact boundary FPR values
    tpr_at_lo = float(np.interp(fpr_min, fpr, tpr))
    tpr_at_hi = float(np.interp(fpr_max, fpr, tpr))

    # Filter to points strictly within range
    mask = (fpr > fpr_min) & (fpr < fpr_max)
    fpr_inner = fpr[mask]
    tpr_inner = tpr[mask]

    # Build trimmed arrays with boundaries
    fpr_trim = np.concatenate([[fpr_min], fpr_inner, [fpr_max]])
    tpr_trim = np.concatenate([[tpr_at_lo], tpr_inner, [tpr_at_hi]])

    # Integrate TPR vs log10(FPR) using trapezoidal rule
    log_fpr = np.log10(fpr_trim)
    area = float(np.trapezoid(tpr_trim, log_fpr))

    # Normalize by the log10 width of the FPR range
    log_width = np.log10(fpr_max) - np.log10(fpr_min)
    return area / log_width


# ------------------------------------------------------------------
# BEDROC
# ------------------------------------------------------------------

def bedroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    alpha: float = 20.0,
) -> float:
    """Compute BEDROC (Boltzmann-Enhanced Discrimination of ROC).

    Measures early enrichment with exponential weighting. With alpha=20,
    approximately 80% of the score comes from the top 8% of the ranked
    list.

    Args:
        y_true: True binary labels {0, 1}.
        y_score: Prediction scores (higher = more likely active).
        alpha: Exponential weighting factor. Default 20.0.

    Returns:
        BEDROC score in [0, 1]. 1 = perfect, ra (active fraction) = random.
        Returns NaN if no actives or only one class present.

    Note:
        Tied scores use stable sort (input order). For models producing
        many ties, consider adding small random jitter to scores.

    References:
        Truchon & Bayly, JCIM 2007, 47, 488-508.
    """
    y_true, y_score = _validate_inputs(y_true, y_score)

    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    n = len(y_true)
    n_actives = int(y_true.sum())

    if n_actives == 0:
        warnings.warn(
            "bedroc: no actives in y_true, returning NaN",
            stacklevel=2,
        )
        return float("nan")

    if n_actives == n:
        # All actives → perfect score
        return 1.0

    ra = n_actives / n  # fraction of actives
    ri = 1.0 - ra       # fraction of inactives

    # Sort by score descending and get 1-based ranks of actives
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    active_ranks = np.where(y_sorted == 1.0)[0] + 1  # 1-based

    # RIE (Relative Information Entropy)
    # Numerator: sum of exponential weights at active positions
    s = float(np.sum(np.exp(-alpha * active_ranks / n)))

    # Denominator: expected sum under random ordering
    # = ra * (1 - exp(-alpha)) / (exp(alpha/n) - 1)
    # Both terms suffer catastrophic cancellation for very small alpha:
    #   exp(alpha/n) - 1 → 0, and 1 - exp(-alpha) → 0.
    # Use expm1() for both: expm1(x) = exp(x)-1 with full precision for |x| << 1.
    #   exp(alpha/n) - 1  =  expm1(alpha/n)
    #   1 - exp(-alpha)   = -expm1(-alpha)
    rie_denom = np.expm1(alpha / n)
    if rie_denom == 0.0:
        # alpha/n underflows even expm1; BEDROC is undefined at this scale.
        warnings.warn(
            f"bedroc: alpha={alpha} too small for n={n} (alpha/n below subnormal), "
            "returning NaN",
            stacklevel=2,
        )
        return float("nan")
    rie_numer = -np.expm1(-alpha)  # = 1 - exp(-alpha), stable for all alpha
    expected = ra * rie_numer / rie_denom

    rie = s / expected

    # BEDROC formula — numerically stable form using negative exponents.
    # Algebraically equivalent to the original sinh/cosh form from
    # Truchon & Bayly but avoids overflow for large alpha (>~1400).
    exp_neg_a = np.exp(-alpha)
    exp_neg_ara = np.exp(-alpha * ra)
    exp_neg_ari = np.exp(-alpha * ri)

    num = ra * (-np.expm1(-alpha))   # = ra * (1 - exp(-alpha)), stable for small alpha
    den = (1.0 + exp_neg_a) - exp_neg_ara - exp_neg_ari

    # Suppress numpy's RuntimeWarning: for very small alpha the formula is 0/0
    # (the BEDROC formula is degenerate as alpha → 0). We detect this below.
    with np.errstate(divide="ignore", invalid="ignore"):
        # Second term: 1/(1-exp(alpha*ri)) rewritten as -exp(-alpha*ri)/(1-exp(-alpha*ri))
        # When alpha*ri is large, exp_neg_ari ≈ 0, so term2 ≈ 0.
        # When alpha is tiny, exp_neg_ari rounds to 1.0 → division by zero (caught below).
        term2 = -exp_neg_ari / (1.0 - exp_neg_ari) if exp_neg_ari > 0.0 else 0.0
        bedroc_score = rie * num / den + term2

    if not np.isfinite(bedroc_score):
        warnings.warn(
            f"bedroc: computation produced non-finite result for alpha={alpha}, "
            "n={n}. Very small alpha may cause numerical degeneracy (formula is "
            "0/0 as alpha → 0). Consider alpha >= 1.0.",
            stacklevel=2,
        )
        return float("nan")

    # Clamp to [0, 1]: floating-point rounding can produce values like 1+2e-16.
    return float(np.clip(bedroc_score, 0.0, 1.0))


# ------------------------------------------------------------------
# Enrichment Factor
# ------------------------------------------------------------------

def enrichment_factor(
    y_true: np.ndarray,
    y_score: np.ndarray,
    percentage: float = 1.0,
) -> float:
    """Compute Enrichment Factor at a given percentage.

    EF@k% measures how many times better than random the model is
    at identifying actives in the top k% of the ranked list.

    EF@k% = (actives_in_top_k% / n_top) / (n_actives / n_total)

    Args:
        y_true: True binary labels {0, 1}.
        y_score: Prediction scores (higher = more likely active).
        percentage: Top percentage to evaluate (1.0 = 1%, 5.0 = 5%).

    Returns:
        Enrichment factor. Random baseline = 1.0.
        Maximum = min(100/percentage, n_total/n_actives).
        Returns NaN if no actives in y_true.
    """
    y_true, y_score = _validate_inputs(y_true, y_score)

    if not (0.0 < percentage <= 100.0):
        raise ValueError(f"percentage must be in (0, 100], got {percentage}")

    n_total = len(y_true)
    n_actives = int(y_true.sum())

    if n_actives == 0:
        warnings.warn(
            "enrichment_factor: no actives in y_true, returning NaN",
            stacklevel=2,
        )
        return float("nan")

    # Sort by score descending
    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    n_top = max(1, ceil(percentage / 100.0 * n_total))
    actives_in_top = int(y_sorted[:n_top].sum())

    # EF = (actives_in_top / n_top) / (n_actives / n_total)
    ef = (actives_in_top / n_top) / (n_actives / n_total)
    return float(ef)


# ------------------------------------------------------------------
# Convenience functions
# ------------------------------------------------------------------

def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all 7 benchmark metrics in one call.

    Args:
        y_true: True binary labels {0, 1}.
        y_score: Prediction scores (higher = more likely active).
        threshold: Score threshold for MCC binarization. Default 0.5.

    Returns:
        Dict with keys: auroc, auprc, mcc, log_auc, bedroc, ef_1pct, ef_5pct.
    """
    return {
        "auroc": auroc(y_true, y_score),
        "auprc": auprc(y_true, y_score),
        "mcc": mcc(y_true, y_score, threshold=threshold),
        "log_auc": log_auc(y_true, y_score),
        "bedroc": bedroc(y_true, y_score),
        "ef_1pct": enrichment_factor(y_true, y_score, percentage=1.0),
        "ef_5pct": enrichment_factor(y_true, y_score, percentage=5.0),
    }


def evaluate_splits(
    y_true: np.ndarray,
    y_score: np.ndarray,
    split_labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Compute all metrics per split fold.

    Args:
        y_true: True binary labels {0, 1}.
        y_score: Prediction scores.
        split_labels: Array of fold labels (string or integer, e.g.,
            ["train", "val", "test"] or [0, 1, 2]). All labels are
            converted to strings in the output dict.
        threshold: Score threshold for MCC binarization.

    Returns:
        Dict mapping fold name to metrics dict.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    split_labels = np.asarray(split_labels).ravel()

    if not (len(y_true) == len(y_score) == len(split_labels)):
        raise ValueError(
            f"y_true, y_score, and split_labels must have the same length, "
            f"got {len(y_true)}, {len(y_score)}, and {len(split_labels)}"
        )

    results = {}
    for fold in np.unique(split_labels):
        mask = split_labels == fold
        n_fold = int(mask.sum())
        if n_fold < 2:
            warnings.warn(
                f"evaluate_splits: fold '{fold}' has {n_fold} sample(s), "
                "returning NaN for all metrics",
                stacklevel=2,
            )
            results[str(fold)] = {
                k: float("nan")
                for k in ("auroc", "auprc", "mcc", "log_auc",
                          "bedroc", "ef_1pct", "ef_5pct")
            }
        else:
            results[str(fold)] = compute_all_metrics(
                y_true[mask], y_score[mask], threshold=threshold
            )
    return results


def summarize_runs(results: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Compute mean and std across multiple runs.

    Args:
        results: List of metric dicts (from compute_all_metrics).

    Returns:
        Dict mapping metric name to {"mean": float, "std": float}.

    Raises:
        ValueError: If results is empty.
    """
    if not results:
        raise ValueError("results must be non-empty")

    keys = set(results[0].keys())
    for i, r in enumerate(results):
        if set(r.keys()) != keys:
            raise ValueError(
                f"results[{i}] has inconsistent keys vs results[0]: "
                f"missing={keys - set(r.keys())}, extra={set(r.keys()) - keys}"
            )
    summary = {}
    for key in keys:
        values = [r[key] for r in results]
        arr = np.array(values, dtype=np.float64)
        # Ignore NaN values when computing mean/std
        summary[key] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr, ddof=0)),
        }
    return summary


def save_results(
    results: dict | list[dict],
    output_path: str | Path,
    format: str = "json",
) -> None:
    """Save metric results to JSON or CSV.

    Args:
        results: Single metrics dict or list of dicts.
        output_path: Destination file path.
        format: "json" or "csv". Default "json".

    Raises:
        ValueError: If format is not "json" or "csv".
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        # Convert NaN to None for JSON compatibility
        def _clean(obj):
            if isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            if isinstance(obj, (float, np.floating)):
                return None if np.isnan(obj) else float(obj)
            if isinstance(obj, (int, np.integer)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            return obj

        with open(output_path, "w") as f:
            json.dump(_clean(results), f, indent=2)

    elif format == "csv":
        import pandas as pd

        def _flatten_dict(d: dict) -> dict:
            """Flatten nested dicts: {"a": {"x": 1}} → {"a_x": 1}."""
            flat = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    for subk, subv in v.items():
                        flat[f"{k}_{subk}"] = subv
                else:
                    flat[k] = v
            return flat

        if isinstance(results, dict):
            results = [_flatten_dict(results)]
        elif isinstance(results, list):
            results = [
                _flatten_dict(r) if any(isinstance(v, dict) for v in r.values()) else r
                for r in results
            ]
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)

    else:
        raise ValueError(f"format must be 'json' or 'csv', got {format!r}")

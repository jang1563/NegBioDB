"""XGBoost baseline for MD metabolite-disease biomarker classification.

Two tasks:
  MD-M1: Binary — is_significant=1 (positive biomarker) vs is_significant=0 (negative)
  MD-M2: Multi-class — disease_category (5-class: cancer/metabolic/neurological/cardiovascular/other)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def train_xgboost_md(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    task: str = "binary",
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    seed: int = 42,
    early_stopping_rounds: int = 20,
):
    """Train XGBoost model for MD classification.

    Args:
        X_train, y_train: Training features and labels.
        X_val, y_val:     Validation data (optional, for early stopping).
        task:             'binary' (MD-M1) or 'multiclass' (MD-M2).
        n_estimators:     Number of boosting rounds.
        max_depth:        Maximum tree depth.
        learning_rate:    Step size shrinkage.
        seed:             Random seed for reproducibility.
        early_stopping_rounds: Stop if validation metric does not improve.

    Returns:
        Trained xgboost.XGBClassifier.
    """
    import xgboost as xgb

    if task == "binary":
        objective = "binary:logistic"
        eval_metric = "logloss"
    else:
        n_classes = len(np.unique(y_train))
        objective = "multi:softprob"
        eval_metric = "mlogloss"

    params = {
        "objective": objective,
        "eval_metric": eval_metric,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "random_state": seed,
        "n_jobs": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "verbosity": 0,
    }
    if task == "multiclass":
        params["num_class"] = n_classes

    model = xgb.XGBClassifier(**params)

    fit_kwargs: dict = {}
    if X_val is not None and y_val is not None and len(y_val) > 0:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        fit_kwargs["verbose"] = False

    model.fit(X_train, y_train, **fit_kwargs)
    logger.info(
        "XGBoost %s trained: %d samples, %d features",
        task, len(X_train), X_train.shape[1],
    )
    return model


def evaluate_md(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str = "binary",
) -> dict:
    """Evaluate MD model and return metrics dict.

    Returns:
        For binary: {auroc, auprc, mcc, accuracy}
        For multiclass: {accuracy, mcc, per_class_accuracy}
    """
    import numpy as np
    _trapz = getattr(np, "trapezoid", np.trapz)

    y_pred = model.predict(X_test)

    try:
        from sklearn.metrics import (
            matthews_corrcoef, accuracy_score, roc_auc_score,
            average_precision_score,
        )
        mcc = matthews_corrcoef(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        metrics: dict = {"mcc": float(mcc), "accuracy": float(acc), "n_test": len(y_test)}

        if task == "binary":
            if len(np.unique(y_test)) > 1:
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics["auroc"] = float(roc_auc_score(y_test, y_prob))
                metrics["auprc"] = float(average_precision_score(y_test, y_prob))
            else:
                metrics["auroc"] = None
                metrics["auprc"] = None
    except Exception as exc:
        logger.warning("Evaluation error: %s", exc)
        metrics = {"mcc": None, "accuracy": None, "n_test": len(y_test)}

    return metrics

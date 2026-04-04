"""XGBoost baseline for GE gene essentiality prediction.

Tabular features (gene stats + cell line one-hot + omics) are well-suited
for gradient boosting. Expected to be the strongest baseline for GE tasks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def train_xgboost_ge(
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
    """Train XGBoost model for GE classification.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data (for early stopping).
        task: 'binary' (GE-M1) or 'multiclass' (GE-M2).
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage.
        seed: Random seed.
        early_stopping_rounds: Stop if no improvement.

    Returns:
        Trained xgboost.XGBClassifier model.
    """
    import xgboost as xgb

    if task == "binary":
        objective = "binary:logistic"
        eval_metric = "logloss"
        n_classes = 2
    else:
        objective = "multi:softprob"
        eval_metric = "mlogloss"
        n_classes = len(np.unique(y_train))

    params = {
        "objective": objective,
        "eval_metric": eval_metric,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    }
    if task == "multiclass":
        params["num_class"] = n_classes

    model = xgb.XGBClassifier(**params)

    eval_set = []
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]

    fit_kwargs = {
        "eval_set": eval_set if eval_set else None,
        "verbose": False,
    }
    if eval_set and early_stopping_rounds:
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

    model.fit(X_train, y_train, **fit_kwargs)

    logger.info(
        "XGBoost trained: %d estimators, best_iteration=%s",
        model.n_estimators,
        getattr(model, "best_iteration", "N/A"),
    )
    return model


def predict_xgboost_ge(
    model,
    X: np.ndarray,
    task: str = "binary",
) -> tuple[np.ndarray, np.ndarray]:
    """Predict with trained XGBoost model.

    Returns:
        (y_pred, y_prob): predicted labels and probabilities.
    """
    y_prob = model.predict_proba(X)
    if task == "binary":
        y_pred = (y_prob[:, 1] >= 0.5).astype(int)
    else:
        y_pred = np.argmax(y_prob, axis=1)

    return y_pred, y_prob


def save_xgboost_model(model, path: Path) -> None:
    """Save XGBoost model to JSON format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info("Saved XGBoost model to %s", path)


def load_xgboost_model(path: Path):
    """Load XGBoost model from JSON format."""
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model

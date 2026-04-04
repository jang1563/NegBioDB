"""XGBoost baseline for VP domain variant pathogenicity prediction.

Follows src/negbiodb_depmap/models/xgboost_ge.py pattern.
"""

import numpy as np

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def train_xgboost_vp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    task: str = "binary",
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    seed: int = 42,
    early_stopping_rounds: int = 15,
):
    """Train XGBoost classifier for VP tasks.

    Args:
        task: 'binary' (VP-M1) or 'multiclass' (VP-M2, 5-way ACMG)
    """
    if xgb is None:
        raise ImportError("xgboost is required: pip install xgboost")

    if task == "binary":
        objective = "binary:logistic"
        eval_metric = "logloss"
        n_classes = None
    else:
        objective = "multi:softprob"
        eval_metric = "mlogloss"
        n_classes = len(np.unique(y_train))

    params = {
        "objective": objective,
        "eval_metric": eval_metric,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "random_state": seed,
        "tree_method": "hist",
        "verbosity": 0,
    }
    if n_classes is not None:
        params["num_class"] = n_classes

    model = xgb.XGBClassifier(**params)

    fit_kwargs = {"verbose": False}
    if X_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        if early_stopping_rounds > 0:
            model.set_params(early_stopping_rounds=early_stopping_rounds)

    model.fit(X_train, y_train, **fit_kwargs)
    return model


def predict_xgboost_vp(
    model,
    X: np.ndarray,
    task: str = "binary",
) -> tuple[np.ndarray, np.ndarray]:
    """Predict with XGBoost model.

    Returns (y_pred, y_prob).
    """
    y_prob = model.predict_proba(X)
    if task == "binary":
        y_pred = (y_prob[:, 1] >= 0.5).astype(int)
    else:
        y_pred = np.argmax(y_prob, axis=1)
    return y_pred, y_prob

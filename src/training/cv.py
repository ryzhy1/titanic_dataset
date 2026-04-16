from __future__ import annotations
from typing import Any, cast
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from src.evaluation.metrics import calculate_classification_metrics
from src.models.sklearn_model import SklearnModel


@dataclass
class CVResult:
    model_name: str
    mean_score: float
    std_score: float
    fold_scores: list[float]


def cross_validate_model(
    model: SklearnModel,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 7,
    shuffle: bool = True,
    random_state: int = 42,
) -> CVResult:
    """Run StratifiedKFold CV and return fold-wise accuracy summary."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    scores: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        fold_model = cast(Any, clone(model))
        fold_model.fit(X_train_fold, y_train_fold)
        preds = fold_model.predict(X_val_fold)
        metrics = calculate_classification_metrics(y_val_fold, preds)
        scores.append(metrics["accuracy"])

    return CVResult(
        model_name=model_name,
        mean_score=float(np.mean(scores)),
        std_score=float(np.std(scores)),
        fold_scores=scores,
    )

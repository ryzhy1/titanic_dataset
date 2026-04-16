from __future__ import annotations

import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.models.factory import build_model
from typing import Any

def tune_single_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    base_config: dict,
    n_trials: int = 20,
    random_state: int = 42,
) -> dict[str, Any]:
    """Minimal Optuna tuner. Can be extended with richer search spaces later."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        config = {k: (v.copy() if isinstance(v, dict) else v) for k, v in base_config.items()}

        if model_name == "random_forest":
            config["random_forest"]["n_estimators"] = trial.suggest_int("n_estimators", 200, 800)
            config["random_forest"]["max_depth"] = trial.suggest_int("max_depth", 3, 10)
        elif model_name == "lightgbm":
            config["lightgbm"]["n_estimators"] = trial.suggest_int("n_estimators", 200, 800)
            config["lightgbm"]["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        elif model_name == "xgboost":
            config["xgboost"]["n_estimators"] = trial.suggest_int("n_estimators", 200, 800)
            config["xgboost"]["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        else:
            raise ValueError(f"Optuna tuning space is not defined for {model_name}")

        model = build_model(model_name, config)
        score = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return {
        "model_name": model_name,
        "best_score": float(study.best_value),
        "best_params": study.best_params,
    }

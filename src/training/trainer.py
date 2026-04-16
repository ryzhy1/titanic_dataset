from __future__ import annotations
from src.models.sklearn_model import SklearnModel

from pathlib import Path
from typing import Any, cast
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.evaluation.metrics import calculate_classification_metrics
from src.models.dnn import TitanicDataset, TitanicNN


def fit_full_model(model: SklearnModel, X: pd.DataFrame, y: pd.Series) -> SklearnModel:
    """Fit a sklearn-style estimator on the full training set."""
    final_model = cast(Any, clone(model))
    final_model.fit(X, y)
    return final_model

def evaluate_holdout(model: SklearnModel, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, float]:
    """Train on one split and return standard metrics on the validation part."""
    model = clone(model)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return calculate_classification_metrics(y_val, preds)

def train_dnn_cv(
    X: pd.DataFrame,
    y: pd.Series,
    dnn_params: dict,
    n_splits: int = 7,
    random_state: int = 42,
    model_dir: str | Path = "artifacts/models",
) -> dict:
    """Train the PyTorch MLP with StratifiedKFold and return CV summary."""
    import torch
    import torch.nn as nn

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    X_np = X.to_numpy(dtype=np.float32)
    y_np = y.to_numpy(dtype=np.int64)

    fold_scores: list[float] = []
    best_global_score = -1.0
    best_global_path: Path | None = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np), start=1):
        X_train_fold, X_val_fold = X_np[train_idx], X_np[val_idx]
        y_train_fold, y_val_fold = y_np[train_idx], y_np[val_idx]

        train_loader = DataLoader(
            TitanicDataset(X_train_fold, y_train_fold),
            batch_size=dnn_params["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            TitanicDataset(X_val_fold, y_val_fold),
            batch_size=dnn_params["batch_size"],
            shuffle=False,
        )

        model = TitanicNN(
            input_dim=X_np.shape[1],
            output_dim=2,
            hidden_dims=dnn_params.get("hidden_dims", [32, 64]),
            dropout=dnn_params.get("dropout", 0.3),
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=dnn_params["learning_rate"],
            weight_decay=dnn_params.get("weight_decay", 0.0),
        )

        best_fold_score = -1.0
        best_fold_path = model_dir / f"dnn_fold_{fold}.pth"

        for _ in range(dnn_params["epochs"]):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            preds_all = []
            y_true_all = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    logits = model(X_batch)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    preds_all.extend(preds.tolist())
                    y_true_all.extend(y_batch.numpy().tolist())

            metrics = calculate_classification_metrics(np.array(y_true_all), np.array(preds_all))
            acc = metrics["accuracy"]

            if acc > best_fold_score:
                best_fold_score = acc
                torch.save(model.state_dict(), best_fold_path)

        fold_scores.append(best_fold_score)

        if best_fold_score > best_global_score:
            best_global_score = best_fold_score
            best_global_path = best_fold_path

    return {
        "model_name": "dnn",
        "mean_score": float(np.mean(fold_scores)),
        "std_score": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
        "best_model_path": str(best_global_path) if best_global_path else None,
    }

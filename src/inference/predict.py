from __future__ import annotations

from pathlib import Path

import pandas as pd


def make_submission(
    model: object,
    X_test: pd.DataFrame,
    raw_test_df: pd.DataFrame,
    id_column: str = "passengerid",
) -> pd.DataFrame:
    preds = model.predict(X_test)
    return pd.DataFrame({
        "PassengerId": raw_test_df[id_column].values,
        "Survived": preds.astype(int),
    })


def save_submission(submission_df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(path, index=False)

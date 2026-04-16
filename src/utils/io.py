from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """создаем папку если ее не сущесвтует"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(model, path)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)

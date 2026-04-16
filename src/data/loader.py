from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_train_test(train_path: str | Path, test_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def build_full_dataframe(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([train_df, test_df], sort=True).reset_index(drop=True)

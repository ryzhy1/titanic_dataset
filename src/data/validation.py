from __future__ import annotations

import pandas as pd


def validate_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> None:
    """При конкатенации проверка что кол-во полей и сами поля совпадают"""
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    if target_col not in train_cols:
        raise ValueError(f"Target column '{target_col}' not found in train dataframe.")

    train_wo_target = train_cols - {target_col}
    if train_wo_target != test_cols:
        missing_in_test = sorted(train_wo_target - test_cols)
        missing_in_train = sorted(test_cols - train_wo_target)
        raise ValueError(
            "Train/test feature mismatch. "
            f"Missing in test: {missing_in_test}; extra in test: {missing_in_train}"
        )

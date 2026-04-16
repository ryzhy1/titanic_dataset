from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def label_encode_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder

    return df, encoders

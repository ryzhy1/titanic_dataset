from __future__ import annotations

import pandas as pd


def cv_results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Конвертация в датафрейм"""
    df = pd.DataFrame(results)
    if df.empty:
        return df
    return df.sort_values(by="mean_score", ascending=False).reset_index(drop=True)

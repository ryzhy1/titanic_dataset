from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def add_cluster_feature(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    n_clusters: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """добавляем фичу кластера на трейне и высчитываем на тесте"""
    train_df = train_df.copy()
    test_df = test_df.copy()

    features = [col for col in train_df.columns if col != target_col]
    X_train = train_df[features].select_dtypes(include="number").copy()
    X_test = test_df[features].select_dtypes(include="number").copy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    train_df["cluster"] = kmeans.fit_predict(X_train_scaled)
    test_df["cluster"] = kmeans.predict(X_test_scaled)
    return train_df, test_df

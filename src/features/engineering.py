import numpy as np
import pandas as pd


INITIAL_MAP = {
    "Master": 0,
    "Miss": 1,
    "Ms": 1,
    "Mme": 1,
    "Mlle": 1,
    "Mrs": 1,
    "Mr": 2,
    "Rare": 3,
}


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Базовый feature-engineering"""
    df = df.copy()

    # достаю кабину
    df["deck"] = df["cabin"].str[0]

    # маппинг пола
    df["sex"] = df["sex"].map({"male": 0, "female": 1})

    # достаем звание(или че это?)
    df["initial"] = df["name"].str.extract(r"([A-Za-z]+)\\.")
    df["initial"] = df["initial"].replace(
        ["Lady", "the Countess", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"],
        "Rare",
    )
    df["initial"] = df["initial"].replace(INITIAL_MAP)

    # заполняем средними значениями по званию
    df["age"] = df["age"].fillna(df.groupby("initial")["age"].transform("mean"))
    df["age"] = df["age"].fillna(df["age"].median()).astype(int)

    # заполняем место посадки
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    # семейный фичи
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_mother"] = ((df["parch"] > 0) & (df["initial"] == 1)).astype(int)

    # объединяем кабины по классам и заполняем неизвестные
    df["deck"] = df["deck"].fillna("U")
    df.loc[df["deck"] == "T", "deck"] = "A"
    df["deck"] = df["deck"].replace({
        "A": "ABC", "B": "ABC", "C": "ABC",
        "D": "DE", "E": "DE",
        "F": "FG", "G": "FG",
    })
    df["has_cabin"] = (df["deck"] != "U").astype(int)

    # нарезаем тариф
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["fare"] = np.log1p(df["fare"])
    df["fare_group"] = pd.qcut(df["fare"], 6, labels=False, duplicates="drop")

    # фича для расчета стоимости билета на члена семьи
    df["ticket_group_size"] = df.groupby("ticket")["ticket"].transform("count")

    # фича материнства
    df["is_mother_with_children"] = df["is_mother"] * df["parch"]

    # разбиваем возраст на группы
    df["age_group"] = 0
    df.loc[(df["age"] >= 7) & (df["age"] < 22), "age_group"] = 1
    df.loc[(df["age"] >= 22) & (df["age"] < 39), "age_group"] = 2
    df.loc[(df["age"] >= 39) & (df["age"] < 50), "age_group"] = 3
    df.loc[df["age"] >= 50, "age_group"] = 4

    # гурппировка размера семьи
    df["family_size_group"] = 0
    df.loc[(df["family_size"] >= 2) & (df["family_size"] < 5), "family_size_group"] = 1
    df.loc[(df["family_size"] >= 5) & (df["family_size"] < 7), "family_size_group"] = 2
    df.loc[(df["family_size"] >= 7) & (df["family_size"] < 12), "family_size_group"] = 3
    df.loc[df["family_size"] >= 12, "family_size_group"] = 4

    # удаляем неиспользуемые столбцы
    df = df.drop(columns=["name", "cabin", "ticket"])
    return df


def split_back_train_test(df: pd.DataFrame, target_col: str = "survived") -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df[target_col].notnull()].copy()
    test_df = df[df[target_col].isnull()].copy()
    return train_df, test_df


def drop_unused_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, drop_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.drop(columns=drop_columns, errors="ignore")
    test_df = test_df.drop(columns=drop_columns, errors="ignore")
    return train_df, test_df

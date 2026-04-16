from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.loader import build_full_dataframe, load_train_test
from src.data.validation import validate_columns
from src.evaluation.reports import cv_results_to_dataframe
from src.features.clustering import add_cluster_feature
from src.features.encoding import label_encode_columns
from src.features.engineering import add_base_features, drop_unused_columns, split_back_train_test
from src.features.preprocessing import cols_to_lower
from src.inference.predict import make_submission, save_submission
from src.models.ensemble import build_bagging, build_soft_voting, build_stacking, build_weighted_soft_voting
from src.models.factory import build_model
from src.training.cv import cross_validate_model
from src.training.trainer import fit_full_model, train_dnn_cv
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_dataframe, save_model
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Titanic ML pipeline.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    return parser.parse_args()


def prepare_features(config: dict, logger) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Load raw data and build final train/test features used by all models."""
    train_df_raw, test_df_raw = load_train_test(
        config["paths"]["train_path"],
        config["paths"]["test_path"],
    )
    validate_columns(train_df_raw, test_df_raw, config["task"]["target"])

    full_df = build_full_dataframe(train_df_raw, test_df_raw)
    full_df = cols_to_lower(full_df)
    full_df = add_base_features(full_df)
    full_df, _ = label_encode_columns(full_df, ["deck", "embarked"])

    train_df, test_df = split_back_train_test(full_df, config["task"]["target_lower"])

    if config["preprocessing"].get("use_kmeans_cluster", True):
        logger.info("Adding KMeans cluster feature.")
        train_df, test_df = add_cluster_feature(
            train_df=train_df,
            test_df=test_df,
            target_col=config["task"]["target_lower"],
            n_clusters=config["preprocessing"].get("n_clusters", 10),
            random_state=config["seed"],
        )

    train_df, test_df = drop_unused_columns(
        train_df,
        test_df,
        config["preprocessing"].get("drop_columns_after_features", []),
    )

    target_col = config["task"]["target_lower"]
    feature_cols = [col for col in train_df.columns if col != target_col]

    X = train_df[feature_cols].select_dtypes(include="number").copy()
    y = train_df[target_col].astype(int).copy()
    X_test = test_df[feature_cols].select_dtypes(include="number").copy()

    return X, y, X_test, test_df_raw


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = get_logger()

    seed_everything(config["seed"])
    logger.info("Preparing directories.")
    ensure_dir(config["paths"]["artifacts_dir"])
    ensure_dir(config["paths"]["submissions_dir"])
    ensure_dir(config["paths"]["models_dir"])
    ensure_dir(config["paths"]["metrics_dir"])
    ensure_dir(config["paths"]["figures_dir"])

    logger.info("Preparing features.")
    X, y, X_test, raw_test_df = prepare_features(config, logger)

    selected_models = config["training"]["selected_models"]
    model_params = config["model_params"]

    base_model_names = ["xgboost", "lightgbm", "gradient_boosting", "random_forest", "knn", "svc"]
    prebuilt_base_models = []
    for name in base_model_names:
        try:
            prebuilt_base_models.append((name, build_model(name, model_params)))
        except ValueError:
            continue

    results: list[dict] = []
    trained_models: dict[str, object] = {}

    for model_name in selected_models:
        logger.info(f"Running model: {model_name}")

        if model_name == "dnn":
            dnn_result = train_dnn_cv(
                X=X,
                y=y,
                dnn_params=model_params["dnn"],
                n_splits=config["cv"]["n_splits"],
                random_state=config["cv"]["random_state"],
                model_dir=config["paths"]["models_dir"],
            )
            results.append(dnn_result)
            continue

        if model_name == "soft_voting":
            model = build_soft_voting(prebuilt_base_models)
        elif model_name == "weighted_soft_voting":
            model = build_weighted_soft_voting(
                prebuilt_base_models,
                model_params["weighted_voting"]["weights"],
            )
        elif model_name == "stacking":
            model = build_stacking(prebuilt_base_models)
        elif model_name == "bagging":
            model = build_bagging(model_params["decision_tree"], model_params["bagging"])
        else:
            model = build_model(model_name, model_params)

        cv_result = cross_validate_model(
            model=model,
            model_name=model_name,
            X=X,
            y=y,
            n_splits=config["cv"]["n_splits"],
            shuffle=config["cv"]["shuffle"],
            random_state=config["cv"]["random_state"],
        )
        results.append({
            "model_name": cv_result.model_name,
            "mean_score": cv_result.mean_score,
            "std_score": cv_result.std_score,
            "fold_scores": cv_result.fold_scores,
        })

        final_model = fit_full_model(model, X, y)
        trained_models[model_name] = final_model

        if config["training"].get("save_all_models", True):
            save_model(final_model, Path(config["paths"]["models_dir"]) / f"{model_name}.joblib")

    leaderboard = cv_results_to_dataframe(results)
    save_dataframe(leaderboard, Path(config["paths"]["metrics_dir"]) / "cv_results.csv")
    logger.info("Saved CV results.")

    non_dnn_results = leaderboard[leaderboard["model_name"] != "dnn"].copy()
    if non_dnn_results.empty:
        logger.info("Only DNN result exists. Submission for DNN is not auto-generated in this version.")
        return

    best_model_name = non_dnn_results.iloc[0]["model_name"]
    logger.info(f"Best sklearn model by CV: {best_model_name}")

    best_model = trained_models[best_model_name]
    submission = make_submission(
        model=best_model,
        X_test=X_test,
        raw_test_df=raw_test_df,
        id_column=config["task"]["id_column"],
    )
    save_submission(submission, Path(config["paths"]["submissions_dir"]) / f"submission_{best_model_name}.csv")
    logger.info("Submission saved.")


if __name__ == "__main__":
    main()

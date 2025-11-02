"""
Model training script for blood culture outcome classification.

This script orchestrates the end-to-end run using helper utilities in
`training_utils.py`. It keeps behavior and outputs the same as before.
"""

import logging
import sys
from typing import List

import pandas as pd

from training_utils import (
    compute_weights,
    cross_validate,
    ensure_dirs,
    get_feature_spaces,
    init_logging,
    load_config,
    project_paths,
    save_feature_list,
    save_model,
    build_model,
    select_features,
    summarize_metrics,
)


# --------------------------------------------------------------------------------------
# Configuration & constants
# --------------------------------------------------------------------------------------


def main() -> None:
    init_logging()
    paths = project_paths()

    # Load config
    try:
        config = load_config(paths["config"])
    except Exception as e:
        logging.error("Failed to load config: %s", e)
        sys.exit(1)

    RANDOM_STATE = int(config["RANDOM_STATE"])
    RANDOM_STATE_BORUTA = int(config["RANDOM_STATE_BORUTA"])

    ensure_dirs([paths["features"], paths["models"], paths["results"]])

    # Load dataset
    data_path = paths["datasets"] / "training_data.csv"
    df_ml = pd.read_csv(data_path)
    y = df_ml["isPOS"]

    feature_spaces = get_feature_spaces()
    model_keys = ["dt", "rf", "xg", "lr"]
    feature_selection_methods = ["boruta"]
    weights_list = [1] 

    results_rows: List[pd.DataFrame] = []

    for feature_space_name, cols in feature_spaces.items():
        X_all = df_ml[cols].copy()

        for model_key in model_keys:
            for weight in weights_list:
                class_weights, pos_weight_xg = compute_weights(y, weight)
                estimator = build_model(
                    model_key,
                    random_state=RANDOM_STATE,
                    class_weights=class_weights,
                    pos_weight_xg=pos_weight_xg,
                )

                logging.info(
                    "Model=%s | Features=%s | weight=%.2f | class_weights=%s | pos_weight_xg=%.4f",
                    model_key,
                    feature_space_name,
                    weight,
                    class_weights,
                    pos_weight_xg,
                )

                for fsm in feature_selection_methods:
                    X_sel, selected_features = select_features(
                        fsm,
                        model_key,
                        estimator,
                        X_all,
                        y,
                        random_state_boruta=RANDOM_STATE_BORUTA,
                        class_weights=class_weights,
                        feature_names=list(cols),
                    )

                    logging.info("Selected %d features via %s", len(selected_features), fsm)

                    # Save selected feature list (preserve original filename pattern)
                    feature_list_path = (
                        paths["features"] / f"{model_key}_{feature_space_name}_{weight}_{fsm}.txt"
                    )
                    save_feature_list(feature_list_path, selected_features)

                    # Cross-validate
                    metrics = cross_validate(
                        estimator,
                        X_sel,
                        y,
                        n_splits=10,
                        random_state=RANDOM_STATE,
                    )

                    # Fit on all data and save model (preserve original filename pattern)
                    estimator.fit(X_sel.values, y.values)
                    model_path = (
                        paths["models"] / f"{model_key}_{feature_space_name}_{weight}_{fsm}.sav"
                    )
                    save_model(model_path, estimator)

                    # Summarize results row and collect
                    summary = summarize_metrics(metrics)
                    row = pd.DataFrame(
                        {
                            "model": f"{model_key}_{feature_space_name}_{weight}_{fsm}",
                            **summary,
                        },
                        index=[0],
                    )
                    results_rows.append(row)

    # Write consolidated CV results
    df_results = pd.concat(results_rows, ignore_index=True)
    results_path = paths["results"] / "results_cross_validation.csv"
    df_results.to_csv(results_path, index=False)
    logging.info("Wrote results to %s", results_path)


if __name__ == "__main__":
    main()
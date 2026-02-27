"""
Model training script for blood culture outcome classification.

This script orchestrates the end-to-end run using helper utilities in
`training_utils.py`. It keeps behavior and outputs the same as before.
"""

import argparse
import io
import logging
import sys
from typing import List

import numpy as np
import pandas as pd

from sklearn.base import clone
from .training_utils import (
    compute_weights,
    ensure_dirs,
    get_feature_spaces,
    init_logging,
    load_config,
    project_paths,
    save_feature_list,
    save_model,
    save_model_metadata,
    build_model,
    select_features,
    summarize_metrics,
    get_robust_n_splits,
    nested_cross_validate,
    plot_roc_curve,
    plot_confusion_matrix,
)
from .reporting import generate_training_report


# --------------------------------------------------------------------------------------
# Configuration & constants
# --------------------------------------------------------------------------------------

ALL_MODELS = ["dt", "rf", "xg", "lr"]
ALL_FS_METHODS = ["boruta", "rfe"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train blood culture outcome classification models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        metavar="MODEL",
        help=(
            "Models to train. Options: dt, rf, xg, lr, all. "
            "Pass 'all' to train every model (default)."
        ),
    )
    parser.add_argument(
        "--fs",
        nargs="+",
        default=["boruta"],
        metavar="METHOD",
        help=(
            "Feature selection methods. Options: boruta, rfe, none, all. "
            "'none' uses all features without selection. "
            "'all' runs every available method. Default: boruta."
        ),
    )
    return parser.parse_args()


def resolve_models(raw: List[str]) -> List[str]:
    """Resolve --models argument into a list of model keys."""
    if "all" in raw:
        return list(ALL_MODELS)
    unknown = set(raw) - set(ALL_MODELS)
    if unknown:
        logging.error("Unknown model(s): %s. Valid options: %s", unknown, ALL_MODELS)
        sys.exit(1)
    return list(dict.fromkeys(raw))  # dedupe, preserve order


def resolve_fs_methods(raw: List[str]) -> List[str]:
    """Resolve --fs argument into a list of FS method strings.

    'none' maps to the internal method name 'all' (use every feature).
    'all'  expands to every available FS method *plus* 'none'.
    """
    if "all" in raw:
        return ["all"] + list(ALL_FS_METHODS)  # "all" = no selection, then each method
    methods: List[str] = []
    for m in raw:
        if m == "none":
            methods.append("all")  # internal name for "no feature selection"
        elif m in ALL_FS_METHODS:
            methods.append(m)
        else:
            logging.error("Unknown FS method '%s'. Valid options: %s, none, all", m, ALL_FS_METHODS)
            sys.exit(1)
    return list(dict.fromkeys(methods))  # dedupe, preserve order


def main() -> None:
    args = parse_args()
    init_logging()

    # Load config first (from the fixed repo-root location) so DATA_DIR is
    # available before resolving other paths.
    from pathlib import Path as _Path
    _repo_root = _Path(__file__).resolve().parent.parent
    try:
        config = load_config(_repo_root / "config.json")
    except Exception as e:
        logging.error("Failed to load config: %s", e)
        sys.exit(1)

    paths = project_paths(config)

    RANDOM_STATE = int(config["RANDOM_STATE"])
    RANDOM_STATE_BORUTA = int(config["RANDOM_STATE_BORUTA"])

    ensure_dirs([paths["features"], paths["models"], paths["results"]])

    # Load dataset
    data_path = paths["datasets"] / "training_data.csv"
    
    # Read manually to filter lines starting with '#' but preserve '#' in column names
    with data_path.open("r") as f:
        lines = [line for line in f if not line.strip().startswith("#")]
    df_ml = pd.read_csv(io.StringIO("".join(lines)))
    
    y = df_ml[config["TARGET_COLUMN"]]

    feature_spaces = get_feature_spaces(config)
    model_keys = resolve_models(args.models)
    feature_selection_methods = resolve_fs_methods(args.fs)
    weights_list = [1]

    logging.info("Models: %s | Feature Selection: %s", model_keys, feature_selection_methods)  

    results_rows = []
    plots_dict = {}

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
                    # 1. Nested Cross-Validation (Unbiased Evaluation)
                    logging.info("Starting Nested CV for %s | %s", model_key, feature_space_name)
                    
                    # Dynamic n_splits based on dataset size
                    n_splits = get_robust_n_splits(y, n_splits_target=10)

                    metrics, agg_data = nested_cross_validate(
                        estimator=estimator,
                        X_all=X_all,
                        y=y,
                        n_splits=n_splits,
                        random_state=RANDOM_STATE,
                        # Feature Selection params
                        fs_method=fsm,
                        model_key=model_key,
                        fs_random_state=RANDOM_STATE_BORUTA,
                        class_weights=class_weights,
                        feature_names=list(cols),
                    )

                    # Generate Plots (from aggregated Nested CV predictions)
                    roc_b64 = plot_roc_curve(agg_data["y_true"], agg_data["y_proba"], title=f"ROC: {model_key} using {feature_space_name}")
                    cm_b64 = plot_confusion_matrix(agg_data["y_true"], agg_data["y_pred"], title=f"Confusion Matrix: {model_key}")
                    
                    plots_dict[f"{model_key}_{feature_space_name}_{weight}_{fsm}"] = {"roc": roc_b64, "cm": cm_b64}

                    # Summarize validation metrics
                    summary = summarize_metrics(metrics)
                    summary.update(
                        {
                            "model": f"{model_key}_{feature_space_name}_{weight}_{fsm}",
                            "feature_space": feature_space_name,
                            "class_weight": weight,
                            "selection_method": fsm,
                        }
                    )
                    results_rows.append(summary)

                    # 2. Final Model Training (Global Feature Selection)
                    # We retain this step to produce the final deployable model artifacts.
                    # Note: The performance of *this* specific model instance is approximated by the Nested CV results above,
                    # but technically acts on slightly different (subset) features if stability is low.
                    logging.info("Training Final Model (Global Selection)...")
                    
                    X_final_sel, final_selected_features = select_features(
                        method=fsm,
                        model_key=model_key,
                        estimator=clone(estimator),
                        X_all=X_all,
                        y=y,
                        random_state_boruta=RANDOM_STATE_BORUTA,
                        class_weights=class_weights,
                        feature_names=list(cols),
                    )
                    
                    if not final_selected_features:
                        logging.warning("No features selected for final model %s. Skipping save.", model_key)
                        continue

                    # Train on full data with selected features
                    final_model = clone(estimator)
                    final_model.fit(X_final_sel.values, y.values)
                    
                    # Save artifacts
                    feature_list_path = (
                        paths["features"]
                        / f"{model_key}_{feature_space_name}_{weight}_{fsm}.txt"
                    )
                    save_feature_list(feature_list_path, final_selected_features)

                    model_path = (
                        paths["models"] / f"{model_key}_{feature_space_name}_{weight}_{fsm}.sav"
                    )
                    save_model(model_path, final_model)
                    
                    # Save metadata
                    metadata_path = paths["models"] / f"{model_key}_{feature_space_name}_{weight}_{fsm}.json"
                    save_model_metadata(
                        metadata_path,
                        model_key=model_key,
                        feature_space=feature_space_name,
                        weight=float(weight),
                        fsm=fsm,
                        features=list(final_selected_features),
                        threshold=0.3, 
                        model_path=model_path.name,
                        feature_list_path=feature_list_path.name,
                    )


    # Write consolidated CV results
    df_results = pd.DataFrame(results_rows)
    results_path = paths["results"] / "results_cross_validation.csv"
    df_results.to_csv(results_path, index=False)
    logging.info("Wrote results to %s", results_path)

    # Generate HTML Report
    dataset_info = {
        "n_total": len(df_ml),
        "n_pos": int(y.sum()),
        "pct_pos": (y.sum() / len(y)) * 100,
        "n_neg": int((y == 0).sum()),
        "pct_neg": ((y == 0).sum() / len(y)) * 100,
    }
    report_path = paths["results"] / "training_report.html"
    generate_training_report(
        output_path=report_path,
        dataset_info=dataset_info,
        results_df=df_results,
        config=config,
        plots=plots_dict,
    )
    logging.info("Wrote training report to %s", report_path)


if __name__ == "__main__":
    main()
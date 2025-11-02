"""
Inference script for blood culture outcome classification models.

Rules:
- Logistic Regression: extract raw-space coefficients from the pipeline and use them
  directly (without the model object) at threshold=0.3. Export coefficients.
- Decision Tree: extract human-readable rules and use those for inference (no model object).
  Export rules in text and JSON.
- Other models (RF, XG): use the model objects for inference.

The features used for inference are loaded from the corresponding features/*.txt files.
Outputs:
- predictions/*.csv for each model (prob and label)
- exports/lr_coeffs_*.csv for LR
- exports/dt_rules_*.txt and dt_rules_*.json for DT
"""

from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from training_utils import (
    init_logging,
    project_paths,
    load_model,
    load_feature_list,
    load_lr_coefficients_csv,
    load_dt_rules_json,
    compute_performance_metrics,
    extract_lr_raw_params_from_pipeline,
    predict_lr_with_raw_params,
    export_lr_coefficients_csv,
    extract_decision_tree_rules,
    render_decision_rules_text,
    predict_with_dt_rules,
    export_confusion_matrix_csv,
    predict_with_lr_pipeline,
    predict_with_dt_model,
    compare_predictions,
)

DEFAULT_THRESHOLD = 0.3


def list_model_artifacts(models_dir: Path, features_dir: Path) -> List[Tuple[Path, Path, str, str, str, str]]:
    """Pair up model files with their feature files based on naming pattern.

    Returns list of tuples: (model_path, feature_path, model_key, feature_space, weight, fsm)
    """
    pairs: List[Tuple[Path, Path, str, str, str, str]] = []
    for model_path in models_dir.glob("*.sav"):
        # pattern: {model}_{features}_{weight}_{fsm}.sav
        name = model_path.stem
        parts = name.split("_")
        # model could be 'dt' or 'lr' or 'rf' or 'xg'; features can contain underscores
        # Reconstruct from right: fsm (last), weight (2nd last), feature_space (rest minus model)
        if len(parts) < 4:
            continue
        model_key = parts[0]
        fsm = parts[-1]
        weight = parts[-2]
        feature_space = "_".join(parts[1:-2])
        feature_path = features_dir / f"{model_key}_{feature_space}_{weight}_{fsm}.txt"
        if feature_path.exists():
            pairs.append((model_path, feature_path, model_key, feature_space, weight, fsm))
    return pairs


essential_newline = "\n"

def ensure_dirs(paths: List[Path]):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def run_inference(
    df: pd.DataFrame,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    validate: bool = False,
) -> pd.DataFrame:
    paths = project_paths()
    models_dir = paths["models"]
    features_dir = paths["features"]
    exports_dir = paths["root"] / "exports"
    predictions_dir = paths["root"] / "predictions"
    ensure_dirs([exports_dir, predictions_dir])

    outputs: List[pd.DataFrame] = []
    metrics_rows: List[dict] = []

    

    for model_path, feature_path, model_key, feature_space, weight, fsm in list_model_artifacts(models_dir, features_dir):
        model_name = f"{model_key}_{feature_space}_{weight}_{fsm}"
        logging.info("Running inference for %s", model_name)
        features = load_feature_list(feature_path)

        features_and_target = features + ["isPOS"]
        df_temp = df.copy()

        if "isPOS" in df.columns:
            df_temp = df_temp[features_and_target]
            # drop rows with missing values in any of the features or target
            df_temp = df_temp.dropna(subset=features_and_target)
            y_true = df_temp["isPOS"].astype(int)
            df_temp = df_temp.drop(columns=["isPOS"])  # ensure we only use features
        else:
            df_temp = df_temp[features]
            df_temp = df_temp.dropna(subset=features)
            y_true = None

        X = df_temp[features].copy()

        if model_key == "lr":
            # Prefer existing exported coefficients; fallback to pipeline extraction
            coeffs_csv = exports_dir / f"lr_coeffs_{feature_space}_{weight}_{fsm}.csv"
            if coeffs_csv.exists():
                feats_order, weights_raw, intercept_raw = load_lr_coefficients_csv(coeffs_csv)
                probs, preds = predict_lr_with_raw_params(
                    X,
                    feature_order=feats_order,
                    weights_raw=weights_raw,
                    intercept_raw=intercept_raw,
                    threshold=threshold,
                )
                pipeline = None
            else:
                pipeline = load_model(model_path)
                params = extract_lr_raw_params_from_pipeline(pipeline, features)
                probs, preds = predict_lr_with_raw_params(
                    X,
                    feature_order=params["features"],
                    weights_raw=params["weights_raw"],
                    intercept_raw=params["intercept_raw"],
                    threshold=threshold,
                )
                # Export coefficients for future runs
                export_lr_coefficients_csv(
                    coeffs_csv, params["features"], params["weights_raw"], params["intercept_raw"]
                )

            # Optional validation: if a pipeline was loaded, compare against it
            if validate and 'pipeline' in locals() and pipeline is not None:
                proba_b, preds_b = predict_with_lr_pipeline(pipeline, X, threshold=threshold)
                report = compare_predictions(probs, preds, proba_b, preds_b)
                (exports_dir / "validation").mkdir(parents=True, exist_ok=True)
                (exports_dir / "validation" / f"validate_lr_{feature_space}_{weight}_{fsm}.json").write_text(
                    json.dumps(report, indent=2)
                )

        elif model_key == "dt":
            # Prefer existing exported rules; fallback to model extraction
            rules_txt = exports_dir / f"dt_rules_{feature_space}_{weight}_{fsm}.txt"
            rules_json = exports_dir / f"dt_rules_{feature_space}_{weight}_{fsm}.json"
            if rules_json.exists():
                tree_dict, saved_thr = load_dt_rules_json(rules_json)
                # Re-annotate with current threshold if different
                from training_utils import mark_dt_threshold_predictions
                tree_dict = mark_dt_threshold_predictions(tree_dict, threshold)
                dt_model = None
            else:
                dt_model = load_model(model_path)
                tree_dict = extract_decision_tree_rules(dt_model, features)
                # Annotate leaves with threshold-based predictions
                from training_utils import mark_dt_threshold_predictions
                tree_dict = mark_dt_threshold_predictions(tree_dict, threshold)
                # Save text (prepend threshold info) and JSON
                lines = render_decision_rules_text(tree_dict)
                header = [f"# probability_threshold={threshold:.6f}"]
                rules_txt.write_text(essential_newline.join(header + lines) + essential_newline)
                export_obj = {"probability_threshold": float(threshold), "tree": tree_dict}
                rules_json.write_text(json.dumps(export_obj, indent=2))

            probs, preds_leaf = predict_with_dt_rules(X, features, tree_dict)
            # Apply probability threshold to DT probabilities
            preds = (probs >= threshold).astype(int)

            # Optional validation: compare against DT model predictions if model loaded
            if validate and 'dt_model' in locals() and dt_model is not None:
                proba_b, preds_b = predict_with_dt_model(dt_model, X, threshold=threshold)
                report = compare_predictions(probs, preds, proba_b, preds_b)
                (exports_dir / "validation").mkdir(parents=True, exist_ok=True)
                (exports_dir / "validation" / f"validate_dt_{feature_space}_{weight}_{fsm}.json").write_text(
                    json.dumps(report, indent=2)
                )

        else:
            # RF, XG: use model directly
            model = load_model(model_path)
            try:
                probs = model.predict_proba(X.values)[:, 1]
            except Exception:
                # fallback to decision function or binary prediction
                if hasattr(model, "decision_function"):
                    margins = model.decision_function(X.values)
                    probs = 1.0 / (1.0 + np.exp(-margins))
                else:
                    preds = model.predict(X.values)
                    probs = preds.astype(float)
            preds = (probs >= threshold).astype(int)

        # Save predictions
        out_payload = {
            "model": model_name,
            "prob_pos": probs,
            "yhat": preds,
        }
        # Include the leaf-based DT predictions for transparency
        if model_key == "dt":
            out_payload["yhat_leaf"] = preds_leaf
        out_df = pd.DataFrame(out_payload)
        out_path = predictions_dir / f"preds_{model_name}.csv"
        out_df.to_csv(out_path, index=False)
        outputs.append(out_df)

        # If ground truth provided, export confusion matrix and collect metrics
        if y_true is not None and len(y_true) == len(out_df):
            cm_path = predictions_dir / f"cm_{model_name}.csv"
            export_confusion_matrix_csv(cm_path, y_true.values.astype(int), preds.astype(int))
            # Metrics summary per model
            perf = compute_performance_metrics(y_true.values.astype(int), preds.astype(int), probs)
            perf_row = {"model": model_name, "threshold": float(threshold)}
            perf_row.update(perf)
            metrics_rows.append(perf_row)

    # Write aggregated metrics if available
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_out = predictions_dir / "metrics_summary.csv"
        metrics_df.to_csv(metrics_out, index=False)
        logging.info("Wrote metrics summary to %s", metrics_out)

    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Inference for blood culture outcome models")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Probability threshold for binary predictions (default: 0.3)")
    parser.add_argument("--validate", action="store_true", help="Validate LR/DT extracted predictions against model objects and write a JSON report")
    args = parser.parse_args()

    init_logging()
    paths = project_paths()
    # Default: if a testing dataset exists, use it; otherwise fall back to training.
    test_path = paths["datasets"] / "testing_sysmex_deduped.csv"
    if test_path.exists():
        df_all = pd.read_csv(test_path)
    else:
        df_all = pd.read_csv(paths["datasets"] / "training_data.csv")


    preds = run_inference(df_all, threshold=args.threshold, validate=args.validate)
    logging.info("Inference complete. Rows: %d", len(preds))


if __name__ == "__main__":
    main()

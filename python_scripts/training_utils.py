"""
Utilities for the blood culture outcome classification training pipeline.

Contains config loading, path management, feature spaces, model builders,
feature selection, cross-validation, metric summarization, and basic I/O helpers.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import base64
import io
import json
import logging
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from boruta import BorutaPy
from imblearn.metrics import specificity_score
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


# --------------------------------------------------------------------------------------
# Configuration & paths
# --------------------------------------------------------------------------------------


# Required keys that must be present in config.json
_REQUIRED_CONFIG_KEYS = ("RANDOM_STATE", "RANDOM_STATE_BORUTA")


def load_config(config_path: Path) -> Dict:
    """Load configuration JSON with required keys.

    Required keys: RANDOM_STATE, RANDOM_STATE_BORUTA
    Optional keys: TARGET_COLUMN (default 'isPOS'), FEATURE_SPACES (dict)
    """
    with config_path.open() as f:
        cfg = json.load(f)
    missing = [k for k in _REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise KeyError(
            f"Missing required config keys in {config_path.name}: {', '.join(missing)}"
        )
    # Apply defaults for optional keys
    cfg.setdefault("TARGET_COLUMN", "isPOS")
    cfg.setdefault("DATA_DIR", "datasets")
    return cfg


def project_paths(cfg: Dict | None = None) -> Dict[str, Path]:
    """Return important project paths relative to this file location.

    Parameters
    ----------
    cfg:
        Optional config dict (as returned by ``load_config``).  When provided,
        the ``DATA_DIR`` key is used to resolve the datasets directory.
        The value may be an absolute path or a path relative to the repo root.
        Defaults to ``"datasets"`` when absent.
    """
    here = Path(__file__).resolve()
    repo_root = here.parent.parent  # move out of python_scripts/
    data_dir_raw = (cfg or {}).get("DATA_DIR", "datasets")
    data_dir = Path(data_dir_raw)
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir
    return {
        "root": repo_root,
        "config": repo_root / "config.json",
        "datasets": data_dir,
        "features": repo_root / "features",
        "models": repo_root / "models",
        "results": repo_root / "results",
    }


def init_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# --------------------------------------------------------------------------------------
# Feature spaces
# --------------------------------------------------------------------------------------


# Default feature spaces (used when config.json does not define FEATURE_SPACES)
_DEFAULT_FEATURE_SPACES: Dict[str, List[str]] = {
    "CBC_DIFF": [
        "WBC(10^9/L)", "RBC(10^12/L)", "HGB(g/L)", "MCV(fL)", "MCHC(g/L)",
        "PLT(10^9/L)", "RDW-CV(%)",
        "NEUT#(10^9/L)", "LYMPH#(10^9/L)", "MONO#(10^9/L)", "EO#(10^9/L)", "BASO#(10^9/L)",
        "NEUT%(%)", "LYMPH%(%)", "MONO%(%)", "EO%(%)", "BASO%(%)",
        "NLR", "MLR",
    ],
    "CBC_DIFF_CPD": [
        "WBC(10^9/L)", "RBC(10^12/L)", "HGB(g/L)", "MCV(fL)", "MCHC(g/L)",
        "PLT(10^9/L)", "RDW-CV(%)",
        "NEUT#(10^9/L)", "LYMPH#(10^9/L)", "MONO#(10^9/L)", "EO#(10^9/L)", "BASO#(10^9/L)",
        "NEUT%(%)", "LYMPH%(%)", "MONO%(%)", "EO%(%)", "BASO%(%)",
        "NLR", "MLR",
        "[NE-SSC(ch)]", "[NE-SFL(ch)]", "[NE-FSC(ch)]",
        "[LY-X(ch)]", "[LY-Y(ch)]", "[LY-Z(ch)]",
        "[MO-X(ch)]", "[MO-Y(ch)]", "[MO-Z(ch)]",
        "[NE-WX]", "[NE-WY]", "[NE-WZ]",
        "[LY-WX]", "[LY-WY]", "[LY-WZ]",
        "[MO-WX]", "[MO-WY]", "[MO-WZ]",
        "IP ABN(WBC)WBC Abn Scattergram",
        "IP ABN(WBC)Neutropenia", "IP ABN(WBC)Neutrophilia",
        "IP ABN(WBC)Lymphopenia", "IP ABN(WBC)Lymphocytosis",
        "IP ABN(WBC)Leukocytopenia", "IP ABN(WBC)Leukocytosis",
        "IP ABN(PLT)Thrombocytopenia",
        "IP SUS(WBC)Blasts/Abn Lympho?", "IP SUS(WBC)Blasts?",
        "IP SUS(WBC)Abn Lympho?", "IP SUS(WBC)Left Shift?",
        "IP SUS(WBC)Atypical Lympho?",
    ],
}


def get_feature_spaces(cfg: Dict | None = None) -> Dict[str, List[str]]:
    """Return feature space configurations.

    Reads from ``cfg["FEATURE_SPACES"]`` if present, otherwise uses built-in
    defaults.  This allows users to define custom feature spaces in
    ``config.json`` without modifying source code.
    """
    if cfg and "FEATURE_SPACES" in cfg:
        return {k: list(v) for k, v in cfg["FEATURE_SPACES"].items()}
    return {k: list(v) for k, v in _DEFAULT_FEATURE_SPACES.items()}


# --------------------------------------------------------------------------------------
# Model building and feature selection
# --------------------------------------------------------------------------------------


def compute_weights(y: pd.Series, weight: float) -> Tuple[Dict[int, float], float]:
    """Compute class weights dict for sklearn and scale_pos_weight for XGBoost."""
    weighting = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=y
    )
    weights = {0: float(weighting[0]), 1: float(weighting[1]) * float(weight)}
    pos_weight_xg = (float(weighting[1]) / float(weighting[0])) * float(weight)
    logging.info("Balanced class weights: %s | pos_weight_xg=%.4f", weights, pos_weight_xg)
    return weights, pos_weight_xg


def build_model(
    model_key: str, *, random_state: int, class_weights: Dict[int, float], pos_weight_xg: float
):
    """Instantiate the estimator for a given key.

    Supported keys: 'rf', 'dt', 'xg', 'lr'
    """
    if model_key == "rf":
        return RandomForestClassifier(
            random_state=random_state, class_weight=class_weights, max_depth=3, n_jobs=-1
        )
    if model_key == "dt":
        return DecisionTreeClassifier(
            random_state=random_state, class_weight=class_weights, max_depth=3
        )
    if model_key == "xg":
        return XGBClassifier(
            learning_rate=0.01,
            max_depth=3,
            n_estimators=100,
            scale_pos_weight=pos_weight_xg,
            random_state=random_state,
            eval_metric="logloss",
        )
    if model_key == "lr":
        return Pipeline(
            steps=[
                ("scaler", RobustScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=random_state, class_weight=class_weights, max_iter=1000
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model key: {model_key}")


def select_features(
    method: str,
    model_key: str,
    estimator,
    X_all: pd.DataFrame,
    y: pd.Series,
    *,
    random_state_boruta: int,
    class_weights: Dict[int, float],
    feature_names: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Select features using the requested method and return reduced X and names.

    WARNING: Data Leakage
    ---------------------
    If this function is called on the entire dataset *before* cross-validation, the resulting
    performance metrics will be optimistically biased. The model "sees" the test data during
    feature selection. For rigorous performance estimation, feature selection should be
    performed inside the CV loop (Nested CV). This pipeline currently uses the "global"
    approach to prioritize final model performance over unbiased estimation.

    method: 'boruta' | 'rfe' | 'all'
    """
    method = method.lower()
    if method == "all":
        return X_all.copy(), list(feature_names)

    if method == "boruta":
        # Boruta requires an estimator with feature_importances_. For LR and DT,
        # use a RF as the feature selector base estimator for stability.
        if model_key in {"dt", "lr"}:
            fs_estimator = RandomForestClassifier(
                random_state=random_state_boruta, class_weight=class_weights, max_depth=3, n_jobs=-1
            )
        else:
            fs_estimator = estimator
        boruta = BorutaPy(fs_estimator, verbose=0, random_state=random_state_boruta)
        boruta.fit(X_all.values, y.values)
        ranks = list(zip(feature_names, boruta.ranking_, boruta.support_))
        selected = [name for name, rank, keep in ranks if rank == 1]
        for name, rank, keep in ranks:
            logging.debug("Feature: %-25s Rank: %s Keep: %s", name, rank, keep)
        return X_all[selected].copy(), selected

    if method == "rfe":
        # RFE requires an estimator with coef_ or feature_importances_. Pipelines
        # may not work directly. Use RF for RFE when needed.
        rfe_estimator = (
            RandomForestClassifier(
                random_state=random_state_boruta, class_weight=class_weights, max_depth=3, n_jobs=-1
            )
            if isinstance(estimator, Pipeline)
            else estimator
        )
        rfe = RFE(estimator=rfe_estimator, n_features_to_select=min(5, X_all.shape[1]))
        rfe.fit(X_all.values, y.values)
        ranks = list(zip(feature_names, rfe.ranking_, rfe.support_))
        selected = [name for name, rank, keep in ranks if rank == 1]
        for name, rank, keep in ranks:
            logging.debug("Feature: %-25s Rank: %s Keep: %s", name, rank, keep)
        return X_all[selected].copy(), selected

    raise ValueError(f"Unknown feature selection method: {method}")


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def get_robust_n_splits(y: pd.Series, n_splits_target: int = 10) -> int:
    """Calculate a robust n_splits for StratifiedKFold based on class counts.

    Ensures n_splits does not exceed the number of samples in the smallest class.
    """
    min_class_samples = y.value_counts().min()
    n_splits = max(2, min(n_splits_target, min_class_samples))
    
    if n_splits < n_splits_target:
        logging.warning(
            "Small dataset detected (%d samples in smallest class). "
            "Reduced n_splits from %d to %d to satisfy StratifiedKFold constraints.",
            min_class_samples, n_splits_target, n_splits
        )
    return n_splits


def nested_cross_validate(
    estimator,
    X_all: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    random_state: int | None = None,
    # Feature selection params
    fs_method: str = "all",
    model_key: str = "",
    fs_random_state: int = 42,
    class_weights: Dict[int, float] = None,
    feature_names: List[str] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Run StratifiedKFold CV with Feature Selection INSIDE the loop (Nested CV).

    Returns:
    1. metrics: Dict of score arrays (one per fold).
    2. aggregate_data: Dict containing consolidated 'y_true', 'y_proba', 'y_pred' for plotting.
    """
    recall_scores: List[float] = []
    precision_scores: List[float] = []
    roc_auc_scores: List[float] = []
    balanced_accuracy_scores: List[float] = []
    specificity_scores: List[float] = []
    j_stat_scores: List[float] = []
    f2_scores: List[float] = []
    
    # Store aggregated predictions for global plotting
    y_true_all = []
    y_proba_all = []
    y_pred_all = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_idx = 1
    for train_index, test_index in skf.split(X_all.values, y.values):
        X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # 1. Feature Selection on TRAIN fold only
        # We pass a fresh clone of estimator if needed, but select_features handles internal logic
        # It needs the base estimator for Boruta/RFE
        X_train_sel, selected_features = select_features(
            method=fs_method,
            model_key=model_key,
            estimator=clone(estimator), # Clone to avoid fitting artifacts
            X_all=X_train,
            y=y_train,
            random_state_boruta=fs_random_state,
            class_weights=class_weights,
            feature_names=feature_names,
        )
        
        # Apply selection to test
        X_test_sel = X_test[selected_features].copy()
        
        logging.debug("Fold %d: Selected %d features", fold_idx, len(selected_features))

        # 2. Train Model
        est_fold = clone(estimator)
        est_fold.fit(X_train_sel.values, y_train.values)
        
        # 3. Predict on Test
        y_hat = est_fold.predict(X_test_sel.values)
        try:
            proba = est_fold.predict_proba(X_test_sel.values)[:, 1]
            auc_score = roc_auc_score(y_test, proba)
        except Exception:
            proba = np.zeros_like(y_hat, dtype=float)
            auc_score = np.nan

        # 4. Metrics
        rec = recall_score(y_test, y_hat, zero_division=0)
        prec = precision_score(y_test, y_hat, zero_division=0)
        bac = balanced_accuracy_score(y_test, y_hat)
        spec = float(specificity_score(y_test, y_hat))
        j_stat = rec + spec - 1
        denom = (4 * prec + rec)
        f2 = (5 * prec * rec) / denom if denom > 0 else 0.0

        recall_scores.append(rec)
        precision_scores.append(prec)
        roc_auc_scores.append(auc_score)
        balanced_accuracy_scores.append(bac)
        specificity_scores.append(spec)
        j_stat_scores.append(j_stat)
        f2_scores.append(f2)
        
        # Aggregate for global plots
        y_true_all.extend(y_test)
        y_proba_all.extend(proba)
        y_pred_all.extend(y_hat)
        
        fold_idx += 1

    # Likelihood ratios with divide-by-zero handling
    recall_arr = np.array(recall_scores)
    spec_arr = np.array(specificity_scores)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr_plus = recall_arr / (1 - spec_arr)
        lr_minus = (1 - recall_arr) / spec_arr

    metrics = {
        "recall": np.array(recall_scores),
        "precision": np.array(precision_scores),
        "roc_auc": np.array(roc_auc_scores),
        "balanced_accuracy": np.array(balanced_accuracy_scores),
        "specificity": spec_arr,
        "j_stat": np.array(j_stat_scores),
        "f2": np.array(f2_scores),
        "lr_plus": lr_plus,
        "lr_minus": lr_minus,
    }
    
    agg_data = {
        "y_true": np.array(y_true_all),
        "y_proba": np.array(y_proba_all),
        "y_pred": np.array(y_pred_all),
    }
    
    return metrics, agg_data


def summarize_metrics(metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Return mean/std for each metric array rounded to 2 decimals."""
    def m(x: np.ndarray) -> float:
        return round(float(np.nanmean(x)), 2)

    def s(x: np.ndarray) -> float:
        return round(float(np.nanstd(x)), 2)

    return {
        "recall_mean": m(metrics["recall"]),
        "recall_std": s(metrics["recall"]),
        "precision_mean": m(metrics["precision"]),
        "precision_std": s(metrics["precision"]),
        "roc_auc_mean": m(metrics["roc_auc"]),
        "roc_auc_std": s(metrics["roc_auc"]),
        "balanced_accuracy_mean": m(metrics["balanced_accuracy"]),
        "balanced_accuracy_std": s(metrics["balanced_accuracy"]),
        "specificity_mean": m(metrics["specificity"]),
        "specificity_std": s(metrics["specificity"]),
        "lr+_mean": m(metrics["lr_plus"]),
        "lr+_std": s(metrics["lr_plus"]),
        "lr-_mean": m(metrics["lr_minus"]),
        "lr-_std": s(metrics["lr_minus"]),
        "j_stat_mean": m(metrics["j_stat"]),
        "j_stat_std": s(metrics["j_stat"]),
        "f2_mean": m(metrics["f2"]),
        "f2_std": s(metrics["f2"]),
    }


def plot_roc_curve(y_true, y_proba, title="ROC Curve") -> str:
    """Generate ROC curve plot and return base64 string."""
    plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix") -> str:
    """Generate Confusion Matrix plot and return base64 string."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    # Ticks might behave oddly with matshow, fix them
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks([0, 1], ['Neg', 'Pos'])
    plt.yticks([0, 1], ['Neg', 'Pos'])
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------


def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


essential_newline = "\n"

def save_feature_list(path: Path, features: List[str]) -> None:
    path.write_text(essential_newline.join(features) + essential_newline)


def save_model(path: Path, model) -> None:
    with path.open("wb") as f:
        pickle.dump(model, f)


# --------------------------------------------------------------------------------------
# Inference utilities
# --------------------------------------------------------------------------------------


def save_model_metadata(
    path: Path,
    *,
    model_key: str,
    feature_space: str,
    weight: float,
    fsm: str,
    features: List[str],
    threshold: float = 0.3,
    **kwargs
) -> None:
    """Save metadata about a trained model config to a JSON file."""
    meta = {
        "model_key": model_key,
        "feature_space": feature_space,
        "weight": float(weight),
        "fsm": fsm,
        "features": features,
        "threshold": float(threshold),
        **kwargs
    }
    path.write_text(json.dumps(meta, indent=2))



def load_model(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_feature_list(path: Path) -> List[str]:
    """Load feature list from a .txt file, one feature per line."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def extract_lr_raw_params_from_pipeline(pipeline: Pipeline, feature_names: List[str]):
    """Extract raw-space logistic regression weights and intercept from a pipeline.

    The pipeline is expected to have ("scaler", RobustScaler) and ("classifier", LogisticRegression).
    This function returns weights/intercept that work directly on the original feature values
    without needing to apply the scaler.
    """
    # Extract components
    scaler: RobustScaler = pipeline.named_steps.get("scaler")
    lr: LogisticRegression = pipeline.named_steps.get("classifier")

    if lr is None or scaler is None:
        raise ValueError("Expected a Pipeline with 'scaler' and 'classifier' steps for LR")

    # Coefficients for positive class
    # For binary classification, lr.coef_ shape is (1, n_features)
    w = lr.coef_[0].astype(float)
    b = float(lr.intercept_[0])

    # RobustScaler attributes
    # Handle missing attributes gracefully
    center = getattr(scaler, "center_", np.zeros_like(w))
    scale = getattr(scaler, "scale_", np.ones_like(w))
    scale_safe = np.where(scale == 0, 1.0, scale)

    # Convert to raw feature space: w'_i = w_i/scale_i; b' = b - sum_i w_i * center_i / scale_i
    w_raw = w / scale_safe
    b_raw = b - np.sum(w * center / scale_safe)

    return {
        "features": list(feature_names),
        "weights_raw": w_raw.tolist(),
        "intercept_raw": float(b_raw),
        "classes": lr.classes_.tolist(),
    }


def predict_lr_with_raw_params(
    X: pd.DataFrame,
    feature_order: List[str],
    weights_raw: List[float],
    intercept_raw: float,
    threshold: float = 0.3,
):
    """Predict probabilities and labels using raw-space logistic params at a threshold."""
    Xsel = X[feature_order].values.astype(float)
    w = np.array(weights_raw, dtype=float)
    z = Xsel.dot(w) + float(intercept_raw)
    p = 1.0 / (1.0 + np.exp(-z))
    yhat = (p >= float(threshold)).astype(int)
    return p, yhat


def export_lr_coefficients_csv(path: Path, feature_names: List[str], weights_raw: List[float], intercept_raw: float) -> None:
    """Export LR coefficients (raw space) to CSV with an Intercept row."""
    rows = [(f, w) for f, w in zip(feature_names, weights_raw)] + [("Intercept", intercept_raw)]
    df = pd.DataFrame(rows, columns=["feature", "weight"])
    df.to_csv(path, index=False)


def load_lr_coefficients_csv(path: Path) -> Tuple[List[str], List[float], float]:
    """Load LR coefficients previously exported via export_lr_coefficients_csv.

    Returns (feature_order, weights_raw, intercept_raw).
    """
    df = pd.read_csv(path)
    if not {"feature", "weight"}.issubset(df.columns):
        raise ValueError(f"Unexpected columns in {path.name}: {list(df.columns)}")
    # Intercept row is labeled 'Intercept'
    intercept_rows = df[df["feature"] == "Intercept"]["weight"].tolist()
    intercept = float(intercept_rows[0]) if intercept_rows else 0.0
    coeffs = df[df["feature"] != "Intercept"]
    features = coeffs["feature"].astype(str).tolist()
    weights = coeffs["weight"].astype(float).tolist()
    return features, weights, intercept


def extract_decision_tree_rules(model: DecisionTreeClassifier, feature_names: List[str]):
    """Extract a JSON-like nested dict of decision rules from a trained DecisionTreeClassifier."""
    tree = model.tree_
    features = tree.feature
    thresholds = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    values = tree.value  # shape (n_nodes, 1, n_classes)

    def node_to_dict(node_id: int):
        left = children_left[node_id]
        right = children_right[node_id]
        if left == -1 and right == -1:
            counts = values[node_id][0]
            total = float(counts.sum())
            prob_pos = float(counts[1] / total) if total > 0 else 0.0
            pred_class = int(np.argmax(counts))
            return {
                "type": "leaf",
                "counts": counts.tolist(),
                "prob_pos": prob_pos,
                "pred_class": pred_class,
            }
        feat_idx = features[node_id]
        return {
            "type": "node",
            "feature": feature_names[feat_idx],
            "threshold": float(thresholds[node_id]),
            "left": node_to_dict(left),
            "right": node_to_dict(right),
        }

    return node_to_dict(0)


def load_dt_rules_json(path: Path):
    """Load decision tree rules from a JSON file exported by inference.

    Supports two shapes:
    - {"probability_threshold": float, "tree": {...}}
    - {...tree dict directly...}
    Returns the tree dict.
    """
    obj = json.loads(path.read_text())
    if isinstance(obj, dict) and "tree" in obj:
        return obj["tree"], obj.get("probability_threshold")
    return obj, None


def render_decision_rules_text(tree_dict, indent: int = 0) -> List[str]:
    """Render human-readable decision rules from a tree dict."""
    lines: List[str] = []

    def recurse(node, prefix: str = ""):
        if node["type"] == "leaf":
            pred_thresh_str = (
                f" pred_thresh={node['pred_thresh']}" if "pred_thresh" in node else ""
            )
            lines.append(
                f"{prefix}=> leaf: pred_class={node['pred_class']} prob_pos={node['prob_pos']:.3f}{pred_thresh_str} counts={node['counts']}"
            )
            return
        # Left branch: feature <= threshold
        lines.append(f"{prefix}if {node['feature']} <= {node['threshold']:.6f}:")
        recurse(node["left"], prefix + "  ")
        # Right branch
        lines.append(f"{prefix}else:  # {node['feature']} > {node['threshold']:.6f}")
        recurse(node["right"], prefix + "  ")

    recurse(tree_dict)
    return lines


def predict_with_dt_rules(X: pd.DataFrame, feature_order: List[str], tree_dict) -> Tuple[np.ndarray, np.ndarray]:
    """Predict using extracted decision tree rules (no sklearn object). Returns (prob_pos, yhat)."""
    def predict_row(row):
        node = tree_dict
        while node["type"] != "leaf":
            feat = node["feature"]
            thr = node["threshold"]
            val = float(row[feat])
            node = node["left"] if val <= thr else node["right"]
        return node["prob_pos"], node["pred_class"]

    probs = []
    preds = []
    for _, row in X.iterrows():
        p, y = predict_row(row)
        probs.append(p)
        preds.append(y)
    return np.array(probs, dtype=float), np.array(preds, dtype=int)


def mark_dt_threshold_predictions(tree_dict, threshold: float):
    """Recursively annotate each leaf with a 'pred_thresh' based on prob_pos and the threshold."""
    def recurse(node):
        if node["type"] == "leaf":
            node["pred_thresh"] = int(float(node.get("prob_pos", 0.0)) >= float(threshold))
            return
        recurse(node["left"])
        recurse(node["right"])

    recurse(tree_dict)
    return tree_dict


def export_confusion_matrix_csv(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Compute and export confusion matrix with labeled rows/cols."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    df.to_csv(path)


def compute_performance_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series | None = None,
) -> Dict[str, float]:
    """Compute a set of binary classification metrics for a single evaluation set.

    Returns a dict with counts and the same core metrics used in training summary:
    - recall, precision, roc_auc (if y_proba is provided), balanced_accuracy,
      specificity, J-statistic, F2 score, LR+, LR-
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n = int(tn + fp + fn + tp)

    # Basic rates
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    bac = balanced_accuracy_score(y_true, y_pred)
    spec = float(specificity_score(y_true, y_pred))
    j_stat = float(recall + spec - 1)
    denom = (4 * precision + recall)
    f2 = float((5 * precision * recall) / denom) if denom > 0 else 0.0

    # LR+/LR- with divide-by-zero handling
    with np.errstate(divide="ignore", invalid="ignore"):
        lr_plus = float(recall / (1 - spec)) if (1 - spec) != 0 else float("inf")
        lr_minus = float((1 - recall) / spec) if spec != 0 else float("inf")

    # ROC AUC only if probabilities provided and y_true contains both classes
    roc_auc = np.nan
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            roc_auc = float(roc_auc_score(y_true, np.asarray(y_proba, dtype=float)))
        except Exception:
            roc_auc = np.nan

    # Accuracy from confusion matrix
    accuracy = float((tp + tn) / n) if n > 0 else 0.0

    return {
        "n": float(n),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "accuracy": accuracy,
        "recall": float(recall),
        "precision": float(precision),
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else np.nan,
        "balanced_accuracy": float(bac),
        "specificity": float(spec),
        "j_stat": float(j_stat),
        "f2": float(f2),
        "lr_plus": float(lr_plus),
        "lr_minus": float(lr_minus),
    }


# --------------------------------------------------------------------------------------
# Validation utilities
# --------------------------------------------------------------------------------------


def predict_with_lr_pipeline(pipeline: Pipeline, X: pd.DataFrame, threshold: float = 0.3):
    proba = pipeline.predict_proba(X.values)[:, 1]
    preds = (proba >= float(threshold)).astype(int)
    return proba, preds


def predict_with_dt_model(model: DecisionTreeClassifier, X: pd.DataFrame, threshold: float = 0.3):
    proba = model.predict_proba(X.values)[:, 1]
    preds = (proba >= float(threshold)).astype(int)
    return proba, preds


def compare_predictions(
    probs_a: np.ndarray,
    preds_a: np.ndarray,
    probs_b: np.ndarray,
    preds_b: np.ndarray,
    *,
    prob_tol: float = 1e-9,
):
    """Compare two sets of predictions; return summary dict."""
    preds_equal = np.array_equal(preds_a.astype(int), preds_b.astype(int))
    prob_diff = np.abs(np.array(probs_a, dtype=float) - np.array(probs_b, dtype=float))
    prob_all_close = np.all(prob_diff <= prob_tol)
    mismatches = int(np.sum(preds_a.astype(int) != preds_b.astype(int)))
    return {
        "preds_equal": bool(preds_equal),
        "mismatch_count": mismatches,
        "prob_all_close": bool(prob_all_close),
        "prob_max_abs_diff": float(np.max(prob_diff)) if prob_diff.size else 0.0,
        "prob_mean_abs_diff": float(np.mean(prob_diff)) if prob_diff.size else 0.0,
        "count": int(len(preds_a)),
        "prob_tol": float(prob_tol),
    }

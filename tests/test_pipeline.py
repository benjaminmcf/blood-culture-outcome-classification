"""
Unit tests for the blood culture outcome classification pipeline.

Run with: pytest tests/ -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier

# Import functions to test (path is set up in conftest.py)
from training_utils import (
    compute_performance_metrics,
    extract_lr_raw_params_from_pipeline,
    predict_lr_with_raw_params,
    extract_decision_tree_rules,
    predict_with_dt_rules,
    mark_dt_threshold_predictions,
    load_lr_coefficients_csv,
    export_lr_coefficients_csv,
    render_decision_rules_text,
    compare_predictions,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Create a small synthetic dataset for testing."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "feature_a": np.random.randn(n),
        "feature_b": np.random.randn(n) * 2 + 1,
        "feature_c": np.random.randn(n) * 0.5,
    })
    # Create labels with some correlation to features
    z = 0.5 * X["feature_a"] - 0.3 * X["feature_b"] + 0.2 * X["feature_c"]
    y = (z + np.random.randn(n) * 0.5 > 0).astype(int)
    return X, pd.Series(y)


@pytest.fixture
def trained_lr_pipeline(sample_data):
    """Train a logistic regression pipeline on sample data."""
    X, y = sample_data
    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
    ])
    pipeline.fit(X.values, y.values)
    return pipeline, list(X.columns)


@pytest.fixture
def trained_dt_model(sample_data):
    """Train a decision tree on sample data."""
    X, y = sample_data
    model = DecisionTreeClassifier(random_state=42, max_depth=3)
    model.fit(X.values, y.values)
    return model, list(X.columns)


# -----------------------------------------------------------------------------
# Test: Performance Metrics
# -----------------------------------------------------------------------------

class TestPerformanceMetrics:
    """Tests for compute_performance_metrics function."""

    def test_perfect_predictions(self):
        """Perfect predictions should give recall=1, specificity=1."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.95])
        
        metrics = compute_performance_metrics(y_true, y_pred, y_proba)
        
        assert metrics["recall"] == 1.0
        assert metrics["specificity"] == 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["balanced_accuracy"] == 1.0
        assert metrics["tp"] == 3
        assert metrics["tn"] == 2
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0

    def test_all_wrong_predictions(self):
        """All wrong predictions should give recall=0, specificity=0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        
        metrics = compute_performance_metrics(y_true, y_pred)
        
        assert metrics["recall"] == 0.0
        assert metrics["specificity"] == 0.0
        assert metrics["tp"] == 0
        assert metrics["fn"] == 2

    def test_roc_auc_requires_proba(self):
        """ROC AUC should be NaN without probabilities."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        metrics = compute_performance_metrics(y_true, y_pred)
        
        assert np.isnan(metrics["roc_auc"])

    def test_j_statistic_calculation(self):
        """J statistic = recall + specificity - 1."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])  # 1 FP, 2 TP, 2 TN
        
        metrics = compute_performance_metrics(y_true, y_pred)
        
        expected_recall = 2 / 2  # TP / (TP + FN) = 2/2 = 1.0
        expected_spec = 2 / 3    # TN / (TN + FP) = 2/3
        expected_j = expected_recall + expected_spec - 1
        
        assert abs(metrics["j_stat"] - expected_j) < 1e-9


# -----------------------------------------------------------------------------
# Test: Logistic Regression Coefficient Extraction
# -----------------------------------------------------------------------------

class TestLRExtraction:
    """Tests for LR coefficient extraction and prediction."""

    def test_raw_params_reproduce_pipeline(self, sample_data, trained_lr_pipeline):
        """Extracted raw-space params should reproduce pipeline predictions exactly."""
        X, y = sample_data
        pipeline, features = trained_lr_pipeline
        
        # Extract raw params
        params = extract_lr_raw_params_from_pipeline(pipeline, features)
        
        # Predict with both methods
        proba_pipeline = pipeline.predict_proba(X.values)[:, 1]
        proba_raw, _ = predict_lr_with_raw_params(
            X, params["features"], params["weights_raw"], params["intercept_raw"], threshold=0.5
        )
        
        # Should match within floating point tolerance
        np.testing.assert_array_almost_equal(proba_pipeline, proba_raw, decimal=9)

    def test_threshold_affects_predictions(self, sample_data, trained_lr_pipeline):
        """Different thresholds should produce different binary predictions."""
        X, _ = sample_data
        pipeline, features = trained_lr_pipeline
        params = extract_lr_raw_params_from_pipeline(pipeline, features)
        
        _, preds_low = predict_lr_with_raw_params(
            X, params["features"], params["weights_raw"], params["intercept_raw"], threshold=0.3
        )
        _, preds_high = predict_lr_with_raw_params(
            X, params["features"], params["weights_raw"], params["intercept_raw"], threshold=0.7
        )
        
        # Lower threshold should produce >= number of positive predictions
        assert preds_low.sum() >= preds_high.sum()

    def test_export_and_load_coefficients(self, trained_lr_pipeline):
        """Exported coefficients should load correctly."""
        pipeline, features = trained_lr_pipeline
        params = extract_lr_raw_params_from_pipeline(pipeline, features)
        
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        
        try:
            export_lr_coefficients_csv(
                path, params["features"], params["weights_raw"], params["intercept_raw"]
            )
            loaded_features, loaded_weights, loaded_intercept = load_lr_coefficients_csv(path)
            
            assert loaded_features == params["features"]
            np.testing.assert_array_almost_equal(loaded_weights, params["weights_raw"])
            assert abs(loaded_intercept - params["intercept_raw"]) < 1e-9
        finally:
            path.unlink()


# -----------------------------------------------------------------------------
# Test: Decision Tree Rule Extraction
# -----------------------------------------------------------------------------

class TestDTExtraction:
    """Tests for decision tree rule extraction and prediction."""

    def test_rules_reproduce_model(self, sample_data, trained_dt_model):
        """Extracted rules should reproduce model predictions exactly."""
        X, _ = sample_data
        model, features = trained_dt_model
        
        # Extract rules
        tree_dict = extract_decision_tree_rules(model, features)
        
        # Predict with both methods
        proba_model = model.predict_proba(X.values)[:, 1]
        proba_rules, _ = predict_with_dt_rules(X, features, tree_dict)
        
        # Should match exactly
        np.testing.assert_array_almost_equal(proba_model, proba_rules, decimal=9)

    def test_tree_structure_is_valid(self, trained_dt_model):
        """Extracted tree should have valid structure."""
        model, features = trained_dt_model
        tree_dict = extract_decision_tree_rules(model, features)
        
        def validate_node(node):
            assert "type" in node
            if node["type"] == "leaf":
                assert "prob_pos" in node
                assert "pred_class" in node
                assert "counts" in node
                assert 0 <= node["prob_pos"] <= 1
            else:
                assert node["type"] == "node"
                assert "feature" in node
                assert "threshold" in node
                assert "left" in node
                assert "right" in node
                validate_node(node["left"])
                validate_node(node["right"])
        
        validate_node(tree_dict)

    def test_threshold_annotation(self, trained_dt_model):
        """mark_dt_threshold_predictions should annotate all leaves."""
        model, features = trained_dt_model
        tree_dict = extract_decision_tree_rules(model, features)
        tree_dict = mark_dt_threshold_predictions(tree_dict, threshold=0.5)
        
        def check_leaves(node):
            if node["type"] == "leaf":
                assert "pred_thresh" in node
                assert node["pred_thresh"] in [0, 1]
            else:
                check_leaves(node["left"])
                check_leaves(node["right"])
        
        check_leaves(tree_dict)

    def test_render_rules_text(self, trained_dt_model):
        """Rendered rules should be non-empty and contain expected keywords."""
        model, features = trained_dt_model
        tree_dict = extract_decision_tree_rules(model, features)
        lines = render_decision_rules_text(tree_dict)
        
        assert len(lines) > 0
        text = "\n".join(lines)
        assert "leaf" in text or "if" in text


# -----------------------------------------------------------------------------
# Test: Prediction Comparison
# -----------------------------------------------------------------------------

class TestComparePrections:
    """Tests for the compare_predictions utility."""

    def test_identical_predictions(self):
        """Identical predictions should report equality."""
        probs = np.array([0.1, 0.5, 0.9])
        preds = np.array([0, 1, 1])
        
        result = compare_predictions(probs, preds, probs, preds)
        
        assert result["preds_equal"] is True
        assert result["prob_all_close"] is True
        assert result["mismatch_count"] == 0
        assert result["prob_max_abs_diff"] == 0.0

    def test_different_predictions(self):
        """Different predictions should be detected."""
        probs_a = np.array([0.1, 0.5, 0.9])
        preds_a = np.array([0, 1, 1])
        probs_b = np.array([0.2, 0.4, 0.8])
        preds_b = np.array([0, 0, 1])
        
        result = compare_predictions(probs_a, preds_a, probs_b, preds_b)
        
        assert result["preds_equal"] is False
        assert result["mismatch_count"] == 1
        assert result["prob_max_abs_diff"] > 0


# -----------------------------------------------------------------------------
# Test: Edge Cases
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_sample(self, trained_lr_pipeline):
        """Pipeline should handle single-sample prediction."""
        pipeline, features = trained_lr_pipeline
        X = pd.DataFrame([[0.5, 1.0, -0.5]], columns=features)
        
        params = extract_lr_raw_params_from_pipeline(pipeline, features)
        proba, preds = predict_lr_with_raw_params(
            X, params["features"], params["weights_raw"], params["intercept_raw"], threshold=0.5
        )
        
        assert len(proba) == 1
        assert len(preds) == 1

    def test_all_same_class(self):
        """Metrics should handle single-class ground truth gracefully."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 1])
        
        metrics = compute_performance_metrics(y_true, y_pred)
        
        assert metrics["recall"] == 0.75  # 3/4 correct
        # ROC AUC should be NaN with single class
        assert np.isnan(metrics["roc_auc"])

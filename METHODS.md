# Methods

## Dataset Requirements

### Input Format

Both training and testing datasets are standard CSV files with a header row. Optional comment lines beginning with `#` are permitted and automatically stripped during loading. Each row represents a single patient episode.

### Target Variable

The target column is configurable via `config.json` (key: `TARGET_COLUMN`). The default is:

- **`isPOS`** — Binary label (1 = positive blood culture, 0 = negative blood culture)

This can be changed to any binary column name to adapt the pipeline to other classification problems (e.g., `isUTI`, `isSepsis`, `isResistant`).

### Feature Spaces

Feature spaces are configurable via `config.json` (key: `FEATURE_SPACES`). Each entry maps a name to a list of column names. The pipeline iterates over all defined feature spaces, training and evaluating models on each.

The default configuration defines two feature spaces:

#### CBC_DIFF (19 features)

Standard complete blood count with white blood cell differential parameters:

| Feature | Unit | Description |
|---------|------|-------------|
| `WBC(10^9/L)` | 10⁹/L | White blood cell count |
| `RBC(10^12/L)` | 10¹²/L | Red blood cell count |
| `HGB(g/L)` | g/L | Haemoglobin |
| `MCV(fL)` | fL | Mean corpuscular volume |
| `MCHC(g/L)` | g/L | Mean corpuscular haemoglobin concentration |
| `PLT(10^9/L)` | 10⁹/L | Platelet count |
| `RDW-CV(%)` | % | Red cell distribution width (coefficient of variation) |
| `NEUT#(10^9/L)` | 10⁹/L | Neutrophil absolute count |
| `LYMPH#(10^9/L)` | 10⁹/L | Lymphocyte absolute count |
| `MONO#(10^9/L)` | 10⁹/L | Monocyte absolute count |
| `EO#(10^9/L)` | 10⁹/L | Eosinophil absolute count |
| `BASO#(10^9/L)` | 10⁹/L | Basophil absolute count |
| `NEUT%(%)` | % | Neutrophil percentage |
| `LYMPH%(%)` | % | Lymphocyte percentage |
| `MONO%(%)` | % | Monocyte percentage |
| `EO%(%)` | % | Eosinophil percentage |
| `BASO%(%)` | % | Basophil percentage |
| `NLR` | ratio | Neutrophil-to-lymphocyte ratio |
| `MLR` | ratio | Monocyte-to-lymphocyte ratio |

#### CBC_DIFF_CPD (50 features)

The full CBC_DIFF feature set plus cell population data and interpretive flags from Sysmex XN-series analysers:

**CPD scatter/fluorescence parameters (18 features):**

| Feature | Description |
|---------|-------------|
| `[NE-SSC(ch)]`, `[NE-SFL(ch)]`, `[NE-FSC(ch)]` | Neutrophil side scatter, fluorescence, forward scatter |
| `[LY-X(ch)]`, `[LY-Y(ch)]`, `[LY-Z(ch)]` | Lymphocyte position parameters |
| `[MO-X(ch)]`, `[MO-Y(ch)]`, `[MO-Z(ch)]` | Monocyte position parameters |
| `[NE-WX]`, `[NE-WY]`, `[NE-WZ]` | Neutrophil population width parameters |
| `[LY-WX]`, `[LY-WY]`, `[LY-WZ]` | Lymphocyte population width parameters |
| `[MO-WX]`, `[MO-WY]`, `[MO-WZ]` | Monocyte population width parameters |

**Interpretive flags (13 features):**

| Feature | Description |
|---------|-------------|
| `IP ABN(WBC)WBC Abn Scattergram` | Abnormal WBC scattergram |
| `IP ABN(WBC)Neutropenia` | Neutropenia flag |
| `IP ABN(WBC)Neutrophilia` | Neutrophilia flag |
| `IP ABN(WBC)Lymphopenia` | Lymphopenia flag |
| `IP ABN(WBC)Lymphocytosis` | Lymphocytosis flag |
| `IP ABN(WBC)Leukocytopenia` | Leukocytopenia flag |
| `IP ABN(WBC)Leukocytosis` | Leukocytosis flag |
| `IP ABN(PLT)Thrombocytopenia` | Thrombocytopenia flag |
| `IP SUS(WBC)Blasts/Abn Lympho?` | Suspect blasts or abnormal lymphocytes |
| `IP SUS(WBC)Blasts?` | Suspect blasts |
| `IP SUS(WBC)Abn Lympho?` | Suspect abnormal lymphocytes |
| `IP SUS(WBC)Left Shift?` | Suspect left shift |
| `IP SUS(WBC)Atypical Lympho?` | Suspect atypical lymphocytes |

## Classification Models

Four classifier types are implemented, all available via scikit-learn or XGBoost:

### Decision Tree (DT)

A single `DecisionTreeClassifier` with balanced class weights and a maximum depth of 3. Decision rules are exported in both human-readable text and structured JSON format, enabling deployment without a Python runtime.

### Random Forest (RF)

A `RandomForestClassifier` ensemble of 100 decision trees with balanced class weights, a maximum depth of 3 per tree, and out-of-bag scoring enabled. The model is deployed via the saved scikit-learn pipeline object.

### XGBoost (XG)

An `XGBClassifier` with gradient-boosted trees. The `scale_pos_weight` parameter is set to the class imbalance ratio (n_negative / n_positive) to handle the typically imbalanced blood culture dataset. 100 estimators are used with a learning rate of 0.01 and a maximum tree depth of 3.

### Logistic Regression (LR)

A `LogisticRegression` model with `RobustScaler` preprocessing, wrapped in a scikit-learn `Pipeline`. `RobustScaler` centres features by their median and scales by their interquartile range (IQR), which is more robust to outliers than standardisation by mean and standard deviation. After training, the pipeline's scaler and classifier coefficients are collapsed into raw-space coefficients and an adjusted intercept, enabling deployment via a simple linear equation:

```
z = Σ(w_i × x_i) + b
p = 1 / (1 + exp(-z))
ŷ = 1 if p ≥ threshold else 0
```

where `w_i = w_lr / s_scaler` and `b = b_lr − Σ(w_lr × c_scaler / s_scaler)`, with `c_scaler` being the median (centre) and `s_scaler` the IQR (scale) computed by `RobustScaler`.

This approach allows inference using only arithmetic operations — suitable for spreadsheet-based or LIMS-integrated deployment.

## Class Imbalance Handling

Blood culture datasets are typically imbalanced, with negative cultures substantially outnumbering positive ones (positive rates often 5–15%). The pipeline addresses this through:

1. **Balanced class weights:** Computed as `n_samples / (n_classes × n_samples_per_class)` and passed to all classifiers that support the `class_weight` parameter
2. **`scale_pos_weight` for XGBoost:** Explicitly set to `n_negative / n_positive`
3. **Probability thresholding:** The default threshold of 0.3 (rather than 0.5) compensates for the prior probability skew during inference

## Feature Selection

### Boruta Algorithm

The pipeline uses the Boruta all-relevant feature selection method (Kursa & Rudnicki, 2010). Boruta is a wrapper around Random Forest that iteratively compares real feature importance against shadow (permuted) features. A feature is confirmed as relevant only if its importance consistently exceeds that of the best shadow feature across multiple iterations.

Implementation details:
- Base estimator: `RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=3)`
- Maximum 100 iterations
- Significance level α = 0.05
- Features confirmed or tentatively accepted by Boruta are retained

### Recursive Feature Elimination (RFE)

The pipeline also supports recursive feature elimination (RFE), which iteratively removes the least important features and rebuilds the model until a specified number of features remains. The implementation targets a compact subset of 5 features (or fewer if the original feature space is smaller). For pipeline-wrapped models (e.g., Logistic Regression with `RobustScaler`), a `RandomForestClassifier` is used as the RFE base estimator since RFE requires direct access to `feature_importances_` or `coef_` attributes.

### No Feature Selection

When run with `--fs none`, all features in the selected feature space are used without filtering. This provides a baseline comparison against the Boruta-selected or RFE-selected subsets.

## Cross-Validation Strategy

### Nested Cross-Validation

The pipeline implements nested (double) cross-validation to prevent optimistic bias from performing feature selection on the same data used for evaluation. This is critical when feature selection methods like Boruta are applied, as selecting features on the full dataset before CV leads to data leakage.

```
┌─────────────────────────────────────────────────┐
│ Outer Loop: k-fold Stratified CV                │
│                                                 │
│  For each fold i:                               │
│    ┌───────────────────────────────────────────┐ │
│    │ Train set (k-1 folds)                     │ │
│    │   1. Feature selection (Boruta)           │ │
│    │   2. Train model on selected features     │ │
│    ├───────────────────────────────────────────┤ │
│    │ Test set (1 fold)                         │ │
│    │   3. Predict using selected features only │ │
│    │   4. Record metrics                       │ │
│    └───────────────────────────────────────────┘ │
│                                                 │
│  Aggregate predictions across all folds         │
│  Compute mean ± SD for each metric              │
└─────────────────────────────────────────────────┘
```

The number of folds adapts dynamically to the dataset size. The target is 10-fold CV, but this is reduced automatically when the minority class has fewer than 10 samples per fold (enforced via `StratifiedKFold` constraints).

### Final Model Training

After nested CV provides unbiased performance estimates, a final model is trained on the **entire** training dataset:

1. Feature selection is performed once on the full dataset (global selection)
2. The model is trained on all samples using the globally-selected features
3. This model and its feature list are saved as deployment artifacts

Note: The nested CV metrics estimate the generalisation performance of the *procedure* (feature selection + training), not of this specific final model instance.

## Performance Metrics

The following metrics are computed during both cross-validation and inference:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Balanced Accuracy** | (Recall + Specificity) / 2 | Overall performance adjusted for class imbalance |
| **Recall (Sensitivity)** | TP / (TP + FN) | Proportion of true positives correctly identified |
| **Specificity** | TN / (TN + FP) | Proportion of true negatives correctly identified |
| **Precision (PPV)** | TP / (TP + FP) | Proportion of positive predictions that are correct |
| **ROC-AUC** | Area under ROC curve | Discrimination ability across all thresholds |
| **Youden's J** | Recall + Specificity − 1 | Net informedness |
| **F2 Score** | (5 × Precision × Recall) / (4 × Precision + Recall) | F-beta with β=2, weighting recall higher |
| **LR+** | Recall / (1 − Specificity) | Positive likelihood ratio |
| **LR−** | (1 − Recall) / Specificity | Negative likelihood ratio |

## Reporting

### Training Report (`results/training_report.html`)

An HTML report generated after training containing:
- Dataset statistics (total samples, class distribution)
- Run configuration (random seeds)
- Cross-validation results table ranked by balanced accuracy
- ROC curves per model (aggregated from nested CV out-of-fold predictions)
- Confusion matrices per model

### Inference Report (`predictions/inference_report.html`)

An HTML report generated after inference containing:
- Input dataset summary
- Performance metrics table (balanced accuracy, recall, specificity, ROC-AUC, precision)
- Per-model prediction counts and positive rates
- ROC curves and confusion matrices (when ground truth available)
- Validation results (when `--validate` flag used)

## Artifact Export and Deployment

### Logistic Regression Coefficients

For each LR model, the pipeline collapses the `RobustScaler → LogisticRegression` pipeline into a single set of raw-space coefficients:

```
w_raw = w_lr / s_scaler
b_raw = b_lr − Σ(w_lr × c_scaler / s_scaler)
```

where `c_scaler` is the median (centre) and `s_scaler` is the IQR (scale) computed by `RobustScaler`.

These are exported to `exports/lr_coeffs_{config}.csv` and can be applied in any environment that supports basic arithmetic (e.g., Excel, SQL, LIMS integration).

### Decision Tree Rules

DT models are exported as:
- **Human-readable text** (`exports/dt_rules_*.txt`) — Indented if-else rules
- **Structured JSON** (`exports/dt_rules_*.json`) — Machine-parseable rule tree

Both formats allow deployment without a Python runtime or trained model object.

### Model Metadata

Each trained model has an accompanying JSON metadata file (`models/*.json`) containing:
- Model type and configuration
- Feature space and selection method
- Selected feature list
- Default probability threshold
- Paths to model and feature list files

## Synthetic Data Generation

### Distribution-Fitted Generator

A script (`python_scripts/generate_synthetic_data.py`) is provided for users with access to real data. This generator fits class-conditional multivariate normal distributions in log-space to actual training data, preserving inter-variable correlations. It supports custom sample sizes, class ratios, and output directories.

## Pipeline Utilities

### Clean Script

The `clean.py` script removes all pipeline output directories (`datasets/`, `models/`, `features/`, `results/`, `predictions/`, `exports/`), enabling a fresh end-to-end test:

```bash
python clean.py              # Remove all outputs
bcoc-train                   # Train models
bcoc-infer                   # Run inference
```

## Software Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| scikit-learn | ≥1.7.2 | Classification, preprocessing, metrics |
| XGBoost | ≥3.0.5 | Gradient-boosted tree classifier |
| Boruta | ≥0.4.3 | All-relevant feature selection |
| pandas | ≥2.3.3 | Data manipulation |
| numpy | ≥2.3.4 | Numerical operations |
| matplotlib | ≥3.10.7 | ROC curves and confusion matrix plots |
| imbalanced-learn | ≥0.14.0 | Resampling utilities |

## Pipeline Configuration

The pipeline is designed to be problem-agnostic. All domain-specific configuration is externalised to `config.json`:

```json
{
    "RANDOM_STATE": 15,
    "RANDOM_STATE_BORUTA": 42,
    "TARGET_COLUMN": "isPOS",
    "FEATURE_SPACES": {
        "FEATURE_SET_A": ["feature_1", "feature_2", "..."],
        "FEATURE_SET_B": ["feature_1", "...", "feature_n"]
    }
}
```

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `RANDOM_STATE` | Yes | — | Seed for model training and cross-validation |
| `RANDOM_STATE_BORUTA` | Yes | — | Seed for Boruta feature selection |
| `TARGET_COLUMN` | No | `isPOS` | Name of the binary outcome column (0/1) |
| `FEATURE_SPACES` | No | Built-in CBC_DIFF / CBC_DIFF_CPD | Dict mapping feature space names to feature column lists |

To adapt the pipeline to a different clinical problem:

1. Set `TARGET_COLUMN` to the binary outcome column in your dataset
2. Define `FEATURE_SPACES` with one or more named sets of predictor variables
3. Ensure input CSVs contain all specified columns
4. Run `bcoc-train` and `bcoc-infer` — no code changes required

## Example Notebook

The `notebooks/Using_Exported_Models_for_Inference.ipynb` notebook demonstrates how to:

1. Load exported LR coefficients and compute predictions using raw-space arithmetic
2. Load exported DT rules (JSON) and traverse the tree for predictions
3. Validate exported artefact predictions against the original sklearn model objects
4. Compute performance metrics and export confusion matrices

The notebook reads pipeline configuration from `config.json` and handles comment lines in synthetic CSV files automatically.

## References

1. Kursa, M. B. & Rudnicki, W. R. (2010). Feature Selection with the Boruta Package. *Journal of Statistical Software*, 36(11), 1–13.
2. McFadden, B. et al. (2023). Machine learning pipeline for blood culture outcome prediction using Sysmex XN-2000 blood sample results in Western Australia. *BMC Infectious Diseases*, 23, 561.
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
4. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proc. KDD*, 785–794.

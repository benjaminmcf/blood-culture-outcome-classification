# blood-culture-outcome-classification
Code associated with research papers relating to blood culture outcome classification

- Machine learning pipeline for blood culture outcome prediction using Sysmex XN2000- blood sample results in Western Australia https://bmcinfectdis.biomedcentral.com/articles/10.1186/s12879-023-08535-y

- Evaluation of machine learning pipeline for blood culture outcome prediction on prospectively collected emergency department data (Under peer review in the Journal of Medical Microbiology)

- Another paper to be confirmed

## Training pipeline

The model training script is in `python_scripts/training.py` and uses helpers from `python_scripts/training_utils.py`. It reads configuration from `config.json`, loads the dataset from `datasets/training_data.csv`, performs feature selection (Boruta), cross-validation, and writes:

- Selected feature lists to `features/` (one file per model/feature-space)
- Trained models to `models/` (pickle `.sav` files)
- Cross-validation summary to `results/results_cross_validation.csv`

### Run

You need Python 3.11+ and the dependencies defined in `pyproject.toml`.

If you use `uv` (recommended):

```bash
uv sync
uv run python python_scripts/training.py
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python python_scripts/training.py
```

Update `config.json` to set seeds: `RANDOM_STATE` and `RANDOM_STATE_BORUTA`.

## Inference

The inference script is `python_scripts/inference.py`.

- Logistic Regression: extracts coefficients from the trained pipeline and applies them directly at a probability threshold of 0.3. Coefficients exported to `exports/lr_coeffs_*.csv`.
- Decision Tree: extracts human-readable rules and applies them (no model object). Rules exported to `exports/dt_rules_*.txt` and `.json`.
- Other models (RF, XG): use the saved model objects for inference.

Predictions are written to `predictions/preds_{model}_{features}_{weight}_{fsm}.csv` with columns: `model`, `prob_pos`, `yhat`.

If the input data contains a ground-truth column `isPOS`, confusion matrices are exported to `predictions/cm_{model}_{features}_{weight}_{fsm}.csv` with labeled rows/columns (`true_0`, `true_1` vs `pred_0`, `pred_1`).

By default, the script uses `datasets/training_data.csv` as input and drops `isPOS` if present. Replace with your test dataset as needed.

Run:

```bash
uv run python python_scripts/inference.py --threshold 0.3 --validate
# or
python python_scripts/inference.py --threshold 0.3 --validate
```

Flags:
- `--threshold <float>`: Probability threshold for binary predictions (applies to LR, DT, RF, XG). Default 0.3.
- `--validate`: For LR and DT, compare extracted-method predictions against the original model objects and write JSON reports under `exports/validation/`.

### Using exported LR coefficients

The LR pipelines are exported to raw-space coefficients so you can predict without the model object. Given a CSV `exports/lr_coeffs_{features}_{weight}_{fsm}.csv` with `feature,weight` rows and an `Intercept` row:

1) Build z = sum_i (weight_i * x_i) + Intercept using the same feature order as in the features file.
2) Convert to probability with p = 1 / (1 + exp(-z)).
3) Apply the same threshold: yhat = 1 if p >= threshold else 0.

These predictions are validated against the original pipeline when `--validate` is used.

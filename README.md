# A Hybrid Tree-Neural Framework for Pre-Transplant Risk Stratification of Acute Kidney Allograft Rejection Incorporating Unique High-Resolution Molecular Histocompatibility Dataset

Accompanying code for the NeurIPS submission.

The method combines XGBoost (tabular features and leaf indices), a PyTorch DNN, VAE + GMM oversampling for class imbalance, and focal loss. Supported outcomes include antibody-mediated rejection (ABMR), T-cell–mediated rejection (TCMR), pooled rejection, and multi-class Banff grades; see `train.py` for details.

---

## Contents

| Item | Description |
|------|-------------|
| `train.py` | Train and evaluate models (Optuna tuning, CV, metrics and plots). |
| `predict.py` | Run inference on new data using a saved model directory. |
| `evaluate_figures.py` | Extended evaluation: ROC/PR curves, confusion matrices, bootstrap metrics, decision curves, summary spreadsheet. |
| `sample_dataset.csv` | Example input table. |
| `ABMR_sample_output.csv` / `TCMR_sample_output.csv` | Example risk-score outputs. |

---

## Dependencies

- **OS:** Tested on macOS and Windows.
- **Python packages:** See imports in each script. Core stack includes PyTorch, XGBoost, scikit-learn, pandas, NumPy, Optuna, `category_encoders`, and joblib. `evaluate_figures.py` additionally uses seaborn, matplotlib, `dcurves`, and writes Excel via `pandas` (requires a suitable engine such as `openpyxl`).
- **Typical setup time:** Under ~30 minutes (environment + installs).

---

## Sample data

- `sample_dataset.csv` — example features/outcomes for running the pipeline.
- `ABMR_sample_output.csv` / `TCMR_sample_output.csv` — example predicted risk scores (probability of the positive class), in the range 0 (low risk) to 1 (high risk).

---

## Usage

**Expected inference runtime:** on the order of seconds for small cohorts (e.g. &lt;30 s for typical sample sizes).

Optional flags are supported by the scripts (for example `--output` and `--trials` on `train.py`, `--output_dir` and `--type` on `predict.py`); the commands below match the original README invocations, using defaults where arguments are omitted.

### Train and develop models

`--train_set` and `--test_set` each accept one or more CSV paths; paired lists must be the same length. By default, runs write under `results/` with subfolders derived from the train file path and outcome (see `train.py`).

#### ABMR prediction

```bash
python train.py \
  --outcome_type outcome_abmr \
  --train_set ./train.csv \
  --test_set ./test_data.csv
```

#### TCMR prediction

```bash
python train.py \
  --outcome_type outcome_tcmr \
  --train_set ./train.csv \
  --test_set ./test_data.csv
```

Other valid `--outcome_type` values: `outcome_rej`, `outcome_banff` (see `train.py` docstring).

### Additional evaluation

#### ABMR prediction

```bash
python evaluate_figures.py \
  --outcome_col outcome_abmr \
  --model_dir ./model_directory \
  --test_paths ./test_data.csv
```

#### TCMR prediction

```bash
python evaluate_figures.py \
  --outcome_col outcome_tcmr \
  --model_dir ./model_directory \
  --test_paths ./test_data.csv
```

### Predict risk scores

#### ABMR prediction

```bash
python predict.py \
  --outcome_col outcome_abmr \
  --model_dir ./model_directory \
  --test_paths ./sample_data.csv
```

#### TCMR prediction

```bash
python predict.py \
  --outcome_col outcome_tcmr \
  --model_dir ./model_directory \
  --test_paths ./sample_data.csv
```

With default options, risk scores are written to `{type}_{outcome_col}_risk_scores.csv` in the current directory (`type` defaults to `unknown` unless you pass `--type` to match the tag used when training).

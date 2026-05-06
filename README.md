# A Hybrid Tree-Neural Framework for Pre-Transplant Risk Stratification of Acute Kidney Allograft Rejection Incorporating Unique High-Resolution Molecular Histocompatibility Dataset

Accompanying code for the NeurIPS submission.

The method combines XGBoost (tabular features and leaf indices), a PyTorch DNN, VAE + GMM oversampling for class imbalance, and focal loss. Supported outcomes include antibody-mediated rejection (ABMR), T-cell–mediated rejection (TCMR), pooled rejection, and multi-class Banff grades; see `KLEAR_development.py` for details.

---

## Contents

| Item | Description |
|------|-------------|
| `KLEAR_development.py` | Train and evaluate models (Optuna tuning, CV, metrics and plots). |
| `KLEAR_run.py` | Run inference on new data using a saved model directory. |
| `KLEAR_evaluate_figures.py` | Extended evaluation: ROC/PR curves, confusion matrices, bootstrap metrics, decision curves, summary spreadsheet. |
| `KLEAR_sample_dataset.csv` | Example input table. |
| `KLEAR_ABMR_sample_output.csv` / `KLEAR_TCMR_sample_output.csv` | Example risk-score outputs. |

---

## Dependencies

- **OS:** Tested on macOS and Windows.
- **Python packages:** See imports in each script. Core stack includes PyTorch, XGBoost, scikit-learn, pandas, NumPy, Optuna, `category_encoders`, and joblib. `KLEAR_evaluate_figures.py` additionally uses seaborn, matplotlib, `dcurves`, and writes Excel via `pandas` (requires a suitable engine such as `openpyxl`).
- **Typical setup time:** Under ~30 minutes (environment + installs).

---

## Sample data

- `KLEAR_sample_dataset.csv` — example features/outcomes for running the pipeline.
- `KLEAR_ABMR_sample_output.csv` / `KLEAR_TCMR_sample_output.csv` — example predicted risk scores (probability of the positive class), in the range 0 (low risk) to 1 (high risk).

---

## Usage

**Expected inference runtime:** on the order of seconds for small cohorts (e.g. &lt;30 s for typical sample sizes).

Optional flags are supported by the scripts (for example `--output` and `--trials` on `KLEAR_development.py`, and `--output_dir` and `--type` on `KLEAR_run.py`); the commands below match the original README invocations, using defaults where arguments are omitted.

### Train and develop models

`--train_set` and `--test_set` each accept one or more CSV paths; paired lists must be the same length. By default, runs write under `results/` with subfolders derived from the train file path and outcome (see `KLEAR_development.py`).

#### ABMR prediction

```bash
python KLEAR_development.py \
  --outcome_type outcome_abmr \
  --train_set ./train.csv \
  --test_set ./test_data.csv
```

#### TCMR prediction

```bash
python KLEAR_development.py \
  --outcome_type outcome_tcmr \
  --train_set ./train.csv \
  --test_set ./test_data.csv
```


### Additional evaluation

#### ABMR prediction

```bash
python KLEAR_evaluate_figures.py \
  --outcome_col outcome_abmr \
  --model_dir ./model_directory \
  --test_paths ./KLEAR_sample_dataset.csv
```

#### TCMR prediction

```bash
python KLEAR_evaluate_figures.py \
  --outcome_col outcome_tcmr \
  --model_dir ./model_directory \
  --test_paths ./KLEAR_sample_dataset.csv
```

### Predict risk scores

#### ABMR prediction

```bash
python KLEAR_run.py \
  --outcome_col outcome_abmr \
  --model_dir ./model_directory \
  --test_paths ./KLEAR_sample_dataset.csv
```

#### TCMR prediction

```bash
python KLEAR_run.py \
  --outcome_col outcome_tcmr \
  --model_dir ./model_directory \
  --test_paths ./KLEAR_sample_dataset.csv
```

With default options, risk scores are written to `{type}_{outcome_col}_risk_scores.csv` in the current directory (`type` defaults to `unknown` unless you pass `--type` to match the tag used when training).

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
| `LLMs/openrouter_runs.py` | Run or resume the reported OpenRouter benchmarks. |
| `LLMs/clinicalbert_runs.py` | Train and evaluate the reported ClinicalBERT models. |

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

---

## OpenRouter and ClinicalBERT benchmark reproduction

The scripts in `LLMs/` reproduce the OpenRouter and ClinicalBERT analyses. The
cohort data and trained checkpoints are not distributed with this repository.

### Installation

Install the OpenRouter dependencies:

```bash
python -m pip install numpy pandas requests scikit-learn
```

Install the additional ClinicalBERT dependencies:

```bash
python -m pip install torch tqdm transformers
```

Run either script with `--help` to see every available option.

### Input data

Provide input CSV files directly through the command line. For OpenRouter, use
`--eval-csv`:

```bash
python LLMs/openrouter_runs.py \
  --cohort geographic \
  --eval-csv ./test_data.csv \
  --validate-only
```

For ClinicalBERT, use `--train-csv` and `--test-csv`:

```bash
python LLMs/clinicalbert_runs.py \
  --dataset srtr_geographic \
  --train-csv ./train.csv \
  --test-csv ./test_data.csv \
  --validate-only
```

The selected `--cohort` or `--dataset` determines the corresponding feature
policy and outcome configuration. Always use `--validate-only` first to check
the input columns, labels, split, and feature types.

### OpenRouter runs

The script does not contain an API key. If `OPENROUTER_API_KEY` is not already
set, an interactive run securely prompts for it without displaying the entered
value. The key is used only to authorize requests and is not saved in the run
outputs. For non-interactive jobs, provide the key through the
`OPENROUTER_API_KEY` environment variable.

Run a small development check with one model:

```bash
python LLMs/openrouter_runs.py \
  --cohort geographic \
  --eval-csv ./test_data.csv \
  --model x-ai/grok-4.3 \
  --limit 10
```

Omitting `--model` runs every model used in the reported benchmark and may
incur substantial API charges. Use `--limit` only for development checks, not
for reported analyses. Rerunning the same command resumes incomplete results;
use `--overwrite` only when existing predictions should be replaced.

The output location is printed when the run starts and can be changed with
`--output-root`. Outputs include per-model predictions, metrics, bootstrap
confidence intervals, token counts, resolved model identifiers, dataset
hashes, run configuration, and a combined model comparison.

The default `--threshold-strategy evaluation-youden` reproduces the reported
spreadsheet analysis. For a new independently evaluated study, consider a
prespecified threshold:

```bash
--threshold-strategy fixed --fixed-threshold 0.5
```

OpenRouter requests transmit the selected input feature values to an external
service and its routed model providers. Use only appropriately de-identified
data and follow applicable institutional approvals and data-use agreements.

### ClinicalBERT runs

Validate the input data without loading BERT:

```bash
python LLMs/clinicalbert_runs.py \
  --dataset srtr_geographic \
  --train-csv ./train.csv \
  --test-csv ./test_data.csv \
  --validate-only
```

Train and evaluate a model:

```bash
python LLMs/clinicalbert_runs.py \
  --dataset srtr_geographic \
  --train-csv ./train.csv \
  --test-csv ./test_data.csv \
  --device auto
```

Supported dataset settings are `low_res`, `imputed`, `high_res`, `srtr`,
`srtr_prospective`, and `srtr_geographic`. Mayo outcomes can be selected with
`--outcome outcome_tcmr`, `--outcome outcome_abmr`, or
`--outcome outcome_rej`. The SRTR settings use the `All rejection` outcome.

Evaluate an existing checkpoint without retraining:

```bash
python LLMs/clinicalbert_runs.py \
  --dataset srtr_geographic \
  --train-csv ./train.csv \
  --test-csv ./test_data.csv \
  --output-dir ./model_directory \
  --evaluate-only
```

The output location is printed when the run starts and can be changed with
`--output-root` or `--output-dir`. Outputs include the checkpoint, saved
backbone and tokenizer, reproducible training and validation splits, dataset
hashes, run configuration, predictions, metrics, and bootstrap confidence
intervals.

The default `--threshold-source evaluation` reproduces the reported results.
For a new study, use `--threshold-source validation` or a prespecified fixed
threshold.

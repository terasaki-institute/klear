# ==============================================================================
# Kidney Transplant Rejection Prediction Pipeline
# Elastic-Net Logistic Regression Baseline + Optuna (5-Fold CV)
# ==============================================================================

"""
Binary / multiclass classification for kidney transplant rejection outcomes
using imputed, low-resolution, or high-resolution HLA typing datasets.

Outcomes for Modeling:
- outcome_tcmr  --> Binary (T-cell mediated rejection)
- outcome_abmr  --> Binary (antibody-mediated rejection)
- outcome_rej   --> Binary (any rejection)
- outcome_banff --> Multiclass (Banff grades)

Modeling Strategy:
- Pure sklearn LogisticRegression with penalty='elasticnet', solver='saga'
- No VAE, GMM oversampling, XGBoost, or leaf-index features
- Preprocessing: median impute + StandardScaler (numeric);
  TargetEncoder (high-cardinality cats); one-hot (low-cardinality cats)
- Optuna hyperparameter tuning with 5-fold stratified CV (maximize macro F1)
- Final retrain on full train set; evaluate on held-out test set + bootstrap CIs

Pipeline Flow:
1. Load train/test CSVs and preprocess (fit on train only)
2. Optuna 5-fold CV on elastic-net logistic regression
3. Retrain best params on full training data
4. Evaluate on test set; save metrics, plots, model, and study

Example:
  python elastic_net_logistic_regression.py \\
    --outcome_type outcome_abmr \\
    --train_set datasets/mayo_sites_merged_highres_train.csv \\
    --test_set datasets/ABMR_TEST.csv \\
    --output results_elasticnet_logreg \\
    --trials 200
"""

# Standard Libraries
import argparse
import warnings
import random
import os
from datetime import datetime

# Data Science
import numpy as np
import pandas as pd
from tqdm import tqdm

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

# Hyperparameter Tuning and Encoding
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from category_encoders import TargetEncoder
import joblib

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Ignore Warnings
warnings.filterwarnings("ignore")

# Globals set in __main__
global_study = None
global_num_trials = None
global_fixed_feature_names = None
DATASET_NAME = "unknown"
OUTCOME_NAME = "outcome_tcmr"
RESULT_DIR = "results_elasticnet_logreg"
num_trials = 100
X = None
y = None
X_test = None
y_test = None
y_count = 2
numeric_features = []
categorical_features = []


# ==============================================================================
# 1. Helpers
# ==============================================================================

def preprocess_fold(X_train_fold, X_val_fold, y_train_fold, X_holdout=None):
    """
    Fit preprocessors on train fold only; transform val (+ optional holdout/test).
    Returns arrays and feature-name list.
    """
    for df in [X_train_fold, X_val_fold] + ([X_holdout] if X_holdout is not None else []):
        if df is None:
            continue
        if "Date_of_Transplant" in df.columns:
            df["Date_of_Transplant"] = pd.to_datetime(df["Date_of_Transplant"], errors="coerce")
            df["Transplant_Year"] = df["Date_of_Transplant"].dt.year
            df["Transplant_Month"] = df["Date_of_Transplant"].dt.month
            df["Transplant_Day"] = df["Date_of_Transplant"].dt.day
            df.drop("Date_of_Transplant", axis=1, inplace=True)

    fold_numeric = [
        c for c in X_train_fold.columns
        if X_train_fold[c].dtype in [np.float64, np.int64, "float64", "int64"]
    ]
    fold_categorical = [
        c for c in X_train_fold.columns
        if X_train_fold[c].dtype == object or str(X_train_fold[c].dtype) == "category"
    ]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_tr_num = numeric_transformer.fit_transform(X_train_fold[fold_numeric]) if fold_numeric else np.empty((len(X_train_fold), 0))
    X_va_num = numeric_transformer.transform(X_val_fold[fold_numeric]) if fold_numeric else np.empty((len(X_val_fold), 0))

    cat_imputer = SimpleImputer(strategy="most_frequent")
    if fold_categorical:
        X_tr_cat = pd.DataFrame(
            cat_imputer.fit_transform(X_train_fold[fold_categorical]),
            columns=fold_categorical,
        )
        X_va_cat = pd.DataFrame(
            cat_imputer.transform(X_val_fold[fold_categorical]),
            columns=fold_categorical,
        )
    else:
        X_tr_cat = pd.DataFrame(index=np.arange(len(X_train_fold)))
        X_va_cat = pd.DataFrame(index=np.arange(len(X_val_fold)))

    high_card = [c for c in fold_categorical if X_tr_cat[c].nunique() > 10]
    low_card = [c for c in fold_categorical if X_tr_cat[c].nunique() <= 10]

    if high_card:
        te = TargetEncoder(cols=high_card)
        X_tr_high = te.fit_transform(X_tr_cat[high_card], y_train_fold)
        X_va_high = te.transform(X_va_cat[high_card])
    else:
        te = None
        X_tr_high = pd.DataFrame(index=np.arange(len(X_train_fold)))
        X_va_high = pd.DataFrame(index=np.arange(len(X_val_fold)))

    X_tr_low = pd.get_dummies(X_tr_cat[low_card], drop_first=True) if low_card else pd.DataFrame(index=np.arange(len(X_train_fold)))
    X_va_low = pd.get_dummies(X_va_cat[low_card], drop_first=True) if low_card else pd.DataFrame(index=np.arange(len(X_val_fold)))
    X_va_low = X_va_low.reindex(columns=X_tr_low.columns, fill_value=0)

    feature_names = (
        list(fold_numeric)
        + [f"{c}_te" for c in high_card]
        + X_tr_low.columns.tolist()
    )

    X_tr = np.nan_to_num(np.hstack([
        X_tr_num,
        X_tr_high.values if len(X_tr_high.columns) else np.empty((len(X_train_fold), 0)),
        X_tr_low.values if len(X_tr_low.columns) else np.empty((len(X_train_fold), 0)),
    ])).astype(float)

    X_va = np.nan_to_num(np.hstack([
        X_va_num,
        X_va_high.values if len(X_va_high.columns) else np.empty((len(X_val_fold), 0)),
        X_va_low.values if len(X_va_low.columns) else np.empty((len(X_val_fold), 0)),
    ])).astype(float)

    artifacts = {
        "numeric_transformer": numeric_transformer,
        "cat_imputer": cat_imputer,
        "te": te,
        "fold_numeric": fold_numeric,
        "fold_categorical": fold_categorical,
        "high_card": high_card,
        "low_card": low_card,
        "ohe_cols": X_tr_low.columns.tolist(),
        "feature_names": feature_names,
    }

    X_ho = None
    if X_holdout is not None:
        X_ho_df = X_holdout.copy()
        for col in fold_numeric:
            if col not in X_ho_df.columns:
                X_ho_df[col] = 0
        for col in fold_categorical:
            if col not in X_ho_df.columns:
                X_ho_df[col] = "missing"
        X_ho_num = numeric_transformer.transform(X_ho_df[fold_numeric]) if fold_numeric else np.empty((len(X_ho_df), 0))
        if fold_categorical:
            X_ho_cat = pd.DataFrame(
                cat_imputer.transform(X_ho_df[fold_categorical]),
                columns=fold_categorical,
            )
        else:
            X_ho_cat = pd.DataFrame(index=np.arange(len(X_ho_df)))
        if high_card:
            X_ho_high = te.transform(X_ho_cat[high_card])
        else:
            X_ho_high = pd.DataFrame(index=np.arange(len(X_ho_df)))
        X_ho_low = pd.get_dummies(X_ho_cat[low_card], drop_first=True) if low_card else pd.DataFrame(index=np.arange(len(X_ho_df)))
        X_ho_low = X_ho_low.reindex(columns=X_tr_low.columns, fill_value=0)
        X_ho = np.nan_to_num(np.hstack([
            X_ho_num,
            X_ho_high.values if len(X_ho_high.columns) else np.empty((len(X_ho_df), 0)),
            X_ho_low.values if len(X_ho_low.columns) else np.empty((len(X_ho_df), 0)),
        ])).astype(float)

    return X_tr, X_va, X_ho, artifacts


def make_elasticnet_logreg(C, l1_ratio, max_iter, class_weight):
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=C,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        class_weight=class_weight,
        multi_class="ovr",
        random_state=SEED,
        n_jobs=-1,
    )


def coef_importances(model, feature_names):
    """Mean absolute coefficient magnitude across classes (normalized)."""
    coef = np.asarray(model.coef_)
    raw = np.mean(np.abs(coef), axis=0)
    if len(raw) < len(feature_names):
        raw = np.pad(raw, (0, len(feature_names) - len(raw)))
    elif len(raw) > len(feature_names):
        raw = raw[: len(feature_names)]
    s = raw.sum()
    norm = raw / s if s > 0 else raw
    return norm, raw


def find_best_threshold(y_true, probs, rejection_positive_class=1):
    """F1-optimal if positives <10%, else Youden's J."""
    positive_ratio = np.mean(y_true == rejection_positive_class)
    if positive_ratio < 0.10:
        precision, recall, thresholds = precision_recall_curve(y_true, probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        return float(best_threshold), "F1-optimal"
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return float(best_threshold), "Youden"


def stratified_bootstrap_indices(y, B=1000, seed=42):
    np.random.seed(seed)
    y = np.asarray(y)
    class_indices = {cls: np.where(y == cls)[0] for cls in np.unique(y)}
    samples = []
    for _ in range(B):
        sample_indices = []
        for cls, cidx in class_indices.items():
            sampled = resample(
                cidx,
                replace=True,
                n_samples=len(cidx),
                random_state=np.random.randint(0, 1_000_000),
            )
            sample_indices.extend(sampled)
        np.random.shuffle(sample_indices)
        samples.append(np.asarray(sample_indices))
    return samples


def bootstrap_evaluation_on_test(model, X_test_combined, y_test, metric_fn, B=1000, seed=42):
    np.random.seed(seed)
    stratified_indices = stratified_bootstrap_indices(y_test, B, seed)
    metrics = []
    for i in tqdm(range(B), desc="Bootstrapping"):
        indices = stratified_indices[i]
        X_bs = X_test_combined[indices]
        y_bs = y_test[indices]
        try:
            if len(np.unique(y_bs)) < 2:
                raise ValueError("Only one class present")
            probs = model.predict_proba(X_bs)
            if y_count == 2:
                preds = (probs[:, 1] >= 0.5).astype(int)
            else:
                preds = np.argmax(probs, axis=1)
            if metric_fn.__name__ == "roc_auc_score":
                score = metric_fn(y_bs, probs[:, 1]) if probs.shape[1] == 2 else metric_fn(
                    y_bs, probs, multi_class="ovr", average="macro"
                )
            elif metric_fn.__name__ == "average_precision_score":
                score = metric_fn(y_bs, probs[:, 1]) if probs.shape[1] == 2 else metric_fn(
                    y_bs, probs, average="macro"
                )
            else:
                score = metric_fn(y_bs, preds, average="macro")
        except ValueError:
            score = np.nan
        metrics.append(score)
    metrics = np.asarray(metrics, dtype=float)
    metrics = metrics[~np.isnan(metrics)]
    if len(metrics) == 0:
        print(f"All bootstraps failed for {metric_fn.__name__}.")
        return np.nan, np.nan, (np.nan, np.nan)
    mean_score = float(np.mean(metrics))
    std_score = float(np.std(metrics))
    ci_lower = float(np.percentile(metrics, 2.5))
    ci_upper = float(np.percentile(metrics, 97.5))
    print(f"\nBootstrapped {metric_fn.__name__}: {mean_score:.4f} +/- {std_score:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    return mean_score, std_score, (ci_lower, ci_upper)


# ==============================================================================
# 2. Optuna Objective (5-Fold CV)
# ==============================================================================

def objective(trial, result_dir):
    global global_fixed_feature_names

    print(f"\n Running Trial {trial.number + 1}/{num_trials}...")

    # Elastic-net hyperparameters only
    logreg_C = trial.suggest_float("logreg_C", 1e-4, 1e2, log=True)
    logreg_l1_ratio = trial.suggest_float("logreg_l1_ratio", 0.0, 1.0)
    logreg_max_iter = trial.suggest_int("logreg_max_iter", 200, 5000, step=100)
    logreg_class_weight = trial.suggest_categorical("logreg_class_weight", [None, "balanced"])

    oof_probs = np.zeros((len(y), y_count), dtype=float)
    oof_labels = y
    fold_importances_norm = []
    fold_importances_raw = []
    cv_results = []
    combined_feature_names_fold = None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train_fold = X.iloc[train_idx].copy().reset_index(drop=True)
        X_val_fold = X.iloc[val_idx].copy().reset_index(drop=True)
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        X_tr, X_va, _, artifacts = preprocess_fold(
            X_train_fold, X_val_fold, y_train_fold
        )
        combined_feature_names_fold = artifacts["feature_names"]

        model = make_elasticnet_logreg(
            C=logreg_C,
            l1_ratio=logreg_l1_ratio,
            max_iter=logreg_max_iter,
            class_weight=logreg_class_weight,
        )
        model.fit(X_tr, y_train_fold)

        val_probs = model.predict_proba(X_va)
        oof_probs[val_idx] = val_probs

        fi_norm, fi_raw = coef_importances(model, combined_feature_names_fold)

        if global_fixed_feature_names is None:
            global_fixed_feature_names = list(combined_feature_names_fold)
        else:
            fixed_len = len(global_fixed_feature_names)
            if len(fi_norm) < fixed_len:
                pad = fixed_len - len(fi_norm)
                fi_norm = np.pad(fi_norm, (0, pad))
                fi_raw = np.pad(fi_raw, (0, pad))
            elif len(fi_norm) > fixed_len:
                fi_norm = fi_norm[:fixed_len]
                fi_raw = fi_raw[:fixed_len]

        fold_importances_norm.append(fi_norm)
        fold_importances_raw.append(fi_raw)

        fold_results_df = pd.DataFrame({
            "Fold": fold_idx,
            "Patient_Index": X.index[val_idx],
            "TrueLabel": y_val_fold,
        })
        for c_idx in range(y_count):
            fold_results_df[f"Prob_Class{c_idx}"] = val_probs[:, c_idx]
        cv_results.append(fold_results_df)

    # OOF metrics
    if y_count == 2:
        oof_auc = roc_auc_score(oof_labels, oof_probs[:, 1])
        oof_auprc = average_precision_score(oof_labels, oof_probs[:, 1])
    else:
        oof_auc = roc_auc_score(oof_labels, oof_probs, multi_class="ovr", average="macro")
        oof_auprc = average_precision_score(oof_labels, oof_probs, average="macro")

    avg_importances_norm = np.mean(fold_importances_norm, axis=0)
    avg_importances_raw = np.mean(fold_importances_raw, axis=0)

    trial.set_user_attr("val_labels", oof_labels)
    trial.set_user_attr("val_probs", oof_probs)
    trial.set_user_attr("avg_importances_norm", avg_importances_norm)
    trial.set_user_attr("avg_importances_raw", avg_importances_raw)
    trial.set_user_attr(
        "final_feature_names",
        global_fixed_feature_names if global_fixed_feature_names is not None else combined_feature_names_fold,
    )
    trial.set_user_attr("oof_auc", oof_auc)
    trial.set_user_attr("oof_auprc", oof_auprc)

    df_cv_results = pd.concat(cv_results, axis=0).sort_values("Patient_Index")
    trial.set_user_attr("cv_results", df_cv_results)

    oof_preds = (
        np.argmax(oof_probs, axis=1)
        if y_count > 2
        else (oof_probs[:, 1] >= 0.5).astype(int)
    )
    cm_val = confusion_matrix(oof_labels, oof_preds, labels=list(range(y_count)))
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_val, annot=True, fmt="d", cmap="Purples")
    plt.title("Confusion Matrix - 5-Fold CV (OOF)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(
        f"{result_dir}/{DATASET_NAME}_{OUTCOME_NAME}_val_confusion_matrix_oof.png",
        dpi=300,
    )
    plt.close()

    val_report_dict = classification_report(oof_labels, oof_preds, output_dict=True)
    pd.DataFrame(val_report_dict).transpose().to_csv(
        f"{result_dir}/{DATASET_NAME}_{OUTCOME_NAME}_val_classification_report_oof.csv"
    )

    macro_f1 = f1_score(oof_labels, oof_preds, average="macro")
    print(
        f"[Trial {trial.number}] Macro F1={macro_f1:.4f}  "
        f"AUC={oof_auc:.4f}  AUPRC={oof_auprc:.4f}"
    )
    return macro_f1


# ==============================================================================
# 3. Best-trial summary plots
# ==============================================================================

def plot_best_trial_summary(best_trial, result_dir):
    trial_number = best_trial.number
    y_val = best_trial.user_attrs["val_labels"]
    y_pred_prob = best_trial.user_attrs["val_probs"]
    best_val_auc = best_trial.user_attrs["oof_auc"]
    best_val_auprc = best_trial.user_attrs["oof_auprc"]
    avg_importances_norm = best_trial.user_attrs["avg_importances_norm"]
    avg_importances_raw = best_trial.user_attrs["avg_importances_raw"]
    final_feature_names = best_trial.user_attrs["final_feature_names"]

    # ROC
    plt.figure(figsize=(8, 6))
    if y_pred_prob.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_val, y_pred_prob[:, 1])
        plt.plot(fpr, tpr, label=f"ROC (AUC={auc(fpr, tpr):.4f})")
    else:
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(y_val)
        all_fpr = None
        mean_tpr = None
        for c in range(y_pred_prob.shape[1]):
            fpr_c, tpr_c, _ = roc_curve(y_bin[:, c], y_pred_prob[:, c])
            plt.plot(fpr_c, tpr_c, label=f"Class {c} vs Rest")
            if all_fpr is None:
                all_fpr = fpr_c
                mean_tpr = np.interp(all_fpr, fpr_c, tpr_c)
            else:
                all_fpr_u = np.unique(np.concatenate([all_fpr, fpr_c]))
                mean_tpr = np.interp(all_fpr_u, all_fpr, mean_tpr) + np.interp(
                    all_fpr_u, fpr_c, tpr_c
                )
                all_fpr = all_fpr_u
        mean_tpr /= y_pred_prob.shape[1]
        plt.plot(
            all_fpr,
            mean_tpr,
            color="black",
            linestyle="--",
            label=f"Macro-avg (AUC={auc(all_fpr, mean_tpr):.4f})",
        )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - Trial {trial_number} (AUC={best_val_auc:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{result_dir}/roc_curve_trial_{trial_number}.png", dpi=300)
    plt.close()

    # Confusion matrix
    y_pred_class = (
        np.argmax(y_pred_prob, axis=1)
        if y_pred_prob.shape[1] > 1
        else (y_pred_prob[:, 1] >= 0.5).astype(int)
    )
    cm = confusion_matrix(y_val, y_pred_class)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - Trial {trial_number}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{result_dir}/cm_trial_{trial_number}.png", dpi=300)
    plt.close()

    # Feature importances (|coef|)
    n = min(len(final_feature_names), len(avg_importances_norm))
    df_norm = pd.DataFrame({
        "feature": list(final_feature_names)[:n],
        "importance_norm": np.asarray(avg_importances_norm)[:n],
    }).sort_values("importance_norm", ascending=False)
    df_norm["rank"] = range(1, len(df_norm) + 1)
    df_norm.to_csv(
        f"{result_dir}/avg_importances_normalized_trial_{trial_number}.csv",
        index=False,
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_norm.head(30), x="importance_norm", y="feature", color="skyblue")
    plt.title(f"Normalized |Coef| Importances - Trial {trial_number}")
    plt.tight_layout()
    plt.savefig(
        f"{result_dir}/feature_importances_normalized_trial_{trial_number}.png",
        dpi=300,
    )
    plt.close()

    df_raw = pd.DataFrame({
        "feature": list(final_feature_names)[:n],
        "importance_raw": np.asarray(avg_importances_raw)[:n],
    }).sort_values("importance_raw", ascending=False)
    df_raw["rank"] = range(1, len(df_raw) + 1)
    df_raw.to_csv(
        f"{result_dir}/avg_importances_raw_trial_{trial_number}.csv",
        index=False,
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_raw.head(30), x="importance_raw", y="feature", color="salmon")
    plt.title(f"Raw |Coef| Importances - Trial {trial_number}")
    plt.tight_layout()
    plt.savefig(
        f"{result_dir}/feature_importances_raw_trial_{trial_number}.png",
        dpi=300,
    )
    plt.close()

    best_trial.user_attrs["cv_results"].to_csv(
        f"{result_dir}/cv_risk_scores_trial_{trial_number}.csv", index=False
    )
    pd.DataFrame([{
        "Best Trial Number": trial_number,
        "Validation OOF AUC": best_val_auc,
        "Validation OOF AUPRC": best_val_auprc,
    }]).to_csv(
        os.path.join(
            result_dir,
            f"best_trial_{trial_number}_validation_metrics_summary.csv",
        ),
        index=False,
    )
    print(f"Best trial visualizations saved for trial #{trial_number}")


# ==============================================================================
# 4. Final test evaluation
# ==============================================================================

def evaluate_best_trial_on_test(
    best_trial, RESULT_DIR, X, y, X_test, y_test, numeric_features, categorical_features
):
    print(
        f"\nRetraining best model (Trial #{best_trial.number}) "
        f"with validation macro-F1 = {best_trial.value:.4f}"
    )
    print("\n=== Evaluating Best Trial on Test Set ===")
    params = best_trial.params

    # Align dtypes
    for col in X_test.columns:
        if col in X.columns and X[col].dtype in ["int64", "float64"]:
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    X_train_df = X.copy().reset_index(drop=True)
    X_test_df = X_test.copy().reset_index(drop=True)
    X_pre, X_test_pre, _, artifacts = _fit_transform_full_train_test(
        X_train_df, X_test_df, y
    )

    model = make_elasticnet_logreg(
        C=params["logreg_C"],
        l1_ratio=params["logreg_l1_ratio"],
        max_iter=params["logreg_max_iter"],
        class_weight=params["logreg_class_weight"],
    )
    model.fit(X_pre, y)

    # Persist preprocessors + model
    joblib.dump(artifacts["numeric_transformer"], os.path.join(RESULT_DIR, "numeric_transformer.joblib"))
    joblib.dump(artifacts["cat_imputer"], os.path.join(RESULT_DIR, "cat_imputer.joblib"))
    if artifacts["te"] is not None:
        joblib.dump(artifacts["te"], os.path.join(RESULT_DIR, "target_encoder.joblib"))
    joblib.dump(artifacts["fold_numeric"], os.path.join(RESULT_DIR, "numeric_features.joblib"))
    joblib.dump(artifacts["high_card"], os.path.join(RESULT_DIR, "high_card_cols.joblib"))
    joblib.dump(artifacts["low_card"], os.path.join(RESULT_DIR, "low_card_cols.joblib"))
    joblib.dump(artifacts["ohe_cols"], os.path.join(RESULT_DIR, "ohe_low_card_cols.joblib"))
    joblib.dump(artifacts["feature_names"], os.path.join(RESULT_DIR, "feature_names.joblib"))

    test_probs = model.predict_proba(X_test_pre)
    train_probs = model.predict_proba(X_pre)

    if y_count == 2:
        train_auc_real = roc_auc_score(y, train_probs[:, 1])
    else:
        train_auc_real = roc_auc_score(y, train_probs, multi_class="ovr", average="macro")
    print(f"Train ROC AUC (full train): {train_auc_real:.4f}")
    pd.DataFrame([{"Corrected Train ROC AUC": round(train_auc_real, 4)}]).to_csv(
        os.path.join(RESULT_DIR, f"{DATASET_NAME}_{OUTCOME_NAME}_train_auc_real.csv"),
        index=False,
    )

    if y_count == 2:
        best_threshold, method = find_best_threshold(y_test, test_probs[:, 1])
        print(f"Best Threshold Chosen ({method}): {best_threshold:.4f}")
        test_preds = (test_probs[:, 1] >= best_threshold).astype(int)
    else:
        best_threshold = None
        method = "Argmax (Multiclass)"
        print(f"Multiclass outcome — predictions will use {method}.")
        test_preds = np.argmax(test_probs, axis=1)

    model_save_path = os.path.join(
        RESULT_DIR, f"{DATASET_NAME}_{OUTCOME_NAME}_best_model.joblib"
    )
    joblib.dump(model, model_save_path)
    print(f"Saved model to: {model_save_path}")

    # PR / AUPRC
    if y_count == 2:
        precision, recall, _ = precision_recall_curve(y_test, test_probs[:, 1])
        auprc = average_precision_score(y_test, test_probs[:, 1])
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"PR Curve (AUPRC = {auprc:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Test PR Curve - Best Trial")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"{RESULT_DIR}/{DATASET_NAME}_{OUTCOME_NAME}_test_pr_curve_best_trial.png",
            dpi=300,
        )
        plt.close()
    else:
        auprc = average_precision_score(y_test, test_probs, average="macro")
    print(f"AUPRC (Average Precision): {auprc:.4f}")

    test_accuracy = accuracy_score(y_test, test_preds)
    if y_count == 2:
        test_auc = roc_auc_score(y_test, test_probs[:, 1])
    else:
        test_auc = roc_auc_score(y_test, test_probs, multi_class="ovr", average="macro")
    test_f1 = f1_score(y_test, test_preds, average="macro")

    print(f"\n[TEST SET RESULTS]")
    print(f"AUC: {test_auc:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Macro F1-score: {test_f1:.4f}")

    cm = confusion_matrix(y_test, test_preds, labels=list(range(y_count)))
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Best Trial on Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(
        f"{RESULT_DIR}/{DATASET_NAME}_{OUTCOME_NAME}_test_confusion_matrix_best_trial.png",
        dpi=300,
    )
    plt.close()

    report_df = pd.DataFrame(
        classification_report(y_test, test_preds, output_dict=True)
    ).transpose()
    report_df.to_csv(
        f"{RESULT_DIR}/{DATASET_NAME}_{OUTCOME_NAME}_test_classification_report_best_trial.csv"
    )

    # Probability histogram
    if y_count == 2:
        plt.figure(figsize=(7, 5))
        for cls in [0, 1]:
            plt.hist(
                test_probs[y_test == cls, 1],
                bins=20,
                alpha=0.6,
                label=f"True {cls}",
            )
        if best_threshold is not None:
            plt.axvline(best_threshold, color="red", linestyle="--", label=f"thr={best_threshold:.3f}")
        plt.xlabel("P(class=1)")
        plt.ylabel("Count")
        plt.title("Test Probability Histogram")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"{RESULT_DIR}/{DATASET_NAME}_{OUTCOME_NAME}_test_probability_histogram.png",
            dpi=300,
        )
        plt.close()

    # ROC
    plt.figure(figsize=(7, 5))
    if y_count == 2:
        fpr, tpr, _ = roc_curve(y_test, test_probs[:, 1])
        plt.plot(fpr, tpr, label=f"AUC={test_auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
    else:
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(y_test)
        for c in range(y_count):
            fpr_c, tpr_c, _ = roc_curve(y_bin[:, c], test_probs[:, c])
            plt.plot(fpr_c, tpr_c, label=f"Class {c}")
        plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Test ROC - Best Trial")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{RESULT_DIR}/{DATASET_NAME}_{OUTCOME_NAME}_test_roc_curve_best_trial.png",
        dpi=300,
    )
    plt.close()

    # Predictions CSV
    pred_cols = {
        "TrueLabel": y_test,
        "PredLabel": test_preds,
    }
    if y_count == 2:
        pred_cols["Prob_Class1"] = test_probs[:, 1]
        pred_cols["threshold"] = best_threshold
        pred_cols["threshold_method"] = method
    else:
        for c in range(y_count):
            pred_cols[f"Prob_Class{c}"] = test_probs[:, c]
    pd.DataFrame(pred_cols).to_csv(
        f"{RESULT_DIR}/{DATASET_NAME}_{OUTCOME_NAME}_test_predictions_best_trial.csv",
        index=False,
    )

    # Bootstrap
    def run_bootstrap(metric_fn, name):
        mean, std, (ci_lower, ci_upper) = bootstrap_evaluation_on_test(
            model, X_test_pre, y_test, metric_fn, B=1000
        )
        return {
            "metric": name,
            "mean": mean,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    bootstrap_results = [
        run_bootstrap(roc_auc_score, "AUC"),
        run_bootstrap(f1_score, "Macro F1"),
        run_bootstrap(average_precision_score, "AUPRC"),
    ]
    pd.DataFrame(bootstrap_results).to_csv(
        f"{RESULT_DIR}/{DATASET_NAME}_{OUTCOME_NAME}_bootstrap_test_metrics.csv",
        index=False,
    )
    print(
        f"\nBootstrapped metrics saved to "
        f"{RESULT_DIR}/{DATASET_NAME}_{OUTCOME_NAME}_bootstrap_test_metrics.csv"
    )


def _fit_transform_full_train_test(X_train_df, X_test_df, y_train):
    """Fit preprocessors on full train; transform train and test."""
    X_tr = X_train_df.copy()
    X_te = X_test_df.copy()

    for df in [X_tr, X_te]:
        if "Date_of_Transplant" in df.columns:
            df["Date_of_Transplant"] = pd.to_datetime(df["Date_of_Transplant"], errors="coerce")
            df["Transplant_Year"] = df["Date_of_Transplant"].dt.year
            df["Transplant_Month"] = df["Date_of_Transplant"].dt.month
            df["Transplant_Day"] = df["Date_of_Transplant"].dt.day
            df.drop("Date_of_Transplant", axis=1, inplace=True)

    fold_numeric = [
        c for c in X_tr.columns
        if X_tr[c].dtype in [np.float64, np.int64, "float64", "int64"]
    ]
    fold_categorical = [
        c for c in X_tr.columns
        if X_tr[c].dtype == object or str(X_tr[c].dtype) == "category"
    ]

    for col in fold_numeric:
        if col not in X_te.columns:
            X_te[col] = 0
    for col in fold_categorical:
        if col not in X_te.columns:
            X_te[col] = "missing"

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_tr_num = numeric_transformer.fit_transform(X_tr[fold_numeric]) if fold_numeric else np.empty((len(X_tr), 0))
    X_te_num = numeric_transformer.transform(X_te[fold_numeric]) if fold_numeric else np.empty((len(X_te), 0))

    cat_imputer = SimpleImputer(strategy="most_frequent")
    if fold_categorical:
        X_tr_cat = pd.DataFrame(cat_imputer.fit_transform(X_tr[fold_categorical]), columns=fold_categorical)
        X_te_cat = pd.DataFrame(cat_imputer.transform(X_te[fold_categorical]), columns=fold_categorical)
    else:
        X_tr_cat = pd.DataFrame(index=np.arange(len(X_tr)))
        X_te_cat = pd.DataFrame(index=np.arange(len(X_te)))

    high_card = [c for c in fold_categorical if X_tr_cat[c].nunique() > 10]
    low_card = [c for c in fold_categorical if X_tr_cat[c].nunique() <= 10]

    if high_card:
        te = TargetEncoder(cols=high_card)
        X_tr_high = te.fit_transform(X_tr_cat[high_card], y_train)
        X_te_high = te.transform(X_te_cat[high_card])
    else:
        te = None
        X_tr_high = pd.DataFrame(index=np.arange(len(X_tr)))
        X_te_high = pd.DataFrame(index=np.arange(len(X_te)))

    X_tr_low = pd.get_dummies(X_tr_cat[low_card], drop_first=True) if low_card else pd.DataFrame(index=np.arange(len(X_tr)))
    X_te_low = pd.get_dummies(X_te_cat[low_card], drop_first=True) if low_card else pd.DataFrame(index=np.arange(len(X_te)))
    X_te_low = X_te_low.reindex(columns=X_tr_low.columns, fill_value=0)

    feature_names = list(fold_numeric) + [f"{c}_te" for c in high_card] + X_tr_low.columns.tolist()

    X_pre = np.nan_to_num(np.hstack([
        X_tr_num,
        X_tr_high.values if len(X_tr_high.columns) else np.empty((len(X_tr), 0)),
        X_tr_low.values if len(X_tr_low.columns) else np.empty((len(X_tr), 0)),
    ])).astype(float)
    X_test_pre = np.nan_to_num(np.hstack([
        X_te_num,
        X_te_high.values if len(X_te_high.columns) else np.empty((len(X_te), 0)),
        X_te_low.values if len(X_te_low.columns) else np.empty((len(X_te), 0)),
    ])).astype(float)

    artifacts = {
        "numeric_transformer": numeric_transformer,
        "cat_imputer": cat_imputer,
        "te": te,
        "fold_numeric": fold_numeric,
        "fold_categorical": fold_categorical,
        "high_card": high_card,
        "low_card": low_card,
        "ohe_cols": X_tr_low.columns.tolist(),
        "feature_names": feature_names,
    }
    return X_pre, X_test_pre, None, artifacts


# ==============================================================================
# 5. Main
# ==============================================================================

if __name__ == "__main__":
    print("\n=== Pure Elastic-Net Logistic Regression + Optuna (no VAE/GMM/XGB) ===")

    parser = argparse.ArgumentParser(
        description="Run Optuna study for elastic-net logistic regression"
    )
    parser.add_argument(
        "--outcome_type",
        type=str,
        required=True,
        help="outcome_abmr, outcome_banff, outcome_tcmr, or outcome_rej",
    )
    parser.add_argument(
        "--train_set",
        nargs="+",
        required=True,
        help="one or more CSV files of training data",
    )
    parser.add_argument(
        "--test_set",
        nargs="+",
        required=True,
        help="one or more CSV files of test data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_elasticnet_logreg",
        help="output folder name",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="number of optuna trials",
    )
    args = parser.parse_args()

    train_array = args.train_set
    test_array = args.test_set
    print(train_array)
    print(test_array)
    if len(train_array) != len(test_array):
        raise ValueError("Training and testing datasets must have the same number of paths.")

    OUTCOME_NAME = args.outcome_type
    OUTPUT_PATH = args.output
    num_trials = args.trials

    for train_path, test_path in zip(train_array, test_array):
        global_fixed_feature_names = None

        tp = train_path.lower()
        if "lowres" in tp or "low_res" in tp or "low-res" in tp:
            DATASET_NAME = "low_res"
        elif "highres" in tp or "high_res" in tp or "high-res" in tp:
            DATASET_NAME = "high_res"
        elif "imputed" in tp or "imp" in tp:
            DATASET_NAME = "imputed"
        else:
            DATASET_NAME = "unknown"

        RESULT_DIR = f"{OUTPUT_PATH}/{DATASET_NAME}/{OUTCOME_NAME}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subfolder_name = f"{num_trials}_trials_{timestamp}"
        run_dir = os.path.join(RESULT_DIR, subfolder_name)
        os.makedirs(run_dir, exist_ok=True)
        RESULT_DIR = run_dir

        data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        outcome_col = OUTCOME_NAME
        y_original = data[outcome_col]
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_original)
        y_count = len(set(y))

        drop_cols = ["outcome_tcmr", "outcome_banff", "outcome_abmr", "outcome_rej", "pid"]
        X = data.drop(columns=drop_cols, errors="ignore")
        X_test = test_data.drop(columns=drop_cols, errors="ignore")
        y_test_original = test_data[outcome_col]
        y_test = label_encoder.transform(y_test_original)

        numeric_features = [c for c in X.columns if X[c].dtype in ["int64", "float64"]]
        categorical_features = [c for c in X.columns if X[c].dtype in ["object", "category"]]

        unique_classes, class_counts = np.unique(y, return_counts=True)
        print("\nClass Distribution:")
        for cls, count in zip(unique_classes, class_counts):
            label = f"Class {cls}" if y_count > 2 else ("No Rejection" if cls == 0 else "Rejection")
            print(f"{label}: {count} samples")

        print(f"\nTrain set: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Test set:  {X_test.shape[0]} samples, {X_test.shape[1]} features")

        metadata = {
            "dataset_name": DATASET_NAME,
            "outcome": OUTCOME_NAME,
            "num_train_samples": X.shape[0],
            "num_test_samples": X_test.shape[0],
            "num_features": X.shape[1],
            "model": "elasticnet_logistic_regression",
        }
        pd.DataFrame(list(metadata.items()), columns=["Attribute", "Value"]).to_csv(
            os.path.join(RESULT_DIR, "dataset_metadata.csv"), index=False
        )

        study_path = os.path.join(RESULT_DIR, "optuna_study.pkl")
        if os.path.exists(study_path):
            study = joblib.load(study_path)
            print("Loaded existing Optuna study.")
        else:
            study = optuna.create_study(direction="maximize")
            print("Created new Optuna study.")

        study.optimize(
            lambda trial: objective(trial, RESULT_DIR),
            n_trials=num_trials,
            show_progress_bar=True,
        )

        print("Saving results to:", RESULT_DIR)
        joblib.dump(study, study_path)
        print(f"\nOptuna study saved to: {study_path}")

        plot_best_trial_summary(study.best_trial, RESULT_DIR)

        for col in X_test.columns:
            if col in X.columns and X[col].dtype in ["int64", "float64"]:
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

        evaluate_best_trial_on_test(
            study.best_trial,
            RESULT_DIR,
            X,
            y,
            X_test,
            y_test,
            numeric_features,
            categorical_features,
        )

        print("\n--- Best Trial ---")
        print(f"  Number: {study.best_trial.number}")
        print(f"  Validation F1 (macro): {study.best_trial.value:.4f}")
        print(f"  Params: {study.best_trial.params}")

        with open(os.path.join(RESULT_DIR, "best_trial_summary.txt"), "w") as f:
            f.write(f"Number: {study.best_trial.number}\n")
            f.write(f"Validation F1 (macro): {study.best_trial.value:.4f}\n")
            f.write(f"Params: {study.best_trial.params}\n")

import os
import numpy as np
import pandas as pd
import joblib
import argparse


# === ARGUMENT PARSING ===
parser = argparse.ArgumentParser(description="Evaluate a trained XGBoost-only model on a new test set.")
parser.add_argument('--model_dir', type=str, required=True, help="Directory containing the trained model and transformers.")
parser.add_argument('--test_paths', type=str, nargs='+', required=True, help="Path to the test dataset CSV file.")
parser.add_argument('--output_dir', type=str, default=".", help="Directory to save output CSV file. Defaults to current directory.")
parser.add_argument('--outcome_col', type=str, default="outcome_tcmr",
                    help="Column name of the outcome variable in the test set.")
parser.add_argument('--type', type=str, default="unknown",
                    help="Type of model to differentiate, e.g., 'high_res', 'low_res', etc.")
args = parser.parse_args()


# === PATHS ===
# The single directory where the XGBoost model and fitted transformers were saved.
MODEL_DIR = args.model_dir
TEST_PATHS = args.test_paths
OUTPUT_DIR = args.output_dir
OUTCOME_COL = args.outcome_col
TYPE = args.type

# Columns dropped by the XGBoost-only training script before preprocessing.
DROP_COLS = ["outcome_tcmr", "outcome_banff", "outcome_abmr", "outcome_rej", "pid"]

# Create output directory if it doesn't exist.
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for TEST_PATH in TEST_PATHS:
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

    # === STEP 1: LOAD TEST DATA ONLY ===
    print(f"Loading test data from: {TEST_PATH}")
    df_test = pd.read_csv(TEST_PATH)

    # Extract ID before dropping columns.
    if "pid" in df_test.columns:
        ids = df_test["pid"].values
    else:
        # If no pid column, use index as ID.
        ids = df_test.index.values

    X_test = df_test.drop(columns=DROP_COLS, errors="ignore")

    # === STEP 2: LOAD THE ENTIRE FITTED PIPELINE (MODEL AND TRANSFORMERS) ===
    print(f"\n--- Loading fitted pipeline objects from: {MODEL_DIR} ---")

    # The XGBoost-only training script saved its fitted classifier with joblib.
    xgb_model = joblib.load(
        os.path.join(MODEL_DIR, f"{TYPE}_{OUTCOME_COL}_xgb_model.pkl")
    )

    # Load the exact feature lists used during training.
    numeric_features = joblib.load(os.path.join(MODEL_DIR, 'numeric_features.joblib'))
    high_card_cols = joblib.load(os.path.join(MODEL_DIR, 'high_card_cols.joblib'))
    low_card_cols = joblib.load(os.path.join(MODEL_DIR, 'low_card_cols.joblib'))
    categorical_features = high_card_cols + low_card_cols

    # Load all fitted transformers.
    numeric_transformer = joblib.load(os.path.join(MODEL_DIR, 'numeric_transformer.joblib'))
    cat_imputer = joblib.load(os.path.join(MODEL_DIR, 'cat_imputer.joblib'))
    target_encoder = joblib.load(os.path.join(MODEL_DIR, 'target_encoder.joblib'))
    train_ohe_cols = joblib.load(os.path.join(MODEL_DIR, 'ohe_low_card_cols.joblib'))
    print("--- Pipeline loaded successfully ---\n")

    # === OPTUNA BEST TRIAL PARAMETERS ===
    study_path = os.path.join(MODEL_DIR, "optuna_study.pkl")
    try:
        study = joblib.load(study_path)
        best_params = study.best_trial.params
        print(f" Successfully loaded study. Best trial was #{study.best_trial.number} with a score of {study.best_trial.value:.4f}.")
        print("Found best hyperparameters:")
        print(best_params)
    except FileNotFoundError:
        print(f" ERROR: Could not find the study file at {study_path}")
        print("Please ensure 'optuna_study.pkl' exists in your MODEL_DIR.")
        raise SystemExit(1)

    # === STEP 3: APPLY THE FITTED PIPELINE TO TEST DATA (TRANSFORM ONLY, NEVER FIT) ===

    # Ensure the test set has all required columns, then restore training order.
    for col in numeric_features + categorical_features:
        if col not in X_test.columns:
            print(f"Warning: Column '{col}' not found in test set. Filling with placeholder.")
            X_test[col] = 0 if col in numeric_features else "missing"
    X_test = X_test[numeric_features + categorical_features]

    # 1. Process numeric features using the loaded numeric transformer.
    X_test_numeric = numeric_transformer.transform(X_test[numeric_features])

    # 2. Process categorical features using the loaded transformers.
    X_test_cat_imp = pd.DataFrame(
        cat_imputer.transform(X_test[categorical_features]),
        columns=categorical_features,
    )
    X_test_high = target_encoder.transform(X_test_cat_imp[high_card_cols])
    X_test_low = pd.get_dummies(X_test_cat_imp[low_card_cols], drop_first=True, dtype=float)

    # Align one-hot encoded columns to match the training set exactly.
    X_test_low_ohe = X_test_low.reindex(columns=train_ohe_cols, fill_value=0.0)

    # 3. Assemble the base preprocessed feature set used by XGBoost.
    X_test_processed = np.hstack([
        X_test_numeric,
        X_test_high.values,
        X_test_low_ohe.values,
    ])
    X_test_processed = np.nan_to_num(X_test_processed).astype(float)

    # === STEP 4: PREDICT AND OUTPUT CSV ===
    print("\n--- Generating predictions on the test set ---")
    y_pred_probs_all = xgb_model.predict_proba(X_test_processed)

    # Risk score is the probability of positive class 1.
    risk_scores = (
        y_pred_probs_all[:, 1]
        if y_pred_probs_all.shape[1] > 1
        else y_pred_probs_all[:, 0]
    )

    output_df = pd.DataFrame({
        'ID': ids,
        'risk_score': risk_scores,
    })

    output_path = os.path.join(OUTPUT_DIR, f"{TYPE}_{OUTCOME_COL}_risk_scores.csv")
    output_df.to_csv(output_path, index=False)
    print(f"✅ Risk scores saved to: {output_path}")
    print(f"   Total samples: {len(output_df)}")
    print(f"   Risk score range: [{risk_scores.min():.4f}, {risk_scores.max():.4f}]")

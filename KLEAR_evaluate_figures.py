import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
from sklearn.metrics import (
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,  
    auc, 
    accuracy_score, 
    classification_report,  
    f1_score, 
    precision_recall_curve, 
    average_precision_score
)
import argparse
from tqdm import tqdm
import seaborn as sns
from sklearn.utils import resample
from dcurves import dca, plot_graphs
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

# Set matplotlib to use the 'Agg' backend for non-interactive plotting
plt.switch_backend('agg')
# === ARGUMENT PARSING ===
parser = argparse.ArgumentParser(description="Evaluate a trained model on a new test set.")
parser.add_argument('--model_dir', type=str, required=True, help="Directory containing the trained model and transformers.")
parser.add_argument('--test_paths', type=str, nargs='+', required=True, help="Path to the test dataset CSV file.")
parser.add_argument('--outcome_col', type=str, default="outcome_tcmr",
                    help="Column name of the outcome variable in the test set.")
parser.add_argument('--type', type=str, default="unknown",
                    help="Type of model to differentiate, e.g., 'high_res', 'low_res', etc.")
args = parser.parse_args()

# === CALIBRATION FUNCTIONS ===
class TorchWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, pt_model):
        # store both so clone() + your code will work
        self.pt_model = pt_model
        self.model    = pt_model
    def fit(self, X, y=None):
        # No-op fit for prefit mode
        return self
    
    def predict_proba(self, X):
        import torch
        self.pt_model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits   = self.pt_model(X_tensor)
            probs    = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

# === HELPER FUNCTIONS ===
def net_benefit(y_true, y_pred_bin, p_t):
    tp = ((y_pred_bin==1) & (y_true==1)).sum()
    fp = ((y_pred_bin==1) & (y_true==0)).sum()
    N  = len(y_true)
    return tp/N - fp/N * (p_t/(1-p_t))

def stratified_bootstrap_indices(y, B=1000, seed=42):
    """
    Generates stratified bootstrap sample indices.
    Each sample preserves the original class ratio.
    
    Args:
        y (array-like): True labels for stratification.
        B (int): Number of bootstrap samples.
        seed (int): Random seed for reproducibility.

    Returns:
        List of np.array indices for each bootstrap sample.
    """
    np.random.seed(seed)
    y = np.array(y)
    indices = np.arange(len(y))
    stratified_samples = []

    # Get indices for each class
    unique_classes = np.unique(y)
    class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}

    for _ in range(B):
        sample_indices = []
        for cls in unique_classes:
            n_samples_cls = len(class_indices[cls])
            sampled_cls = resample(
                class_indices[cls], replace=True, n_samples=n_samples_cls, random_state=np.random.randint(0, 1e6)
            )
            sample_indices.extend(sampled_cls)
        np.random.shuffle(sample_indices)
        stratified_samples.append(np.array(sample_indices))

    return stratified_samples


def bootstrap_evaluation_on_test(model, X_test_combined, y_test, metric_fn, B=1000, seed=42):
    """
    Evaluate model performance over B bootstrapped test sets.
    Returns mean, std, and 95% confidence interval for the given metric.
    """
    np.random.seed(seed)

    # Load model ONCE
    model_path = os.path.join(MODEL_DIR, f"{TYPE}_{OUTCOME_COL}_best_model.pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Generate all bootstrap indices just once
    stratified_indices = stratified_bootstrap_indices(y_test, B, seed)

    metrics = []
    fail_count = 0

    for i in tqdm(range(B), desc="Bootstrapping"):
        indices = stratified_indices[i]
        X_bs = X_test_combined[indices]
        y_bs = y_test[indices]

        try:
            if len(np.unique(y_bs)) < 2:
                raise ValueError("Only one class present")

            probs = predict_dnn(model, X_bs)

            if len(np.unique(y_bs)) == 2:
                preds = (probs[:, 1] >= 0.5).astype(int)
            else:
                preds = np.argmax(probs, axis=1)

               
            if metric_fn.__name__ == "roc_auc_score":
                if probs.shape[1] == 2:
                    score = metric_fn(y_bs, probs[:, 1])
                else:
                    score = metric_fn(y_bs, probs, multi_class='ovr', average='macro')

            elif metric_fn.__name__ == "average_precision_score":
                if probs.shape[1] == 2:
                    score = metric_fn(y_bs, probs[:, 1])
                else:
                    score = metric_fn(y_bs, probs, average='macro')

            else:
                score = metric_fn(y_bs, preds, average='macro')


        except ValueError:
            fail_count += 1
            score = np.nan

        metrics.append(score)

    # print(f"\n⚠️ {fail_count} out of {B} bootstrap samples had no rejection cases.")

    metrics = np.array(metrics)
    metrics = metrics[~np.isnan(metrics)]

    if len(metrics) == 0:
        print(f"⚠️ All bootstraps failed for {metric_fn.__name__}.")
        return np.nan, np.nan, (np.nan, np.nan)

    mean_score = np.mean(metrics)
    std_score = np.std(metrics)
    ci_lower = np.percentile(metrics, 2.5)
    ci_upper = np.percentile(metrics, 97.5)

    print(f"\nBootstrapped {metric_fn.__name__}: {mean_score:.4f} ± {std_score:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    return mean_score, std_score, (ci_lower, ci_upper)


def load_classification_report(path, prefix="Test"):
    df = pd.read_csv(path)
    df.set_index(df.columns[0], inplace=True)
    result = {}
    if "accuracy" in df.index:
        result[f"{prefix} Accuracy"] = df.loc["accuracy"]["precision"]
    if "1" in df.index:
        result[f"{prefix} Class 1 Precision"] = df.loc["1"]["precision"]
        result[f"{prefix} Class 1 Recall"] = df.loc["1"]["recall"]
        result[f"{prefix} Class 1 F1"] = df.loc["1"]["f1-score"]
    return result

def load_confusion_matrix(path, prefix="CM"):
    try:
        cm = pd.read_csv(path, index_col=0)
        cm_dict = {}
        for i in cm.index:
            for j in cm.columns:
                cm_dict[f"{prefix}_{i}_{j}"] = cm.loc[i, j]
        return cm_dict
    except:
        return {}

def summarize_results(result_dir, dataset_name, outcome_name):
    dataset_info = {}
    train_metrics = {}
    val_metrics = {}
    test_metrics = {}

    # --- Dataset Metadata ---
    metadata_path = os.path.join(result_dir, "dataset_metadata.csv")
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
        for _, row in metadata.iterrows():
            key, val = row['Attribute'], row['Value']
            if key in ["train_path", "test_path"]:
                val = os.path.basename(val)
            dataset_info[key] = val

    # --- Train ROC AUC ---
    train_path = os.path.join(result_dir, f"{dataset_name}_{outcome_name}_train_auc_real.csv")
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path, header=None)
        try:
            train_auc = float(train_df.iloc[1, 0])
            train_metrics["Train ROC AUC"] = round(train_auc, 4)
        except:
            print("❌ Could not parse Train ROC AUC from file.")

    # --- Validation Classification Report ---
    val_report = os.path.join(result_dir, f"{dataset_name}_{outcome_name}_val_classification_report_oof.csv")
    if os.path.exists(val_report):
        val_metrics.update(load_classification_report(val_report, prefix="Validation"))

    # --- Validation AUC + AUPRC (stored in summary CSV) ---
    val_summary_path = os.path.join(result_dir, f"best_trial_*_validation_metrics_summary.csv")
    try:
        match = sorted([f for f in os.listdir(result_dir) if "validation_metrics_summary.csv" in f])[0]
        val_summary_df = pd.read_csv(os.path.join(result_dir, match))
        val_metrics["Validation AUC"] = float(val_summary_df["Validation OOF AUC"][0])
        val_metrics["Validation AUPRC"] = float(val_summary_df["Validation OOF AUPRC"][0])
    except:
        print("⚠️ Could not load validation AUC/AUPRC from best_trial summary.")
        
    # --- Add optimized F1 from Optuna study --- 
    study_path = os.path.join(result_dir, "optuna_study.pkl")
    if os.path.exists(study_path):
        try:
            study = joblib.load(study_path)
            val_metrics["Validation Macro F1 (optimized)"] = round(study.best_trial.value, 4)
        except:
            print("⚠️ Could not load Optuna study to get macro F1.") 


    # --- Test Classification Report ---
    test_report = os.path.join(result_dir, f"{dataset_name}_{outcome_name}_test_classification_report_best_trial.csv")
    if os.path.exists(test_report):
        test_metrics.update(load_classification_report(test_report, prefix="Test"))

    # --- Bootstrapped Metrics ---
    bootstrap_path = os.path.join(result_dir, f"{dataset_name}_{outcome_name}_bootstrap_test_metrics.csv")
    if os.path.exists(bootstrap_path):
        boot_df = pd.read_csv(bootstrap_path)
        for _, row in boot_df.iterrows():
            metric = row["metric"]
            test_metrics[f"Test {metric} (mean)"] = round(row["mean"], 4)
            test_metrics[f"Test {metric} (std)"] = round(row["std"], 4)
            test_metrics[f"Test {metric} 95% CI"] = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        try:
            test_metrics["Test ROC AUC"] = boot_df.loc[boot_df['metric'] == 'AUC', 'mean'].values[0]
            test_metrics["Test PR AUC"] = boot_df.loc[boot_df['metric'] == 'AUPRC', 'mean'].values[0]
        except:
            pass 

    # --- Confusion Matrices (flattened) ---
    test_cm_csv = os.path.join(result_dir, f"{dataset_name}_{outcome_name}_test_confusion_matrix_best_trial.csv")
    val_cm_csv = os.path.join(result_dir, f"{dataset_name}_{outcome_name}_val_confusion_matrix_oof.csv")
    if os.path.exists(val_cm_csv):
        val_metrics.update(load_confusion_matrix(val_cm_csv, prefix="Val_CM"))
    if os.path.exists(test_cm_csv):
        test_metrics.update(load_confusion_matrix(test_cm_csv, prefix="Test_CM"))

    # === Combine and Save ===
    ordered_summary = {}
    ordered_summary.update(dataset_info)
    ordered_summary.update(train_metrics)
    ordered_summary.update(val_metrics)
    ordered_summary.update(test_metrics)

    summary_df = pd.DataFrame(list(ordered_summary.items()), columns=["Metric", "Value"])
    output_path = os.path.join(result_dir, "comprehensive_summary.xlsx")
    summary_df.to_excel(output_path, index=False)
    print(f"✅ Summary saved to: {output_path}") 

def find_best_threshold(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr  # Youden's J statistic
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    # Handle the case where the optimal threshold is 1.0, which can be problematic
    if best_threshold >= 1.0:
        return thresholds[best_idx - 1] if best_idx > 0 else 0.999

    return best_threshold
def flatten_leaf_indices(leaf_indices):
    """
    Reshapes the leaf indices output from XGBoost for multi-class models.
    """
    if leaf_indices.ndim == 3:
        n_samples, n_estimators, n_outputs = leaf_indices.shape
        leaf_indices = leaf_indices.reshape(n_samples, n_estimators * n_outputs)
    return leaf_indices

# === PATHS ===
# The single directory where all models AND transformers were saved from the training script
MODEL_DIR = args.model_dir 
TEST_PATHS = args.test_paths  # Assuming a single test path for simplicity
OUTCOME_COL = args.outcome_col
TYPE = args.type
# Columns to drop from the test set before processing
DROP_COLS = ["outcome_tcmr", "outcome_banff", "outcome_abmr", "outcome_rej", "pid"]
for TEST_PATH in TEST_PATHS:
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Test file not found: {TEST_PATH}")
    
    if "15" in TEST_PATH:
        PERCENTAGE = 15.0
    elif "10" in TEST_PATH:
        PERCENTAGE = 10.0
    elif "6.9" in TEST_PATH:
        PERCENTAGE = 6.9
    elif "5" in TEST_PATH:
        PERCENTAGE = 5.0
    elif "3.8" in TEST_PATH:
        PERCENTAGE = 3.8
    else:
        PERCENTAGE = 0.0  # Default to 0% if no match found

    PERCENTAGE_DIR = os.path.join(MODEL_DIR, f"{PERCENTAGE}%")
    if not os.path.exists(PERCENTAGE_DIR):
        os.makedirs(PERCENTAGE_DIR)

    # === STEP 1: LOAD TEST DATA ONLY ===
    print(f"Loading test data from: {TEST_PATH}")
    df_test = pd.read_csv(TEST_PATH)
    y_test = df_test[OUTCOME_COL].values
    X_test = df_test.drop(columns=DROP_COLS, errors="ignore")

    # === STEP 2: LOAD THE ENTIRE FITTED PIPELINE (MODELS AND TRANSFORMERS) ===
    print(f"\n--- Loading fitted pipeline objects from: {MODEL_DIR} ---")

    # Load trained models
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODEL_DIR, f"{TYPE}_{OUTCOME_COL}_xgb_model.bin"))

    # Load the exact feature lists used during training
    numeric_features = joblib.load(os.path.join(MODEL_DIR, 'numeric_features.joblib'))
    high_card_cols = joblib.load(os.path.join(MODEL_DIR, 'high_card_cols.joblib'))
    low_card_cols = joblib.load(os.path.join(MODEL_DIR, 'low_card_cols.joblib'))
    categorical_features = high_card_cols + low_card_cols

    # Load all the fitted transformers
    numeric_transformer = joblib.load(os.path.join(MODEL_DIR, 'numeric_transformer.joblib'))
    cat_imputer = joblib.load(os.path.join(MODEL_DIR, 'cat_imputer.joblib'))
    target_encoder = joblib.load(os.path.join(MODEL_DIR, 'target_encoder.joblib'))
    train_ohe_cols = joblib.load(os.path.join(MODEL_DIR, 'ohe_low_card_cols.joblib'))
    ohe_leaf_encoder = joblib.load(os.path.join(MODEL_DIR, 'ohe_leaf_encoder.joblib'))
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
        exit() # Exit the script if params can't be loaded

    # === STEP 3: APPLY THE FITTED PIPELINE TO TEST DATA (TRANSFORM ONLY, NEVER FIT) ===

    # Sanity check: Ensure test set has all required columns, fill if necessary
    for col in numeric_features + categorical_features:
        if col not in X_test.columns:
            print(f"Warning: Column '{col}' not found in test set. Filling with placeholder.")
            X_test[col] = 0 if col in numeric_features else "missing"
    X_test = X_test[numeric_features + categorical_features] # Ensure correct order

    # 1. Process numeric features using the loaded numeric_transformer
    X_test_numeric = numeric_transformer.transform(X_test[numeric_features])

    # 2. Process categorical features using the loaded transformers
    X_test_cat_imp = pd.DataFrame(cat_imputer.transform(X_test[categorical_features]), columns=categorical_features)
    X_test_high = target_encoder.transform(X_test_cat_imp[high_card_cols])
    X_test_low = pd.get_dummies(X_test_cat_imp[low_card_cols], drop_first=True, dtype=float)
    # Align one-hot encoded columns to match the training set exactly
    X_test_low_ohe = X_test_low.reindex(columns=train_ohe_cols, fill_value=0.0)

    # 3. Assemble the base preprocessed test set
    X_test_processed = np.hstack([X_test_numeric, X_test_high.values, X_test_low_ohe.values])
    X_test_processed = np.nan_to_num(X_test_processed).astype(float)

    # 4. Get XGBoost leaf indices for the test set
    leaf_test_flat = flatten_leaf_indices(xgb_model.apply(X_test_processed))

    # 5. Transform leaves using the loaded leaf encoder
    X_test_leaf_ohe = ohe_leaf_encoder.transform(leaf_test_flat)

    # 6. Assemble the final feature set for the DNN
    X_test_combined = np.hstack([X_test_processed, X_test_leaf_ohe])
    X_test_combined = np.nan_to_num(X_test_combined).astype(float)


    # === STEP 4: DEFINE AND LOAD THE DNN MODEL ===
    class PyTorchDNN(torch.nn.Module):
        def __init__(self, input_dim, num_neurons, dropout_rate, l2_reg, num_classes=2):
            super(PyTorchDNN, self).__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(input_dim, num_neurons),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(num_neurons, num_neurons // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(num_neurons // 2, num_classes),
            )
            self.l2_reg = l2_reg

        def forward(self, x):
            return self.model(x)
        
    def predict_dnn(model, X):
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = model(X_tensor)                
            probs = torch.softmax(logits, dim=1)     
            return probs.numpy()                     

    input_dim = X_test_combined.shape[1]
    print(f"Input dimension for DNN: {input_dim}")

    model = PyTorchDNN(
        input_dim=input_dim,
        num_neurons=best_params['dnn_num_neurons'],
        dropout_rate=best_params['dnn_dropout_rate'],
        l2_reg=best_params['dnn_l2_regularizer']
    )


    # Load the saved model weights. This should now work without error.
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{TYPE}_{OUTCOME_COL}_best_model.pt"), weights_only=True))
    model.eval()
    print("✅ PyTorch DNN model loaded successfully.")

    # === CALIBRATION ===
    wrapped = TorchWrapper(model)
    wrapped.classes_ = np.unique(y_test)     
    wrapped.n_classes_ = len(wrapped.classes_)

    calibrator = CalibratedClassifierCV(estimator=wrapped, method='sigmoid', cv="prefit")
    calibrator.fit(X_test_combined, y_test)



    # === STEP 5: PREDICT AND EVALUATE ===
    print("\n--- Generating predictions on the test set ---")
    with torch.no_grad():
        x_tensor = torch.FloatTensor(X_test_combined)
        y_pred_logits = model(x_tensor)
        y_pred_probs_all = torch.nn.functional.softmax(y_pred_logits, dim=1).numpy()
        y_pred_probs = y_pred_probs_all[:, 1]

        #calibrate the probabilities for the DCA
        calibrator.fit(X_test_combined, y_test)          
        y_pred_probs_cal = calibrator.predict_proba(X_test_combined)[:,1]


    # Use a standard 0.5 threshold for initial classification
    best_threshold = find_best_threshold(y_test, y_pred_probs)
    y_pred = (y_pred_probs >= best_threshold).astype(int)
    y_pred_class = (y_pred_probs >= 0.5).astype(int)  # For comparison with 0.5 threshold

    print(f"\nPredicted Positives: {np.sum(y_pred)} / {len(y_pred)}")
    print(f"Actual Positives:    {np.sum(y_test)} / {len(y_test)}")

    # --- Classification Metrics ---
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

    # --- Plotting ---
    print("\n--- Generating evaluation plots ---")

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)
    avg_prec = average_precision_score(y_test, y_pred_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Avg Precision (AUPRC) = {avg_prec:.3f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve on Test Set")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PERCENTAGE_DIR}/{PERCENTAGE}precision_recall_curve.png")
    plt.close('all')  # Close all plots to free memory

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PERCENTAGE_DIR}/{PERCENTAGE}roc_curve.png")
    plt.close('all')  # Close all plots to free memory

    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Youden Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'])
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig(f"{PERCENTAGE_DIR}/{PERCENTAGE}_youden_confusion_matrix.png")
    plt.close('all')  # Close all plots to free memory


    # Confusion 0.5 Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred_class)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion 0.5 Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'])
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig(f"{PERCENTAGE_DIR}/{PERCENTAGE}_0.5_confusion_matrix.png")
    plt.close('all')  # Close all plots to free memory

    print("--- Evaluation complete. Plots saved to current directory. ---")
        # --- Bootstrap Evaluation ---
    def run_bootstrap(metric_fn, name):
        mean, std, (ci_lower, ci_upper) = bootstrap_evaluation_on_test(
            model, X_test_combined, y_test, metric_fn, B=1000
        )
        return {
            "metric": name,
            "mean": mean,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }

    bootstrap_results = []
    bootstrap_results.append(run_bootstrap(roc_auc_score, "AUC"))
    bootstrap_results.append(run_bootstrap(f1_score, "Macro F1"))
    bootstrap_results.append(run_bootstrap(average_precision_score, "AUPRC")) 

    df_bootstrap = pd.DataFrame(bootstrap_results)
    df_bootstrap.to_csv(f"{PERCENTAGE_DIR}/{TYPE}_{OUTCOME_COL}_bootstrap_test_metrics.csv", index=False)
    print(f"\nBootstrapped metrics saved to {PERCENTAGE_DIR}/{TYPE}_{OUTCOME_COL}_bootstrap_test_metrics.csv")

    # --- DCA (Decision Curve Analysis) ---

    treshholds = np.linspace(0.01, 0.20, 40)   # or whatever window makes clinical sense
    net_benefit_vals = []
    for t in treshholds:
        preds_t = (y_pred_probs_cal >= t).astype(int)
        net_benefit_vals.append(net_benefit(y_test, preds_t, t))

    best_idx    = np.argmax(net_benefit_vals)
    best_thresh = treshholds[best_idx]
    print(f"→ Max net-benefit at Pₜ = {best_thresh:.3f}, NB = {net_benefit_vals[best_idx]:.4f}")

    #Scope of DCA
    delta = 0.15
    lower = max(0.0, best_thresh - delta)
    upper = min(1.0, best_thresh + delta)

    # New thresholds grid
    zoom_ths = np.linspace(lower, upper, 50)

    # Recompute net-benefit on that zoomed grid
    zoom_nb = [net_benefit(y_test, (y_pred_probs_cal>=t).astype(int), t) for t in zoom_ths]

    # Choose y-axis limits just beyond your min/max NB in this window
    ymin = min(zoom_nb) - 0.01
    ymax = max(zoom_nb) + 0.01

    # 1. Create a DataFrame .
    dca_df = pd.DataFrame({
        OUTCOME_COL: y_test,
        'Model': y_pred_probs_cal  
    })

    # 2. Perform Decision Curve Analysis.
    dca_results = dca(
        data=dca_df,
        outcome=OUTCOME_COL,
        modelnames=['Model'],
        thresholds=zoom_ths
    )


    # 3. Plot the graphs and save the figure.
    plot_graphs(
        plot_df=dca_results,
        graph_type='net_benefit',
        y_limits=[ymin, ymax], 
        file_name=f"{PERCENTAGE_DIR}/{TYPE}_{OUTCOME_COL}_dca.png",
        smooth_frac=0.5
    )

    print(f" Decision Curve Analysis plot saved.")
    plt.close('all')  

summarize_results(MODEL_DIR, TYPE, OUTCOME_COL) 

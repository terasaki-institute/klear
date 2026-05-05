# ==============================================================================
# Kidney Transplant Rejection Prediction Pipeline (XGBoost + DNN + VAE+GMM)
# ==============================================================================

""" 
The purpose of this script is to perform binary classification for various kidney transplant rejection outcomes,
using either imputed, low-resolution, or high-resolution HLA typing datasets.
 
Outcomes for Modeling: 
- outcome_tcmr  --> Binary classification (e.g., T-cell mediated rejection vs. no rejection)
- outcome_abmr  --> Binary classification (e.g., antibody-mediated rejection)
- outcome_rej   --> Binary classification (e.g., any rejection vs. none)
- outcome_banff --> Multi-class classification (e.g., multiple Banff rejection grades) 

Modeling Strategy:
- This pipeline combines: 
    - XGBoost: to model tabular data and extract tree-based leaf indices
    - PyTorch DNN: to further learn from raw + XGBoost features
    - Variational Autoencoder (VAE) + Gaussian Mixture Model (GMM): to handle severe class imbalance through synthetic oversampling of minority classes 
    - Focal Loss: improves learning on minority class by down-weighting easy examples 

Pipeline Flow:
1. Load dataset and apply preprocessing
2. Apply VAE+GMM oversampling within each CV fold 
3. Train XGBoost and extract leaf embeddings
4. Concatenate features and train DNN using Focal Loss
5. Tune hyperparameters using Optuna with 5-fold CV
6. Evaluate on test set and save metrics + plots 

All outputs (confusion matrix, AUC, F1-score, feature importances, ROC curves, AUPRC) are saved to the results directory.
"""

# Standard Libraries 
import argparse
import warnings 
import random 
import os 
from tqdm import tqdm # timer bar 
import copy

# Data Science 
import numpy as np
import pandas as pd 
from datetime import datetime

# Data Visualization 
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset 

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, LabelEncoder
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
    average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture 
from sklearn.utils import resample

# Hyperparameter Tuning and Encoding 
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING) # removes repetitive trial printouts but keeps critical errors/warnings
from category_encoders import TargetEncoder 
import joblib 

# Reproducibility 
SEED = 42

# Set seed for standard libraries
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
# Set seed for PyTorch
torch.manual_seed(SEED)

# Set seed for CUDA (if available)
if torch.cuda.is_available():
    print("✅ GPU available. Setting CUDA seeds for reproducibility.")
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print("⚠️ GPU not available. Running on CPU.")

import warnings
warnings.filterwarnings('ignore')

global_study = None  
global_num_trials = None 
global_fixed_feature_names = None  
 
# ==============================================================================
# 1. Dataset Configuration and Loading 
# ==============================================================================
"""
DATASET_NAME = "imputed"  # "imputed" or "low_res" or "high_res" 
OUTCOME_NAME = "outcome_tcmr"  
 
# Set paths based on dataset type 
if DATASET_NAME == "imputed":  
    train_path = "/Users/aaryamecwan/Dropbox/Aarya/MS_Mayo_cor/data/mayo_datasets_03112025/Imputed HLA datasets/mayo_sites_merged_imputed_train.csv"
    test_path = "/Users/aaryamecwan/Dropbox/Aarya/MS_Mayo_cor/data/shared_datasets/lowres_imputed_test_imputed.csv" 
 
elif DATASET_NAME == "low_res":  
    # PC/Dropbox 
    train_path = "/Users/aaryamecwan/Dropbox/Aarya/MS_Mayo_cor/data/mayo_datasets_03112025/Low-res HLA datasets/mayo_sites_merged_lowres_train.csv"
    test_path = "/Users/aaryamecwan/Dropbox/Aarya/MS_Mayo_cor/data/shared_datasets/lowres_highres_test_lowres.csv" 
    # HPC 
    # train_path = "/mnt/md0/Public/ms_mayo_cor/data/mayo_datasets_03112025/Low-res HLA datasets/mayo_sites_merged_lowres_train.csv"
    # test_path  = "/mnt/md0/Public/ms_mayo_cor/data/shared_datasets/lowres_highres_test_lowres.csv"

elif DATASET_NAME == "high_res":   
    # PC/Dropbox 
    train_path = "/Users/aaryamecwan/Dropbox/Aarya/MS_Mayo_cor/data/mayo_datasets_03112025/High-res HLA datasets/mayo_sites_merged_highres_train.csv"
    test_path  = "/Users/aaryamecwan/Dropbox/Aarya/MS_Mayo_cor/data/shared_datasets/lowres_highres_test_highres.csv" 
    # HPC 
    # train_path = "/mnt/md0/Public/ms_mayo_cor/data/mayo_datasets_03112025/High-res HLA datasets/mayo_sites_merged_highres_train.csv"
    # test_path  = "/mnt/md0/Public/ms_mayo_cor/data/shared_datasets/lowres_highres_test_highres.csv"
  
#TOP PERCENT DATASET
if DATASET_PERCENT == "75":
    columns_to_keep = ['don_race', 'don_ethnicity', 'prev_kt', 'don_abo', 'rec_race', 'thymo_ind', 'pc12_tcmr', 'abo', 'pc11', 
                       'pc10_tcmr', 'end_cpra', 'pc3_tcmr', 'rec_age', 'cmv', 'cmv_mm', 'pc6_tcmr', 'kdpi', 'pc13_tcmr', 'pc1_tcmr']
elif DATASET_PERCENT == "90":
    columns_to_keep = ['don_race', 'don_ethnicity', 'prev_kt', 'don_abo', 'rec_race', 'thymo_ind', 'pc12_tcmr', 'abo', 'pc11']
print("\nDataset: ", DATASET_NAME)   
print("Outcome: ", OUTCOME_NAME)   
 
# Output directory 
# RESULT_DIR = f"/mnt/md0/Public/ms_mayo_cor/results/{DATASET_NAME}/{OUTCOME_NAME}" 
# RESULT_DIR = f"/home/aaryamecwan/results/{DATASET_NAME}_{OUTCOME_NAME}"
RESULT_DIR = f"results/{DATASET_NAME}/{OUTCOME_NAME}"
os.makedirs(RESULT_DIR, exist_ok=True)
     
# ------------------------------------------------------------------------------
# Dataset Loading and Preprocessing
# ------------------------------------------------------------------------------

# Load Train Data 
data = pd.read_csv(train_path) 
# data = pd.read_csv("/Users/aaryamecwan/Dropbox/Aarya/MS_Mayo_cor/data/mayo_datasets_03112025/Low-res HLA datasets/mayo_sites_merged_lowres_train.csv")

# Define outcome column to predict 
outcome_col = OUTCOME_NAME

# Extract labels  
y_original = data[outcome_col]  
label_encoder = LabelEncoder() 
y = label_encoder.fit_transform(y_original)
y_count  = len(set(y)) 

# Drop target + other outcomes from training features 
if DATASET_PERCENT == "75" or DATASET_PERCENT == "90":
    X = data[columns_to_keep]
else:
    drop_cols = ["outcome_tcmr", "outcome_banff", "outcome_abmr", "outcome_rej", "pid"] 
    X = data.drop(columns=drop_cols, errors="ignore") 

# Load Test Data 
test_data = pd.read_csv(test_path)  
y_test_original = test_data[outcome_col]
y_test = label_encoder.transform(y_test_original) 
if DATASET_PERCENT == "75" or DATASET_PERCENT == "90":
    X_test = test_data[columns_to_keep]
else:
    X_test = test_data.drop(columns=drop_cols, errors="ignore")


# Identify numeric and categorical features
numeric_features = [col for col in X.columns if X[col].dtype in ["int64", "float64"]]
categorical_features = [col for col in X.columns if X[col].dtype in ["object", "category"]]

# Print class distribution
unique_classes, class_counts = np.unique(y, return_counts=True)
print("\nClass Distribution:")
for cls, count in zip(unique_classes, class_counts):
    label = f"Class {cls}" if y_count > 2 else ("No Rejection" if cls == 0 else "Rejection")
    print(f"{label}: {count} samples") 

print(f"\nTrain set: {X.shape[0]} samples, {X.shape[1]} engineering features")
print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} engineered features") 

"""

# ==============================================================================
# 2. Model Components
# (XGBoost, PyTorch DNN, VAE+GMM Oversampling)
# ==============================================================================

# This section defines how we build, train, make predictions with neural network 
# Use a deep neural network (DNN) as a second-stage classifier that takes in features from the data 
# and additional features from XGBoost (like leaf indices) 
# DNN is trained to classify patient outcomes (e.g., rejection or no rejection) 

# ------------------------------------------------------------------------------
# 2.1 PyTorch DNN 
# ------------------------------------------------------------------------------
class PyTorchDNN(nn.Module): 
    
    # ---- Model Architecture ---- 
    def __init__(self, input_dim, num_neurons, dropout_rate, l2_reg, num_classes=2):
        super(PyTorchDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, num_neurons),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_neurons, num_neurons // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_neurons // 2, num_classes), 
        )
        self.l2_reg = l2_reg

    def forward(self, x): 
        return self.model(x)  
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # weight for minority class
        self.gamma = gamma  # how much to focus on hard examples
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)  # pt = predicted prob for the true class
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss 

# ---- DNN Training ---- 
# Instantiates the model, optimizer, loss function 
# Normally, CrossEntropyLoss penalizes all mistakes equally 
# Focal Loss adds a weighting factor 
def create_dnn(input_dim, num_neurons, dropout_rate, l2_reg, lr, use_focal_loss=False):
    model = PyTorchDNN(input_dim, num_neurons, dropout_rate, l2_reg, num_classes=y_count)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    if use_focal_loss:
        criterion = FocalLoss(alpha=2, gamma=2)  # try alpha=2 for imbalanced class
    else:
        criterion = nn.CrossEntropyLoss() 
    return model, optimizer, criterion

# Trains the model using mini-batch gradient descent
def train_dnn(model, optimizer, criterion, train_loader, val_loader, epochs, patience = 3):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # Learning rate scheduler: Reduces LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=patience//2)

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            
            # Gradient Clipping (for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                val_logits = model(X_val)
                loss = criterion(val_logits, y_val)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Early Stopping and Best Model Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the state of the best model
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break
            
        # --- Learning Rate Scheduler Step ---
        scheduler.step(avg_val_loss)
        
    # Load the best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return model

# ---- DNN Prediction ---- 
# Makes probability predictions using trained DNN 
def predict_dnn(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(X_tensor)                
        probs = torch.softmax(logits, dim=1)     
        return probs.numpy()                     

# Returns raw output (logits) instead of probabilities 
def predict_dnn_logits(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(X_tensor)  
        return logits.numpy()
    
# ------------------------------------------------------------------------------
# 2.2 VAE + GMM for OverSampling 
# ------------------------------------------------------------------------------
# This section combats class imbalance using synthetic data generation 
# It uses VAE to learn compressed representations of minority class samples 
# It uses GMM to model and sample from that compressed space to generate realistic synthetic samples 

# Defines the autoencoder architecture that learns a compressed (latent) space 
class TabularVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8, hidden_dim=64):
        super(TabularVAE, self).__init__()
        self.encoder = nn.Sequential( 
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        ) 
 
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def elbo_loss(self, recon_x, x, mu, logvar):
        # Reconstruction term (MSE here, but you can swap to BCE if you prefer)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        # KL divergence term
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Return summed ELBO; if you want per-sample, divide by batch_size here
        return recon_loss + kld

# Combines reconstruction loss and KL-divergence to train the VAE 
def vae_loss_fn(x_recon, x, mu, logvar):
    recon_loss = nn.MSELoss()(x_recon, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

# Trains the VAE on minority class data 
def train_tabular_vae(model, data_loader, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            batch = batch[0]
            optimizer.zero_grad()
            x_recon, mu, logvar = model(batch)
            loss = vae_loss_fn(x_recon, batch, mu, logvar)
            loss.backward()
            optimizer.step() 
            total_loss += loss.item() 
            
        # if globals().get("PRINT_VAE_LOSS", False) and epoch in [0, epochs // 2, epochs - 1]:
        #     print(f"VAE Epoch {epoch+1}/{epochs}, Loss={total_loss:.4f}")

# Uses the trained VAE decoder to generate synthetic data 
def generate_synthetic_vae(model, n_samples, device='cpu'):
    model.eval()
    with torch.no_grad():
        latent_dim = model.fc_mu.out_features
        z = torch.randn(n_samples, latent_dim).to(device)
        generated = model.decode(z)
    return generated.cpu().numpy()

# Fits a GMM on latent space and samples synthetic data
def train_gmm(data, n_components=5, random_state=42):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=random_state
    )
    gmm.fit(data)
    return gmm 

def sample_gmm(gmm, n_samples):
    synthetic_data, _ = gmm.sample(n_samples)
    return synthetic_data

# ------------------------------------------------------------------------------
# Main function: VAE + GMM Oversampling for Minority Class
# ------------------------------------------------------------------------------
def vae_gmm_oversample(
    X_train,
    y_train,
    X_val,
    y_val,
    minority_class,
    vae_latent_dim=8,
    vae_hidden_dim=64,
    vae_lr=1e-3,
    vae_epochs=50,
    gmm_n_components=5,
    patience=10,
    device='cpu'
):
    """
    X_train, y_train:  training set
    X_val,   y_val:    your held-out validation set
    """
    # 1) Separate minority vs majority in TRAIN
    X_min = X_train[y_train == minority_class]
    X_maj = X_train[y_train != minority_class]
    y_maj = y_train[y_train != minority_class]

    minor_count, major_count = len(X_min), len(X_maj)
    if minor_count == 0 or minor_count >= major_count:
        return X_train, y_train
    needed = major_count - minor_count

    # 2) Build DataLoaders for TRAIN and for YOUR VAL
    tensor_min = torch.tensor(X_min, dtype=torch.float32, device=device)
    train_ds = TensorDataset(tensor_min)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # filter minority examples out of your provided validation set
    X_val_min = X_val[y_val == minority_class]
    tensor_val = torch.tensor(X_val_min, dtype=torch.float32, device=device)
    val_ds = TensorDataset(tensor_val)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 3) VAE + EarlyStopping setup
    vae_model = TabularVAE(
        input_dim = X_train.shape[1],
        latent_dim = vae_latent_dim,
        hidden_dim = vae_hidden_dim
    ).to(device)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=vae_lr)

    class EarlyStopping:
        def __init__(self, patience=15, delta=1e-4):
            self.patience, self.delta = patience, delta
            self.best_loss = float('inf')
            self.counter = 0
            self.should_stop = False

        def __call__(self, val_loss):
            if val_loss + self.delta < self.best_loss:
                self.best_loss, self.counter = val_loss, 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True

    early_stop = EarlyStopping(patience=patience)

    # 4) Training loop with YOUR val_loader
    for epoch in range(vae_epochs):
        vae_model.train()
        for batch, in train_loader:
            recon, mu, logvar = vae_model(batch)
            loss = vae_model.elbo_loss(recon, batch, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation pass
        vae_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch, in val_loader:
                recon, mu, logvar = vae_model(batch)
                val_loss += vae_model.elbo_loss(recon, batch, mu, logvar).item()
        val_loss /= len(val_loader)
        print(f"[VAE] Epoch {epoch:02d} — val ELBO: {val_loss:.4f}")

        early_stop(val_loss)
        if early_stop.should_stop:
            print(f"[VAE] Early stopping at epoch {epoch:02d}")
            break

    # 5) Generate & GMM‐resample as before
    vae_synth_count = minor_count * 3
    vae_synthetic = generate_synthetic_vae(vae_model, vae_synth_count, device=device)

    combined_min = np.vstack([X_min, vae_synthetic])
    gmm = train_gmm(combined_min, n_components=gmm_n_components, random_state=SEED)
    final_synth = sample_gmm(gmm, needed)

    X_res = np.vstack([X_train, final_synth])
    y_res = np.concatenate([y_train, np.full(needed, minority_class, dtype=int)])
    return X_res, y_res

# ==============================================================================
# 3. Training Helpers 
# ==============================================================================

# Flattens the 3D leaf index output from XGBoost for DNN input 
def flatten_leaf_indices(leaf_indices):
    if leaf_indices.ndim == 3:
        n_samples, n_estimators, n_outputs = leaf_indices.shape
        leaf_indices = leaf_indices.reshape(n_samples, n_estimators * n_outputs)
    return leaf_indices


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
    model_path = os.path.join(RESULT_DIR, f"{DATASET_NAME}_{OUTCOME_NAME}_best_model.pt")
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

            if y_count == 2:
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

# ==============================================================================
# 4. Optuna Objective and Callbacks
# ==============================================================================

# --------------------------------------------------------------------- 
# 4.1 Objective Function with Strict 5-Fold CV 
# ---------------------------------------------------------------------
def objective(trial, result_dir): 
    
    global X_test 
    # Print trial progress 
    print(f"\n Running Trial {trial.number + 1}/{num_trials}...") 

    # --- Suggest hyperparameters --- 
    xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 100, 500, step=100)
    xgb_max_depth = trial.suggest_int("xgb_max_depth", 3, 8) 
    xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.001, 0.3, log=True)
    xgb_subsample = trial.suggest_float("xgb_subsample", 0.5, 1.0)
    xgb_reg_lambda = trial.suggest_float("xgb_reg_lambda", 0.01, 10.0, log=True)
    xgb_early_stopping_rounds = trial.suggest_int("xgb_early_stopping_rounds", 2, 10, step=2)
    dnn_lr = trial.suggest_float("dnn_lr", 1e-5, 1e-2, log=True)
    dnn_l2_regularizer = trial.suggest_float("dnn_l2_regularizer", 1e-6, 1e-2, log=True)
    dnn_num_neurons = trial.suggest_int("dnn_num_neurons", 32, 256, step=32)
    dnn_epoch = trial.suggest_int("dnn_epoch", 20, 40, step=10)
    dnn_dropout_rate = trial.suggest_float("dnn_dropout_rate", 0.1, 0.6, step=0.1)
    batch_size = trial.suggest_categorical("dnn_batch_size", [32, 64, 128])
    dnn_early_stopping_rounds = trial.suggest_int("dnn_early_stopping_rounds", 1, 5, step=1)

    vae_latent_dim = trial.suggest_int("vae_latent_dim", 4, 16, step=4)
    vae_hidden_dim = trial.suggest_int("vae_hidden_dim", 32, 128, step=32)
    vae_lr         = trial.suggest_float("vae_lr", 1e-4, 1e-2, log=True)
    vae_epochs     = trial.suggest_int("vae_epochs", 50, 200, step=50)
    vae_early_stopping_rounds = trial.suggest_int("vae_early_stopping_rounds", 2, 10, step=2)
    gmm_n_components = trial.suggest_int("gmm_n_components", 2, 12, step=1)

    # --- Tracking objects ---
    oof_probs = np.zeros((len(y), y_count), dtype=float)
    oof_labels = y
    fold_importances_norm = []
    fold_importances_raw = []
    cv_results = []
    cv_results1 = []

        # --- 5-Fold Stratified CV ---
    # StratifiedKold ensures each fold of cross-validation maintains original class distribution 
    # guaranteeing each training/validation split has a similar class balance 
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        
        # ------------------------------
        # (A) Split and Preprocess Fold
        # ------------------------------
        # Split into training and validation sets 
        X_train_fold = X.iloc[train_idx].copy().reset_index(drop=True)
        X_val_fold   = X.iloc[val_idx].copy().reset_index(drop=True)
        y_train_fold = y[train_idx] 
        y_val_fold   = y[val_idx]

        # If there's a transplant date column, extract year/month/day and drop it
        # (needed for date-aware models or keeping time consistent)
        # Handle 'Date_of_Transplant' in train and val
        for df in [X_train_fold, X_val_fold]:
            if 'Date_of_Transplant' in df.columns:
                df['Date_of_Transplant'] = pd.to_datetime(df['Date_of_Transplant'], errors='coerce')
                df['Transplant_Year'] = df['Date_of_Transplant'].dt.year
                df['Transplant_Month'] = df['Date_of_Transplant'].dt.month
                df['Transplant_Day'] = df['Date_of_Transplant'].dt.day
                df.drop('Date_of_Transplant', axis=1, inplace=True)

        # Identify numeric and categorical features 
        fold_numeric_feats = [col for col in X_train_fold.columns if X_train_fold[col].dtype in [np.float64, np.int64]]
        fold_categorical_feats = [col for col in X_train_fold.columns if X_train_fold[col].dtype == object or str(X_train_fold[col].dtype) == 'category']

        # Numeric preprocessing: Impute missing + Standard Scale 
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        X_train_fold_numeric = numeric_transformer.fit_transform(X_train_fold[fold_numeric_feats])
        X_val_fold_numeric   = numeric_transformer.transform(X_val_fold[fold_numeric_feats]) 
        
        # Categorical preprocessing: Impute missing with mode (most frequent) 
        if len(fold_categorical_feats) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_train_fold_cat = pd.DataFrame(cat_imputer.fit_transform(X_train_fold[fold_categorical_feats]), columns=fold_categorical_feats)
            X_val_fold_cat   = pd.DataFrame(cat_imputer.transform(X_val_fold[fold_categorical_feats]), columns=fold_categorical_feats)

            # 3. Split categorical into high-cardinality vs low-cardinality
            high_card_cols = [col for col in fold_categorical_feats if X_train_fold_cat[col].nunique() > 10]
            low_card_cols  = [col for col in fold_categorical_feats if X_train_fold_cat[col].nunique() <= 10]

            # Encode high-cardinality features with TargetEncoder 
            if len(high_card_cols) > 0:
                te = TargetEncoder(cols=high_card_cols) 
                X_train_fold_high = te.fit_transform(X_train_fold_cat[high_card_cols], y_train_fold)
                X_val_fold_high   = te.transform(X_val_fold_cat[high_card_cols])
            else:
                X_train_fold_high = np.empty((X_train_fold.shape[0], 0))
                X_val_fold_high   = np.empty((X_val_fold.shape[0], 0))

            # Encode low-cardinality features with OneHotEncoder 
            if len(low_card_cols) > 0:
                X_train_fold_low = X_train_fold_cat[low_card_cols]
                X_val_fold_low   = X_val_fold_cat[low_card_cols]
                X_train_fold_low_ohe = pd.get_dummies(X_train_fold_low, drop_first=True)
                X_val_fold_low_ohe   = pd.get_dummies(X_val_fold_low, drop_first=True)

                # Align validation set one-hot columns to training set columns 
                X_val_fold_low_ohe = X_val_fold_low_ohe.reindex(columns=X_train_fold_low_ohe.columns, fill_value=0)
                train_ohe_cols = X_train_fold_low_ohe.columns
            else:
                X_train_fold_low_ohe = pd.DataFrame()
                X_val_fold_low_ohe   = pd.DataFrame()
                train_ohe_cols = pd.Index([])

            # Record column names 
            high_card_encoded_names = [f"{col}_te" for col in high_card_cols]
            low_card_encoded_names = train_ohe_cols.tolist()
        else:
            # No categorical features
            X_train_fold_high = np.empty((X_train_fold.shape[0], 0))
            X_val_fold_high   = np.empty((X_val_fold.shape[0], 0))
            X_train_fold_low_ohe = pd.DataFrame()
            X_val_fold_low_ohe   = pd.DataFrame()
            high_card_encoded_names = []
            low_card_encoded_names = []
            train_ohe_cols = pd.Index([])

        numeric_feature_names = fold_numeric_feats
        combined_feature_names_fold = (
            numeric_feature_names
            + high_card_encoded_names
            + low_card_encoded_names
        )

        # Final preprocessed matrices for training and validation 
        if len(fold_categorical_feats) > 0:
            X_train_fold_pre = np.hstack([
                X_train_fold_numeric,
                X_train_fold_high.values,
                X_train_fold_low_ohe.values
            ])
            X_val_fold_pre = np.hstack([
                X_val_fold_numeric,
                X_val_fold_high.values,
                X_val_fold_low_ohe.values
            ])
        else:
            X_train_fold_pre = X_train_fold_numeric
            X_val_fold_pre = X_val_fold_numeric

        X_train_fold_pre = np.nan_to_num(X_train_fold_pre).astype(float)
        X_val_fold_pre   = np.nan_to_num(X_val_fold_pre).astype(float)
        
        # ------------------------------
        # Preprocess the Test Set 
        # ------------------------------
        
        # Ensure test has all numeric and categorical columns 
        missing_numeric = set(numeric_features) - set(X_test.columns)
        for col in missing_numeric: 
            X_test[col] = 0 # Fill missing numeric columns with 0 
        
        if len(categorical_features) > 0:
            missing_categorical = set(categorical_features) - set(X_test.columns)
            for col in missing_categorical:
                X_test[col] = "missing" # Fill missing categorical columns with a placeholder 

        # Reorder numeric columns to match training 
        X_test_numeric = numeric_transformer.transform(X_test[numeric_features])

        # Reorder categorical columns to match training 
        if len(categorical_features) > 0:
            X_test_cat = pd.DataFrame(cat_imputer.transform(X_test[categorical_features]), columns = categorical_features)
            
            # Apply encoders (TargetEncoder + OneHotEncoder)
            if len(high_card_cols) > 0:
                X_test_high = te.transform(X_test_cat[high_card_cols])  # Target encoded high-cardinality 
            else:
                X_test_high = np.empty((X_test.shape[0], 0))
                
            if len(low_card_cols) > 0:
                X_test_low = X_test_cat[low_card_cols]                  # Low-cardinality categorical columnns 
                X_test_low_ohe = pd.get_dummies(X_test_low, drop_first=True) # One-hot encode 
                
                # Align one-hot encoded columns with training 
                X_test_low_ohe = X_test_low_ohe.reindex(columns=train_ohe_cols, fill_value=0)
            else:
                X_test_low_ohe = pd.DataFrame()
        else:
            X_test_high = np.empty((X_test.shape[0], 0))
            X_test_low_ohe = pd.DataFrame()
 
        # Final assembled test matrix 
        if len(categorical_features) > 0:
            X_test_pre = np.hstack([ 
                X_test_numeric,
                X_test_high.values,
                X_test_low_ohe.values
            ]) 
        else:
            X_test_pre = X_test_numeric
        X_test_pre = np.nan_to_num(X_test_pre).astype(float)
        
        # ------------------------------
        # (B) Apply VAE + GMM Oversampling
        # ------------------------------
        # this is applied within each fold to oversample the minority class in the training set 
        # here we ensure that in each training fold, we're generating synthetic data for class 1 (minority) 
        mc = set(y)
        mc = list(mc)
        mc.remove(0) # 0 is the majority class
        X_res, y_res = X_train_fold_pre, y_train_fold
        
        for minority_class in  mc:
            # print(f"[Fold {fold_idx}] Using VAE+GMM oversampling on class={minority_class}.")
            X_res, y_res = vae_gmm_oversample(
                X_train_fold_pre,
                y_train_fold,
                X_val_fold_pre,
                y_val_fold,
                minority_class=minority_class,
                vae_latent_dim=vae_latent_dim,
                vae_hidden_dim=vae_hidden_dim,
                vae_lr=vae_lr,
                vae_epochs=vae_epochs,
                gmm_n_components=gmm_n_components,
                device='cpu', # changed from cuda to cpu,
                patience=vae_early_stopping_rounds,
            )
        # ------------------------------
        # (C) Train XGBoost + Extract Feature Importance
        # ------------------------------
        
        # Add class weighting for imbalance
        scale_pos_weight_val = (y_res == 0).sum() / (y_res == 1).sum()

        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=y_count,  
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            subsample=xgb_subsample,
            reg_lambda=xgb_reg_lambda,
            use_label_encoder=False,
            eval_metric='mlogloss',
            early_stopping_rounds=xgb_early_stopping_rounds,
            # scale_pos_weight=scale_pos_weight_val, 
            random_state=SEED
        )
        xgb_model.fit(X_res, y_res, eval_set=[(X_val_fold_pre, y_val_fold)], verbose=False)

        booster = xgb_model.get_booster()
        feature_importances_norm = xgb_model.feature_importances_
        raw_importances_dict = booster.get_score(importance_type="total_gain")
        raw_imports_list = [
            raw_importances_dict.get(f"f{i}", 0.0)
            for i in range(len(combined_feature_names_fold))
        ]
 
        # fold_importances_norm.append(feature_importances_norm)
        # Pad or trim to match length of combined_feature_names_fold
        fi_len = len(combined_feature_names_fold)
        feature_importances_norm = feature_importances_norm[:fi_len] if len(feature_importances_norm) >= fi_len else np.pad(
            feature_importances_norm, (0, fi_len - len(feature_importances_norm)), constant_values=0)
        
        raw_imports_list = raw_imports_list[:fi_len] if len(raw_imports_list) >= fi_len else raw_imports_list + [0.0] * (fi_len - len(raw_imports_list))
        
        global global_fixed_feature_names 

        if global_fixed_feature_names is None:
            global_fixed_feature_names = combined_feature_names_fold
        else:
            # Re-align fold's features with global feature set
            # Pad or trim as needed
            current_len = len(combined_feature_names_fold)
            fixed_len = len(global_fixed_feature_names)

            if current_len < fixed_len:
                # Pad feature importances to match
                pad_size = fixed_len - current_len
                feature_importances_norm = np.pad(feature_importances_norm, (0, pad_size), constant_values=0)
                raw_imports_list += [0.0] * pad_size
                # Optionally: print a debug statement
            elif current_len > fixed_len:
                # Truncate feature importances to match
                feature_importances_norm = feature_importances_norm[:fixed_len]
                raw_imports_list = raw_imports_list[:fixed_len]
                # Optionally: print a debug statement

        fold_importances_norm.append(feature_importances_norm)
        fold_importances_raw.append(raw_imports_list)

        # The leaf indices from XGBoost are used as additioanl features for the DNN 
        # XGBoost and DNN are combined at the feature level 
        
        # XGBClassifier().apply() returns the leaf node index for each sample and each tree 
        X_train_leaf = flatten_leaf_indices(xgb_model.apply(X_res)) 
        X_val_leaf   = flatten_leaf_indices(xgb_model.apply(X_val_fold_pre))

        # The leaf indices are categorical - one-hot encode them 
        ohe_leaf = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_train_leaf_ohe = ohe_leaf.fit_transform(X_train_leaf)
        X_val_leaf_ohe   = ohe_leaf.transform(X_val_leaf)
        
        # ------------------------------
        # (D) Train PyTorch DNN
        # ------------------------------

        # concatenate them with original input features 
        X_train_combined = np.hstack([X_res, X_train_leaf_ohe]) 
        X_val_combined   = np.hstack([X_val_fold_pre, X_val_leaf_ohe])

        # resulting combined matrix is passed to DNN to learn final representations and output probabilities 
        X_train_combined = np.nan_to_num(X_train_combined).astype(float) # X_res + XGBoost leaf nodes (used for training the model)
        X_val_combined   = np.nan_to_num(X_val_combined).astype(float)

        input_dim = X_train_combined.shape[1]
        
        model, optimizer, criterion = create_dnn(
            input_dim,
            dnn_num_neurons,
            dnn_dropout_rate,
            dnn_l2_regularizer,
            dnn_lr, 
            use_focal_loss=True # Enables focal loss for imbalanced classification 
        )

        train_dataset = TensorDataset(
            torch.tensor(X_train_combined, dtype=torch.float32),
            torch.tensor(y_res, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_combined, dtype=torch.float32),
            torch.tensor(y_val_fold, dtype=torch.long) 
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        train_dnn(model, optimizer, criterion, train_loader, val_loader, epochs=dnn_epoch, patience=dnn_early_stopping_rounds)
        
        # ------------------------------
        # (E) Predict on Validation Fold
        # ------------------------------

        val_probs = predict_dnn(model, X_val_combined)      
        val_logits = predict_dnn_logits(model, X_val_combined)

        oof_probs[val_idx] = val_probs  
        
        # ------------------------------
        # (F) Track Fold Results
        # ------------------------------

        fold_results_df = pd.DataFrame({
            "Fold": fold_idx,
            "Patient_Index": X.index[val_idx],
            "TrueLabel": y_val_fold,
        })
        for c in range(y_count):
            fold_results_df[f"Prob_Class{c}"] = val_probs[:, c]

        cv_results.append(fold_results_df)

        fold_results_logit = pd.DataFrame({
            "Fold": fold_idx,
            "Patient_Index": X.index[val_idx],
            "TrueLabel": y_val_fold,
        })
        for c in range(y_count):
            fold_results_logit[f"Logit_Class{c}"] = val_logits[:, c]

        cv_results1.append(fold_results_logit)
        
        #--------------------- End of CV ---------------------  
        
    # ------------------------------------------------------------
    # After CV: Evaluate OOF Performance and save Results
    # ------------------------------------------------------------

    # Calculate AUC from out-of-fold probabilities 
    if y_count == 2:
        oof_auc = roc_auc_score(oof_labels, oof_probs[:, 1])  # binary classification
        oof_auprc = average_precision_score(oof_labels, oof_probs[:, 1])
    else:
        oof_auc = roc_auc_score(oof_labels, oof_probs, multi_class='ovr', average='macro')  # multiclass
        oof_auprc = average_precision_score(oof_labels, oof_probs, average='macro')
        
    # Sanity check: All folds must produce feature importances of same length 
    assert all(len(f) == len(global_fixed_feature_names) for f in fold_importances_norm), f"Inconsistent lengths even after fix! Expected {len(global_fixed_feature_names)}."

    # Average feature importances across folds 
    avg_importances_norm = np.mean(fold_importances_norm, axis=0)
    avg_importances_raw  = np.mean(fold_importances_raw, axis=0)

    # Save pre-trial metadata 
    trial.set_user_attr("val_labels", oof_labels)
    trial.set_user_attr("val_probs", oof_probs)
    trial.set_user_attr("oof_auc", oof_auc) # ADDED: Store OOF AUC
    trial.set_user_attr("oof_auprc", oof_auprc) # ADDED: Store OOF AUPRC
    trial.set_user_attr("avg_importances_norm", avg_importances_norm)
    trial.set_user_attr("avg_importances_raw", avg_importances_raw)
    trial.set_user_attr("final_feature_names", combined_feature_names_fold)

    # Combine and save per-fold risk scores and logits
    df_cv_results = pd.concat(cv_results, axis=0).sort_values("Patient_Index")
    trial.set_user_attr("cv_results", df_cv_results)
    df_cv_results1 = pd.concat(cv_results1, axis=0).sort_values("Patient_Index")
    trial.set_user_attr("cv_results1", df_cv_results1) 
    
    # Generate and save OOF confusion matrix 
    oof_preds = np.argmax(oof_probs, axis=1)
    cm_val = confusion_matrix(oof_labels, oof_preds, labels=list(range(y_count))) 
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Purples')
    plt.title("Confusion Matrix - 5-Fold CV (OOF)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{result_dir}/{DATASET_NAME}_{OUTCOME_NAME}_val_confusion_matrix_oof.png", dpi=300)
    plt.close()
    
    # Generate and save OOF classification report 
    val_report_dict = classification_report(oof_labels, oof_preds, output_dict=True)
    val_report_df = pd.DataFrame(val_report_dict).transpose()
    val_report_df.to_csv(f"{result_dir}/{DATASET_NAME}_{OUTCOME_NAME}_val_classification_report_oof.csv")

    # Can return macro-F1 here if you prefer it over AUC
    # macro_f1 = val_report_dict["macro avg"]["f1-score"]
    # return macro_f1

    #return oof_auc 
   # Calculate F1-score from out-of-fold predictions
    oof_preds = np.argmax(oof_probs, axis=1) if y_count > 2 else (oof_probs[:, 1] >= 0.5).astype(int)
    macro_f1 = f1_score(oof_labels, oof_preds, average='macro')
    print(f"[Trial {trial.number}] Macro F1-score (OOF): {macro_f1:.4f}") 

    return macro_f1

def find_best_threshold(y_true, probs, rejection_positive_class=1): 
    """
    Decide optimal threshold based on class balance.
    
    If positive class is very rare (<10%), use F1-optimal threshold.
    Else, use Youden's J statistic.
    
    Args:
        y_true: true binary labels (0/1)
        probs: model predicted probabilities for positive class
        rejection_positive_class: which class index represents 'Rejection'
    
    Returns:
        best_threshold: float
        method: str ("Youden" or "F1")
    """
    positive_ratio = np.mean(y_true == rejection_positive_class)

    if positive_ratio < 0.10:
        # Positive class is <10% of data -> use F1-optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_true, probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        return best_threshold, "F1-optimal"
    
    else:
        # Use Youden's J statistic
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        return best_threshold, "Youden"
    
    # ---------------------------------------------------------------------
# 4.3 Callback to Plot ROC, Confusion Matrix & Feature Importances 
# ---------------------------------------------------------------------
def callback_plot_roc_cm(study, trial, result_dir):

    if study.best_trial.number == trial.number:
        y_val = trial.user_attrs["val_labels"]
        y_pred_prob = trial.user_attrs["val_probs"]
        new_best_auc = trial.user_attrs["oof_auc"] # Get from user_attrs
        new_best_auprc = trial.user_attrs["oof_auprc"] # ADDED: Get AUPRC from user_attrs

        avg_importances_norm = trial.user_attrs["avg_importances_norm"]
        avg_importances_raw  = trial.user_attrs["avg_importances_raw"]
        final_feature_names  = trial.user_attrs["final_feature_names"]
        
        if y_pred_prob.shape[1] == 2:  # Binary classification
            new_best_auc = roc_auc_score(y_val, y_pred_prob[:, 1])
        else:
            new_best_auc = roc_auc_score(y_val, y_pred_prob, multi_class='ovr', average='macro')
        
        print(f"\n[Callback] New best trial #{trial.number}, AUC = {new_best_auc:.4f}, AUPRC = {new_best_auprc:.4f}") 

        plt.figure(figsize=(8,6))

        if y_pred_prob.shape[1] == 2:  # Binary ROC Curve
            fpr, tpr, _ = roc_curve(y_val, y_pred_prob[:, 1])
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC={auc_score:.4f})")
        else:
            lb = LabelBinarizer()
            lb.fit(y_val)  
            y_val_binarized = lb.transform(y_val)  

            class_fpr = {}
            class_tpr = {}
            for c in range(y_pred_prob.shape[1]):
                fpr_c, tpr_c, _ = roc_curve(y_val_binarized[:, c], y_pred_prob[:, c])
                class_fpr[c] = fpr_c
                class_tpr[c] = tpr_c
                plt.plot(fpr_c, tpr_c, label=f"Class {c} vs Rest")

            all_fpr = np.unique(np.concatenate([class_fpr[c] for c in class_fpr]))
            mean_tpr = np.zeros_like(all_fpr)
            for c in class_fpr:
                mean_tpr += np.interp(all_fpr, class_fpr[c], class_tpr[c])
            mean_tpr /= len(class_fpr)
            macro_auc = auc(all_fpr, mean_tpr)

            plt.plot(all_fpr, mean_tpr, label=f"Macro-average (AUC={macro_auc:.4f})", color='black', linestyle='--', linewidth=2) 
        
        # ROC curve (fold average) for a specific Optuna trial 
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves - Trial {trial.number} (AUC={new_best_auc:.4f})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{result_dir}/{DATASET_NAME}_{OUTCOME_NAME}_val_roc_curve_trial_{trial.number}.png", dpi=300)
        plt.close()
        
        # ADDED -------------------------- 
        y_pred_class = np.argmax(y_pred_prob, axis=1) if y_pred_prob.shape[1] > 1 else (y_pred_prob[:, 1] > 0.5).astype(int)
        cm = confusion_matrix(y_val, y_pred_class, labels=list(range(y_pred_prob.shape[1])))
        # --------------------------------

        plt.figure(figsize=(6,4)) 
        sns.heatmap( 
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f"Pred={i}" for i in range(y_count)],
            yticklabels=[f"True={i}" for i in range(y_count)]
        )
        # Confusion matrix for same trial 
        plt.title(f"Confusion Matrix - Trial {trial.number}")
        plt.xlabel("Predicted")  
        plt.ylabel("Actual") 
        plt.tight_layout()
        plt.savefig(f"{result_dir}/cm_trial_{trial.number}.png", dpi=300)
        plt.close()

        # Ensure feature names and importances are the same length
        min_len = min(len(final_feature_names), len(avg_importances_norm), len(avg_importances_raw))
        if len(final_feature_names) != len(avg_importances_norm) or len(final_feature_names) != len(avg_importances_raw):
            print(f"Warning: feature_names ({len(final_feature_names)}), norm ({len(avg_importances_norm)}), raw ({len(avg_importances_raw)}) length mismatch. Truncating to {min_len}.")
        final_feature_names = final_feature_names[:min_len]
        avg_importances_norm = avg_importances_norm[:min_len]
        avg_importances_raw = avg_importances_raw[:min_len]

        # Feature importances (normalized)
        df_norm = pd.DataFrame({"feature": final_feature_names, "importance_norm": avg_importances_norm})
        df_norm = df_norm.sort_values("importance_norm", ascending=False)
        df_norm["rank"] = range(1, len(df_norm) + 1)
        df_norm.to_csv(f"{result_dir}/avg_importances_normalized_trial_{trial.number}.csv", index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_norm, x="importance_norm", y="feature", color="skyblue")
        plt.title(f"Normalized Feature Importances - Trial {trial.number}")
        plt.tight_layout()
        plt.savefig(f"{result_dir}/feature_importances_normalized_trial_{trial.number}.png", dpi=300)
        plt.close()

        # Feature importances (raw)
        df_raw = pd.DataFrame({"feature": final_feature_names, "importance_raw": avg_importances_raw})
        df_raw = df_raw.sort_values("importance_raw", ascending=False)
        df_raw["rank"] = range(1, len(df_raw) + 1)
        df_raw.to_csv(f"{result_dir}/avg_importances_raw_trial_{trial.number}.csv", index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_raw, x="importance_raw", y="feature", color="salmon")
        plt.title(f"Raw Feature Importances - Trial {trial.number}")
        plt.tight_layout()
        plt.savefig(f"{result_dir}/feature_importances_raw_trial_{trial.number}.png", dpi=300)
        plt.close()

        # Out-of-fold validation probabilities 
        df_cv_results = trial.user_attrs["cv_results"]
        df_cv_results.to_csv(
            f"{result_dir}/cv_risk_scores_trial_{trial.number}.csv",
            index=False
        )
        print(f"OOF risk scores saved to: result_dir/cv_risk_scores_trial_{trial.number}.csv")
        
def plot_best_trial_summary(best_trial, result_dir):
    
    # print("\n[Final Summary] Saving plots and reports for the best trial only...")

    trial_number = best_trial.number
    y_val = best_trial.user_attrs["val_labels"]
    y_pred_prob = best_trial.user_attrs["val_probs"]
    final_feature_names = best_trial.user_attrs["final_feature_names"]
    avg_importances_norm = best_trial.user_attrs["avg_importances_norm"]
    avg_importances_raw  = best_trial.user_attrs["avg_importances_raw"]
    best_val_auc = best_trial.user_attrs.get("oof_auc", best_trial.value) 
    best_val_auprc = best_trial.user_attrs.get("oof_auprc", "N/A") 

    # Ensure all arrays are the same length to avoid ValueError
    min_len = min(len(final_feature_names), len(avg_importances_norm), len(avg_importances_raw))
    if len(final_feature_names) != len(avg_importances_norm) or len(final_feature_names) != len(avg_importances_raw):
        print(f"Warning: feature_names ({len(final_feature_names)}), norm ({len(avg_importances_norm)}), raw ({len(avg_importances_raw)}) length mismatch. Truncating to {min_len}.")
    final_feature_names = final_feature_names[:min_len]
    avg_importances_norm = avg_importances_norm[:min_len]
    avg_importances_raw = avg_importances_raw[:min_len]

    print(f"\n--- Best Trial Summary (Trial #{trial_number}) ---")
    print(f"  Validation OOF AUC: {best_val_auc:.4f}")
    if isinstance(best_val_auprc, float):
        print(f"  Validation OOF AUPRC: {best_val_auprc:.4f}") 
    else:
        print(f"  Validation OOF AUPRC: {best_val_auprc}")


    y_pred_class = np.argmax(y_pred_prob, axis=1)

    # ROC Curve 
    plt.figure(figsize=(8, 6))
    if y_pred_prob.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_val, y_pred_prob[:, 1])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC={auc_score:.4f})")
    else:
        # multi-class (optional if you're not doing multi) 
        for i in range(y_pred_prob.shape[1]):
            y_val_binary = np.where(y_val == i, 1, 0)

            fpr, tpr, thresholds = roc_curve(y_val_binary, y_pred_prob[:, i])
            auc_score = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"Class {i} ROC Curve (AUC={auc_score:.4f})")
    
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Test ROC Curve - Best Trial")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{result_dir}/roc_curve_trial_{trial_number}.png", dpi=300)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred_class)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - Trial {trial_number}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{result_dir}/cm_trial_{trial_number}.png", dpi=300)
    plt.close()

    # Feature importances (normalized)
    df_norm = pd.DataFrame({"feature": final_feature_names, "importance_norm": avg_importances_norm})
    df_norm = df_norm.sort_values("importance_norm", ascending=False)
    df_norm["rank"] = range(1, len(df_norm) + 1)
    df_norm.to_csv(f"{result_dir}/avg_importances_normalized_trial_{trial_number}.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_norm, x="importance_norm", y="feature", color="skyblue")
    plt.title(f"Normalized Feature Importances - Trial {trial_number}")
    plt.tight_layout()
    plt.savefig(f"{result_dir}/feature_importances_normalized_trial_{trial_number}.png", dpi=300)
    plt.close()

    # Feature importances (raw)
    df_raw = pd.DataFrame({"feature": final_feature_names, "importance_raw": avg_importances_raw})
    df_raw = df_raw.sort_values("importance_raw", ascending=False)
    df_raw["rank"] = range(1, len(df_raw) + 1)
    df_raw.to_csv(f"{result_dir}/avg_importances_raw_trial_{trial_number}.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_raw, x="importance_raw", y="feature", color="salmon")
    plt.title(f"Raw Feature Importances - Trial {trial_number}")
    plt.tight_layout()
    plt.savefig(f"{result_dir}/feature_importances_raw_trial_{trial_number}.png", dpi=300)
    plt.close()

  
    # Save OOF predictions
    best_trial.user_attrs["cv_results"].to_csv(f"{result_dir}/cv_risk_scores_trial_{trial_number}.csv", index=False)

    val_metrics = {
        "Best Trial Number": trial_number,
        "Validation OOF AUC": best_val_auc,
        "Validation OOF AUPRC": best_val_auprc
    }
    val_metrics_df = pd.DataFrame([val_metrics])
    val_metrics_df.to_csv(os.path.join(result_dir, f"best_trial_{trial_number}_validation_metrics_summary.csv"), index=False)
    print(f"Best trial validation metrics saved to: {result_dir}/best_trial_{trial_number}_validation_metrics_summary.csv")

    print(f"Final best trial visualizations and reports saved for trial #{trial_number}")

# ==============================================================================
# 5. Final Test Set Evaluation
# ==============================================================================

# This function evaluates the best trial on the test set 
# def evaluate_best_trial_on_test(best_trial, result_dir):
def evaluate_best_trial_on_test(best_trial, RESULT_DIR, X, y, X_test, y_test, numeric_features, categorical_features): 
     
    print(f"\n🚀 Retraining best model (Trial #{best_trial.number}) with validation AUC = {best_trial.value:.4f}")
    
    print("\n=== Evaluating Best Trial on Test Set ===")
     
    # Extract best params
    params = best_trial.params

    # Preprocess training data again for final training 
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_imputer = SimpleImputer(strategy='most_frequent') 
    
    # Ensure test set has all training numeric features
    missing_test_numeric = set(numeric_features) - set(X_test.columns)
    for col in missing_test_numeric:
        X_test[col] = 0  # or np.nan if you prefer imputation
    print("Saving feature lists")
    joblib.dump(numeric_features, os.path.join(RESULT_DIR, 'numeric_features.joblib'))
    joblib.dump(categorical_features, os.path.join(RESULT_DIR, 'categorical_features.joblib'))

    # # Then reorder just to be safe
    # X_test = X_test[numeric_features]
    
    # if len(numeric_features) == 0:
    #     raise ValueError("❌ No numeric features found in test set after filtering. Check your column names or update feature lists.")
    # print("[DEBUG] Categorical features missing from X_test:")
    # print([col for col in categorical_features if col not in X_test.columns])
    
    # Ensure all expected categorical columns are present in X_test
    if len(categorical_features) > 0:
        for col in categorical_features:
            if col not in X_test.columns:
                X_test[col] = "missing"  # or np.nan, but "missing" is safer for categorical
    
    X_numeric = numeric_transformer.fit_transform(X[numeric_features])
    X_test_numeric = numeric_transformer.transform(X_test[numeric_features])
    joblib.dump(numeric_transformer, os.path.join(RESULT_DIR, 'numeric_transformer.joblib'))

    if len(categorical_features) > 0:
        X_cat = pd.DataFrame(cat_imputer.fit_transform(X[categorical_features]), columns=categorical_features)
        X_test_cat = pd.DataFrame(cat_imputer.transform(X_test[categorical_features]), columns=categorical_features)
        joblib.dump(cat_imputer, os.path.join(RESULT_DIR, 'cat_imputer.joblib'))

        # Encode
        high_card_cols = [col for col in categorical_features if X_cat[col].nunique() > 10] 
        low_card_cols  = [col for col in categorical_features if X_cat[col].nunique() <= 10]
        joblib.dump(high_card_cols, os.path.join(RESULT_DIR, 'high_card_cols.joblib')) # <-- SAVE LIST
        joblib.dump(low_card_cols, os.path.join(RESULT_DIR, 'low_card_cols.joblib'))   # <-- SAVE LIST

        if len(high_card_cols) > 0:
            te = TargetEncoder(cols=high_card_cols)
            X_high = te.fit_transform(X_cat[high_card_cols], y)
            X_test_high = te.transform(X_test_cat[high_card_cols])
            joblib.dump(te, os.path.join(RESULT_DIR, 'target_encoder.joblib')) # <-- SAVE FITTED ENCODER
        else:
            X_high = np.empty((X.shape[0], 0))
            X_test_high = np.empty((X_test.shape[0], 0))

        if len(low_card_cols) > 0:
            X_low = pd.get_dummies(X_cat[low_card_cols], drop_first=True)
            X_test_low = pd.get_dummies(X_test_cat[low_card_cols], drop_first=True)
            train_ohe_cols = X_low.columns.tolist()
            joblib.dump(train_ohe_cols, os.path.join(RESULT_DIR, 'ohe_low_card_cols.joblib')) # <-- SAVE OHE COLUMNS
            X_test_low = X_test_low.reindex(columns=X_low.columns, fill_value=0.0)
        else:
            X_low = pd.DataFrame()
            X_test_low = pd.DataFrame()
            train_ohe_cols = []
    else:
        # No categorical features
        X_high = np.empty((X.shape[0], 0))
        X_test_high = np.empty((X_test.shape[0], 0))
        X_low = pd.DataFrame()
        X_test_low = pd.DataFrame()
        train_ohe_cols = []
        high_card_cols = []
        low_card_cols = []
        joblib.dump(high_card_cols, os.path.join(RESULT_DIR, 'high_card_cols.joblib'))
        joblib.dump(low_card_cols, os.path.join(RESULT_DIR, 'low_card_cols.joblib'))
        joblib.dump(train_ohe_cols, os.path.join(RESULT_DIR, 'ohe_low_card_cols.joblib'))

    if len(categorical_features) > 0:
        X_pre = np.hstack([X_numeric, X_high.values, X_low.values]) # real training features (processed)
        X_test_pre = np.hstack([X_test_numeric, X_test_high.values, X_test_low.values])
    else:
        X_pre = X_numeric
        X_test_pre = X_test_numeric
    X_pre = np.nan_to_num(X_pre).astype(float)
    X_test_pre = np.nan_to_num(X_test_pre).astype(float)
    
    globals()["PRINT_VAE_LOSS"] = True

    # VAE+GMM Oversampling on training set
    X_res, y_res = X_pre, y
    mc = list(set(y))
    mc.remove(0)
    for minority_class in mc: 
        X_res, y_res = vae_gmm_oversample(
            X_res, y_res, X_test_pre, y_test,
             minority_class=minority_class,
            vae_latent_dim=params['vae_latent_dim'],
            vae_hidden_dim=params['vae_hidden_dim'],
            vae_lr=params['vae_lr'],
            vae_epochs=params['vae_epochs'],
            gmm_n_components=params['gmm_n_components'],
            device='cpu',  # Use CPU for final training
            patience=params['vae_early_stopping_rounds']
        )

    # XGBoost
    XGB_SAVE_PATH = os.path.join(RESULT_DIR, f"{DATASET_NAME}_{OUTCOME_NAME}_xgb_model.bin")
    scale_pos_weight_val = (y_res == 0).sum() / (y_res == 1).sum()
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=y_count,
        n_estimators=params["xgb_n_estimators"],
        max_depth=params["xgb_max_depth"],
        learning_rate=params["xgb_learning_rate"],
        subsample=params["xgb_subsample"],
        reg_lambda=params["xgb_reg_lambda"],
        use_label_encoder=False,
        eval_metric='mlogloss',
        early_stopping_rounds=params["xgb_early_stopping_rounds"],
        # scale_pos_weight=scale_pos_weight_val, 
        random_state=SEED
    )
    xgb_model.fit(X_res, y_res, eval_set=[(X_test_pre, y_test)], verbose=False)

    # Save the XGBoost model
    xgb_model.save_model(XGB_SAVE_PATH)
    print(f"XGBoost model saved to: {XGB_SAVE_PATH}")

    # Add leaf indices
    X_train_leaf = flatten_leaf_indices(xgb_model.apply(X_res))
    X_test_leaf = flatten_leaf_indices(xgb_model.apply(X_test_pre))

    ohe_leaf = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=float)
    X_train_leaf_ohe = ohe_leaf.fit_transform(X_train_leaf)
    joblib.dump(ohe_leaf, os.path.join(RESULT_DIR, 'ohe_leaf_encoder.joblib'))

    X_test_leaf_ohe = ohe_leaf.transform(X_test_leaf)

    X_train_combined = np.hstack([X_res, X_train_leaf_ohe])
    X_test_combined = np.hstack([X_test_pre, X_test_leaf_ohe])
    X_train_combined = np.nan_to_num(X_train_combined).astype(float)
    X_test_combined = np.nan_to_num(X_test_combined).astype(float)

    # DNN
    model, optimizer, criterion = create_dnn(
        input_dim=X_train_combined.shape[1],
        num_neurons=params["dnn_num_neurons"],
        dropout_rate=params["dnn_dropout_rate"],
        l2_reg=params["dnn_l2_regularizer"], 
        lr=params["dnn_lr"], 
        use_focal_loss=True # Add focal loss 
    ) 
    train_dataset = TensorDataset( 
        torch.tensor(X_train_combined, dtype=torch.float32), 
        torch.tensor(y_res, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=params["dnn_batch_size"], shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=params["dnn_batch_size"], shuffle=False)  # Dummy val
    
    # Enable clean DNN loss printout for final best trial
    globals()["PRINT_VAE_LOSS"] = True
    globals()["PRINT_DNN_LOSS"] = True 

    train_dnn(model, optimizer, criterion, train_loader, val_loader, epochs=params["dnn_epoch"], patience=params["dnn_early_stopping_rounds"])
        
    test_probs = predict_dnn(model, X_test_combined) 
    
    # # ---- Train ROC AUC on the training set ---- 
    # # get the model's probability predictions for every training example
    # train_probs = predict_dnn(model, X_train_combined)

    # if y_count == 2:
    #     # binary: compare true training labels y_res vs. P(rejection)
    #     train_auc = roc_auc_score(y_res, train_probs[:, 1])
    # else:
    #     # multiclass: supply the full probability matrix
    #     train_auc = roc_auc_score( 
    #         y_res,
    #         train_probs,
    #         multi_class='ovr',
    #         average='macro' 
    #     )

    # # save or print
    # train_results = {"Train ROC AUC": round(train_auc, 4)}
    # train_results_df = pd.DataFrame([train_results])
    # train_results_df.to_csv(
    #     os.path.join(RESULT_DIR, f"{DATASET_NAME}_{OUTCOME_NAME}_train_metrics.csv"),
    #     index=False
    # )
    # print(f"Train ROC AUC: {train_auc:.4f}") 
    
    # ==== FIXED: Correct Train AUC (on original training data only) ==== 

    # 1. Get XGBoost leaf indices for original (non-oversampled) data
    X_train_leaf_real = flatten_leaf_indices(xgb_model.apply(X_pre))
    X_train_leaf_real_ohe = ohe_leaf.transform(X_train_leaf_real) 

    # 2. Combine original features with leaf indices 
    X_pre_combined = np.hstack([X_pre, X_train_leaf_real_ohe]) # X_pre + XGBoost leaf nodes (used to evaluate on real training data)
    X_pre_combined = np.nan_to_num(X_pre_combined).astype(float) 

    # 3. Predict on original data and compute AUC
    train_probs_real = predict_dnn(model, X_pre_combined)
    if y_count == 2:
        train_auc_real = roc_auc_score(y, train_probs_real[:, 1]) 
    else:
        train_auc_real = roc_auc_score(y, train_probs_real, multi_class='ovr', average='macro')

    # 4. Log and save
    print(f"✅ Corrected Train ROC AUC (real data only): {train_auc_real:.4f}")
    pd.DataFrame([{"Corrected Train ROC AUC": round(train_auc_real, 4)}]).to_csv(
        os.path.join(RESULT_DIR, f"{DATASET_NAME}_{OUTCOME_NAME}_train_auc_real.csv"), index=False
    )

    # --- Decide Best Threshold for Binary (Youden vs F1) ---
    if y_count == 2:
        best_threshold, method = find_best_threshold(y_test, test_probs[:, 1])
        print(f"Best Threshold Chosen ({method}): {best_threshold:.4f}")
    else:
        # Multiclass → ignore thresholds; always use argmax
        best_threshold = None
        method = "Argmax (Multiclass)"
        print(f"Multiclass outcome — predictions will use {method}.")
        
    # test_preds = np.argmax(test_probs, axis=1) 
    # Binary → apply best threshold 
    if y_count == 2:
        test_preds = (test_probs[:, 1] >= best_threshold).astype(int)
    # Multiclass → still use argmax
    else:
        test_preds = np.argmax(test_probs, axis=1)
    
    # Save the trained model
    model_save_path = os.path.join(RESULT_DIR, f"{DATASET_NAME}_{OUTCOME_NAME}_best_model.pt")  
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved model to: {model_save_path}")
 

    # --- PR Curve and AUPRC ---
    # binary - directly plot PR curves 
    if y_count == 2:
        precision, recall, thresholds = precision_recall_curve(y_test, test_probs[:, 1])
        auprc = average_precision_score(y_test, test_probs[:, 1])
        
        # Plot PR Curve
        plt.figure(figsize=(6,5))
        plt.plot(recall, precision, label=f"PR Curve (AUPRC = {auprc:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Test PR Curve - Best Trial")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/{DATASET_NAME}_{OUTCOME_NAME}_test_pr_curve_best_trial.png", dpi=300)
        plt.close()
    # multiclass - compute macro-averaged AUPRC for simplicity 
    else:
        precision = recall = thresholds = None  # (optional: skip plotting for multiclass)
        auprc = average_precision_score(y_test, test_probs, average="macro")

    # precision, recall, thresholds = precision_recall_curve(y_test, test_probs[:, 1])
    # auprc = average_precision_score(y_test, test_probs[:, 1])
    
    print(f"AUPRC (Average Precision): {auprc:.4f}") 

    # # --- Decide Best Threshold for Binary (Youden vs F1) ---
    # if y_count == 2:
    #     best_threshold, method = find_best_threshold(y_test, test_probs[:, 1])
    #     print(f"Best Threshold Chosen ({method}): {best_threshold:.4f}")
    # else:
    #     # Multiclass → ignore thresholds; always use argmax
    #     best_threshold = None
    #     method = "Argmax (Multiclass)"
    #     print(f"Multiclass outcome — predictions will use {method}.")


    # Save metrics 
    test_accuracy = accuracy_score(y_test, test_preds)
    if y_count == 2:
        test_auc = roc_auc_score(y_test, test_probs[:, 1])
    else:
        test_auc = roc_auc_score(y_test, test_probs, multi_class='ovr', average='macro')
    
 
    # Macro averages F1 equally across all classes, regardless of class size 
    # Weighted F1 - biased toward majority classes 
    test_f1 = f1_score(y_test, test_preds, average='macro')  # Or 'weighted' or 'binary' as needed
    
    print(f"\n[TEST SET RESULTS]")
    print(f"AUC: {test_auc:.4f}") 
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Macro F1-score: {test_f1:.4f}") 

    # Confusion matrix on test set using best trial 
    cm = confusion_matrix(y_test, test_preds, labels=list(range(y_count)))
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - Best Trial on Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/{DATASET_NAME}_{OUTCOME_NAME}_test_confusion_matrix_best_trial.png", dpi=300)
    plt.close() 

    # Classification report
    # Test set precision, recall, F1, support
    report_df = pd.DataFrame(classification_report(y_test, test_preds, output_dict=True)).transpose()
    report_df.to_csv(f"{RESULT_DIR}/{DATASET_NAME}_{OUTCOME_NAME}_test_classification_report_best_trial.csv")

    # print("Test set confusion matrix and classification report saved.") 
    
    # Histogram of Test Probabilities 
    if y_count == 2: 
        plt.figure(figsize=(8, 5))
        plt.hist(test_probs[y_test == 0][:, 1], bins=30, alpha=0.6, label="No Rejection", color="skyblue")
        plt.hist(test_probs[y_test == 1][:, 1], bins=30, alpha=0.6, label="Rejection", color="salmon")
        if best_threshold is not None: 
            plt.axvline(best_threshold, color='black', linestyle='--', label=f"Youden's J Threshold: {best_threshold:.2f}")

        plt.xlabel("Predicted Probability of Rejection")
        plt.ylabel("Frequency")
        plt.title("Histogram of Test Set Probabilities")
        plt.legend() 
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/{DATASET_NAME}_{OUTCOME_NAME}_test_probability_histogram.png", dpi=300)
        plt.close() 

    # --- Plot and Save ROC Curve for Test Set ---
    
    # Final ROC curve on test set 
    if y_count == 2:
        fpr, tpr, _ = roc_curve(y_test, test_probs[:, 1])
        roc_auc = auc(fpr, tpr) 

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Test ROC Curve - Best Trial")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/{DATASET_NAME}_{OUTCOME_NAME}_test_roc_curve_best_trial.png", dpi=300)
        plt.close() 
        
    # multiclass - build on ROC curve per class, then compute a macro-average to summarize performance across all classes 
    else:
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test_binarized = lb.transform(y_test)

        fpr = dict()
        tpr = dict()
        roc_auc = dict() 
        for i in range(y_count):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], test_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute macro-average ROC curve
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(y_count)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(y_count):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= y_count

        plt.figure(figsize=(6, 5))
        macro_auc = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, label=f"Macro-average ROC (AUC={macro_auc:.4f})", color='black', linestyle='--')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Test ROC Curve - Best Trial")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/{DATASET_NAME}_{OUTCOME_NAME}_test_roc_curve_best_trial.png", dpi=300)
        plt.close()

        
    # --- Save Test Predictions ---     
    # Binary - apply a threshold to a single probability to decide if it's class 0 or 1 
    if y_count == 2:
        test_preds = (test_probs[:, 1] >= best_threshold).astype(int) 
    # Multiclass - pick class with highest probability (argmax)
    else: 
        test_preds = np.argmax(test_probs, axis=1)

    df_preds = pd.DataFrame({ 
        "TrueLabel": y_test,  
        "PredictedLabel": test_preds
    })
    for i in range(test_probs.shape[1]):
        df_preds[f"Prob_Class{i}"] = test_probs[:, i]  

    # Raw test predictions and probabilities
    df_preds.to_csv(f"{RESULT_DIR}/{DATASET_NAME}_{OUTCOME_NAME}_test_predictions_best_trial.csv", index=False)
    # print("Test predictions saved.") 
    
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
    df_bootstrap.to_csv(f"{RESULT_DIR}/{DATASET_NAME}_{OUTCOME_NAME}_bootstrap_test_metrics.csv", index=False)
    print(f"\nBootstrapped metrics saved to {RESULT_DIR}/{DATASET_NAME}_{OUTCOME_NAME}_bootstrap_test_metrics.csv")

 
# ==============================================================================
# 6. Main Execution Block - Run Optuna Study (5-Fold CV)
# ==============================================================================
# Optuna is an automatic hyperparameter optimization software framework 

if __name__ == "__main__": 
    
    print("\n=== XGBoost + Leaf Indices + DNN + (VAE+GMM) Oversampling [Multiclass] ===")

    parser = argparse.ArgumentParser(description='Run Optuna study for DNN model')
    parser.add_argument('--outcome_type', type=str, required=True, help='outcome_abmr, outcome_banff, outcome_tcmr, or outcome_rej')
    parser.add_argument('--train_set',
                        nargs='+',         
                        required=True,
                        help='one or more CSV files of training data')
    parser.add_argument('--test_set',
                        nargs='+',         
                        required=True,
                        help='one or more CSV files of test data')
    parser.add_argument('--output', type=str, default="results", help='output folder name')
    parser.add_argument('--trials', type=int, default=1000, help='number of optuna trials')
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

    # Loop over each train/test pair
    for train_path, test_path in zip(train_array, test_array):
        # Determine dataset name from train_path
        if "lowres" in train_path or "low" in train_path:
            DATASET_NAME = "low_res"
        elif "highres" in train_path or "high" in train_path:
            DATASET_NAME = "high_res"
        elif "imputed" in train_path or "imp" in train_path:
            DATASET_NAME = "imputed"
        else:
            DATASET_NAME = "unknown"
        
        RESULT_DIR = f"{OUTPUT_PATH}/{DATASET_NAME}/{OUTCOME_NAME}"
        
        # Create a new subdirectory like "10_trials" under RESULT_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subfolder_name = f"{num_trials}_trials_{timestamp}" 
        run_dir = os.path.join(RESULT_DIR, subfolder_name)
        os.makedirs(run_dir, exist_ok=True) 

        RESULT_DIR = run_dir 
        
        # Load data for this train/test pair
        data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        # Define outcome column to predict
        outcome_col = OUTCOME_NAME

        # Extract labels
        y_original = data[outcome_col]
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_original)
        y_count = len(set(y))

        # Drop target + other outcomes from training features
        drop_cols = ["outcome_tcmr", "outcome_banff", "outcome_abmr", "outcome_rej", "pid"]
        X = data.drop(columns=drop_cols, errors="ignore")
        X_test = test_data.drop(columns=drop_cols, errors="ignore")
        y_test_original = test_data[outcome_col]
        y_test = label_encoder.transform(y_test_original)

        # Identify numeric and categorical features
        numeric_features = [col for col in X.columns if X[col].dtype in ["int64", "float64"]]
        categorical_features = [col for col in X.columns if X[col].dtype in ["object", "category"]]

        # Print class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print("\nClass Distribution:")
        for cls, count in zip(unique_classes, class_counts):
            label = f"Class {cls}" if y_count > 2 else ("No Rejection" if cls == 0 else "Rejection")
            print(f"{label}: {count} samples") 

        print(f"\nTrain set: {X.shape[0]} samples, {X.shape[1]} engineering features")
        print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} engineered features") 
        

        # Save dataset info to CSV
        metadata = {
            "dataset_name": DATASET_NAME,
            "outcome": OUTCOME_NAME,
            "train_path": train_path,
            "test_path": test_path,
            "num_train_samples": X.shape[0],
            "num_test_samples": X_test.shape[0],
            "num_features": X.shape[1]
        }
        metadata_df = pd.DataFrame(list(metadata.items()), columns=["Attribute", "Value"])
        metadata_df.to_csv(os.path.join(RESULT_DIR, "dataset_metadata.csv"), index=False)

        # Path to save/reload study
        study_path = os.path.join(RESULT_DIR, "optuna_study.pkl")

        # Check if study already exists and load it if it does
        if os.path.exists(study_path):
            study = joblib.load(study_path)
            print("Loaded existing Optuna study.") 
        else:
            study = optuna.create_study(direction="maximize")
            print("Created new Optuna study.")

        # Run trials 
        study.optimize(
            lambda trial: objective(trial, RESULT_DIR),
            n_trials=num_trials,
            show_progress_bar=True,
            # callbacks=[callback_plot_roc_cm]
        ) 
        
        print("Saving results to:", RESULT_DIR) 

        # Save the Optuna study
        joblib.dump(study, study_path)
        print(f"\nOptuna study saved to: {study_path}")

        # Visualize and Evaluate
        plot_best_trial_summary(study.best_trial, RESULT_DIR) 
        # evaluate_best_trial_on_test(study.best_trial, RESULT_DIR) 
        
        # 🔧 Force numeric-looking test columns to be treated as numeric (match training) 
        '''
        for col in X_test.columns:
            if col in X.columns and X[col].dtype in ["int64", "float64"]:
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

        if len(numeric_features) == 0:
            raise ValueError("No numeric features found in test set. Please inspect column types.")
        '''
        # evaluate_best_trial_on_test(study.best_trial, RESULT_DIR, X, y, X_test, y_test, numeric_features, categorical_features)

        print("\n--- Best Trial ---")
        print(f"  Number: {study.best_trial.number}")
        print(f"  Validation F1 (macro): {study.best_trial.value:.4f}") 
        print(f"  Params: {study.best_trial.params}") 

        # Save the same summary to a text file in RESULT_DIR
        summary_lines = [
            f"Number: {study.best_trial.number}",
            f"Validation F1 (macro): {study.best_trial.value:.4f}",
            f"Params: {study.best_trial.params}"
        ]
        with open(os.path.join(RESULT_DIR, "best_trial_summary.txt"), "w") as f:
            for line in summary_lines:
                f.write(line + "\n")

#!/usr/bin/env python3
"""Train and evaluate the ClinicalBERT models reported in the results sheet.

This publication entry point consolidates the original training script and the
separate SRTR prospective/geographic inference and bootstrap utilities.  It
supports the Mayo HLA datasets and all three SRTR split types from one CLI.

Consolidated provenance: ``clinical_BERT.py``, the ClinicalBERT portion of
``bootstrap_srtr_prospective_cis.py``, and
``bootstrap_clinicalbert_srtr_geographic_ci.py``.

Examples
--------
Install the runtime dependencies with Python 3.10 or newer::

    python -m pip install numpy pandas scikit-learn torch tqdm transformers

Validate the reported high-resolution Mayo cohort::

    python clinicalbert_runs.py --dataset high_res --outcome outcome_tcmr \
        --data-root /path/to/cohort_csvs --validate-only

Train the reported geographic SRTR model::

    python clinicalbert_runs.py --dataset srtr_geographic --outcome outcome_rej

Evaluate an existing checkpoint without retraining::

    python clinicalbert_runs.py --dataset srtr_geographic \
        --output-dir /path/to/existing/run --evaluate-only

The spreadsheet's thresholded ClinicalBERT metrics chose an F1-optimal
threshold on the evaluation cohort.  ``--threshold-source evaluation`` remains
the default solely to reproduce that reported analysis.  For an independent
test analysis, use ``--threshold-source validation`` (recommended for new
experiments) or ``--threshold-source fixed --fixed-threshold 0.5``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import BertConfig, BertModel, BertTokenizer, get_scheduler


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    os.environ.get("KIDNEY_DATA_ROOT", str(SCRIPT_DIR / "data"))
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("KIDNEY_RESULTS_ROOT", str(SCRIPT_DIR / "results"))
) / "clinicalbert"

MAYO_OUTCOMES = ("outcome_tcmr", "outcome_abmr", "outcome_rej")
SRTR_LABEL = "All rejection"
OUTCOME_COLUMNS = {
    "outcome_tcmr",
    "outcome_banff",
    "outcome_abmr",
    "outcome_rej",
}

# Exact clean-feature policy used by the SRTR ClinicalBERT prospective and
# geographic runs reported in the spreadsheet.
SRTR_DROP_COLUMNS = {
    *OUTCOME_COLUMNS,
    "Acute rejection",
    "Chronic rejection",
    "All rejection",
    "all rejection",
    "Unnamed: 0",
    "pid",
    "PERS_ID",
    "PX_ID",
    "TRR_ID",
    "TX_ID",
    "REC_HISTO_TX_ID",
    "DONOR_ID",
    "REC_ACUTE_REJ_EPISODE",
    "ORG_AR",
    "DAYS_AR",
    "Graft status",
    "Patient status",
    "TFL_FAIL_DT",
    "TFL_FAIL_CAUSE_TY",
    "TFL_ENDTXFU",
    "TFL_LASTATUS",
    "TFL_LAFUDATE",
    "TIME_LFU",
    "REC_CUR_PX_STAT",
    "REC_GRAFT_STAT",
    "REC_PX_STAT",
    "REC_PX_STAT_DT",
    "REC_POSTX_LOS",
    "REC_DISCHRG_DT",
    "REC_ADMISSION_DT",
    "REC_DISCHRG_CREAT",
    "REC_FIRST_WEEK_DIAL",
    "REC_CUR_CTR_ID",
    "REC_TX_DT",
    "DON_RECOV_DT",
    "CAN_LISTING_DT",
    "REC_ORG_RECEIVED_ON",
}
MAYO_DROP_COLUMNS = {*OUTCOME_COLUMNS, "pid"}


@dataclass(frozen=True)
class DatasetPreset:
    train_csv: Path
    test_csv: Path
    source: str


@dataclass(frozen=True)
class ModelSettings:
    backbone: str
    max_length: int
    dropout: float
    alpha: float
    gamma: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int
    patience: int
    validation_fraction: float
    seed: int


def dataset_presets(data_root: Path) -> dict[str, DatasetPreset]:
    mayo_root = data_root / "test_datasets_new_features"
    # The 2025-08-22 high-resolution snapshot has the 319/82 train/test sample
    # counts reported in the spreadsheet.  The current unsuffixed directory was
    # later replaced by a smaller 130/33 snapshot.
    high_res_root = data_root / "test_datasets_new_features_20250822"
    return {
        "low_res": DatasetPreset(
            mayo_root / "mayo_sites_merged_lowres_train_v2.csv",
            mayo_root / "mayo_sites_merged_lowres_test_v2.csv",
            "mayo",
        ),
        "imputed": DatasetPreset(
            mayo_root / "mayo_sites_merged_imputed_train_v2.csv",
            mayo_root / "mayo_sites_merged_imputed_test_v2.csv",
            "mayo",
        ),
        "high_res": DatasetPreset(
            high_res_root / "mayo_sites_merged_highres_train_v2.csv",
            high_res_root / "mayo_sites_merged_highres_test_v2.csv",
            "mayo",
        ),
        "srtr": DatasetPreset(
            data_root / "SRTR_train.csv",
            data_root / "SRTR_test.csv",
            "srtr",
        ),
        "srtr_prospective": DatasetPreset(
            data_root / "SRTR_train_prospective.csv",
            data_root / "SRTR_test_prospective.csv",
            "srtr",
        ),
        "srtr_geographic": DatasetPreset(
            data_root / "SRTR_train_geographic.csv",
            data_root / "SRTR_test_geographic.csv",
            "srtr",
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or evaluate the reported ClinicalBERT models."
    )
    parser.add_argument(
        "--dataset",
        choices=(
            "low_res",
            "imputed",
            "high_res",
            "srtr",
            "srtr_prospective",
            "srtr_geographic",
        ),
        required=True,
    )
    parser.add_argument("--outcome", choices=MAYO_OUTCOMES, default="outcome_rej")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Directory containing the cohort CSVs and Mayo dataset folders "
            "(default: LLMs/data or $KIDNEY_DATA_ROOT)."
        ),
    )
    parser.add_argument("--train-csv", type=Path, help="Override preset training CSV.")
    parser.add_argument("--test-csv", type=Path, help="Override preset test CSV.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory used when --output-dir is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Exact run directory, including an existing run for evaluation.",
    )
    parser.add_argument(
        "--backbone", default="emilyalsentzer/Bio_ClinicalBERT"
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--threshold-source",
        choices=("evaluation", "validation", "fixed"),
        default="evaluation",
        help=(
            "evaluation reproduces the reported metrics; validation is "
            "recommended for a new independent test analysis."
        ),
    )
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Load an existing checkpoint and run validation/test inference.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow a training run to replace an existing checkpoint.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate paths, labels, split, and feature types without loading BERT.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_length < 2:
        raise ValueError("--max-length must be at least 2")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 because the head uses BatchNorm")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("--epochs and --patience must be positive")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")
    if not 0 <= args.fixed_threshold <= 1:
        raise ValueError("--fixed-threshold must be between 0 and 1")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates cannot be negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")


def set_reproducibility(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def software_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for distribution in (
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
        "tqdm",
    ):
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = "not installed"
    return versions


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def ensure_binary_labels(values: pd.Series, label_column: str, path: Path) -> pd.Series:
    labels = pd.to_numeric(values, errors="raise").astype(int)
    unique = sorted(labels.dropna().unique().tolist())
    if unique != [0, 1]:
        raise ValueError(
            f"Expected binary 0/1 labels in {label_column!r} at {path}; found {unique}"
        )
    return labels


def split_and_select_features(
    full_train: pd.DataFrame,
    test: pd.DataFrame,
    label_column: str,
    source: str,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    train, validation = train_test_split(
        full_train,
        test_size=validation_fraction,
        random_state=seed,
        stratify=full_train[label_column],
    )
    drop_columns = SRTR_DROP_COLUMNS if source == "srtr" else MAYO_DROP_COLUMNS
    features = [column for column in full_train.columns if column not in drop_columns]
    if not features:
        raise ValueError("Feature policy removed every input column")
    missing_validation = set(features) - set(validation.columns)
    missing_test = set(features) - set(test.columns)
    if missing_validation or missing_test:
        raise ValueError(
            "Feature mismatch: "
            f"validation missing={sorted(missing_validation)}, "
            f"test missing={sorted(missing_test)}"
        )
    train_features = train[features]
    text_columns = train_features.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numerical_columns = train_features.select_dtypes(include=[np.number]).columns.tolist()
    unsupported = [
        column
        for column in features
        if column not in set(text_columns) | set(numerical_columns)
    ]
    if unsupported:
        raise ValueError(f"Unsupported feature dtypes for columns: {unsupported}")

    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        numerical = frame[numerical_columns].to_numpy(dtype=np.float32)
        if not np.isfinite(numerical).all():
            bad = np.argwhere(~np.isfinite(numerical))[0]
            raise ValueError(
                f"Non-finite numerical value in {name}: row={bad[0]}, "
                f"column={numerical_columns[int(bad[1])]!r}"
            )
    return train, validation, features, text_columns, numerical_columns


def frame_to_arrays(
    frame: pd.DataFrame,
    label_column: str,
    text_columns: list[str],
    numerical_columns: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    # This deliberately retains the original reported representation: values
    # are concatenated in column order without adding column names.
    texts = frame[text_columns].apply(
        lambda row: " ".join(row.astype(str)), axis=1
    ).tolist()
    numerical = frame[numerical_columns].to_numpy(dtype=np.float32)
    labels = frame[label_column].to_numpy(dtype=np.int64)
    return texts, numerical, labels


class ClinicalDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        numerical: np.ndarray,
        labels: np.ndarray,
        tokenizer: BertTokenizer,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.numerical = numerical
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].flatten(),
            "attention_mask": encoded["attention_mask"].flatten(),
            "numerical_features": torch.from_numpy(self.numerical[index]),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


class FocalLoss(nn.Module):
    """Binary focal loss used in the reported ClinicalBERT runs."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)
        selected = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
        alpha = torch.where(labels == 1, self.alpha, 1 - self.alpha)
        return (-alpha * (1 - selected) ** self.gamma * torch.log(selected + 1e-8)).mean()


class ClinicalBertWithNumericalFeatures(nn.Module):
    """Bio_ClinicalBERT CLS embedding plus the reported numerical MLP head."""

    def __init__(
        self,
        backbone_source: str | Path,
        numerical_feature_count: int,
        dropout: float,
        initialize_backbone_weights: bool = True,
    ) -> None:
        super().__init__()
        if initialize_backbone_weights:
            self.bert = BertModel.from_pretrained(str(backbone_source))
        else:
            # A legacy custom checkpoint contains the full BERT state.  Build
            # from its config here and let load_checkpoint restore all weights,
            # avoiding misleading warnings about the custom numerical head.
            self.bert = BertModel(BertConfig.from_pretrained(str(backbone_source)))
        combined_size = self.bert.config.hidden_size + numerical_feature_count
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(combined_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 2)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.elu = nn.ELU(alpha=1.0)
        self.relu = nn.ReLU()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numerical_features: torch.Tensor,
    ) -> torch.Tensor:
        cls = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        x = self.dropout(torch.cat((cls, numerical_features), dim=1))
        x = self.dropout(self.leaky_relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.elu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = self.dropout(self.relu(self.bn4(self.fc4(x))))
        return self.fc5(x)


def make_loaders(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    label_column: str,
    text_columns: list[str],
    numerical_columns: list[str],
    tokenizer: BertTokenizer,
    settings: ModelSettings,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    arrays = [
        frame_to_arrays(frame, label_column, text_columns, numerical_columns)
        for frame in (train, validation, test)
    ]
    datasets = [
        ClinicalDataset(*values, tokenizer, settings.max_length) for values in arrays
    ]
    train_labels = arrays[0][2]
    counts = np.bincount(train_labels, minlength=2)
    if (counts == 0).any():
        raise ValueError(f"Training split is missing a class: counts={counts.tolist()}")
    sample_weights = (1.0 / counts)[train_labels]
    generator = torch.Generator().manual_seed(settings.seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    common = {
        "batch_size": settings.batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(datasets[0], sampler=sampler, **common),
        DataLoader(datasets[1], shuffle=False, **common),
        DataLoader(datasets[2], shuffle=False, **common),
    )


def f1_optimal_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_values = 2 * precision * recall / (precision + recall + 1e-6)
    index = int(np.argmax(f1_values))
    return float(thresholds[index]) if index < len(thresholds) else 0.5


def youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    return float(thresholds[finite][np.argmax(tpr[finite] - fpr[finite])])


def reported_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, str]:
    if np.mean(labels == 1) < 0.10:
        return f1_optimal_threshold(labels, scores), "F1-optimal"
    return youden_threshold(labels, scores), "Youden"


def predict(
    model: ClinicalBertWithNumericalFeatures,
    loader: DataLoader,
    loss_function: FocalLoss,
    device: torch.device,
    description: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    weighted_loss = 0.0
    with torch.inference_mode():
        for batch in tqdm(loader, desc=description):
            batch_labels = batch["labels"].to(device)
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                numerical_features=batch["numerical_features"].to(device),
            )
            loss = loss_function(logits, batch_labels)
            weighted_loss += loss.item() * len(batch_labels)
            labels.extend(batch_labels.cpu().tolist())
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return (
        np.asarray(labels, dtype=int),
        np.asarray(probabilities, dtype=float),
        weighted_loss / len(labels),
    )


def metrics_at_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    loss: float,
    threshold: float,
    threshold_method: str,
) -> dict[str, Any]:
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "loss": loss,
        "threshold": threshold,
        "threshold_method": threshold_method,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_samples": int(len(labels)),
    }


def bootstrap_cis(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    if replicates == 0:
        return {}
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    if not len(negative) or not len(positive):
        return {}
    rng = np.random.default_rng(seed)
    values = np.empty((replicates, 3), dtype=float)
    for index in range(replicates):
        sample = np.concatenate(
            (
                rng.choice(negative, len(negative), replace=True),
                rng.choice(positive, len(positive), replace=True),
            )
        )
        rng.shuffle(sample)
        y_sample = labels[sample]
        score_sample = scores[sample]
        pred_sample = score_sample >= threshold
        values[index] = (
            f1_score(y_sample, pred_sample, average="macro", zero_division=0),
            roc_auc_score(y_sample, score_sample),
            average_precision_score(y_sample, score_sample),
        )
    lower, upper = np.percentile(values, [2.5, 97.5], axis=0)
    return {
        "macro_f1_ci_lower": float(lower[0]),
        "macro_f1_ci_upper": float(upper[0]),
        "roc_auc_ci_lower": float(lower[1]),
        "roc_auc_ci_upper": float(upper[1]),
        "auprc_ci_lower": float(lower[2]),
        "auprc_ci_upper": float(upper[2]),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def checkpoint_candidates(output_dir: Path) -> tuple[Path, ...]:
    return output_dir / "checkpoint.pt", output_dir / "pytorch_model.bin"


def existing_checkpoint(output_dir: Path) -> Path | None:
    return next((path for path in checkpoint_candidates(output_dir) if path.is_file()), None)


def save_checkpoint(
    model: ClinicalBertWithNumericalFeatures,
    tokenizer: BertTokenizer,
    output_dir: Path,
    settings: ModelSettings,
    feature_columns: list[str],
    text_columns: list[str],
    numerical_columns: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "settings": asdict(settings),
            "feature_columns": feature_columns,
            "text_columns": text_columns,
            "numerical_columns": numerical_columns,
        },
        output_dir / "checkpoint.pt",
    )
    tokenizer.save_pretrained(output_dir / "tokenizer")
    model.bert.save_pretrained(output_dir / "backbone")


def load_checkpoint(
    model: ClinicalBertWithNumericalFeatures,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    incompatible = model.load_state_dict(state, strict=False)
    allowed_unexpected = {"classifier.weight", "classifier.bias"}
    unexpected = set(incompatible.unexpected_keys) - allowed_unexpected
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            f"Incompatible checkpoint {checkpoint_path}: "
            f"missing={incompatible.missing_keys}, unexpected={sorted(unexpected)}"
        )


def model_sources(output_dir: Path, backbone: str) -> tuple[str | Path, str | Path]:
    backbone_dir = output_dir / "backbone"
    tokenizer_dir = output_dir / "tokenizer"
    if backbone_dir.is_dir():
        backbone_source: str | Path = backbone_dir
    elif (output_dir / "config.json").is_file():  # Legacy reported run.
        backbone_source = output_dir
    else:
        backbone_source = backbone
    if tokenizer_dir.is_dir():
        tokenizer_source: str | Path = tokenizer_dir
    elif (output_dir / "tokenizer_config.json").is_file():
        tokenizer_source = output_dir
    else:
        tokenizer_source = backbone
    return backbone_source, tokenizer_source


def train_model(
    model: ClinicalBertWithNumericalFeatures,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_function: FocalLoss,
    device: torch.device,
    settings: ModelSettings,
    output_dir: Path,
    tokenizer: BertTokenizer,
    feature_columns: list[str],
    text_columns: list[str],
    numerical_columns: list[str],
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=settings.epochs * len(train_loader),
    )
    best_loss = math.inf
    best_f1 = -math.inf
    best_precision = -math.inf
    epochs_without_improvement = 0

    for epoch in range(1, settings.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for batch in tqdm(train_loader, desc=f"Training {epoch}/{settings.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            labels = batch["labels"].to(device)
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                numerical_features=batch["numerical_features"].to(device),
            )
            loss = loss_function(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum())
            seen += len(labels)

        val_labels, val_scores, val_loss = predict(
            model,
            validation_loader,
            loss_function,
            device,
            f"Validation {epoch}/{settings.epochs}",
        )
        val_threshold, val_method = reported_threshold(val_labels, val_scores)
        val_metrics = metrics_at_threshold(
            val_labels, val_scores, val_loss, val_threshold, val_method
        )
        print(
            f"Epoch {epoch}: train_loss={total_loss / seen:.6f}, "
            f"train_accuracy={correct / seen:.6f}, val_loss={val_loss:.6f}, "
            f"val_f1={val_metrics['f1']:.6f}, "
            f"val_precision={val_metrics['precision']:.6f}"
        )

        # This reproduces the original checkpoint rule: save when any one of
        # validation loss, positive-class F1, or precision improves.
        improved = (
            val_loss < best_loss
            or val_metrics["f1"] > best_f1
            or val_metrics["precision"] > best_precision
        )
        if improved:
            best_loss = min(best_loss, val_loss)
            best_f1 = max(best_f1, val_metrics["f1"])
            best_precision = max(best_precision, val_metrics["precision"])
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                tokenizer,
                output_dir,
                settings,
                feature_columns,
                text_columns,
                numerical_columns,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings.patience:
                print(f"Early stopping after epoch {epoch}")
                break


def write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    settings: ModelSettings,
    train_csv: Path,
    test_csv: Path,
    source: str,
    label_column: str,
    feature_columns: list[str],
    text_columns: list[str],
    numerical_columns: list[str],
    train_count: int,
    validation_count: int,
    test_count: int,
) -> None:
    config = {
        "dataset": args.dataset,
        "source": source,
        "requested_outcome": args.outcome,
        "effective_outcome": "outcome_rej" if source == "srtr" else args.outcome,
        "label_column": label_column,
        "train_csv": str(train_csv.resolve()),
        "test_csv": str(test_csv.resolve()),
        "train_csv_sha256": sha256_file(train_csv),
        "test_csv_sha256": sha256_file(test_csv),
        "n_train_split": train_count,
        "n_validation_split": validation_count,
        "n_test": test_count,
        "feature_policy": (
            "shared_srtr_clean_no_post_transplant"
            if source == "srtr"
            else "mayo_outcomes_and_pid_excluded"
        ),
        "feature_columns": feature_columns,
        "text_columns": text_columns,
        "numerical_columns": numerical_columns,
        "settings": asdict(settings),
        "threshold_source": args.threshold_source,
        "fixed_threshold": args.fixed_threshold,
        "bootstrap_replicates": args.bootstrap_replicates,
        "software_versions": software_versions(),
        "methodological_note": (
            "The evaluation threshold is optimized on evaluation labels to "
            "reproduce the reported spreadsheet metrics."
            if args.threshold_source == "evaluation"
            else "The evaluation threshold is independent of evaluation labels."
        ),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_reproducibility(args.seed)
    preset = dataset_presets(args.data_root)[args.dataset]
    train_csv = (args.train_csv or preset.train_csv).expanduser()
    test_csv = (args.test_csv or preset.test_csv).expanduser()
    if not train_csv.is_file() or not test_csv.is_file():
        raise FileNotFoundError(f"Missing train/test CSV: {train_csv}, {test_csv}")
    source = preset.source
    label_column = SRTR_LABEL if source == "srtr" else args.outcome
    effective_outcome = "outcome_rej" if source == "srtr" else args.outcome
    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir
        else args.output_root.expanduser() / args.dataset / effective_outcome
    )

    full_train = pd.read_csv(train_csv, low_memory=False)
    test = pd.read_csv(test_csv, low_memory=False)
    if label_column not in full_train or label_column not in test:
        raise ValueError(f"Missing outcome column {label_column!r}")
    full_train[label_column] = ensure_binary_labels(
        full_train[label_column], label_column, train_csv
    )
    test[label_column] = ensure_binary_labels(test[label_column], label_column, test_csv)
    train, validation, features, text_columns, numerical_columns = split_and_select_features(
        full_train,
        test,
        label_column,
        source,
        args.validation_fraction,
        args.seed,
    )
    settings = ModelSettings(
        backbone=args.backbone,
        max_length=args.max_length,
        dropout=args.dropout,
        alpha=args.alpha,
        gamma=args.gamma,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Outcome: {effective_outcome} ({label_column})")
    print(f"Train/validation/test: {len(train)}/{len(validation)}/{len(test)}")
    print(
        "Text/numerical/total features: "
        f"{len(text_columns)}/{len(numerical_columns)}/{len(features)}"
    )
    print(f"Output: {output_dir}")
    if args.validate_only:
        print("Validation complete; BERT was not loaded.")
        return

    if args.evaluate_only and not output_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {output_dir}")
    checkpoint = existing_checkpoint(output_dir)
    if args.evaluate_only and checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {output_dir}")
    if not args.evaluate_only and checkpoint is not None and not args.overwrite:
        raise FileExistsError(
            f"Checkpoint already exists at {checkpoint}; use --evaluate-only or --overwrite"
        )
    if args.evaluate_only:
        backbone_source, tokenizer_source = model_sources(output_dir, args.backbone)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_run_config(
            output_dir,
            args,
            settings,
            train_csv,
            test_csv,
            source,
            label_column,
            features,
            text_columns,
            numerical_columns,
            len(train),
            len(validation),
            len(test),
        )
        train.to_csv(output_dir / "train_split.csv", index=False)
        validation.to_csv(output_dir / "validation_split.csv", index=False)
        backbone_source, tokenizer_source = args.backbone, args.backbone

    tokenizer = BertTokenizer.from_pretrained(str(tokenizer_source))
    model = ClinicalBertWithNumericalFeatures(
        backbone_source,
        len(numerical_columns),
        args.dropout,
        initialize_backbone_weights=(
            not args.evaluate_only or (output_dir / "backbone").is_dir()
        ),
    )
    device = resolve_device(args.device)
    model.to(device)
    loaders = make_loaders(
        train,
        validation,
        test,
        label_column,
        text_columns,
        numerical_columns,
        tokenizer,
        settings,
        args.num_workers,
    )
    loss_function = FocalLoss(args.alpha, args.gamma)

    if args.evaluate_only:
        load_checkpoint(model, checkpoint, device)
    else:
        train_model(
            model,
            loaders[0],
            loaders[1],
            loss_function,
            device,
            settings,
            output_dir,
            tokenizer,
            features,
            text_columns,
            numerical_columns,
        )
        checkpoint = existing_checkpoint(output_dir)
        if checkpoint is None:
            raise RuntimeError("Training ended without saving a checkpoint")
        load_checkpoint(model, checkpoint, device)

    val_labels, val_scores, val_loss = predict(
        model, loaders[1], loss_function, device, "Final validation"
    )
    test_labels, test_scores, test_loss = predict(
        model, loaders[2], loss_function, device, "Test"
    )
    validation_threshold, validation_method = reported_threshold(val_labels, val_scores)
    if args.threshold_source == "evaluation":
        threshold, threshold_method = reported_threshold(test_labels, test_scores)
        threshold_method = f"evaluation-{threshold_method}"
    elif args.threshold_source == "validation":
        threshold, threshold_method = validation_threshold, f"validation-{validation_method}"
    else:
        threshold, threshold_method = args.fixed_threshold, "fixed"

    metrics = metrics_at_threshold(
        test_labels, test_scores, test_loss, threshold, threshold_method
    )
    metrics.update(
        bootstrap_cis(
            test_labels,
            test_scores,
            threshold,
            args.bootstrap_replicates,
            args.seed,
        )
    )
    metrics["validation_threshold"] = validation_threshold
    metrics["validation_threshold_method"] = validation_method
    pd.DataFrame(
        {
            "y_true": test_labels,
            "rejection_probability": test_scores,
            "prediction": (test_scores >= threshold).astype(int),
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "test_metrics.csv", index=False)
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

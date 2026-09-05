#!/usr/bin/env python3
"""Run the reported SRTR OpenRouter rejection benchmarks.

This publication entry point consolidates the original benchmark, continuation,
and invalid-row retry scripts.  It supports the three SRTR cohorts reported in
the results spreadsheet: random, prospective, and geographic.

Consolidated provenance: ``openrouter_sota_benchmark.py``,
``continue_openrouter_geographic_invalid_5.py``, the three
``retry_openrouter_*`` shell launchers, and the OpenRouter portion of
``bootstrap_srtr_prospective_cis.py``.

Examples
--------
Install the runtime dependencies with Python 3.10 or newer::

    python -m pip install numpy pandas requests scikit-learn

Validate a cohort without contacting OpenRouter::

    python openrouter_runs.py --cohort geographic \
        --data-root /path/to/srtr_csvs --validate-only

Run every reported model (``OPENROUTER_API_KEY`` must be set)::

    python openrouter_runs.py --cohort geographic

Run or resume selected models::

    python openrouter_runs.py --cohort prospective \
        --model x-ai/grok-4.3 --model qwen/qwen3.7-max

The reported spreadsheet metrics use a Youden threshold estimated on the
evaluation cohort (``--threshold-strategy evaluation-youden``).  That default
is retained for exact methodological traceability.  For a prespecified
threshold, use ``--threshold-strategy fixed --fixed-threshold 0.5``.

The historical ``*-latest`` model aliases are also retained because those are
the identifiers sent during the reported runs.  Aliases can change over time,
so every new prediction records the model identifier returned by OpenRouter.
Pass explicit ``--model`` values to use version-pinned models in a new study.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    os.environ.get("KIDNEY_DATA_ROOT", str(SCRIPT_DIR / "data"))
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("KIDNEY_RESULTS_ROOT", str(SCRIPT_DIR / "results"))
) / "openrouter"

LABEL_COLUMN = "All rejection"
ROW_ID_COLUMN = "PERS_ID"
DEFAULT_THRESHOLD = 0.5
DEFAULT_BOOTSTRAP_REPLICATES = 1_000
DEFAULT_SEED = 42

REPORTED_MODEL_IDS = (
    "~openai/gpt-latest",
    "~anthropic/claude-opus-latest",
    "~anthropic/claude-sonnet-latest",
    "~google/gemini-pro-latest",
    "x-ai/grok-4.3",
    "deepseek/deepseek-v4-pro",
    "qwen/qwen3.7-max",
    "moonshotai/kimi-k2.7-code",
)

LONG_REASONING_MODELS = {
    "~google/gemini-pro-latest",
    "moonshotai/kimi-k2.7-code",
}
MODELS_REQUIRING_REASONING = {"moonshotai/kimi-k2.7-code"}

# Feature policy used by the prospective and geographic runs.  Labels, stable
# identifiers, and post-transplant/follow-up variables are excluded.
SRTR_CLEAN_DROP_COLUMNS = {
    "outcome_tcmr",
    "outcome_banff",
    "outcome_abmr",
    "outcome_rej",
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

# The first random-split run predates the expanded clean-feature policy.  This
# exact historical exclusion list is preserved so its prompts can be recreated.
SRTR_RANDOM_REPORTED_DROP_COLUMNS = {
    "Acute rejection",
    "All rejection",
    "CAN_LISTING_DT",
    "Chronic rejection",
    "DAYS_AR",
    "DONOR_ID",
    "DON_RECOV_DT",
    "Graft status",
    "PERS_ID",
    "PX_ID",
    "Patient status",
    "REC_ACUTE_REJ_EPISODE",
    "REC_ADMISSION_DT",
    "REC_DISCHRG_DT",
    "REC_GRAFT_STAT",
    "REC_HISTO_TX_ID",
    "REC_ORG_RECEIVED_ON",
    "REC_PX_STAT_DT",
    "REC_TX_DT",
    "TFL_ENDTXFU",
    "TFL_FAIL_CAUSE_TY",
    "TFL_FAIL_DT",
    "TFL_LAFUDATE",
    "TFL_LASTATUS",
    "TRR_ID",
    "TX_ID",
    "Unnamed: 0",
}

SYSTEM_PROMPT = """You are a clinical prediction model for kidney transplant rejection.
Use only the patient information in the user message.

Predict a binary outcome:
- 0 = no rejection
- 1 = rejection

Return exactly one JSON object with no markdown and no explanation:
{"label": 0, "rejection_probability": 0.0}

Rules:
- label must be 0 or 1.
- rejection_probability must be a number from 0 to 1 representing the probability of rejection.
- label must be 1 when rejection_probability is at least 0.5 and 0 otherwise.
"""

USER_TEMPLATE = """Patient information:
{patient_prompt}

Predict kidney rejection for this patient. Return only the required JSON object."""


@dataclass(frozen=True)
class CohortPreset:
    eval_csv: Path
    run_name: str
    feature_policy: str
    drop_columns: frozenset[str]


def cohort_presets(data_root: Path) -> dict[str, CohortPreset]:
    """Return paths and exact feature policies for the reported cohorts."""
    return {
        "random": CohortPreset(
            eval_csv=data_root / "SRTR_test.csv",
            run_name="sota_openrouter_srtr",
            feature_policy="reported_random_legacy_exclusions",
            drop_columns=frozenset(SRTR_RANDOM_REPORTED_DROP_COLUMNS),
        ),
        "prospective": CohortPreset(
            eval_csv=data_root / "SRTR_test_prospective.csv",
            run_name="sota_openrouter_srtr_prospective_clean_features",
            feature_policy="shared_srtr_clean_no_post_transplant",
            drop_columns=frozenset(SRTR_CLEAN_DROP_COLUMNS),
        ),
        "geographic": CohortPreset(
            eval_csv=data_root / "SRTR_test_geographic.csv",
            run_name="sota_openrouter_srtr_geographic_clean_features",
            feature_policy="shared_srtr_clean_no_post_transplant",
            drop_columns=frozenset(SRTR_CLEAN_DROP_COLUMNS),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or resume the reported SRTR OpenRouter benchmarks."
    )
    parser.add_argument(
        "--cohort",
        choices=("random", "prospective", "geographic"),
        required=True,
        help="Reported SRTR test cohort to evaluate.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Directory containing SRTR_test.csv, SRTR_test_prospective.csv, "
            "and SRTR_test_geographic.csv (default: LLMs/data or "
            "$KIDNEY_DATA_ROOT)."
        ),
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        help="Override the selected cohort's evaluation CSV.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory for run outputs.",
    )
    parser.add_argument(
        "--run-name",
        help="Override the spreadsheet-aligned run directory name.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Model identifier to run; repeat to select multiple models.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N rows (for development, not reporting).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing predictions for the selected models instead of resuming.",
    )
    parser.add_argument(
        "--invalid-retry-passes",
        type=int,
        default=3,
        help="Additional passes over invalid/missing predictions (default: 3).",
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=("evaluation-youden", "fixed"),
        default="evaluation-youden",
        help=(
            "Threshold method. evaluation-youden reproduces reported results; "
            "fixed uses --fixed-threshold."
        ),
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Prespecified threshold when --threshold-strategy=fixed.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
        help="Class-stratified bootstrap replicates; use 0 to omit CIs.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-max-tokens", type=int, default=4096)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--request-interval", type=float, default=0.25)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate data and write run metadata without API requests.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.invalid_retry_passes < 0:
        raise ValueError("--invalid-retry-passes cannot be negative")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates cannot be negative")
    if not 0.0 <= args.fixed_threshold <= 1.0:
        raise ValueError("--fixed-threshold must be between 0 and 1")
    if args.request_retries < 1:
        raise ValueError("--request-retries must be at least 1")


def get_api_key() -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key
    if not sys.stdin.isatty():
        raise RuntimeError("Set OPENROUTER_API_KEY before running non-interactively.")
    api_key = getpass.getpass("OPENROUTER_API_KEY: ").strip()
    if not api_key:
        raise RuntimeError("No OpenRouter API key supplied.")
    return api_key


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def software_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for distribution in ("numpy", "pandas", "requests", "scikit-learn"):
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = "not installed"
    return versions


def feature_columns(columns: Iterable[str], drop_columns: frozenset[str]) -> list[str]:
    return [column for column in columns if column not in drop_columns]


def is_missing(value: Any) -> bool:
    return value is None or (
        isinstance(value, float) and math.isnan(value)
    ) or (isinstance(value, str) and not value.strip())


def format_value(value: Any) -> str:
    if is_missing(value):
        return "not available"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_patient_prompt(row: pd.Series, columns: list[str]) -> str:
    lines = ["Kidney transplant recipient and donor profile from SRTR."]
    lines.extend(
        f"- {column.replace('_', ' ')}: {format_value(row[column])}"
        for column in columns
    )
    return "\n".join(lines)


def load_evaluation_data(
    csv_path: Path,
    drop_columns: frozenset[str],
    limit: int | None,
) -> tuple[pd.DataFrame, list[str], pd.Index]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")
    source = pd.read_csv(csv_path, low_memory=False)
    if LABEL_COLUMN not in source:
        raise ValueError(f"Missing label column {LABEL_COLUMN!r}: {csv_path}")
    if limit is not None:
        source = source.head(limit).copy()
    labels = pd.to_numeric(source[LABEL_COLUMN], errors="raise").astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"{LABEL_COLUMN!r} must contain only 0/1 labels")

    columns = feature_columns(source.columns, drop_columns)
    if not columns:
        raise ValueError("Feature policy removed every input column")
    records = []
    for position, (_, row) in enumerate(source.iterrows()):
        records.append(
            {
                "row_id": row.get(ROW_ID_COLUMN, position + 1),
                "line_number": position + 2,
                "patient_prompt": build_patient_prompt(row, columns),
                "y_true": int(labels.iloc[position]),
            }
        )
    return pd.DataFrame(records), columns, source.columns


def response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "integer", "enum": [0, 1]},
            "rejection_probability": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
        },
        "required": ["label", "rejection_probability"],
        "additionalProperties": False,
    }


def request_modes(model_id: str) -> tuple[str, ...]:
    if model_id in MODELS_REQUIRING_REASONING:
        return ("json_schema", "json_object", "plain")
    return (
        "json_schema_no_reasoning",
        "json_object_no_reasoning",
        "json_schema",
        "json_object",
        "plain",
    )


def make_payload(
    model_id: str,
    patient_prompt: str,
    mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(patient_prompt=patient_prompt),
            },
        ],
        "temperature": args.temperature,
        "max_tokens": (
            args.reasoning_max_tokens
            if model_id in LONG_REASONING_MODELS
            else args.max_tokens
        ),
    }
    if mode.endswith("_no_reasoning"):
        payload["reasoning"] = {"effort": "none", "exclude": True}
    if mode.startswith("json_schema"):
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "kidney_rejection_prediction",
                "strict": True,
                "schema": response_schema(),
            },
        }
    elif mode.startswith("json_object"):
        payload["response_format"] = {"type": "json_object"}
    return payload


def request_prediction(
    session: requests.Session,
    model_id: str,
    patient_prompt: str,
    api_key: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openrouter.ai/",
        "X-OpenRouter-Title": "SRTR Kidney Rejection Benchmark",
    }
    attempted_errors: list[str] = []
    for mode in request_modes(model_id):
        payload = make_payload(model_id, patient_prompt, mode, args)
        last_error = "request failed"
        for attempt in range(args.request_retries):
            try:
                response = session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=args.request_timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    if attempt + 1 < args.request_retries:
                        time.sleep(2**attempt + random.random())
                        continue
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"HTTP {response.status_code} ({mode}): {response.text[:500]}"
                    )
                data = response.json()
                choice = data["choices"][0]
                content = (choice.get("message") or {}).get("content") or ""
                finish_reason = choice.get("finish_reason")
                if content.strip():
                    return {
                        "ok": True,
                        "content": content,
                        "usage": data.get("usage") or {},
                        "finish_reason": finish_reason,
                        "request_mode": mode,
                        "resolved_model_id": data.get("model"),
                        "error": None,
                    }
                last_error = f"empty response ({mode}; finish_reason={finish_reason})"
            except Exception as exc:  # API and transport failures are row-level data.
                last_error = str(exc)
                if attempt + 1 < args.request_retries:
                    time.sleep(2**attempt + random.random())
        attempted_errors.append(last_error)
    return {
        "ok": False,
        "content": "",
        "usage": {},
        "finish_reason": None,
        "request_mode": None,
        "resolved_model_id": None,
        "error": " | ".join(attempted_errors),
    }


def parse_model_output(text: str) -> tuple[int, float]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty model response")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            # Compatibility with early responses formatted as (no-rejection,
            # rejection, label), where probabilities were sometimes percentages.
            match = re.search(
                r"\(?\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*"
                r"([0-9]+(?:\.[0-9]+)?)\s*,\s*([01])\s*\)?",
                raw,
            )
            if not match:
                raise ValueError(f"response is not valid JSON: {raw[:300]}")
            probability = float(match.group(2))
            if probability > 1:
                probability /= 100
            return int(match.group(3)), min(max(probability, 0.0), 1.0)
    if not isinstance(parsed, dict):
        raise ValueError("response JSON is not an object")
    label = int(parsed["label"])
    probability = float(parsed["rejection_probability"])
    if label not in (0, 1) or not 0 <= probability <= 1:
        raise ValueError(f"invalid prediction: label={label}, probability={probability}")
    return label, probability


def finite_probability(row: dict[str, Any]) -> bool:
    try:
        return bool(row.get("api_ok")) and math.isfinite(
            float(row["rejection_probability"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def credit_error(error: Any) -> bool:
    value = str(error or "").lower()
    return any(
        phrase in value
        for phrase in ("402 payment required", "insufficient credits", "requires more credits")
    )


def read_prediction_map(
    path: Path, model_id: str, expected_lines: set[int]
) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    predictions: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for jsonl_line, text in enumerate(handle, start=1):
            try:
                row = json.loads(text, parse_constant=lambda _: math.nan)
                line_number = int(row["line_number"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                print(f"Ignoring malformed {path.name} line {jsonl_line}: {exc}")
                continue
            if row.get("model_id") != model_id or line_number not in expected_lines:
                continue
            if line_number not in predictions or finite_probability(row):
                predictions[line_number] = row
    return predictions


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
    temporary.replace(path)


def queried_row(
    session: requests.Session,
    model_id: str,
    row: pd.Series,
    api_key: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result = request_prediction(session, model_id, row["patient_prompt"], api_key, args)
    if not result["ok"] and credit_error(result["error"]):
        raise RuntimeError(f"OpenRouter credit error: {result['error']}")
    model_label: float | int = math.nan
    probability = math.nan
    parse_error: str | None = None
    if result["ok"]:
        try:
            model_label, probability = parse_model_output(result["content"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            parse_error = str(exc)
    usage = result["usage"]
    return {
        "model_id": model_id,
        "resolved_model_id": result["resolved_model_id"],
        "row_id": row["row_id"],
        "line_number": int(row["line_number"]),
        "y_true": int(row["y_true"]),
        "model_label": model_label,
        "pred_label_0_5": int(probability >= 0.5) if math.isfinite(probability) else math.nan,
        "rejection_probability": probability,
        "api_ok": result["ok"],
        "api_error": result["error"],
        "parse_error": parse_error,
        "finish_reason": result["finish_reason"],
        "request_mode": result["request_mode"],
        "prompt_tokens": float(usage.get("prompt_tokens") or 0),
        "completion_tokens": float(usage.get("completion_tokens") or 0),
        "total_tokens": float(usage.get("total_tokens") or 0),
        "raw_output": result["content"],
    }


def run_model(
    model_id: str,
    evaluation: pd.DataFrame,
    output_root: Path,
    api_key: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    model_dir = output_root / re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")
    model_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = model_dir / "predictions.jsonl"
    expected_lines = set(evaluation["line_number"].astype(int))
    predictions = (
        {} if args.overwrite else read_prediction_map(jsonl_path, model_id, expected_lines)
    )
    session = requests.Session()

    for pass_number in range(args.invalid_retry_passes + 1):
        pending = [
            row
            for _, row in evaluation.iterrows()
            if not finite_probability(predictions.get(int(row["line_number"]), {}))
        ]
        if not pending:
            break
        print(
            f"{model_id}: pass {pass_number + 1}/{args.invalid_retry_passes + 1}; "
            f"{len(pending)} rows pending",
            flush=True,
        )
        for position, row in enumerate(pending, start=1):
            prediction = queried_row(session, model_id, row, api_key, args)
            predictions[int(row["line_number"])] = prediction
            ordered = [
                predictions[line]
                for line in evaluation["line_number"].astype(int)
                if line in predictions
            ]
            write_jsonl(jsonl_path, ordered)
            print(
                f"{model_id}: {position}/{len(pending)} "
                f"CSV-line={row['line_number']} probability="
                f"{prediction['rejection_probability']}",
                flush=True,
            )
            time.sleep(args.request_interval)

    ordered = [
        predictions.get(
            int(row["line_number"]),
            {
                "model_id": model_id,
                "row_id": row["row_id"],
                "line_number": int(row["line_number"]),
                "y_true": int(row["y_true"]),
                "api_ok": False,
                "api_error": "missing prediction after retry passes",
                "parse_error": None,
                "rejection_probability": math.nan,
            },
        )
        for _, row in evaluation.iterrows()
    ]
    write_jsonl(jsonl_path, ordered)
    frame = pd.DataFrame(ordered)
    frame.to_csv(model_dir / "predictions.csv", index=False)
    return frame


def safe_metric(function: Callable[..., float], *values: Any, **kwargs: Any) -> float:
    try:
        return float(function(*values, **kwargs))
    except ValueError:
        return math.nan


def youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return DEFAULT_THRESHOLD
    fpr, tpr, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return DEFAULT_THRESHOLD
    return float(thresholds[finite][np.argmax(tpr[finite] - fpr[finite])])


def point_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "n_samples": int(labels.size),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "roc_auc": safe_metric(roc_auc_score, labels, scores),
        "pr_auc": safe_metric(average_precision_score, labels, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def stratified_bootstrap_cis(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, float]:
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
        "pr_auc_ci_lower": float(lower[2]),
        "pr_auc_ci_upper": float(upper[2]),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def summarize_model(
    model_id: str,
    predictions: pd.DataFrame,
    model_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    valid = predictions[
        predictions["rejection_probability"].map(
            lambda value: isinstance(value, (int, float, np.integer, np.floating))
            and math.isfinite(float(value))
        )
    ].copy()
    summary: dict[str, Any] = {
        "model_id": model_id,
        "resolved_model_ids": sorted(
            str(value)
            for value in valid.get(
                "resolved_model_id", pd.Series(dtype=str)
            ).dropna().unique()
        ),
        "n_records": int(len(predictions)),
        "n_valid_predictions": int(len(valid)),
        "coverage": float(len(valid) / len(predictions)) if len(predictions) else 0.0,
        "n_api_errors": int(
            predictions.get("api_error", pd.Series(dtype=object)).notna().sum()
        ),
        "n_parse_errors": int(
            predictions.get("parse_error", pd.Series(dtype=object)).notna().sum()
        ),
        "total_tokens": int(
            pd.to_numeric(predictions.get("total_tokens", 0), errors="coerce").fillna(0).sum()
        ),
        "threshold_strategy": args.threshold_strategy,
    }
    if valid.empty:
        summary["error"] = "no valid predictions"
    else:
        labels = valid["y_true"].to_numpy(dtype=int)
        scores = valid["rejection_probability"].to_numpy(dtype=float)
        threshold = (
            youden_threshold(labels, scores)
            if args.threshold_strategy == "evaluation-youden"
            else args.fixed_threshold
        )
        summary.update(point_metrics(labels, scores, threshold))
        summary.update(
            stratified_bootstrap_cis(
                labels, scores, threshold, args.bootstrap_replicates, args.seed
            )
        )
    with (model_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    return summary


def write_run_metadata(
    path: Path,
    args: argparse.Namespace,
    preset: CohortPreset,
    eval_csv: Path,
    run_name: str,
    models: list[str],
    features: list[str],
    source_columns: pd.Index,
) -> None:
    metadata = {
        "cohort": args.cohort,
        "eval_csv": str(eval_csv.resolve()),
        "eval_csv_sha256": sha256_file(eval_csv),
        "run_name": run_name,
        "requested_model_ids": models,
        "label_column": LABEL_COLUMN,
        "feature_policy": preset.feature_policy,
        "feature_columns": features,
        "dropped_columns_present": [
            column for column in source_columns if column in preset.drop_columns
        ],
        "threshold_strategy": args.threshold_strategy,
        "fixed_threshold": args.fixed_threshold,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "reasoning_max_tokens": args.reasoning_max_tokens,
        "request_timeout": args.request_timeout,
        "request_retries": args.request_retries,
        "request_interval": args.request_interval,
        "limit": args.limit,
        "software_versions": software_versions(),
        "methodological_note": (
            "evaluation-youden estimates the operating threshold on the same "
            "evaluation labels used for thresholded metrics; this reproduces the "
            "reported spreadsheet analysis."
            if args.threshold_strategy == "evaluation-youden"
            else "A prespecified fixed threshold is used."
        ),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    preset = cohort_presets(args.data_root)[args.cohort]
    eval_csv = (args.eval_csv or preset.eval_csv).expanduser()
    run_name = args.run_name or preset.run_name
    models = list(dict.fromkeys(args.models or REPORTED_MODEL_IDS))
    evaluation, features, source_columns = load_evaluation_data(
        eval_csv, preset.drop_columns, args.limit
    )
    output_root = args.output_root.expanduser() / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    write_run_metadata(
        output_root / "run_config.json",
        args,
        preset,
        eval_csv,
        run_name,
        models,
        features,
        source_columns,
    )

    print(f"Cohort: {args.cohort}")
    print(f"Input: {eval_csv}")
    print(f"Rows: {len(evaluation)}")
    print(f"Features: {len(features)} ({preset.feature_policy})")
    print(f"Labels:\n{evaluation['y_true'].value_counts().sort_index().to_string()}")
    print(f"Output: {output_root}")
    if args.validate_only:
        print("Validation complete; no API requests were made.")
        return

    api_key = get_api_key()
    summaries: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    for model_id in models:
        print(f"\nRunning {model_id}", flush=True)
        predictions = run_model(model_id, evaluation, output_root, api_key, args)
        model_dir = output_root / re.sub(
            r"[^A-Za-z0-9._-]+", "_", model_id
        ).strip("_")
        summaries.append(summarize_model(model_id, predictions, model_dir, args))
        all_predictions.append(predictions)

    comparison = pd.DataFrame(summaries)
    comparison.to_csv(output_root / "model_comparison.csv", index=False)
    comparison.to_json(
        output_root / "model_comparison.json", orient="records", indent=2
    )
    pd.concat(all_predictions, ignore_index=True).to_csv(
        output_root / "all_predictions.csv", index=False
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

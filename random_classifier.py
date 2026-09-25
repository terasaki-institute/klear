# ==============================================================================
# Random Classifier Baseline (Table 1)
# ==============================================================================
"""
Reproduce the "Random classifier" row used in manuscript Table 1.

Closed-form (binary, positive prevalence P = n_pos / n):

  ROC-AUC  = 0.50
      Random scores independent of labels.

  AUPRC    = P
      Expected average precision for a random / uninformative score
      equals the positive class prevalence.
      Examples (high-res train, n=319):
        TCMR  P=22/319 ≈ 0.069  -> 0.07
        ABMR  P=12/319 ≈ 0.038  -> 0.04

  Macro F1 = 0.5 * [ P/(P+0.5) + (1-P)/(1.5-P) ]
      Expected macro-F1 for a *uniform* random hard classifier that
      predicts class 0 or 1 each with probability 1/2 (not prevalence-matched).

      Derivation (binary, pred ~ Bern(0.5) independent of y):
        recall_1  = 0.5
        prec_1    = P
        F1_1      = P / (P + 0.5)

        recall_0  = 0.5
        prec_0    = 1 - P
        F1_0      = (1-P) / (1.5 - P)

        Macro F1  = (F1_0 + F1_1) / 2
                  = 0.5 * ( P/(P+0.5) + (1-P)/(1.5-P) )

      Examples:
        TCMR  -> ≈ 0.386  -> 0.39
        ABMR  -> ≈ 0.364  -> 0.36

Usage
-----
  python random_classifier.py ^
    --data datasets/mayo_sites_merged_highres_train.csv ^
    --outcome_col outcome_tcmr ^
    --output random_baselines_tcmr.csv

  python random_classifier.py ^
    --data datasets/mayo_sites_merged_highres_train.csv ^
    --outcome_col outcome_abmr ^
    --positive_count 12 --n_total 319   # optional: force Table 1 counts
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


def macro_f1_uniform_random(P: float) -> float:
    """
    Expected macro F1 for uniform random hard classifier (P(pred=1)=0.5).
    P = positive prevalence.
    """
    P = float(P)
    if not (0.0 <= P <= 1.0):
        raise ValueError(f"Prevalence P must be in [0,1], got {P}")
    f1_pos = P / (P + 0.5)
    f1_neg = (1.0 - P) / (1.5 - P)
    return 0.5 * (f1_pos + f1_neg)


def random_classifier_table1(P: float, n_pos=None, n_total=None, decimals=None):
    """
    Table-1 random classifier metrics from prevalence P.

    If decimals is an int, round like the manuscript (AUPRC/F1 to 2 dp, AUC 0.50).
    """
    P = float(P)
    auc = 0.50
    auprc = P
    macro_f1 = macro_f1_uniform_random(P)

    if decimals is not None:
        auc = round(auc, decimals)
        auprc = round(auprc, decimals)
        macro_f1 = round(macro_f1, decimals)

    out = {
        "n_pos": n_pos,
        "n_total": n_total,
        "prevalence_P": P,
        "AUPRC": auprc,
        "Macro_F1": macro_f1,
        "ROC_AUC": auc,
        "AUPRC_formula": "P",
        "Macro_F1_formula": "0.5 * (P/(P+0.5) + (1-P)/(1.5-P))",
        "ROC_AUC_formula": "0.5",
    }
    return out


def metrics_from_labels(y, positive_label=1, decimals=None):
    y = np.asarray(y)
    n_total = len(y)
    n_pos = int(np.sum(y == positive_label))
    P = n_pos / n_total if n_total else 0.0
    row = random_classifier_table1(P, n_pos=n_pos, n_total=n_total, decimals=decimals)
    row["positive_label"] = positive_label
    return row


def metrics_from_csv(path, outcome_col, decimals=None):
    df = pd.read_csv(path)
    if outcome_col not in df.columns:
        raise KeyError(f"'{outcome_col}' not in {path}. Columns: {list(df.columns)}")

    y_raw = df[outcome_col].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    # After LabelEncoder, positive class is typically the minority / label 1
    # Prefer encoded 1 if present; else use max label.
    pos = 1 if 1 in np.unique(y) else int(np.max(y))
    row = metrics_from_labels(y, positive_label=pos, decimals=decimals)
    row["dataset"] = os.path.basename(path)
    row["outcome_col"] = outcome_col
    row["label_mapping"] = {int(i): str(c) for i, c in enumerate(le.classes_)}
    row["class_counts"] = {
        int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))
    }
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Table-1 random classifier baselines (AUC / AUPRC / Macro F1)"
    )
    parser.add_argument(
        "--data",
        nargs="*",
        default=None,
        help="Optional labeled CSV(s). If omitted, use --positive_count/--n_total.",
    )
    parser.add_argument(
        "--outcome_col",
        type=str,
        default=None,
        help="Outcome column (required with --data)",
    )
    parser.add_argument(
        "--positive_count",
        type=int,
        default=None,
        help="n_pos (e.g. 22 for TCMR, 12 for ABMR on n=319)",
    )
    parser.add_argument(
        "--n_total",
        type=int,
        default=None,
        help="n_total (e.g. 319)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional row name when using --positive_count/--n_total",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=2,
        dest="decimals",
        help="Round metrics to this many decimals (Table 1 uses 2). Use -1 for full precision.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="random_classifier_baselines.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    decimals = None if args.decimals < 0 else args.decimals
    rows = []

    # Explicit counts (exact Table 1 inputs)
    if args.positive_count is not None or args.n_total is not None:
        if args.positive_count is None or args.n_total is None:
            raise ValueError("Provide both --positive_count and --n_total")
        P = args.positive_count / args.n_total
        row = random_classifier_table1(
            P,
            n_pos=args.positive_count,
            n_total=args.n_total,
            decimals=decimals,
        )
        row["dataset"] = args.name or f"n_pos={args.positive_count}_n={args.n_total}"
        row["outcome_col"] = args.outcome_col or ""
        rows.append(row)

    if args.data:
        if not args.outcome_col:
            raise ValueError("--outcome_col is required with --data")
        for path in args.data:
            rows.append(metrics_from_csv(path, args.outcome_col, decimals=decimals))

    if not rows:
        # Demo: reproduce Table 1 random rows
        print("No inputs given — reproducing Table 1 random classifier rows:\n")
        for name, n_pos, n_tot in [("TCMR", 22, 319), ("ABMR", 12, 319)]:
            row = random_classifier_table1(
                n_pos / n_tot, n_pos=n_pos, n_total=n_tot, decimals=decimals
            )
            row["dataset"] = name
            row["outcome_col"] = f"outcome_{name.lower()}"
            rows.append(row)

    out_df = pd.DataFrame(rows)

    # Print manuscript-style preview
    print("Random classifier baselines (Table 1 formulas)")
    print("  AUC   = 0.50")
    print("  AUPRC = P")
    print("  Macro F1 = 0.5 * (P/(P+0.5) + (1-P)/(1.5-P))  [uniform random preds]\n")
    cols = [
        c
        for c in [
            "dataset",
            "outcome_col",
            "n_pos",
            "n_total",
            "prevalence_P",
            "AUPRC",
            "Macro_F1",
            "ROC_AUC",
        ]
        if c in out_df.columns
    ]
    print(out_df[cols].to_string(index=False))

    out_df.to_csv(args.output, index=False)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()

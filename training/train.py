from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "machine_data.csv"
MODEL_PATH = BASE_DIR / "models" / "fault_model.joblib"
MODEL_VERSION = "v1.3"

FEATURES = [
    "temperature",
    "pressure",
    "vibration",
    "run_hours",
    "alarm_count",
    "maintenance_days",
]
TARGET = "fault"
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)
MIN_RECALL = 0.75
FALSE_NEGATIVE_COST = 10_000
FALSE_POSITIVE_COST = 100
FP_COST_SCENARIOS = (100, 500, 1_000, 2_000, 5_000)


def metrics_at_threshold(
    y_true: pd.Series,
    probabilities,
    threshold: float,
    false_negative_cost: int = FALSE_NEGATIVE_COST,
    false_positive_cost: int = FALSE_POSITIVE_COST,
) -> dict:
    y_pred = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    business_cost = (fn * false_negative_cost) + (fp * false_positive_cost)
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "business_cost": int(business_cost),
    }


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n--- {title} ---")
    print(f'Threshold: {metrics["threshold"]:.2f}')
    print("Confusion matrix:")
    print([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    print(
        f'TN={metrics["tn"]} FP={metrics["fp"]} '
        f'FN={metrics["fn"]} TP={metrics["tp"]}'
    )
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall: {metrics["recall"]:.4f}')
    print(f'F1: {metrics["f1"]:.4f}')
    print(f'Business cost: ${metrics["business_cost"]:,}')


def threshold_candidates(
    y_true: pd.Series,
    probabilities,
    min_recall: float,
    false_negative_cost: int = FALSE_NEGATIVE_COST,
    false_positive_cost: int = FALSE_POSITIVE_COST,
) -> list[dict]:
    candidates = []
    for threshold in np.arange(0.01, 1.00, 0.01):
        metrics = metrics_at_threshold(
            y_true,
            probabilities,
            threshold,
            false_negative_cost=false_negative_cost,
            false_positive_cost=false_positive_cost,
        )
        if metrics["recall"] >= min_recall:
            candidates.append(metrics)

    if not candidates:
        raise ValueError(f"No threshold satisfies minimum recall {min_recall:.2%}")
    return candidates


def choose_threshold_by_precision(y_true: pd.Series, probabilities, min_recall: float) -> dict:
    candidates = threshold_candidates(y_true, probabilities, min_recall)
    return max(
        candidates,
        key=lambda item: (item["precision"], item["f1"], item["threshold"]),
    )


def choose_threshold_by_cost(
    y_true: pd.Series,
    probabilities,
    min_recall: float,
    false_negative_cost: int = FALSE_NEGATIVE_COST,
    false_positive_cost: int = FALSE_POSITIVE_COST,
) -> dict:
    candidates = threshold_candidates(
        y_true,
        probabilities,
        min_recall,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    return min(
        candidates,
        key=lambda item: (
            item["business_cost"],
            -item["recall"],
            -item["precision"],
            -item["threshold"],
        ),
    )


def print_cost_sensitivity(y_true: pd.Series, probabilities) -> None:
    print("\n--- Business cost sensitivity analysis on VALIDATION ---")
    print(f"FN cost fixed at ${FALSE_NEGATIVE_COST:,}")
    print("FP COST | BEST THRESHOLD | RECALL | PRECISION | FP | FN | TOTAL COST")
    for fp_cost in FP_COST_SCENARIOS:
        choice = choose_threshold_by_cost(
            y_true,
            probabilities,
            MIN_RECALL,
            false_negative_cost=FALSE_NEGATIVE_COST,
            false_positive_cost=fp_cost,
        )
        print(
            f'${fp_cost:>6,} | {choice["threshold"]:>14.2f} | '
            f'{choice["recall"]:>6.2%} | {choice["precision"]:>9.2%} | '
            f'{choice["fp"]:>2} | {choice["fn"]:>2} | '
            f'${choice["business_cost"]:,}'
        )


def train() -> None:
    df = pd.read_csv(DATA_PATH)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )

    validation_fraction = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_fraction,
        random_state=42,
        stratify=y_train_val,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    validation_probabilities = pipeline.predict_proba(X_val)[:, 1]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Training fault rate: {y_train.mean():.2%}")
    print(f"Validation fault rate: {y_val.mean():.2%}")
    print(f"Test fault rate: {y_test.mean():.2%}")
    print(
        f"Business costs: FN=${FALSE_NEGATIVE_COST:,}, "
        f"FP=${FALSE_POSITIVE_COST:,}"
    )

    print("\n--- Validation reference thresholds ---")
    for threshold in THRESHOLDS:
        metrics = metrics_at_threshold(y_val, validation_probabilities, threshold)
        print(
            f'{threshold:.2f}: precision={metrics["precision"]:.4f} '
            f'recall={metrics["recall"]:.4f} f1={metrics["f1"]:.4f} '
            f'FP={metrics["fp"]} FN={metrics["fn"]} '
            f'cost=${metrics["business_cost"]:,}'
        )

    precision_choice = choose_threshold_by_precision(
        y_val, validation_probabilities, MIN_RECALL
    )
    cost_choice = choose_threshold_by_cost(
        y_val, validation_probabilities, MIN_RECALL
    )

    print_metrics("Precision-optimized threshold on VALIDATION", precision_choice)
    print_metrics("Business-cost-optimized threshold on VALIDATION", cost_choice)
    print(f"Business requirement: validation recall >= {MIN_RECALL:.0%}")

    # This is an analysis only: vary the assumed cost of a false alarm while
    # keeping the trained model and validation predictions exactly the same.
    print_cost_sensitivity(y_val, validation_probabilities)

    # The production artifact still uses the agreed baseline business costs.
    # Freeze that threshold before touching the test set.
    chosen = cost_choice
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    final_test = metrics_at_threshold(y_test, test_probabilities, chosen["threshold"])
    print_metrics("Final evaluation on untouched TEST set", final_test)

    artifact = {
        "model": pipeline,
        "model_version": MODEL_VERSION,
        "threshold": chosen["threshold"],
        "features": FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "min_recall_requirement": MIN_RECALL,
        "false_negative_cost": FALSE_NEGATIVE_COST,
        "false_positive_cost": FALSE_POSITIVE_COST,
        "threshold_selection_method": "minimum_business_cost_with_recall_constraint",
        "precision_optimized_validation_evaluation": precision_choice,
        "validation_evaluation": chosen,
        "test_evaluation": final_test,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nTraining completed with {len(df)} total samples")
    print(f"Model artifact saved to {MODEL_PATH}")
    print(f"Artifact model version: {MODEL_VERSION}")
    print(f'Artifact threshold: {chosen["threshold"]:.2f}')
    print("Threshold selection: minimum business cost with recall constraint")


if __name__ == "__main__":
    train()

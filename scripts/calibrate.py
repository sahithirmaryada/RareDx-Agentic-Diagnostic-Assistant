#!/usr/bin/env python3
"""Calibrate RareDx top-1 confidence against benchmark evaluation results.

This script learns a lightweight logistic calibration model from an evaluation
CSV produced by scripts/evaluate.py. It treats top1_correct as the target and
exports cross-validated confidence estimates plus reliability metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SOURCE = "scripts/calibrate.py"
EPSILON = 1e-9

FEATURE_NAMES = [
    "top_score",
    "score_margin",
    "num_candidates",
    "graph_candidate_count",
    "citation_candidate_count",
    "icd10_candidate_count",
    "top_graph_path_count",
    "top_citation_count",
    "top_has_icd10",
    "top_evidence_badge_count",
    "top_matched_symptom_count",
    "top_matched_gene_count",
]


@dataclass
class CalibrationExample:
    row_index: int
    case_id: str
    strategy: str
    difficulty: str
    label: int
    features: list[float]
    top_score: float
    score_margin: float
    top_disease: str
    top_orphacode: str


@dataclass
class CalibrationModel:
    model_type: str
    feature_names: list[str]
    intercept: float
    coefficients: list[float]
    means: list[float]
    scales: list[float]
    constant_probability: float | None = None

    def predict_proba(self, features: list[float]) -> float:
        if self.constant_probability is not None:
            return self.constant_probability

        z = self.intercept
        for value, coefficient, center, scale in zip(
            features,
            self.coefficients,
            self.means,
            self.scales,
        ):
            z += coefficient * ((value - center) / scale)
        return sigmoid(z)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.model_type,
            "feature_names": self.feature_names,
            "intercept": self.intercept,
            "coefficients": self.coefficients,
            "means": self.means,
            "scales": self.scales,
            "constant_probability": self.constant_probability,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit benchmark calibration for RareDx top-1 confidence."
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Evaluation results CSV from scripts/evaluate.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON calibration artifact path.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=None,
        help="Output CSV with cross-validated calibrated confidence per row.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional label for deterministic output file names.",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        default=[],
        help="Optional strategy filter. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds. Default: 5.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=5,
        help="Reliability bins for ECE/MCE. Default: 5.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4000,
        help="Gradient-descent epochs for logistic calibration. Default: 4000.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Gradient-descent learning rate. Default: 0.05.",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=0.1,
        help="L2 regularization strength. Default: 0.1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for cross-validation splits. Default: 17.",
    )
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2.")
    if args.bins < 2:
        parser.error("--bins must be at least 2.")
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if args.l2 < 0:
        parser.error("--l2 must be non-negative.")
    return args


def load_examples(results_path: Path, strategies: set[str]) -> list[CalibrationExample]:
    if not results_path.exists():
        raise FileNotFoundError(f"Evaluation results CSV not found: {results_path}")

    examples: list[CalibrationExample] = []
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_results_fields(reader.fieldnames or [])
        for row_index, row in enumerate(reader, start=1):
            if row.get("status") != "ok":
                continue
            if strategies and row.get("strategy") not in strategies:
                continue
            examples.append(example_from_row(row_index, row))

    if not examples:
        strategy_text = f" for strategies {sorted(strategies)}" if strategies else ""
        raise ValueError(f"No successful evaluation rows found{strategy_text}.")
    return examples


def validate_results_fields(fieldnames: list[str]) -> None:
    required = {
        "case_id",
        "strategy",
        "difficulty",
        "status",
        "top1_correct",
        "top_score",
        "num_candidates",
        "graph_candidate_count",
        "citation_candidate_count",
        "icd10_candidate_count",
        "top_disease",
        "top_orphacode",
        "ranked_candidates_json",
    }
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(
            "Results CSV is missing required fields: " + ", ".join(sorted(missing))
        )


def example_from_row(row_index: int, row: dict[str, str]) -> CalibrationExample:
    candidates = parse_candidates(row.get("ranked_candidates_json", "[]"))
    top_candidate = candidates[0] if candidates else {}
    second_candidate = candidates[1] if len(candidates) > 1 else {}

    top_score = parse_float(row.get("top_score")) or parse_float(top_candidate.get("score"))
    second_score = parse_float(second_candidate.get("score"))
    score_margin = top_score - second_score if second_candidate else top_score
    difficulty = row.get("difficulty", "")

    features = [
        top_score,
        score_margin,
        parse_float(row.get("num_candidates")),
        parse_float(row.get("graph_candidate_count")),
        parse_float(row.get("citation_candidate_count")),
        parse_float(row.get("icd10_candidate_count")),
        parse_float(top_candidate.get("graph_path_count")),
        parse_float(top_candidate.get("citation_count")),
        1.0 if top_candidate.get("icd10") else 0.0,
        float(len(top_candidate.get("evidence_badges", []) or [])),
        float(len(top_candidate.get("matched_symptoms", []) or [])),
        float(len(top_candidate.get("matched_genes", []) or [])),
    ]

    return CalibrationExample(
        row_index=row_index,
        case_id=row.get("case_id", ""),
        strategy=row.get("strategy", ""),
        difficulty=difficulty,
        label=int(parse_float(row.get("top1_correct"))),
        features=features,
        top_score=top_score,
        score_margin=score_margin,
        top_disease=row.get("top_disease", ""),
        top_orphacode=row.get("top_orphacode", ""),
    )


def parse_candidates(value: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def parse_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def calibrate_examples(
    examples: list[CalibrationExample],
    folds: int,
    bins: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> dict[str, Any]:
    indices = list(range(len(examples)))
    fold_indices = make_stratified_folds(examples, min(folds, len(examples)), seed)
    predictions = [0.0 for _ in examples]

    for test_indices in fold_indices:
        train_indices = [index for index in indices if index not in set(test_indices)]
        train_examples = [examples[index] for index in train_indices]
        model = fit_calibration_model(
            train_examples,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        for index in test_indices:
            predictions[index] = model.predict_proba(examples[index].features)

    full_model = fit_calibration_model(
        examples,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    labels = [example.label for example in examples]
    metrics = calibration_metrics(labels, predictions, bins)

    return {
        "metrics": metrics,
        "model": full_model.to_dict(),
        "predictions": predictions,
        "folds": len(fold_indices),
    }


def make_stratified_folds(
    examples: list[CalibrationExample],
    folds: int,
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    positives = [index for index, example in enumerate(examples) if example.label == 1]
    negatives = [index for index, example in enumerate(examples) if example.label == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    fold_indices = [[] for _ in range(folds)]
    for bucket in [positives, negatives]:
        for offset, index in enumerate(bucket):
            fold_indices[offset % folds].append(index)

    return [fold for fold in fold_indices if fold]


def fit_calibration_model(
    examples: list[CalibrationExample],
    epochs: int,
    learning_rate: float,
    l2: float,
) -> CalibrationModel:
    labels = [example.label for example in examples]
    positive_rate = clipped_probability(mean(labels))
    if len(set(labels)) < 2:
        return CalibrationModel(
            model_type="constant",
            feature_names=FEATURE_NAMES,
            intercept=logit(positive_rate),
            coefficients=[0.0 for _ in FEATURE_NAMES],
            means=[0.0 for _ in FEATURE_NAMES],
            scales=[1.0 for _ in FEATURE_NAMES],
            constant_probability=positive_rate,
        )

    raw_features = [example.features for example in examples]
    means, scales = feature_stats(raw_features)
    features = [standardize(row, means, scales) for row in raw_features]
    weights = [0.0 for _ in FEATURE_NAMES]
    intercept = logit(positive_rate)

    for _ in range(epochs):
        intercept_grad = 0.0
        weight_grads = [0.0 for _ in FEATURE_NAMES]
        for row, label in zip(features, labels):
            prediction = sigmoid(intercept + dot(weights, row))
            error = prediction - label
            intercept_grad += error
            for index, value in enumerate(row):
                weight_grads[index] += error * value

        n = len(features)
        intercept -= learning_rate * (intercept_grad / n)
        for index in range(len(weights)):
            gradient = (weight_grads[index] / n) + (l2 * weights[index])
            weights[index] -= learning_rate * gradient

    return CalibrationModel(
        model_type="logistic_regression",
        feature_names=FEATURE_NAMES,
        intercept=intercept,
        coefficients=weights,
        means=means,
        scales=scales,
    )


def feature_stats(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    means = []
    scales = []
    for column in zip(*rows):
        column_mean = mean(column)
        variance = mean((value - column_mean) ** 2 for value in column)
        scale = math.sqrt(variance)
        means.append(column_mean)
        scales.append(scale if scale > EPSILON else 1.0)
    return means, scales


def standardize(row: list[float], means: list[float], scales: list[float]) -> list[float]:
    return [(value - center) / scale for value, center, scale in zip(row, means, scales)]


def calibration_metrics(labels: list[int], predictions: list[float], bins: int) -> dict[str, Any]:
    reliability_bins = reliability(labels, predictions, bins)
    total = len(labels)
    brier = mean((prediction - label) ** 2 for prediction, label in zip(predictions, labels))
    nll = mean(
        -(
            label * math.log(clipped_probability(prediction))
            + (1 - label) * math.log(1 - clipped_probability(prediction))
        )
        for prediction, label in zip(predictions, labels)
    )
    ece = sum((bin_row["count"] / total) * bin_row["abs_gap"] for bin_row in reliability_bins)
    mce = max((bin_row["abs_gap"] for bin_row in reliability_bins), default=0.0)
    predicted_positive_rate = mean(predictions)
    observed_accuracy = mean(labels)

    return {
        "cases": total,
        "observed_top1_accuracy": round(observed_accuracy, 6),
        "mean_calibrated_confidence": round(predicted_positive_rate, 6),
        "brier_score": round(brier, 6),
        "negative_log_likelihood": round(nll, 6),
        "expected_calibration_error": round(ece, 6),
        "maximum_calibration_error": round(mce, 6),
        "reliability_bins": reliability_bins,
    }


def reliability(labels: list[int], predictions: list[float], bins: int) -> list[dict[str, Any]]:
    rows = []
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        selected = []
        for prediction, label in zip(predictions, labels):
            in_bin = (
                lower <= prediction <= upper
                if bin_index == bins - 1
                else lower <= prediction < upper
            )
            if in_bin:
                selected.append((prediction, label))

        if selected:
            avg_prediction = mean(prediction for prediction, _ in selected)
            observed = mean(label for _, label in selected)
            gap = abs(avg_prediction - observed)
        else:
            avg_prediction = 0.0
            observed = 0.0
            gap = 0.0

        rows.append(
            {
                "bin": bin_index + 1,
                "lower": round(lower, 6),
                "upper": round(upper, 6),
                "count": len(selected),
                "avg_confidence": round(avg_prediction, 6),
                "observed_accuracy": round(observed, 6),
                "abs_gap": round(gap, 6),
            }
        )
    return rows


def write_artifact(
    output_path: Path,
    results_path: Path,
    examples: list[CalibrationExample],
    calibration: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    strategies = sorted({example.strategy for example in examples})
    artifact = {
        "source": SOURCE,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results_path": str(results_path),
        "strategies": strategies,
        "target": "top1_correct",
        "scope": "top1_diagnostic_confidence",
        "note": (
            "Calibrated confidence estimates probability that the top-ranked "
            "candidate is correct for the evaluated benchmark distribution."
        ),
        "feature_names": FEATURE_NAMES,
        "training": {
            "rows": len(examples),
            "folds": calibration["folds"],
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "seed": args.seed,
        },
        "cross_validated_metrics": calibration["metrics"],
        "model": calibration["model"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_predictions(
    output_path: Path,
    examples: list[CalibrationExample],
    predictions: list[float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row_index",
        "case_id",
        "strategy",
        "difficulty",
        "top_disease",
        "top_orphacode",
        "top1_correct",
        "top_score",
        "score_margin",
        "calibrated_confidence",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for example, prediction in zip(examples, predictions):
            writer.writerow(
                {
                    "row_index": example.row_index,
                    "case_id": example.case_id,
                    "strategy": example.strategy,
                    "difficulty": example.difficulty,
                    "top_disease": example.top_disease,
                    "top_orphacode": example.top_orphacode,
                    "top1_correct": example.label,
                    "top_score": round(example.top_score, 6),
                    "score_margin": round(example.score_margin, 6),
                    "calibrated_confidence": round(prediction, 6),
                }
            )


def default_output_paths(results_path: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    label = args.timestamp or results_path.stem
    if not args.timestamp and label.startswith("results_"):
        label = label[len("results_") :]
    output = args.output or results_path.with_name(f"calibration_{label}.json")
    predictions = args.predictions_output or results_path.with_name(
        f"calibration_predictions_{label}.csv"
    )
    return output, predictions


def print_summary(
    calibration: dict[str, Any],
    output_path: Path,
    predictions_path: Path,
) -> None:
    metrics = calibration["metrics"]
    print("\nRareDx Calibration Summary")
    print("==========================")
    print(f"Cases: {metrics['cases']}")
    print(f"Observed Top-1 accuracy: {metrics['observed_top1_accuracy']:.3f}")
    print(f"Mean calibrated confidence: {metrics['mean_calibrated_confidence']:.3f}")
    print(f"Brier score: {metrics['brier_score']:.3f}")
    print(f"ECE: {metrics['expected_calibration_error']:.3f}")
    print(f"MCE: {metrics['maximum_calibration_error']:.3f}")
    print(f"NLL: {metrics['negative_log_likelihood']:.3f}")
    print("\nReliability")
    print("-----------")
    for row in metrics["reliability_bins"]:
        print(
            f"{row['lower']:.1f}-{row['upper']:.1f}: "
            f"predicted={row['avg_confidence']:.3f}, "
            f"observed={row['observed_accuracy']:.3f}, "
            f"n={row['count']}"
        )
    print(f"\nCalibration JSON: {output_path}")
    print(f"Per-case predictions CSV: {predictions_path}")


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 35.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -35.0))
    return z / (1.0 + z)


def clipped_probability(value: float) -> float:
    return min(max(float(value), 1e-6), 1 - 1e-6)


def logit(probability: float) -> float:
    probability = clipped_probability(probability)
    return math.log(probability / (1.0 - probability))


def main() -> int:
    args = parse_args()
    results_path = args.results if args.results.is_absolute() else PROJECT_ROOT / args.results
    output_path, predictions_path = default_output_paths(results_path, args)
    examples = load_examples(results_path, set(args.strategy))
    calibration = calibrate_examples(
        examples,
        folds=args.folds,
        bins=args.bins,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    write_artifact(output_path, results_path, examples, calibration, args)
    write_predictions(predictions_path, examples, calibration["predictions"])
    print_summary(calibration, output_path, predictions_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

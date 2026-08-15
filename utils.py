"""Shared data preparation and evaluation helpers for model training scripts."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


LABEL_NAMES = ("weakness", "MAT")
DIMENSION_METRICS = ("accuracy", "precision", "recall", "f1")
METRIC_NAMES = tuple(
    f"{label_name}_{metric_name}"
    for label_name in LABEL_NAMES
    for metric_name in DIMENSION_METRICS
) + tuple(f"average_{metric_name}" for metric_name in DIMENSION_METRICS)
SPLIT_FILES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}


def add_common_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_model_id: str,
    default_output_dir: str,
    default_epochs: int,
    default_batch_size: int,
    default_train_limit: int = 100,
) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("curated_dataset_steps"),
        help="Directory containing train.csv, validation.csv, and test.csv.",
    )
    parser.add_argument(
        "--model-id",
        default=default_model_id,
        help="Pretrained model ID or local model path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(default_output_dir),
        help="Directory for epoch checkpoints, final model, and metrics.",
    )
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument(
        "--train-limit",
        type=int,
        default=default_train_limit,
        help="Maximum training rows; use 0 for the complete training split.",
    )
    parser.add_argument("--seed", type=int, default=42)


def validate_common_training_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.train_limit < 0:
        parser.error("--train-limit cannot be negative")


def normalize_binary_matrix(values: Any, name: str) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    matrix = np.asarray(values, dtype=np.int64)
    if matrix.ndim == 1:
        if matrix.size == 0:
            matrix = matrix.reshape(0, len(LABEL_NAMES))
        elif matrix.size == len(LABEL_NAMES):
            matrix = matrix.reshape(1, len(LABEL_NAMES))
    if matrix.ndim != 2 or matrix.shape[1] != len(LABEL_NAMES):
        raise ValueError(
            f"{name} must have shape (rows, {len(LABEL_NAMES)}), got {matrix.shape}"
        )
    if not np.isin(matrix, (0, 1)).all():
        raise ValueError(f"{name} contains values other than 0 and 1")
    return matrix


def compute_binary_metrics(predictions: Any, references: Any) -> dict[str, float]:
    """Return binary metrics per dimension and their unweighted macro means."""
    predictions_array = normalize_binary_matrix(predictions, "predictions")
    references_array = normalize_binary_matrix(references, "references")
    if predictions_array.shape != references_array.shape:
        raise ValueError(
            "Predictions and references must have identical shapes, got "
            f"{predictions_array.shape} and {references_array.shape}"
        )

    metrics: dict[str, float] = {}
    scores: dict[str, list[float]] = {
        metric_name: [] for metric_name in DIMENSION_METRICS
    }
    for index, label_name in enumerate(LABEL_NAMES):
        predicted = predictions_array[:, index]
        expected = references_array[:, index]
        true_positives = int(np.sum((predicted == 1) & (expected == 1)))
        false_positives = int(np.sum((predicted == 1) & (expected == 0)))
        true_negatives = int(np.sum((predicted == 0) & (expected == 0)))
        false_negatives = int(np.sum((predicted == 0) & (expected == 1)))
        row_count = true_positives + false_positives + true_negatives + false_negatives
        precision_denominator = true_positives + false_positives
        recall_denominator = true_positives + false_negatives
        f1_denominator = 2 * true_positives + false_positives + false_negatives
        dimension_scores = {
            "accuracy": (true_positives + true_negatives) / row_count
            if row_count
            else 0.0,
            "precision": true_positives / precision_denominator
            if precision_denominator
            else 0.0,
            "recall": true_positives / recall_denominator
            if recall_denominator
            else 0.0,
            "f1": 2 * true_positives / f1_denominator if f1_denominator else 0.0,
        }
        for metric_name, score in dimension_scores.items():
            metrics[f"{label_name}_{metric_name}"] = float(score)
            scores[metric_name].append(float(score))
    for metric_name, metric_scores in scores.items():
        metrics[f"average_{metric_name}"] = float(np.mean(metric_scores))
    return metrics


def parse_label(value: Any) -> list[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Label value: {value!r}") from exc
    if not isinstance(value, (list, tuple)) or len(value) != len(LABEL_NAMES):
        raise ValueError(f"Label must be [weakness, MAT], got {value!r}")
    label = [int(item) for item in value]
    if any(item not in (0, 1) for item in label):
        raise ValueError(f"Label must contain only zeroes and ones, got {value!r}")
    return label


def prepare_dataset(dataset: Any) -> Any:
    required = {"Function", "Label"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    def prepare_row(row: Mapping[str, Any]) -> dict[str, Any]:
        return {"text": row["Function"], "label": parse_label(row["Label"])}

    return dataset.map(prepare_row, desc="Preparing text and labels").select_columns(
        ["text", "label"]
    )


def stratified_sample(dataset: Any, limit: int, seed: int) -> Any:
    """Select up to limit rows while retaining every available joint label."""
    if limit == 0 or len(dataset) <= limit:
        return dataset.shuffle(seed=seed)

    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, label in enumerate(dataset["label"]):
        grouped[tuple(parse_label(label))].append(index)
    if limit < len(grouped):
        raise ValueError(
            f"--train-limit {limit} is smaller than the {len(grouped)} joint label "
            "classes and cannot retain every class"
        )

    rng = random.Random(seed)
    selected: list[int] = []
    remaining_by_class: dict[tuple[int, int], list[int]] = {}
    for key, indices in sorted(grouped.items()):
        rng.shuffle(indices)
        selected.append(indices[0])
        remaining_by_class[key] = indices[1:]

    remaining_slots = limit - len(selected)
    population = sum(len(indices) for indices in remaining_by_class.values())
    exact = {
        key: remaining_slots * len(indices) / population
        for key, indices in remaining_by_class.items()
    }
    counts = {
        key: min(len(remaining_by_class[key]), int(exact[key])) for key in exact
    }
    unallocated = remaining_slots - sum(counts.values())
    ranked = sorted(
        exact,
        key=lambda key: (exact[key] - int(exact[key]), len(remaining_by_class[key]), key),
        reverse=True,
    )
    while unallocated:
        made_progress = False
        for key in ranked:
            if counts[key] < len(remaining_by_class[key]):
                counts[key] += 1
                unallocated -= 1
                made_progress = True
                if unallocated == 0:
                    break
        if not made_progress:
            raise RuntimeError("Could not allocate the requested stratified sample")

    for key, count in counts.items():
        selected.extend(remaining_by_class[key][:count])
    rng.shuffle(selected)
    return dataset.select(selected)


def random_fraction_sample(dataset: Any, fraction: float, seed: int) -> Any:
    """Randomly retain ``fraction`` of a dataset using a reproducible shuffle."""
    if not 0 < fraction <= 1:
        raise ValueError("Training fraction must be greater than 0 and at most 1")
    if fraction == 1 or len(dataset) == 0:
        return dataset

    retained_rows = max(1, int(len(dataset) * fraction))
    return dataset.shuffle(seed=seed).select(range(retained_rows))


def validate_training_labels(dataset: Any) -> None:
    matrix = normalize_binary_matrix(dataset["label"], "training labels")
    for index, label_name in enumerate(LABEL_NAMES):
        observed = set(matrix[:, index].tolist())
        if observed != {0, 1}:
            raise ValueError(
                f"Training sample needs both 0 and 1 for {label_name}; found {sorted(observed)}"
            )


def load_prepared_splits(
    load_dataset: Any,
    data_dir: Path,
    train_limit: int,
    seed: int,
    train_fraction: float = 1.0,
) -> tuple[Any, Any, Any]:
    data_files = {name: str(data_dir / filename) for name, filename in SPLIT_FILES.items()}
    missing = [path for path in data_files.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing curated split files: "
            + ", ".join(missing)
            + ". Run curate_dataset.py first."
        )

    data = load_dataset("csv", data_files=data_files)
    train_dataset = stratified_sample(
        prepare_dataset(data["train"]), train_limit, seed
    )
    train_dataset = random_fraction_sample(train_dataset, train_fraction, seed)
    validation_dataset = prepare_dataset(data["validation"])
    test_dataset = prepare_dataset(data["test"])
    validate_training_labels(train_dataset)
    return train_dataset, validation_dataset, test_dataset


def extract_evaluation_metrics(
    metrics: Mapping[str, Any], metric_key_prefix: str | None = None
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in METRIC_NAMES:
        prefixed_name = f"{metric_key_prefix}_{name}" if metric_key_prefix else name
        source_name = prefixed_name if prefixed_name in metrics else name
        if source_name not in metrics:
            raise KeyError(f"Evaluation results are missing {prefixed_name!r}")
        result[name] = float(metrics[source_name])
    return result


def write_metrics(output_dir: Path, metrics: Mapping[str, Mapping[str, float]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("split", *METRIC_NAMES))
        writer.writeheader()
        for split_name, split_metrics in metrics.items():
            writer.writerow({"split": split_name, **split_metrics})

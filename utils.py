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


LABEL_SCHEMAS = {
    2: ("weakness", "MAT"),
    4: ("OSPR", "DEVIGN", "BIGVUL", "MAT"),
}
# Backward-compatible names for callers that explicitly operate on two labels.
LABEL_NAMES = LABEL_SCHEMAS[2]
DIMENSION_METRICS = ("accuracy", "precision", "recall", "f1")


def metric_names(label_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        f"{label_name}_{metric_name}"
        for label_name in label_names
        for metric_name in DIMENSION_METRICS
    ) + tuple(f"average_{metric_name}" for metric_name in DIMENSION_METRICS)


METRIC_NAMES = metric_names(LABEL_NAMES)
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


def label_names_for_count(label_count: int) -> tuple[str, ...]:
    try:
        return LABEL_SCHEMAS[label_count]
    except KeyError as exc:
        supported = ", ".join(str(count) for count in sorted(LABEL_SCHEMAS))
        raise ValueError(
            f"Labels must contain {supported} values, found {label_count}"
        ) from exc


def normalize_binary_matrix(
    values: Any, name: str, label_names: tuple[str, ...] | None = None
) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    matrix = np.asarray(values, dtype=np.int64)
    if matrix.ndim == 1:
        if matrix.size == 0:
            if label_names is None:
                raise ValueError(f"Cannot infer the label schema from empty {name}")
            matrix = matrix.reshape(0, len(label_names))
        else:
            inferred_names = label_names_for_count(matrix.size)
            if label_names is None:
                label_names = inferred_names
            matrix = matrix.reshape(1, matrix.size)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix, got {matrix.shape}")
    if label_names is None:
        label_names = label_names_for_count(matrix.shape[1])
    if matrix.shape[1] != len(label_names):
        raise ValueError(
            f"{name} must have shape (rows, {len(label_names)}), got {matrix.shape}"
        )
    if not np.isin(matrix, (0, 1)).all():
        raise ValueError(f"{name} contains values other than 0 and 1")
    return matrix


def compute_binary_metrics(
    predictions: Any,
    references: Any,
    label_names: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Return binary metrics per dimension and their unweighted macro means."""
    references_array = normalize_binary_matrix(references, "references", label_names)
    if label_names is None:
        label_names = label_names_for_count(references_array.shape[1])
    predictions_array = normalize_binary_matrix(predictions, "predictions", label_names)
    if predictions_array.shape != references_array.shape:
        raise ValueError(
            "Predictions and references must have identical shapes, got "
            f"{predictions_array.shape} and {references_array.shape}"
        )

    metrics: dict[str, float] = {}
    scores: dict[str, list[float]] = {
        metric_name: [] for metric_name in DIMENSION_METRICS
    }
    for index, label_name in enumerate(label_names):
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


def parse_label(
    value: Any, label_names: tuple[str, ...] | None = None
) -> list[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Label value: {value!r}") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Label must be a list, got {value!r}")
    detected_names = label_names_for_count(len(value))
    if label_names is not None and detected_names != label_names:
        raise ValueError(
            f"Label must follow [{', '.join(label_names)}], got {value!r}"
        )
    label = [int(item) for item in value]
    if any(item not in (0, 1) for item in label):
        raise ValueError(f"Label must contain only zeroes and ones, got {value!r}")
    return label


def prepare_dataset(dataset: Any, label_names: tuple[str, ...]) -> Any:
    required = {"Function", "Label"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    def prepare_row(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "text": row["Function"],
            "label": parse_label(row["Label"], label_names),
        }

    return dataset.map(prepare_row, desc="Preparing text and labels").select_columns(
        ["text", "label"]
    )


def stratified_sample(dataset: Any, limit: int, seed: int) -> Any:
    """Select up to limit rows while retaining every available joint label."""
    if limit == 0 or len(dataset) <= limit:
        return dataset.shuffle(seed=seed)

    grouped: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, label in enumerate(dataset["label"]):
        grouped[tuple(parse_label(label))].append(index)
    if limit < len(grouped):
        raise ValueError(
            f"--train-limit {limit} is smaller than the {len(grouped)} joint label "
            "classes and cannot retain every class"
        )

    rng = random.Random(seed)
    for indices in grouped.values():
        rng.shuffle(indices)

    exact = {
        key: limit * len(indices) / len(dataset) for key, indices in grouped.items()
    }
    counts = {key: 1 for key in grouped}
    while sum(counts.values()) < limit:
        eligible = [key for key in grouped if counts[key] < len(grouped[key])]
        if not eligible:
            raise RuntimeError("Could not allocate the requested stratified sample")
        key = max(
            eligible,
            key=lambda candidate: (
                exact[candidate] - counts[candidate],
                len(grouped[candidate]),
                candidate,
            ),
        )
        counts[key] += 1

    selected = [
        index
        for key, count in counts.items()
        for index in grouped[key][:count]
    ]
    rng.shuffle(selected)
    return dataset.select(selected)


def stratified_fraction_sample(dataset: Any, fraction: float, seed: int) -> Any:
    """Retain a fraction while preserving the joint label distribution."""
    if not 0 < fraction <= 1:
        raise ValueError("Training fraction must be greater than 0 and at most 1")
    if fraction == 1 or len(dataset) == 0:
        return dataset

    retained_rows = max(1, int(len(dataset) * fraction))
    return stratified_sample(dataset, retained_rows, seed)


def validate_training_labels(dataset: Any, label_names: tuple[str, ...]) -> None:
    matrix = normalize_binary_matrix(dataset["label"], "training labels", label_names)
    for index, label_name in enumerate(label_names):
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
) -> tuple[Any, Any, Any, tuple[str, ...]]:
    data_files = {name: str(data_dir / filename) for name, filename in SPLIT_FILES.items()}
    missing = [path for path in data_files.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing curated split files: "
            + ", ".join(missing)
            + ". Run curate_dataset.py first."
        )

    data = load_dataset("csv", data_files=data_files)
    empty_splits = [name for name, dataset in data.items() if len(dataset) == 0]
    if empty_splits:
        raise ValueError(f"Dataset splits cannot be empty: {empty_splits}")
    split_label_names = {
        name: label_names_for_count(len(parse_label(dataset[0]["Label"])))
        for name, dataset in data.items()
    }
    if len(set(split_label_names.values())) != 1:
        raise ValueError(f"Dataset splits use different label schemas: {split_label_names}")
    label_names = next(iter(split_label_names.values()))
    train_dataset = stratified_sample(
        prepare_dataset(data["train"], label_names), train_limit, seed
    )
    train_dataset = stratified_fraction_sample(train_dataset, train_fraction, seed)
    validation_dataset = prepare_dataset(data["validation"], label_names)
    test_dataset = prepare_dataset(data["test"], label_names)
    validate_training_labels(train_dataset, label_names)
    return train_dataset, validation_dataset, test_dataset, label_names


def extract_evaluation_metrics(
    metrics: Mapping[str, Any],
    metric_key_prefix: str | None = None,
    label_names: tuple[str, ...] = LABEL_NAMES,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in metric_names(label_names):
        prefixed_name = f"{metric_key_prefix}_{name}" if metric_key_prefix else name
        source_name = prefixed_name if prefixed_name in metrics else name
        if source_name not in metrics:
            raise KeyError(f"Evaluation results are missing {prefixed_name!r}")
        result[name] = float(metrics[source_name])
    return result


def write_metrics(
    output_dir: Path,
    metrics: Mapping[str, Mapping[str, float]],
    label_names: tuple[str, ...] = LABEL_NAMES,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("split", *metric_names(label_names))
        )
        writer.writeheader()
        for split_name, split_metrics in metrics.items():
            writer.writerow({"split": split_name, **split_metrics})

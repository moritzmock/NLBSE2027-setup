#!/usr/bin/env python3
"""Fine-tune SetFit on the dataset produced by ``curate_dataset.py``.

Training uses a seeded, joint-label-stratified 5% of the training split by
default. Every epoch is saved, validation runs after every epoch, and final
metrics cover validation and test.
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, Sequence

from utils import (
    add_common_training_arguments,
    compute_binary_metrics,
    extract_evaluation_metrics,
    load_prepared_splits,
    stratified_sample,
    validate_common_training_arguments,
    write_metrics,
)


# Retain the original public name for callers and tests.
compute_metrics = compute_binary_metrics


def iter_contrastive_pairs(
    sentences: Sequence[str],
    labels: Sequence[Sequence[int]],
    pairs_per_kind: int,
    seed: int,
) -> Iterator[dict[str, Any]]:
    """Draw balanced multilabel pairs without enumerating all combinations."""
    if len(sentences) != len(labels):
        raise ValueError("Sentences and labels must contain the same number of rows")
    if pairs_per_kind < 1:
        raise ValueError("pairs_per_kind must be at least 1")

    grouped: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[tuple(int(value) for value in label)].append(index)

    candidates: dict[bool, list[tuple[list[int], list[int], int]]] = {
        True: [],
        False: [],
    }
    group_items = sorted(grouped.items())
    for left_position, (left_label, left_indices) in enumerate(group_items):
        for right_label, right_indices in group_items[left_position:]:
            same_group = left_label == right_label
            pair_count = (
                len(left_indices) * (len(left_indices) + 1) // 2
                if same_group
                else len(left_indices) * len(right_indices)
            )
            is_positive = any(
                left_value and right_value
                for left_value, right_value in zip(left_label, right_label)
            )
            candidates[is_positive].append((left_indices, right_indices, pair_count))

    if not candidates[True] or not candidates[False]:
        raise ValueError("Training labels must allow both positive and negative pairs")

    cumulative_weights: dict[bool, list[int]] = {}
    for pair_kind, groups in candidates.items():
        total = 0
        cumulative_weights[pair_kind] = []
        for _, _, weight in groups:
            total += weight
            cumulative_weights[pair_kind].append(total)

    rng = random.Random(seed)
    for _ in range(pairs_per_kind):
        for pair_kind in (True, False):
            weights = cumulative_weights[pair_kind]
            group_index = bisect.bisect_right(weights, rng.randrange(weights[-1]))
            left_indices, right_indices, _ = candidates[pair_kind][group_index]
            left_index = rng.choice(left_indices)
            right_index = rng.choice(right_indices)
            yield {
                "sentence_1": sentences[left_index],
                "sentence_2": sentences[right_index],
                "label": float(pair_kind),
            }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_training_arguments(
        parser,
        default_model_id="sentence-transformers/paraphrase-MiniLM-L6-v2",
        default_output_dir="models/setfit-2",
        default_epochs=5,
        default_batch_size=32,
        default_train_limit=0,
    )
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.05,
        help=(
            "Fraction of selected training rows to retain while preserving the "
            "joint weakness/MAT distribution."
        ),
    )
    parser.add_argument(
        "--embedding-eval-limit",
        type=int,
        default=1000,
        help=(
            "Maximum stratified validation rows used during embedding training; "
            "the final validation evaluation still uses every row."
        ),
    )
    args = parser.parse_args(argv)
    validate_common_training_arguments(parser, args)
    if args.num_iterations < 1:
        parser.error("--num-iterations must be at least 1")
    if not 0 < args.train_fraction <= 1:
        parser.error("--train-fraction must be greater than 0 and at most 1")
    if args.embedding_eval_limit < 0:
        parser.error("--embedding-eval-limit cannot be negative")
    return args


def import_training_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        from datasets import Dataset, Features, Value, load_dataset
        from sentence_transformers import losses
        from setfit import SetFitModel, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install them with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    class MemoryEfficientTrainer(Trainer):
        """SetFit trainer that samples contrastive pairs in bounded memory."""

        def get_dataset(
            self,
            x: list[str],
            y: list[list[int]],
            args: Any,
            max_pairs: int = -1,
        ) -> tuple[Any, Any]:
            if (
                args.num_iterations is None
                or args.loss is not losses.CosineSimilarityLoss
            ):
                return super().get_dataset(x, y, args, max_pairs=max_pairs)

            pairs_per_kind = args.num_iterations * len(x)
            if max_pairs != -1:
                pairs_per_kind = min(pairs_per_kind, max_pairs // 2)
            dataset = Dataset.from_generator(
                iter_contrastive_pairs,
                features=Features(
                    {
                        "sentence_1": Value("string"),
                        "sentence_2": Value("string"),
                        "label": Value("float32"),
                    }
                ),
                keep_in_memory=False,
                gen_kwargs={
                    "sentences": x,
                    "labels": y,
                    "pairs_per_kind": pairs_per_kind,
                    "seed": args.seed,
                },
            )
            return dataset, args.loss(self.model.model_body)

    return load_dataset, SetFitModel, MemoryEfficientTrainer, TrainingArguments


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(args)
    load_dataset, SetFitModel, Trainer, TrainingArguments = import_training_dependencies()
    train_dataset, validation_dataset, test_dataset = load_prepared_splits(
        load_dataset,
        args.data_dir,
        args.train_limit,
        args.seed,
        train_fraction=args.train_fraction,
    )
    embedding_validation_dataset = stratified_sample(
        validation_dataset, args.embedding_eval_limit, args.seed
    )

    checkpoint_dir = args.output_dir / "checkpoints"
    final_model_dir = args.output_dir / "final"
    model = SetFitModel.from_pretrained(
        args.model_id,
        multi_target_strategy="multi-output",
    )
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=None,
        logging_strategy="epoch",
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=embedding_validation_dataset,
        metric=compute_binary_metrics,
    )

    print(f"Training on {len(train_dataset)} instances")
    print(
        "Embedding validation uses "
        f"{len(embedding_validation_dataset)} of {len(validation_dataset)} instances"
    )
    trainer.train()
    trainer.model.save_pretrained(str(final_model_dir))

    metrics = {
        "validation": extract_evaluation_metrics(
            trainer.evaluate(validation_dataset, metric_key_prefix="validation"),
            "validation",
        ),
        "test": extract_evaluation_metrics(
            trainer.evaluate(test_dataset, metric_key_prefix="test"), "test"
        ),
    }
    write_metrics(args.output_dir, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Final model: {final_model_dir}")
    print(f"Epoch checkpoints: {checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fine-tune SetFit on the dataset produced by ``curate_dataset.py``.

Training uses at most 100 stratified examples by default. Every epoch is saved,
validation runs after every epoch, and final metrics cover validation and test.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

from utils import (
    add_common_training_arguments,
    compute_binary_metrics,
    extract_evaluation_metrics,
    load_prepared_splits,
    validate_common_training_arguments,
    write_metrics,
)


# Retain the original public name for callers and tests.
compute_metrics = compute_binary_metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_training_arguments(
        parser,
        default_model_id="sentence-transformers/paraphrase-MiniLM-L6-v2",
        default_output_dir="models/setfit",
        default_epochs=5,
        default_batch_size=32,
    )
    parser.add_argument("--num-iterations", type=int, default=20)
    args = parser.parse_args(argv)
    validate_common_training_arguments(parser, args)
    if args.num_iterations < 1:
        parser.error("--num-iterations must be at least 1")
    return args


def import_training_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        from datasets import load_dataset
        from setfit import SetFitModel, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install them with: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return load_dataset, SetFitModel, Trainer, TrainingArguments


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(args)
    load_dataset, SetFitModel, Trainer, TrainingArguments = import_training_dependencies()
    train_dataset, validation_dataset, test_dataset = load_prepared_splits(
        load_dataset, args.data_dir, args.train_limit, args.seed
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
        eval_dataset=validation_dataset,
        metric=compute_binary_metrics,
    )

    print(f"Training on {len(train_dataset)} instances")
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

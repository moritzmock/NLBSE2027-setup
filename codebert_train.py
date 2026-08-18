#!/usr/bin/env python3
"""Fine-tune CodeBERT on the dataset produced by ``curate_dataset.py``.

Training uses at most 100 stratified examples by default. Every epoch is saved,
validation runs after every epoch, and final metrics cover validation and test.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from typing import Any, Sequence

import numpy as np

from utils import (
    add_common_training_arguments,
    compute_binary_metrics,
    extract_evaluation_metrics,
    load_prepared_splits,
    validate_common_training_arguments,
    write_metrics,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_training_arguments(
        parser,
        default_model_id="microsoft/codebert-base",
        default_output_dir="models/codebert",
        default_epochs=5,
        default_batch_size=8,
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args(argv)
    validate_common_training_arguments(parser, args)
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    if args.max_length < 1:
        parser.error("--max-length must be at least 1")
    return args


def import_training_dependencies() -> tuple[Any, ...]:
    try:
        from datasets import load_dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install them with: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return (
        load_dataset,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )


def sigmoid(logits: Any) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -500, 500)))


def compute_metrics(
    eval_prediction: Any, label_names: tuple[str, ...] | None = None
) -> dict[str, float]:
    logits = eval_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = (sigmoid(logits) >= 0.5).astype(np.int64)
    return compute_binary_metrics(
        predictions, eval_prediction.label_ids, label_names=label_names
    )


def tokenize_dataset(dataset: Any, tokenizer: Any, max_length: int) -> Any:
    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )
        encoded["labels"] = [
            [float(value) for value in label] for label in batch["label"]
        ]
        return encoded

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing functions with CodeBERT",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    (
        load_dataset,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    ) = import_training_dependencies()
    train_dataset, validation_dataset, test_dataset, label_names = load_prepared_splits(
        load_dataset, args.data_dir, args.train_limit, args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    train_dataset = tokenize_dataset(train_dataset, tokenizer, args.max_length)
    validation_dataset = tokenize_dataset(
        validation_dataset, tokenizer, args.max_length
    )
    test_dataset = tokenize_dataset(test_dataset, tokenizer, args.max_length)

    id2label = {index: name for index, name in enumerate(label_names)}
    label2id = {name: index for index, name in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=len(label_names),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    checkpoint_dir = args.output_dir / "checkpoints"
    final_model_dir = args.output_dir / "final"
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=None,
        logging_strategy="epoch",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=partial(compute_metrics, label_names=label_names),
    )

    print(f"Training on {len(train_dataset)} instances")
    trainer.train()
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    metrics = {
        "validation": extract_evaluation_metrics(
            trainer.evaluate(
                validation_dataset, metric_key_prefix="validation"
            ),
            "validation",
            label_names,
        ),
        "test": extract_evaluation_metrics(
            trainer.evaluate(test_dataset, metric_key_prefix="test"),
            "test",
            label_names,
        ),
    }
    write_metrics(args.output_dir, metrics, label_names)
    print(f"Detected label order: [{', '.join(label_names)}]")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Final model: {final_model_dir}")
    print(f"Epoch checkpoints: {checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

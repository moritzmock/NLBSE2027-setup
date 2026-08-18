import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from datasets import Dataset

import utils as training_utils


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "codebert_train.py"
SPEC = importlib.util.spec_from_file_location("codebert_training_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
codebert_training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(codebert_training)


class CodeBERTTrainingTests(unittest.TestCase):
    def test_default_configuration_uses_codebert_and_100_training_rows(self):
        args = codebert_training.parse_args([])

        self.assertEqual(args.model_id, "microsoft/codebert-base")
        self.assertEqual(args.train_limit, 100)
        self.assertEqual(args.output_dir, Path("models/codebert"))

    def test_metrics_apply_sigmoid_and_report_both_dimensions(self):
        evaluation = SimpleNamespace(
            predictions=np.array([[4.0, -4.0], [-4.0, 4.0], [4.0, 4.0]]),
            label_ids=np.array([[1, 0], [1, 1], [1, 0]]),
        )

        metrics = codebert_training.compute_metrics(evaluation)

        self.assertEqual(set(metrics), set(training_utils.METRIC_NAMES))
        self.assertAlmostEqual(metrics["weakness_f1"], 0.8)
        self.assertAlmostEqual(metrics["weakness_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["weakness_precision"], 1.0)
        self.assertAlmostEqual(metrics["weakness_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["MAT_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["MAT_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["MAT_precision"], 0.5)
        self.assertAlmostEqual(metrics["MAT_recall"], 1.0)
        self.assertAlmostEqual(metrics["average_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["average_precision"], 0.75)
        self.assertAlmostEqual(metrics["average_recall"], 5 / 6)
        self.assertAlmostEqual(metrics["average_f1"], (0.8 + 2 / 3) / 2)

    def test_metrics_dynamically_support_four_dimensions(self):
        evaluation = SimpleNamespace(
            predictions=np.array([[4.0, -4.0, 4.0, -4.0]]),
            label_ids=np.array([[1, 0, 1, 0]]),
        )

        metrics = codebert_training.compute_metrics(
            evaluation, training_utils.LABEL_SCHEMAS[4]
        )

        self.assertEqual(
            set(metrics),
            set(training_utils.metric_names(training_utils.LABEL_SCHEMAS[4])),
        )
        self.assertEqual(metrics["OSPR_f1"], 1.0)
        self.assertEqual(metrics["DEVIGN_accuracy"], 1.0)
        self.assertEqual(metrics["BIGVUL_f1"], 1.0)
        self.assertEqual(metrics["MAT_accuracy"], 1.0)

    def test_sigmoid_is_stable_for_extreme_logits(self):
        probabilities = codebert_training.sigmoid([[-1000.0, 1000.0]])

        self.assertTrue(np.isfinite(probabilities).all())
        self.assertLess(probabilities[0, 0], 1e-100)
        self.assertEqual(probabilities[0, 1], 1.0)

    def test_tokenization_emits_float_multilabel_targets(self):
        class FakeTokenizer:
            def __call__(self, texts, truncation, max_length):
                self.call = (texts, truncation, max_length)
                return {
                    "input_ids": [[1, 2] for _ in texts],
                    "attention_mask": [[1, 1] for _ in texts],
                }

        tokenizer = FakeTokenizer()
        dataset = Dataset.from_dict(
            {"text": ["int main() {}", "void f() {}"], "label": [[1, 0], [0, 1]]}
        )

        tokenized = codebert_training.tokenize_dataset(dataset, tokenizer, 512)

        self.assertEqual(tokenizer.call[1:], (True, 512))
        self.assertEqual(tokenized.column_names, ["input_ids", "attention_mask", "labels"])
        self.assertEqual(tokenized["labels"], [[1.0, 0.0], [0.0, 1.0]])

    def test_validation_and_test_are_evaluated_only_after_training(self):
        events = []
        train_dataset = ["train"]
        validation_dataset = ["validation"]
        test_dataset = ["test"]
        metric_values = {name: 1.0 for name in training_utils.METRIC_NAMES}

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, _model_id):
                return cls()

            def save_pretrained(self, _path):
                events.append("save-tokenizer")

        class FakeModel:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        class FakeDataCollator:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

        class FakeTrainingArguments:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeTrainer:
            instance = None

            def __init__(self, **kwargs):
                type(self).instance = self
                self.kwargs = kwargs

            def train(self):
                events.append("train")

            def save_model(self, _path):
                events.append("save-model")

            def evaluate(self, dataset, metric_key_prefix):
                split = "validation" if dataset is validation_dataset else "test"
                events.append(f"evaluate:{split}")
                return {
                    f"{metric_key_prefix}_{name}": value
                    for name, value in metric_values.items()
                }

        with (
            mock.patch.object(
                codebert_training,
                "import_training_dependencies",
                return_value=(
                    object(),
                    FakeModel,
                    FakeTokenizer,
                    FakeDataCollator,
                    FakeTrainer,
                    FakeTrainingArguments,
                ),
            ),
            mock.patch.object(
                codebert_training,
                "load_prepared_splits",
                return_value=(
                    train_dataset,
                    validation_dataset,
                    test_dataset,
                    training_utils.LABEL_NAMES,
                ),
            ),
            mock.patch.object(
                codebert_training,
                "tokenize_dataset",
                side_effect=lambda dataset, _tokenizer, _max_length: dataset,
            ),
            mock.patch.object(codebert_training, "write_metrics"),
        ):
            exit_code = codebert_training.main(["--output-dir", "unused"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            events,
            [
                "train",
                "save-model",
                "save-tokenizer",
                "evaluate:validation",
                "evaluate:test",
            ],
        )
        trainer = FakeTrainer.instance
        self.assertIsNotNone(trainer)
        self.assertNotIn("eval_dataset", trainer.kwargs)
        self.assertEqual(trainer.kwargs["args"].kwargs["eval_strategy"], "no")


if __name__ == "__main__":
    unittest.main()

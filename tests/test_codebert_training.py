import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()

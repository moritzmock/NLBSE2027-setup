import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import utils as training_utils


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "setfit_train.py"
SPEC = importlib.util.spec_from_file_location("setfit_training_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
setfit_training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setfit_training)


class FakeDataset:
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, key):
        if key == "label":
            return self.labels
        raise KeyError(key)

    def shuffle(self, seed):
        return self

    def select(self, indices):
        return FakeDataset([self.labels[index] for index in indices])


class SetFitTrainingTests(unittest.TestCase):
    def test_defaults_retain_random_five_percent_of_full_training_split(self):
        args = setfit_training.parse_args([])

        self.assertEqual(args.train_limit, 0)
        self.assertEqual(args.train_fraction, 0.05)

    def test_metrics_contain_both_dimensions_and_average(self):
        metrics = setfit_training.compute_metrics(
            [[1, 0], [0, 1], [1, 1]],
            [[1, 0], [1, 1], [1, 0]],
        )

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

    def test_stratified_sample_uses_100_rows_and_retains_joint_labels(self):
        labels = [[0, 0]] * 120 + [[0, 1]] * 30 + [[1, 0]] * 80 + [[1, 1]] * 20
        dataset = FakeDataset(labels)

        sampled = training_utils.stratified_sample(dataset, limit=100, seed=42)

        self.assertEqual(len(sampled), 100)
        self.assertEqual({tuple(label) for label in sampled.labels}, set(map(tuple, labels)))

    def test_random_fraction_sample_retains_five_percent(self):
        dataset = FakeDataset([[0, 0]] * 100)

        sampled = training_utils.random_fraction_sample(dataset, fraction=0.05, seed=42)

        self.assertEqual(len(sampled), 5)

    def test_metrics_are_written_as_json_and_csv(self):
        validation_metrics = {
            name: (index + 1) / 100
            for index, name in enumerate(training_utils.METRIC_NAMES)
        }
        test_metrics = {name: value + 0.1 for name, value in validation_metrics.items()}
        metrics = {
            "validation": validation_metrics,
            "test": test_metrics,
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)

            training_utils.write_metrics(output_dir, metrics)

            with (output_dir / "metrics.json").open(encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), metrics)
            with (output_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, ["split", *training_utils.METRIC_NAMES])
            self.assertEqual(rows[0]["split"], "validation")
            self.assertEqual(float(rows[0]["weakness_accuracy"]), 0.01)

    def test_prefixed_transformer_metrics_are_normalized(self):
        expected = {
            name: (index + 1) / 100
            for index, name in enumerate(training_utils.METRIC_NAMES)
        }
        metrics = {f"validation_{name}": value for name, value in expected.items()}
        metrics["validation_loss"] = 0.1

        normalized = training_utils.extract_evaluation_metrics(metrics, "validation")

        self.assertEqual(normalized, expected)


if __name__ == "__main__":
    unittest.main()

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
    def test_metrics_contain_both_dimensions_and_average(self):
        metrics = setfit_training.compute_metrics(
            [[1, 0], [0, 1], [1, 1]],
            [[1, 0], [1, 1], [1, 0]],
        )

        self.assertAlmostEqual(metrics["weakness_f1"], 0.8)
        self.assertAlmostEqual(metrics["MAT_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["average_f1"], (0.8 + 2 / 3) / 2)

    def test_stratified_sample_uses_100_rows_and_retains_joint_labels(self):
        labels = [[0, 0]] * 120 + [[0, 1]] * 30 + [[1, 0]] * 80 + [[1, 1]] * 20
        dataset = FakeDataset(labels)

        sampled = training_utils.stratified_sample(dataset, limit=100, seed=42)

        self.assertEqual(len(sampled), 100)
        self.assertEqual({tuple(label) for label in sampled.labels}, set(map(tuple, labels)))

    def test_metrics_are_written_as_json_and_csv(self):
        metrics = {
            "validation": {"weakness_f1": 0.5, "MAT_f1": 0.25, "average_f1": 0.375},
            "test": {"weakness_f1": 0.75, "MAT_f1": 0.5, "average_f1": 0.625},
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)

            training_utils.write_metrics(output_dir, metrics)

            with (output_dir / "metrics.json").open(encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), metrics)
            csv_text = (output_dir / "metrics.csv").read_text(encoding="utf-8")
            self.assertIn("split,weakness_f1,MAT_f1,average_f1", csv_text)
            self.assertIn("validation,0.5,0.25,0.375", csv_text)

    def test_prefixed_transformer_metrics_are_normalized(self):
        metrics = {
            "validation_loss": 0.1,
            "validation_weakness_f1": 0.5,
            "validation_MAT_f1": 0.25,
            "validation_average_f1": 0.375,
        }

        normalized = training_utils.extract_f1_metrics(metrics, "validation")

        self.assertEqual(
            normalized,
            {"weakness_f1": 0.5, "MAT_f1": 0.25, "average_f1": 0.375},
        )


if __name__ == "__main__":
    unittest.main()

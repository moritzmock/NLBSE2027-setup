import tempfile
import unittest
from pathlib import Path

from datasets import Dataset

import curate_dataset_4labels
import utils


class TrainingLabelSchemaTests(unittest.TestCase):
    def test_supported_label_orders_are_fixed_by_vector_length(self) -> None:
        self.assertEqual(utils.label_names_for_count(2), ("weakness", "MAT"))
        self.assertEqual(
            utils.label_names_for_count(4),
            ("OSPR", "DEVIGN", "BIGVUL", "MAT"),
        )
        self.assertEqual(
            utils.label_names_for_count(4), curate_dataset_4labels.LABEL_COLUMNS
        )
        with self.assertRaisesRegex(ValueError, "must contain 2, 4 values"):
            utils.label_names_for_count(3)

    def test_four_label_metrics_use_all_dimensions_in_the_correct_order(self) -> None:
        label_names = utils.LABEL_SCHEMAS[4]
        metrics = utils.compute_binary_metrics(
            [[1, 0, 1, 0], [0, 1, 0, 1]],
            [[1, 0, 0, 0], [0, 1, 0, 1]],
            label_names,
        )

        self.assertEqual(set(metrics), set(utils.metric_names(label_names)))
        self.assertEqual(metrics["OSPR_accuracy"], 1.0)
        self.assertEqual(metrics["DEVIGN_accuracy"], 1.0)
        self.assertEqual(metrics["BIGVUL_accuracy"], 0.5)
        self.assertEqual(metrics["MAT_accuracy"], 1.0)

    def test_loader_detects_four_labels_and_validates_every_instance(self) -> None:
        valid_rows = {
            "Function": ["int safe(void) {}", "int unsafe(void) {}"],
            "Label": ["[0,0,0,0]", "[1,1,1,1]"],
        }
        datasets = {
            name: Dataset.from_dict(valid_rows)
            for name in utils.SPLIT_FILES
        }

        with tempfile.TemporaryDirectory() as directory:
            data_dir = self.create_split_placeholders(Path(directory))
            loaded = utils.load_prepared_splits(
                lambda *_args, **_kwargs: datasets,
                data_dir,
                train_limit=0,
                seed=42,
            )

        train, validation, test, label_names = loaded
        self.assertEqual(label_names, utils.LABEL_SCHEMAS[4])
        self.assertCountEqual(train["label"], [[0, 0, 0, 0], [1, 1, 1, 1]])
        self.assertEqual(len(validation["label"][0]), 4)
        self.assertEqual(len(test["label"][0]), 4)

    def test_loader_rejects_mixed_instance_lengths(self) -> None:
        datasets = {
            name: Dataset.from_dict(
                {
                    "Function": ["int a(void) {}", "int b(void) {}"],
                    "Label": ["[0,0,0,0]", "[1,1]"],
                }
            )
            for name in utils.SPLIT_FILES
        }

        with tempfile.TemporaryDirectory() as directory:
            data_dir = self.create_split_placeholders(Path(directory))
            with self.assertRaisesRegex(
                ValueError, "Label must follow.*OSPR.*DEVIGN.*BIGVUL.*MAT"
            ):
                utils.load_prepared_splits(
                    lambda *_args, **_kwargs: datasets,
                    data_dir,
                    train_limit=0,
                    seed=42,
                )

    @staticmethod
    def create_split_placeholders(data_dir: Path) -> Path:
        for filename in utils.SPLIT_FILES.values():
            (data_dir / filename).write_text("placeholder\n", encoding="utf-8")
        return data_dir


if __name__ == "__main__":
    unittest.main()

import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import curate_dataset_4labels as curate


class CurateDatasetFourLabelTests(unittest.TestCase):
    def test_step_3_keeps_source_labels_separate_and_renames_w_to_ospr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "formatted.csv"
            output_path = root / "labels.csv"
            fieldnames = list(curate.BASE_COLUMNS)
            rows = [
                self.row(source="OSPR", w="1"),
                self.row(source="devign", w="0", devign="1", mat="1"),
                self.row(source="big-vul", w="1", big_vul="1"),
            ]
            with input_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            count, distribution = curate.run_step_3(
                input_path, output_path, total=len(rows)
            )

            self.assertEqual(count, 3)
            self.assertEqual(
                distribution,
                Counter({(1, 0, 0, 0): 1, (0, 1, 0, 1): 1, (1, 0, 1, 0): 1}),
            )
            with output_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                output_rows = list(reader)
            self.assertEqual(
                [row["OSPR"] for row in output_rows], ["1", "0", "1"]
            )
            self.assertEqual(
                [row["DEVIGN"] for row in output_rows], ["0", "1", "0"]
            )
            self.assertEqual(
                [row["BIGVUL"] for row in output_rows], ["0", "0", "1"]
            )
            for old_column in curate.SOURCE_LABEL_COLUMNS:
                self.assertNotIn(old_column, reader.fieldnames or [])

            final_path = root / "rebalanced.csv"
            final_count, final_distribution = curate.run_step_5(
                output_path,
                final_path,
                total=count,
                original=distribution,
                available=distribution,
                seed=42,
            )
            self.assertEqual(final_count, count)
            self.assertEqual(final_distribution, distribution)
            with final_path.open(newline="", encoding="utf-8") as handle:
                final_rows = list(csv.DictReader(handle))
            self.assertCountEqual(
                [row["Label"] for row in final_rows],
                ["[1,0,0,0]", "[0,1,0,1]", "[1,0,1,0]"],
            )

    def test_four_value_label_round_trip(self) -> None:
        distribution = Counter({(1, 0, 0, 1): 7, (0, 1, 0, 0): 3})
        serialized = curate.serialize_distribution(distribution)

        self.assertEqual(curate.deserialize_distribution(serialized), distribution)
        self.assertEqual(curate.label_stratum("[1,0,0,1]"), (1, 0, 0, 1))

    @staticmethod
    def row(
        *,
        source: str,
        w: str = "0",
        devign: str = "",
        big_vul: str = "",
        mat: str = "0",
    ) -> dict[str, str]:
        row = {column: "" for column in curate.BASE_COLUMNS}
        row.update(
            {
                "Projectname": "example",
                "Filepath": "example.c",
                "Function": "int example(void) { return 0; }",
                "PS": "0",
                "MAT": mat,
                "Devign": devign,
                "Big-Vul": big_vul,
                "W": w,
                "SecI": "0",
                "SourceDataset": source,
            }
        )
        return row


if __name__ == "__main__":
    unittest.main()

import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import curate_dataset


class CurateDatasetTests(unittest.TestCase):
    def test_as_bool_accepts_csv_encodings_and_rejects_unknown_labels(self) -> None:
        for value in ("", "0", "0.0", "false", "No", "nan"):
            self.assertEqual(curate_dataset.as_bool(value), 0)
        for value in ("1", "1.0", "true", "YES"):
            self.assertEqual(curate_dataset.as_bool(value), 1)
        with self.assertRaises(ValueError):
            curate_dataset.as_bool("maybe")

    def test_target_counts_restore_joint_distribution_by_downsampling(self) -> None:
        original = Counter({(0, 0): 60, (0, 1): 20, (1, 0): 15, (1, 1): 5})
        available = Counter({(0, 0): 54, (0, 1): 18, (1, 0): 9, (1, 1): 4})

        targets = curate_dataset.target_stratum_counts(original, available)

        self.assertEqual(targets, Counter({(0, 0): 36, (0, 1): 12, (1, 0): 9, (1, 1): 3}))
        self.assertTrue(all(targets[key] <= available[key] for key in original))

    def test_final_stratum_uses_vulnerability_then_mat_only(self) -> None:
        row = {"Vulnerable": "1", "MAT": "0", "PS": "1", "SecI": "1"}

        self.assertEqual(curate_dataset.row_stratum(row), (1, 0))

    def test_split_allocation_has_exact_global_ratio(self) -> None:
        distribution = Counter({(0, 0): 61, (0, 1): 19, (1, 0): 13, (1, 1): 7})

        allocations = curate_dataset.allocate_split_counts(distribution)
        totals = {
            name: sum(counts[name] for counts in allocations.values())
            for name in curate_dataset.SPLIT_RATIOS
        }

        self.assertEqual(totals, {"train": 70, "validation": 10, "test": 20})
        self.assertEqual(
            {key: sum(counts.values()) for key, counts in allocations.items()},
            dict(distribution),
        )

    def test_small_pipeline_merges_labels_filters_tokens_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "Dataset"
            output_dir = root / "steps"
            rows_by_source = {
                "OSPR": [self.row("int a(){return 0;}", w="1")],
                "big-vul": [
                    self.row("int b(){return 1;}", big_vul="1", ps="1"),
                    self.row("int huge(){" + "x++;" * 100 + "}", big_vul="0"),
                    self.row("int safe(){return 3;}", big_vul="0"),
                ],
                "devign": [self.row("int c(){return 2;}", devign="1", mat="1")],
            }
            for source, rows in rows_by_source.items():
                folder = input_dir / source
                folder.mkdir(parents=True)
                self.write_input(folder / "complete.csv", rows)

            exit_code = curate_dataset.main(
                [
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--token-limit",
                    "30",
                    "--workers",
                    "2",
                ]
            )

            self.assertEqual(exit_code, 0)
            for filename in curate_dataset.STEP_FILENAMES.values():
                self.assertTrue((output_dir / filename).is_file())
            with (output_dir / curate_dataset.STEP_FILENAMES[3]).open(
                newline="", encoding="utf-8"
            ) as handle:
                merged = list(csv.DictReader(handle))
            self.assertEqual(
                [row["Vulnerable"] for row in merged], ["1", "1", "0", "0", "1"]
            )
            self.assertNotIn("W", merged[0])
            self.assertNotIn("Big-Vul", merged[0])
            self.assertNotIn("Devign", merged[0])
            with (output_dir / curate_dataset.STEP_FILENAMES[4]).open(
                newline="", encoding="utf-8"
            ) as handle:
                filtered = list(csv.DictReader(handle))
            self.assertEqual(len(filtered), 4)
            self.assertTrue(all(int(row["TokenCount"]) <= 30 for row in filtered))
            with (output_dir / curate_dataset.STEP_FILENAMES[5]).open(
                newline="", encoding="utf-8"
            ) as handle:
                final_reader = csv.DictReader(handle)
                final_rows = list(final_reader)
            self.assertEqual(
                final_reader.fieldnames,
                ["Projectname", "Filepath", "Function", "Label"],
            )
            self.assertTrue(final_rows)
            self.assertTrue(all(row["Projectname"] == "example" for row in final_rows))
            self.assertTrue(all(row["Filepath"] == "example.c" for row in final_rows))
            self.assertTrue(
                all(
                    row["Label"] in {"[0,0]", "[0,1]", "[1,0]", "[1,1]"}
                    for row in final_rows
                )
            )
            split_rows = []
            for filename in curate_dataset.SPLIT_FILENAMES.values():
                with (output_dir / filename).open(newline="", encoding="utf-8") as handle:
                    split_reader = csv.DictReader(handle)
                    rows = list(split_reader)
                self.assertEqual(split_reader.fieldnames, final_reader.fieldnames)
                split_rows.extend(rows)
            self.assertCountEqual(
                [(row["Function"], row["Label"]) for row in split_rows],
                [(row["Function"], row["Label"]) for row in final_rows],
            )

            # A second invocation exercises checkpoint reuse without rewriting.
            mtimes = {path: path.stat().st_mtime_ns for path in output_dir.glob("*.csv")}
            self.assertEqual(
                curate_dataset.main(
                    ["--input-dir", str(input_dir), "--output-dir", str(output_dir)]
                ),
                0,
            )
            self.assertEqual(
                mtimes, {path: path.stat().st_mtime_ns for path in output_dir.glob("*.csv")}
            )

    @staticmethod
    def row(
        function: str,
        *,
        w: str = "0",
        big_vul: str = "",
        devign: str = "",
        ps: str = "0",
        mat: str = "0",
        seci: str = "0",
    ) -> dict[str, str]:
        return {
            "Projectname": "example",
            "Commit-ID": "abc",
            "Filepath": "example.c",
            "Function": function,
            "LeadingComment": "",
            "PS": ps,
            "MAT": mat,
            "Devign": devign,
            "Big-Vul": big_vul,
            "W": w,
            "SecI": seci,
        }

    @staticmethod
    def write_input(path: Path, rows: list[dict[str, str]]) -> None:
        fieldnames = list(rows[0])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()

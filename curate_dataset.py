#!/usr/bin/env python3
"""Build a formatted, token-limited, class-balanced MADE-WIC dataset.

The pipeline is deliberately split into durable CSV checkpoints.  An existing
checkpoint is reused on the next run; pass ``--force-from STEP`` to rebuild a
step and everything after it.

Steps:
    1. Combine the three MADE-WIC CSV files.
    2. Format every function with clang-format.
    3. Merge Devign, Big-Vul, and W into one boolean Vulnerable label.
    4. Count tiktoken tokens and discard functions over the configured limit.
    5. Downsample joint (Vulnerable, MAT) strata to restore the pre-filter
       distribution, then emit Projectname, Filepath, Function, and a
       [Vulnerable, MAT] label.
    6. Create deterministic, stratified train/validation/test splits using a
       70/10/20 ratio.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence, TextIO, TypeVar

try:
    import tiktoken
except ImportError as exc:  # pragma: no cover - depends on the local environment
    raise SystemExit(
        "Missing dependency 'tiktoken'. Install it in the project environment "
        "with: ./env/bin/python -m pip install tiktoken tqdm"
    ) from exc

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - depends on the local environment
    raise SystemExit(
        "Missing dependency 'tqdm'. Install it in the project environment "
        "with: ./env/bin/python -m pip install tqdm"
    ) from exc


T = TypeVar("T")
Row = dict[str, str]
Stratum = tuple[int, int]

INPUT_FILES = (
    ("OSPR", Path("OSPR/complete.csv")),
    ("big-vul", Path("big-vul/complete.csv")),
    ("devign", Path("devign/complete.csv")),
)
BASE_COLUMNS = (
    "Projectname",
    "Commit-ID",
    "Filepath",
    "Function",
    "LeadingComment",
    "PS",
    "MAT",
    "Devign",
    "Big-Vul",
    "W",
    "SecI",
    "SourceDataset",
)
VULNERABILITY_COLUMNS = ("Devign", "Big-Vul", "W")
ANNOTATION_COLUMNS = ("PS", "MAT", "SecI")
STEP_FILENAMES = {
    1: "01_loaded.csv",
    2: "02_clang_formatted.csv",
    3: "03_labels_merged.csv",
    4: "04_token_filtered.csv",
    5: "05_rebalanced.csv",
}
SPLIT_FILENAMES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}
SPLIT_RATIOS = {"train": 0.7, "validation": 0.1, "test": 0.2}
MANIFEST_NAME = "manifest.json"

# CSV Function values contain many physical newlines, so file line counts do not
# equal row counts. Checkpoint row counts are recorded here for accurate tqdm bars.
csv.field_size_limit(sys.maxsize)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("MADE-WIC/Dataset"),
        help="Directory containing OSPR, big-vul, and devign folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("curated_dataset_steps"),
        help="Directory for resumable checkpoints and the manifest.",
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=500,
        help="Maximum number of tokens allowed per function (default: 500).",
    )
    parser.add_argument(
        "--encoding",
        default="cl100k_base",
        help="tiktoken encoding name (default: cl100k_base).",
    )
    parser.add_argument(
        "--clang-format",
        dest="clang_format",
        default="clang-format",
        help="clang-format executable name or path.",
    )
    parser.add_argument(
        "--clang-style",
        default="LLVM",
        help="clang-format style name or inline configuration (default: LLVM).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Concurrent clang-format processes (default: up to 8).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for deterministic downsampling."
    )
    parser.add_argument(
        "--force-from",
        type=int,
        choices=range(1, 7),
        metavar="STEP",
        help="Rebuild this step and all later checkpoints.",
    )
    args = parser.parse_args(argv)
    if args.token_limit < 0:
        parser.error("--token-limit must be non-negative")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def load_manifest(output_dir: Path) -> dict[str, object]:
    path = output_dir / MANIFEST_NAME
    if not path.exists():
        return {"steps": {}}
    try:
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Cannot read checkpoint manifest {path}: {exc}") from exc
    manifest.setdefault("steps", {})
    return manifest


def save_manifest(output_dir: Path, manifest: Mapping[str, object]) -> None:
    path = output_dir / MANIFEST_NAME
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def step_record(manifest: Mapping[str, object], step: int) -> dict[str, object]:
    steps = manifest.get("steps", {})
    if not isinstance(steps, dict):
        return {}
    record = steps.get(str(step), {})
    return record if isinstance(record, dict) else {}


def record_step(
    output_dir: Path,
    manifest: dict[str, object],
    step: int,
    path: Path,
    rows: int,
    **metadata: object,
) -> None:
    steps = manifest.setdefault("steps", {})
    if not isinstance(steps, dict):
        raise RuntimeError("Manifest 'steps' value is not an object")
    steps[str(step)] = {"file": path.name, "rows": rows, **metadata}
    save_manifest(output_dir, manifest)


def should_run(step: int, path: Path, force_from: int | None) -> bool:
    return not path.exists() or (force_from is not None and step >= force_from)


@contextmanager
def atomic_csv_writer(path: Path, fieldnames: Sequence[str]) -> Iterator[csv.DictWriter]:
    """Write a checkpoint atomically so interrupted runs are never considered done."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            yield writer
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def atomic_split_writers(
    paths: Mapping[str, Path], fieldnames: Sequence[str]
) -> Iterator[dict[str, csv.DictWriter]]:
    """Atomically publish a related set of split CSV files."""
    temporary_paths = {
        name: path.with_suffix(path.suffix + ".tmp") for name, path in paths.items()
    }
    handles: dict[str, TextIO] = {}
    try:
        writers: dict[str, csv.DictWriter] = {}
        for name, temporary_path in temporary_paths.items():
            handle = temporary_path.open("w", newline="", encoding="utf-8")
            handles[name] = handle
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writers[name] = writer
        yield writers
        for handle in handles.values():
            handle.close()
        for name, path in paths.items():
            temporary_paths[name].replace(path)
    except BaseException:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
        raise


def open_reader(path: Path) -> tuple[TextIO, csv.DictReader]:
    handle = path.open(newline="", encoding="utf-8")
    return handle, csv.DictReader(handle)


def require_columns(path: Path, fieldnames: Sequence[str] | None, required: Iterable[str]) -> None:
    missing = set(required).difference(fieldnames or ())
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def as_bool(value: object) -> int:
    """Normalize common CSV boolean forms, treating empty values as false."""
    normalized = str(value).strip().lower()
    if normalized in {"", "0", "0.0", "false", "f", "no", "n", "nan", "none"}:
        return 0
    if normalized in {"1", "1.0", "true", "t", "yes", "y"}:
        return 1
    raise ValueError(f"Unsupported boolean label: {value!r}")


def row_stratum(row: Mapping[str, str]) -> Stratum:
    vulnerable = as_bool(row.get("Vulnerable", "0"))
    mat = as_bool(row.get("MAT", "0"))
    return vulnerable, mat


def run_step_1(input_dir: Path, output_path: Path) -> tuple[int, dict[str, int]]:
    rows = 0
    per_dataset: dict[str, int] = {}
    with atomic_csv_writer(output_path, BASE_COLUMNS) as writer:
        for source_name, relative_path in tqdm(
            INPUT_FILES, desc="Step 1/6 datasets", unit="dataset"
        ):
            input_path = input_dir / relative_path
            if not input_path.is_file():
                raise FileNotFoundError(f"Missing MADE-WIC input: {input_path}")
            handle, reader = open_reader(input_path)
            try:
                require_columns(
                    input_path, reader.fieldnames, ("Function", *ANNOTATION_COLUMNS, "W")
                )
                dataset_rows = 0
                for row in tqdm(reader, desc=f"  loading {source_name}", unit="rows", leave=False):
                    normalized = {column: row.get(column, "") for column in BASE_COLUMNS}
                    normalized["SourceDataset"] = source_name
                    writer.writerow(normalized)
                    rows += 1
                    dataset_rows += 1
                per_dataset[source_name] = dataset_rows
            finally:
                handle.close()
    return rows, per_dataset


def format_function(
    function: str, clang_format: str, clang_style: str
) -> tuple[str, str | None]:
    if not function.strip():
        return function, None
    process = subprocess.run(
        [
            clang_format,
            f"--style={clang_style}",
            "--assume-filename=function.cpp",
        ],
        input=function,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        message = process.stderr.strip() or f"exit status {process.returncode}"
        return function, message
    return process.stdout.rstrip("\n"), None


def bounded_parallel_map(
    function: Callable[[T], T],
    items: Iterable[T],
    workers: int,
    max_pending: int | None = None,
) -> Iterator[T]:
    """Ordered executor map with bounded memory for very large CSV files."""
    pending_limit = max_pending or workers * 4
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: deque[Future[T]] = deque()
        for _ in range(pending_limit):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass


def run_step_2(
    input_path: Path,
    output_path: Path,
    total: int | None,
    clang_format: str,
    clang_style: str,
    workers: int,
) -> tuple[int, int]:
    executable = shutil.which(clang_format)
    if executable is None:
        raise FileNotFoundError(f"clang-format executable not found: {clang_format}")
    handle, reader = open_reader(input_path)
    require_columns(input_path, reader.fieldnames, ("Function",))
    fieldnames = reader.fieldnames or []
    failures = 0

    def format_row(row: Row) -> Row:
        formatted, error = format_function(row["Function"], executable, clang_style)
        row["Function"] = formatted
        # The private key travels only inside the executor and is ignored by CSV.
        row["__format_error"] = error or ""
        return row

    try:
        with atomic_csv_writer(output_path, fieldnames) as writer:
            formatted_rows = bounded_parallel_map(format_row, reader, workers)
            rows = 0
            for row in tqdm(formatted_rows, total=total, desc="Step 2/6 clang-format", unit="rows"):
                failures += bool(row.pop("__format_error"))
                writer.writerow(row)
                rows += 1
    finally:
        handle.close()
    return rows, failures


def run_step_3(
    input_path: Path, output_path: Path, total: int | None
) -> tuple[int, Counter[Stratum]]:
    handle, reader = open_reader(input_path)
    require_columns(
        input_path, reader.fieldnames, (*VULNERABILITY_COLUMNS, *ANNOTATION_COLUMNS)
    )
    fieldnames = [
        column for column in (reader.fieldnames or []) if column not in VULNERABILITY_COLUMNS
    ]
    function_index = fieldnames.index("Function")
    fieldnames.insert(function_index + 1, "Vulnerable")
    distribution: Counter[Stratum] = Counter()
    try:
        with atomic_csv_writer(output_path, fieldnames) as writer:
            rows = 0
            for row in tqdm(reader, total=total, desc="Step 3/6 merge labels", unit="rows"):
                row["Vulnerable"] = str(
                    int(any(as_bool(row.get(column, "")) for column in VULNERABILITY_COLUMNS))
                )
                distribution[row_stratum(row)] += 1
                writer.writerow(row)
                rows += 1
    finally:
        handle.close()
    return rows, distribution


def encode_batches(
    rows: Iterable[Row], encoding: object, batch_size: int, workers: int
) -> Iterator[tuple[Row, int]]:
    batch: list[Row] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            functions = [item["Function"] for item in batch]
            token_lists = encoding.encode_ordinary_batch(functions, num_threads=workers)
            yield from zip(batch, map(len, token_lists))
            batch = []
    if batch:
        functions = [item["Function"] for item in batch]
        token_lists = encoding.encode_ordinary_batch(functions, num_threads=workers)
        yield from zip(batch, map(len, token_lists))


def run_step_4(
    input_path: Path,
    output_path: Path,
    total: int | None,
    encoding_name: str,
    token_limit: int,
    workers: int,
) -> tuple[int, int, Counter[Stratum]]:
    encoding = tiktoken.get_encoding(encoding_name)
    handle, reader = open_reader(input_path)
    require_columns(
        input_path,
        reader.fieldnames,
        ("Function", "Vulnerable", *ANNOTATION_COLUMNS),
    )
    fieldnames = list(reader.fieldnames or [])
    function_index = fieldnames.index("Function")
    fieldnames.insert(function_index + 1, "TokenCount")
    distribution: Counter[Stratum] = Counter()
    removed = 0
    try:
        with atomic_csv_writer(output_path, fieldnames) as writer:
            rows = 0
            encoded_rows = encode_batches(reader, encoding, batch_size=256, workers=workers)
            for row, token_count in tqdm(
                encoded_rows, total=total, desc="Step 4/6 tiktoken filter", unit="rows"
            ):
                if token_count > token_limit:
                    removed += 1
                    continue
                row["TokenCount"] = str(token_count)
                distribution[row_stratum(row)] += 1
                writer.writerow(row)
                rows += 1
    finally:
        handle.close()
    return rows, removed, distribution


def deserialize_distribution(value: object) -> Counter[Stratum]:
    result: Counter[Stratum] = Counter()
    if not isinstance(value, dict):
        return result
    for key, count in value.items():
        vulnerable, mat = str(key).split(",", maxsplit=1)
        result[(int(vulnerable), int(mat))] = int(count)
    return result


def serialize_distribution(distribution: Mapping[Stratum, int]) -> dict[str, int]:
    return {
        f"{vulnerable},{mat}": distribution.get((vulnerable, mat), 0)
        for vulnerable in (0, 1)
        for mat in (0, 1)
    }


def target_stratum_counts(
    original: Mapping[Stratum, int], available: Mapping[Stratum, int]
) -> Counter[Stratum]:
    """Find the largest near-proportional joint sample that fits available rows."""
    original_total = sum(original.values())
    if original_total == 0:
        raise ValueError("Cannot reconstruct a distribution from an empty dataset")
    positive = [key for key, count in original.items() if count > 0]
    if any(available.get(key, 0) == 0 for key in positive):
        missing = [key for key in positive if available.get(key, 0) == 0]
        raise ValueError(
            "Token filtering removed every row from required strata: "
            f"{missing}; distribution cannot be reconstructed by downsampling"
        )

    # Continuous upper bound. Floors and largest remainders make the result
    # integer while keeping every stratum at or below its available count.
    sample_total = math.floor(
        min(available.get(key, 0) * original_total / original[key] for key in positive)
    )
    exact = {key: sample_total * original[key] / original_total for key in positive}
    targets: Counter[Stratum] = Counter(
        {key: min(available.get(key, 0), math.floor(value)) for key, value in exact.items()}
    )
    remaining = sample_total - sum(targets.values())
    ranked = sorted(
        positive,
        key=lambda key: (exact[key] - math.floor(exact[key]), key),
        reverse=True,
    )
    for key in ranked:
        if remaining == 0:
            break
        if targets[key] < available.get(key, 0):
            targets[key] += 1
            remaining -= 1
    return targets


def choose_ordinals_to_remove(
    available: Mapping[Stratum, int], targets: Mapping[Stratum, int], seed: int
) -> dict[Stratum, set[int]]:
    rng = random.Random(seed)
    selected: dict[Stratum, set[int]] = {}
    for key, count in available.items():
        remove_count = count - targets.get(key, 0)
        if remove_count < 0:
            raise ValueError(f"Target exceeds available rows for stratum {key}")
        selected[key] = set(rng.sample(range(count), remove_count))
    return selected


def allocate_split_counts(
    distribution: Mapping[Stratum, int],
) -> dict[Stratum, dict[str, int]]:
    """Round stratified counts while preserving exact global split sizes."""
    total = sum(distribution.values())
    split_names = tuple(SPLIT_RATIOS)
    global_exact = {name: total * SPLIT_RATIOS[name] for name in split_names}
    global_targets = {name: math.floor(global_exact[name]) for name in split_names}
    global_remaining = total - sum(global_targets.values())
    global_ranked = sorted(
        split_names,
        key=lambda name: (global_exact[name] - global_targets[name], name),
        reverse=True,
    )
    for name in global_ranked[:global_remaining]:
        global_targets[name] += 1

    allocations: dict[Stratum, dict[str, int]] = {}
    fractions: dict[tuple[Stratum, str], float] = {}
    for key, count in distribution.items():
        allocations[key] = {}
        for name in split_names:
            exact = count * SPLIT_RATIOS[name]
            allocations[key][name] = math.floor(exact)
            fractions[(key, name)] = exact - math.floor(exact)

    row_deficits = {
        key: distribution[key] - sum(allocations[key].values()) for key in distribution
    }
    column_deficits = {
        name: global_targets[name]
        - sum(allocations[key][name] for key in distribution)
        for name in split_names
    }
    while sum(row_deficits.values()):
        candidates = [
            (fractions[(key, name)], key, name)
            for key in distribution
            for name in split_names
            if row_deficits[key] > 0 and column_deficits[name] > 0
        ]
        if not candidates:
            raise RuntimeError("Unable to allocate exact stratified split counts")
        _, key, name = max(candidates)
        allocations[key][name] += 1
        row_deficits[key] -= 1
        column_deficits[name] -= 1
    return allocations


def build_split_assignments(
    allocations: Mapping[Stratum, Mapping[str, int]], seed: int
) -> dict[Stratum, bytearray]:
    """Return compact random split assignments: train=0, validation=1, test=2."""
    rng = random.Random(seed)
    assignments: dict[Stratum, bytearray] = {}
    for key, counts in allocations.items():
        total = sum(counts.values())
        validation_count = counts["validation"]
        test_count = counts["test"]
        held_out = rng.sample(range(total), validation_count + test_count)
        values = bytearray(total)
        for ordinal in held_out[:validation_count]:
            values[ordinal] = 1
        for ordinal in held_out[validation_count:]:
            values[ordinal] = 2
        assignments[key] = values
    return assignments


def run_step_5(
    input_path: Path,
    output_path: Path,
    total: int | None,
    original: Mapping[Stratum, int],
    available: Mapping[Stratum, int],
    seed: int,
) -> tuple[int, Counter[Stratum]]:
    targets = target_stratum_counts(original, available)
    removals = choose_ordinals_to_remove(available, targets, seed)
    seen: Counter[Stratum] = Counter()
    output_distribution: Counter[Stratum] = Counter()
    handle, reader = open_reader(input_path)
    require_columns(
        input_path,
        reader.fieldnames,
        ("Projectname", "Filepath", "Function", "Vulnerable", "MAT"),
    )
    try:
        final_columns = ("Projectname", "Filepath", "Function", "Label")
        with atomic_csv_writer(output_path, final_columns) as writer:
            rows = 0
            for row in tqdm(
                reader,
                total=total,
                desc="Step 5/6 restore distribution",
                unit="rows",
            ):
                key = row_stratum(row)
                ordinal = seen[key]
                seen[key] += 1
                if ordinal in removals.get(key, set()):
                    continue
                vulnerable, mat = key
                writer.writerow(
                    {
                        "Projectname": row["Projectname"],
                        "Filepath": row["Filepath"],
                        "Function": row["Function"],
                        "Label": f"[{vulnerable},{mat}]",
                    }
                )
                output_distribution[key] += 1
                rows += 1
    finally:
        handle.close()
    if output_distribution != targets:
        raise RuntimeError(
            f"Rebalancing produced {output_distribution}, expected {targets}"
        )
    return rows, output_distribution


def label_stratum(value: str) -> Stratum:
    try:
        label = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Label value: {value!r}") from exc
    if not isinstance(label, list) or len(label) != 2:
        raise ValueError(f"Label must be [Vulnerable, MAT], got {value!r}")
    return as_bool(label[0]), as_bool(label[1])


def run_step_6(
    input_path: Path,
    output_paths: Mapping[str, Path],
    total: int | None,
    distribution: Mapping[Stratum, int],
    seed: int,
) -> dict[str, int]:
    allocations = allocate_split_counts(distribution)
    assignments = build_split_assignments(allocations, seed)
    split_names = tuple(SPLIT_RATIOS)
    assignment_names = split_names
    seen: Counter[Stratum] = Counter()
    output_counts = {name: 0 for name in split_names}
    handle, reader = open_reader(input_path)
    final_columns = ("Projectname", "Filepath", "Function", "Label")
    require_columns(input_path, reader.fieldnames, final_columns)
    try:
        with atomic_split_writers(output_paths, final_columns) as writers:
            for row in tqdm(
                reader,
                total=total,
                desc="Step 6/6 split dataset",
                unit="rows",
            ):
                key = label_stratum(row["Label"])
                ordinal = seen[key]
                seen[key] += 1
                try:
                    split_name = assignment_names[assignments[key][ordinal]]
                except (KeyError, IndexError) as exc:
                    raise RuntimeError(
                        f"Step 5 distribution metadata does not match rows for {key}"
                    ) from exc
                writers[split_name].writerow(row)
                output_counts[split_name] += 1
    finally:
        handle.close()
    if seen != Counter(distribution):
        raise RuntimeError(
            f"Split input distribution {seen} does not match metadata {distribution}"
        )
    expected_counts = {
        name: sum(counts[name] for counts in allocations.values())
        for name in split_names
    }
    if output_counts != expected_counts:
        raise RuntimeError(f"Split counts {output_counts} do not match {expected_counts}")
    return output_counts


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.output_dir)
    paths = {
        step: args.output_dir / filename for step, filename in STEP_FILENAMES.items()
    }
    split_paths = {
        name: args.output_dir / filename for name, filename in SPLIT_FILENAMES.items()
    }

    if should_run(1, paths[1], args.force_from):
        rows, per_dataset = run_step_1(args.input_dir, paths[1])
        record_step(args.output_dir, manifest, 1, paths[1], rows, per_dataset=per_dataset)
    else:
        tqdm.write(f"Step 1/6: using checkpoint {paths[1]}")

    if should_run(2, paths[2], args.force_from):
        rows, failures = run_step_2(
            paths[1],
            paths[2],
            int(step_record(manifest, 1).get("rows", 0)) or None,
            args.clang_format,
            args.clang_style,
            args.workers,
        )
        record_step(
            args.output_dir,
            manifest,
            2,
            paths[2],
            rows,
            clang_format_failures=failures,
            clang_style=args.clang_style,
        )
        if failures:
            tqdm.write(
                f"Warning: clang-format failed for {failures} rows; "
                "their original text was retained."
            )
    else:
        tqdm.write(f"Step 2/6: using checkpoint {paths[2]}")

    if should_run(3, paths[3], args.force_from):
        rows, distribution = run_step_3(
            paths[2], paths[3], int(step_record(manifest, 2).get("rows", 0)) or None
        )
        record_step(
            args.output_dir,
            manifest,
            3,
            paths[3],
            rows,
            distribution=serialize_distribution(distribution),
        )
    else:
        tqdm.write(f"Step 3/6: using checkpoint {paths[3]}")

    if should_run(4, paths[4], args.force_from):
        rows, removed, distribution = run_step_4(
            paths[3],
            paths[4],
            int(step_record(manifest, 3).get("rows", 0)) or None,
            args.encoding,
            args.token_limit,
            args.workers,
        )
        record_step(
            args.output_dir,
            manifest,
            4,
            paths[4],
            rows,
            removed_over_token_limit=removed,
            token_limit=args.token_limit,
            encoding=args.encoding,
            distribution=serialize_distribution(distribution),
        )
    else:
        tqdm.write(f"Step 4/6: using checkpoint {paths[4]}")

    if should_run(5, paths[5], args.force_from):
        original = deserialize_distribution(step_record(manifest, 3).get("distribution"))
        available = deserialize_distribution(step_record(manifest, 4).get("distribution"))
        if not original or not available:
            raise RuntimeError(
                "Steps 3 and 4 need distribution metadata. Rebuild with --force-from 3."
            )
        rows, distribution = run_step_5(
            paths[4],
            paths[5],
            int(step_record(manifest, 4).get("rows", 0)) or None,
            original,
            available,
            args.seed,
        )
        record_step(
            args.output_dir,
            manifest,
            5,
            paths[5],
            rows,
            seed=args.seed,
            distribution=serialize_distribution(distribution),
        )
    else:
        tqdm.write(f"Step 5/6: using checkpoint {paths[5]}")

    split_checkpoint_exists = all(path.exists() for path in split_paths.values())
    if not split_checkpoint_exists or (
        args.force_from is not None and args.force_from <= 6
    ):
        distribution = deserialize_distribution(
            step_record(manifest, 5).get("distribution")
        )
        if not distribution:
            raise RuntimeError(
                "Step 5 needs distribution metadata. Rebuild with --force-from 5."
            )
        split_counts = run_step_6(
            paths[5],
            split_paths,
            int(step_record(manifest, 5).get("rows", 0)) or None,
            distribution,
            args.seed,
        )
        record_step(
            args.output_dir,
            manifest,
            6,
            split_paths["train"],
            sum(split_counts.values()),
            files={name: path.name for name, path in split_paths.items()},
            split_counts=split_counts,
            ratios=SPLIT_RATIOS,
            seed=args.seed,
        )
    else:
        tqdm.write(
            "Step 6/6: using checkpoints "
            + ", ".join(str(path) for path in split_paths.values())
        )

    tqdm.write(
        "Dataset splits: "
        + ", ".join(f"{name}={path}" for name, path in split_paths.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

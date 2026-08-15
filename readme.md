# MADE-WIC Dataset Curation, SetFit, and CodeBERT Training

This repository builds a unified function-level dataset from the three
MADE-WIC datasets and fine-tunes either a SetFit or CodeBERT multi-output
classifier on the result. The two prediction targets are:

1. Whether a function contains a weakness or vulnerability.
2. Whether the function is positive according to MAT.

The final label is stored as `[weakness, MAT]`. For example, `[1,0]` denotes a
function that is positive for the aggregated weakness label and negative for
MAT.

## Requirements

The pipeline requires Python 3.12 or a compatible recent Python version.
`clang-format` must be available on `PATH` because every function is formatted
before tokenization.

Check the formatter installation with:

```bash
clang-format --version
```

Create a virtual environment and install the dataset-curation and model
training dependencies:

```bash
python3 -m venv env
./env/bin/python -m pip install -r requirements.txt
```

Verify that the main dependencies can be imported:

```bash
./env/bin/python -c "import datasets, numpy, setfit, tiktoken, torch, tqdm"
```

## Input data

Place the MADE-WIC repository in the project root with this structure:

```text
MADE-WIC/
`-- Dataset/
    |-- OSPR/
    |   `-- complete.csv
    |-- big-vul/
    |   `-- complete.csv
    `-- devign/
        `-- complete.csv
```

The curation pipeline reads all three files. It does not modify the source
datasets.

## Build the curated dataset

Run the complete pipeline with:

```bash
./env/bin/python curate_dataset.py
```

The pipeline performs six stages:

1. Loads and combines OSPR, Big-Vul, and Devign.
2. Formats each function with `clang-format` using the LLVM style by default.
3. Combines `Devign`, `Big-Vul`, and `W` with a boolean OR into one weakness
   label.
4. Counts tokens with the `cl100k_base` tiktoken encoding and removes
   functions containing more than 500 tokens.
5. Restores the pre-filter joint distribution of the weakness and MAT labels
   by deterministically downsampling overrepresented classes.
6. Creates stratified train, validation, and test splits using a 70/10/20
   ratio.

Every stage displays a progress bar. The formatting stage runs several
`clang-format` processes concurrently; use `--workers` to change the number of
workers.

Example with four workers:

```bash
./env/bin/python curate_dataset.py --workers 4
```

The complete dataset is large. Formatting every function can take a
considerable amount of time and the intermediate CSV checkpoints require
additional disk space.

## Checkpoints and resumed runs

Dataset checkpoints are written to `curated_dataset_steps` by default:

```text
01_loaded.csv
02_clang_formatted.csv
03_labels_merged.csv
04_token_filtered.csv
05_rebalanced.csv
manifest.json
train.csv
validation.csv
test.csv
```

A completed checkpoint is reused automatically in later runs. Temporary files
ending in `.tmp` are incomplete and are not treated as completed stages.

Use `--force-from` to rebuild a stage and every downstream stage. For example,
to rerun token filtering, rebalancing, and splitting:

```bash
./env/bin/python curate_dataset.py --force-from 4
```

To regenerate only the train, validation, and test files:

```bash
./env/bin/python curate_dataset.py --force-from 6
```

Changing a setting does not invalidate an existing checkpoint automatically.
Use `--force-from` at the first stage affected by the setting change.

Common options include:

```text
--token-limit 500
--encoding cl100k_base
--clang-style LLVM
--workers 8
--seed 42
--output-dir curated_dataset_steps
```

Run `./env/bin/python curate_dataset.py --help` for the complete option list.

## Curated output format

Each split contains four columns:

| Column | Description |
| --- | --- |
| `Projectname` | Source project name from MADE-WIC. |
| `Filepath` | Original path of the source file. |
| `Function` | Function body after `clang-format`. |
| `Label` | Two-dimensional label in `[weakness, MAT]` order. |

The possible label values are:

| Label | Meaning |
| --- | --- |
| `[0,0]` | No aggregated weakness and MAT negative. |
| `[1,0]` | Aggregated weakness positive and MAT negative. |
| `[0,1]` | No aggregated weakness and MAT positive. |
| `[1,1]` | Aggregated weakness positive and MAT positive. |

The split operation is deterministic for a given seed and preserves the joint
label distribution as closely as integer split sizes allow. The manifest
records row counts, distributions, settings, and formatting failures for each
stage.

## Fine-tune SetFit

The training script reads the three curated split files. By default, it uses a
seeded random sample containing 5% of the training split:

```bash
./env/bin/python setfit_train.py
```

The default model is
`sentence-transformers/paraphrase-MiniLM-L6-v2` with SetFit's `multi-output`
strategy. Training uses five epochs, a batch size of 32, and 20 contrastive
pair-generation iterations.

Validation is performed after every epoch. After training, the final model is
evaluated on both the validation and test splits. Test data is never used for
training.

Run on the complete training split with:

```bash
./env/bin/python setfit_train.py --train-fraction 1
```

Example with a different model and output directory:

```bash
./env/bin/python setfit_train.py \
  --model-id BAAI/bge-small-en-v1.5 \
  --epochs 10 \
  --output-dir models/setfit-bge
```

Run `./env/bin/python setfit_train.py --help` for all training options.

## Fine-tune CodeBERT

The CodeBERT training script uses the same prepared splits, stratified
100-instance default sample, label order, evaluation metrics, and checkpoint
policy as the SetFit script:

```bash
./env/bin/python codebert_train.py
```

The default model is `microsoft/codebert-base`. It is loaded with a
two-dimensional multi-label sequence-classification head. Functions are
truncated to at most 512 CodeBERT tokens, and predictions use a sigmoid
threshold of 0.5 for each label dimension.

Validation is performed after every epoch. The final model is evaluated on
both the validation and test splits.

Run CodeBERT on the complete training split with:

```bash
./env/bin/python codebert_train.py --train-limit 0
```

Example with different optimization settings:

```bash
./env/bin/python codebert_train.py \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 0.00002 \
  --output-dir models/codebert-experiment
```

Run `./env/bin/python codebert_train.py --help` for all CodeBERT options.

## Slurm jobs

Two Slurm submission files are provided for full-dataset training on one 80 GB
GPU. Submit them from the project root:

```bash
sbatch setfit_train.sbatch
sbatch codebert_train.sbatch
```

Both jobs use the `ml-vuln` account, `gpu-low` partition, `gpu80g` constraint,
eight CPU cores, 80 GB of host memory, and a maximum runtime of 20 days. They
load CUDA and Python 3.11.7, activate the `env` virtual environment, verify the
three dataset splits, and train with `--train-limit 0`.

Arguments placed after the submission filename are forwarded to the training
script. For example:

```bash
sbatch codebert_train.sbatch --epochs 10 --batch-size 16
sbatch setfit_train.sbatch --train-limit 100
```

Set `DATA_DIR` when the split files are not in `curated_dataset_steps`:

```bash
sbatch --export=ALL,DATA_DIR=/data/path/to/splits codebert_train.sbatch
```

Create the virtual environment on the cluster with the same Python module used
by the jobs. Virtual environments copied from another Python version or
operating system are not portable.

## Training outputs and metrics

The default SetFit training output is:

```text
models/setfit/
|-- checkpoints/
|-- final/
|-- metrics.csv
`-- metrics.json
```

`checkpoints` retains the model checkpoint saved at every epoch. `final`
contains the final trained model.

CodeBERT uses the same structure below `models/codebert`.

Both metrics files contain results for the validation and test splits. Each
metric is calculated independently for the weakness and MAT dimensions, then
averaged across both dimensions:

| Metric | Description |
| --- | --- |
| `weakness_accuracy` | Binary accuracy for the aggregated weakness dimension. |
| `weakness_precision` | Positive-class precision for the weakness dimension. |
| `weakness_recall` | Positive-class recall for the weakness dimension. |
| `weakness_f1` | Positive-class F1 for the weakness dimension. |
| `MAT_accuracy` | Binary accuracy for the MAT dimension. |
| `MAT_precision` | Positive-class precision for the MAT dimension. |
| `MAT_recall` | Positive-class recall for the MAT dimension. |
| `MAT_f1` | Positive-class F1 for the MAT dimension. |
| `average_accuracy` | Unweighted mean of both accuracy values. |
| `average_precision` | Unweighted mean of both precision values. |
| `average_recall` | Unweighted mean of both recall values. |
| `average_f1` | Unweighted mean of both F1 values. |

## Troubleshooting

### A dataset stage appears to restart

Confirm that the corresponding CSV checkpoint and `manifest.json` exist in
the selected output directory. A `.tmp` file indicates that the previous run
stopped before completing that stage.

### `clang-format` is not found

Install `clang-format` with the system package manager or pass its absolute
path:

```bash
./env/bin/python curate_dataset.py --clang-format /path/to/clang-format
```

### Training dependencies cannot be imported

Install the training dependencies into the interpreter used to launch the
script:

```bash
./env/bin/python -m pip install -r requirements.txt
```

### A 100-instance training sample lacks a binary class

The SetFit multi-output classifier requires both zero and one examples for
each output dimension. The training script samples jointly by label and stops
with an explicit error if the available training data cannot satisfy this
condition. Increase `--train-limit` or inspect the curated label distribution.

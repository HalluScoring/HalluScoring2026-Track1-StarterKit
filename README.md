# HalluScoring 2026 Track 1 - Starter Kit

A complete pipeline for detecting hallucinations in Arabic LLM outputs using fine-tuned BERT models. This starter kit supports **both subtasks** of Track 1 (1.1 and 1.2), enabling you to build and evaluate hallucination detection systems with ease.

## Overview

The HalluScoring challenge focuses on detecting factual hallucinations in model-generated answers to Arabic questions. This kit provides:

- **4 BERT Models**: CamelBERT, ARBERT, MARBERTv2, and multilingual BERT
- **Subtask Support**: Separate pipelines for Task 1.1 and Task 1.2 with their own data
- **No preprocessing step**: Training and inference read the provided `.xlsx`/`.csv` files directly, auto-detecting the format and mapping columns
- **Complete Workflow**: Training → Inference

### Task Definitions

- **Task 1.1**: Train and evaluate on the Subtask 1.1 set
  - Default data: `task1.1_data/task1.1_train.xlsx`, `task1.1_data/task1.1_dev.xlsx`

- **Task 1.2**: Train and evaluate on the Subtask 1.2 set
  - Default data: `task1.2_data/task1.2_train.xlsx`, `task1.2_data/task1.2_dev.xlsx`

## Project Structure

```
├── README.md              # This file
├── train.py               # Train BERT models for hallucination detection
├── predict.py             # Run inference and generate predictions
├── task1.1_data/          # Subtask 1.1 train/dev files
│   ├── task1.1_train.xlsx
│   └── task1.1_dev.xlsx
├── task1.2_data/          # Subtask 1.2 train/dev files
│   ├── task1.2_train.xlsx
│   └── task1.2_dev.xlsx
└── outputs/               # Generated during training / inference
    ├── task1.1/
    │   ├── camelbert/best_model/
    │   ├── arbert/best_model/
    │   ├── marbert/best_model/
    │   ├── mbert/best_model/
    │   ├── predictions_<model>.csv   # written by predict.py
    │   └── all_dev_results.json
    └── task1.2/
        └── ... (same layout)
```

## Data Format

No conversion step is required. Files are read directly and the loader picks the
reader from the file's actual content (magic bytes), so a mislabeled `.csv`/`.xlsx`
still loads correctly.

The starter-kit `.xlsx` headers are mapped automatically to canonical names:

| Starter-kit header | Canonical name  |
| ------------------ | --------------- |
| `ID`               | `id`            |
| `Question`         | `question`      |
| `Gold Answer`      | `gold_answer`   |
| `Generator_model`  | `model_name`    |
| `Generated_answer` | `model_answer`  |
| `Label`            | `label`         |

A pre-mapped `.csv` with canonical columns (`question`, `model_answer`,
`[label]`, `[model_name]`) also works directly.

## Installation

### Prerequisites

- Python 3.8+
- GPU recommended (NVIDIA CUDA)

### Setup

```bash
cd HalluScoring2026-Track1-StarterKit
pip install torch transformers pandas openpyxl scikit-learn scipy
```

## Training

By default, paths are resolved automatically from `--data_dir` (default: current
directory) using the `task<X>_data/` layout, so the simplest invocation is:

### Train Task 1.1

```bash
python train.py --task 1.1
```

### Train Task 1.2

```bash
python train.py --task 1.2
```

### Train Both Tasks Back-to-Back

```bash
python train.py --task 1.1 1.2
```

### Override Data Paths

For a single task, use the generic flags:

```bash
python train.py --task 1.1 \
    --train_path task1.1_data/task1.1_train.xlsx \
    --dev_path   task1.1_data/task1.1_dev.xlsx
```

For both tasks at once, use the task-specific flags:

```bash
python train.py --task 1.1 1.2 \
    --train_path_task1_1 task1.1_data/task1.1_train.xlsx \
    --dev_path_task1_1   task1.1_data/task1.1_dev.xlsx \
    --train_path_task1_2 task1.2_data/task1.2_train.xlsx \
    --dev_path_task1_2   task1.2_data/task1.2_dev.xlsx
```

### Train Specific Models Only

By default, all 4 models are trained. To train only specific models:

```bash
python train.py --task 1.1 --models camelbert marbert
```

**Available models**: `camelbert`, `arbert`, `marbert`, `mbert` (or `all`)

| Key         | Model                                         |
| ----------- | --------------------------------------------- |
| `camelbert` | `CAMeL-Lab/bert-base-arabic-camelbert-mix`    |
| `arbert`    | `UBC-NLP/ARBERT`                              |
| `marbert`   | `UBC-NLP/MARBERTv2`                           |
| `mbert`     | `google-bert/bert-base-multilingual-cased`    |

### Training Hyperparameters

| Flag              | Default     |
| ----------------- | ----------- |
| `--max_length`    | 512         |
| `--batch_size`    | 16          |
| `--num_epochs`    | 5           |
| `--learning_rate` | 2e-5        |
| `--warmup_ratio`  | 0.1         |
| `--weight_decay`  | 0.01        |
| `--seed`          | 42          |
| `--output_dir`    | `./outputs` |

### Training Output

For each task and model, the script will:

- Validate and load every split up-front (fails fast on a bad path/column)
- Display training progress with loss and metrics each epoch
- Select and save the best checkpoint per model based on AUC-ROC, to
  `outputs/task<X>/<model>/best_model/`
- Print a final summary table (AUC-ROC, F1-Macro, AUC-PR per model)
- Write combined dev metrics to `outputs/all_dev_results.json`

## Inference

`predict.py` runs the best saved models on a test/dev file and writes per-model
predictions. It detects labels automatically (or force with `--has_labels`).

### Run Predictions — Task 1.1

```bash
python predict.py --task 1.1 --test_path test.xlsx
```

### Run Predictions — Task 1.2

```bash
python predict.py --task 1.2 --test_path test.xlsx
```

### Evaluate on Dev Set (with labels)

If your file contains a `Label`/`label` column, metrics are computed
automatically. You can also force it:

```bash
python predict.py --task 1.2 \
    --test_path task1.2_data/task1.2_dev.xlsx --has_labels
```

This computes and prints:

- **AUC-ROC**: Overall hallucination detection performance
- **AUC-PR**: Area under the precision-recall curve
- **F1-Macro**: Balanced F1 score
- **Per-generator breakdown**

### Run Specific Models Only

```bash
python predict.py --task 1.1 --test_path test.xlsx --models camelbert arbert
```

### Inference Output

For each model, predictions are saved to
`outputs/task<X>/predictions_<model>.csv`. Each row is the original input plus:

- `prob_hallucinated` — predicted probability of hallucination
- `predicted_label` — 0/1 (threshold 0.5)

When labels are present, a summary table is printed and metrics are written to
`outputs/task<X>/test_results.json`.

## Complete Workflow Example

```bash
# 1. Train all models for both tasks (uses default task1.1_data/ & task1.2_data/)
python train.py --task 1.1 1.2

# 2. Generate predictions for Task 1.1
python predict.py --task 1.1 --test_path test_task1.1.xlsx

# 3. Generate predictions for Task 1.2
python predict.py --task 1.2 --test_path test_task1.2.xlsx
```

## Citation

Released soon...

## Support

Join the discord: <a href="https://discord.gg/G7s48MRdTq" target="_blank">HalluScoring 2026 Challenge Discord</a> for questions, discussions, and updates.

## License

This starter kit is provided for the HalluScoring 2026 challenge.

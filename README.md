# HalluScoring 2026 Track 1 - Starter Kit

A complete pipeline for detecting hallucinations in Arabic LLM outputs using fine-tuned BERT models. This starter kit supports **both subtasks** of Track 1, enabling you to build and evaluate hallucination detection systems with ease.

## Overview

The HalluScoring challenge focuses on detecting factual hallucinations in model-generated answers to Arabic questions. This kit provides:

- **3 BERT Models**: CamelBERT, ArBERT, MarBERT, and multilingual BERT
- **Subtask Support**: Separate pipelines for Task 1 and Task 2 with different evaluation sets
- **Complete Workflow**: Data preprocessing → Training → Inference

### Task Definitions

- **Task 1 (Subtask 1)**: Train and evaluate on standard hallucination detection set
  - Training: `train_long.csv`
  - Evaluation: `dev_long.csv`

- **Task 2 (Subtask 2)**: Train on same data but evaluate on Task 2 distribution
  - Training: `train_long.csv`
  - Evaluation: `dev_task2_long.csv`

## Project Structure

```
├── README.md              # This file
├── preprocess.py          # Convert wide format to long format
├── train.py               # Train BERT models for hallucination detection
├── predict.py             # Run inference and generate predictions
└── outputs/               # Generated during training
    ├── task1/
    │   ├── camelbert/best_model/
    │   ├── arbert/best_model/
    │   ├── marbert/best_model/
    │   └── mbert/best_model/
    └── task2/
        ├── camelbert/best_model/
        ├── arbert/best_model/
        ├── marbert/best_model/
        └── mbert/best_model/
```

## Installation

### Prerequisites

- Python 3.8+
- GPU recommended (NVIDIA CUDA)

### Setup

```bash
# Clone or download the starter kit
cd HalluScoring2026-Track1-StarterKit

# or manually:
pip install torch transformers pandas openpyxl scikit-learn scipy
```

## Quick Start

### Step 1: Prepare Your Data

Steps to be specified soon...

#### Convert to Long Format

The first step is converting wide format data to long format for training:

```bash
# Single file conversion
python preprocess.py --input train.csv --output train_long.csv

# Multiple files at once (recommended)
python preprocess.py \
    --input train.csv --output train_long.csv \
    --input dev.csv --output dev_long.csv \
    --input dev_task2.csv --output dev_task2_long.csv
```

## Training

### Train Task 1 (Subtask 1)

```bash
python train.py --task 1 \
    --train_path train_long.csv \
    --dev_path dev_long.csv
```

This will:

1. Train all 4 BERT models (CamelBERT, ArBERT, MarBERT, mBERT)
2. Validate each model on `dev_long.csv`
3. Select the best checkpoint for each model using AUC-ROC
4. Save models to `outputs/task1/<model_name>/best_model/`

### Train Task 2 (Subtask 2)

```bash
python train.py --task 2 \
    --train_path train_long.csv \
    --dev_path dev_task2_long.csv
```

This uses the same training data as Task 1 but validates on the Task 2 distribution.

### Train Both Tasks Back-to-Back

```bash
python train.py --task 1 2 \
    --train_path train_long.csv \
    --dev_path_task1 dev_long.csv \
    --dev_path_task2 dev_task2_long.csv
```

### Train Specific Models Only

By default, all 4 models are trained. To train only specific models:

```bash
python train.py --task 1 \
    --train_path train_long.csv \
    --dev_path dev_long.csv \
    --models camelbert marbert
```

**Available models**: `camelbert`, `arbert`, `marbert`, `mbert`

### Training Output

For each task and model, the script will:

- Display training progress with loss and metrics
- Evaluate on the dev set every epoch
- Save the best model based on AUC-ROC score
- Print final metrics including AUC-ROC, AUC-PR, F1-Macro, and per-model breakdown

## Inference

### Run Predictions on Test Set - Task 1

```bash
# Using wide format (XLSX) - automatic conversion
python predict.py --task 1 --test_path test.xlsx

# Using long format (CSV) - already preprocessed
python predict.py --task 1 --test_path test_long.csv
```

### Run Predictions on Test Set - Task 2

```bash
python predict.py --task 2 --test_path test_task2.xlsx
```

### Evaluate on Dev Set (with labels)

If your test file contains label columns, use `--has_labels`:

```bash
python predict.py --task 1 --test_path dev_long.csv --has_labels
```

This will compute and print:

- **AUC-ROC**: Overall hallucination detection performance
- **AUC-PR**: Area under precision-recall curve
- **F1-Macro**: Balanced F1 score
- **Per-model breakdown**: Performance for each model

### Run Specific Models Only

```bash
python predict.py --task 1 --test_path test.xlsx --models camelbert arbert
```

### Output

Predictions are saved to `predictions_task<N>.json` with:

```json
{
  "model1": {
    "predictions": [0, 1, 0, ...],
    "probabilities": [0.1, 0.9, 0.2, ...],
    "metrics": {
      "auc_roc": 0.85,
      "auc_pr": 0.82,
      "f1_macro": 0.83
    }
  },
  ...
}
```

## Complete Workflow Example

Here's a typical end-to-end workflow:

```bash
# 1. Prepare data - convert wide format to long
python preprocess.py \
    --input train.xlsx --output train_long.csv \
    --input dev.xlsx --output dev_long.csv \
    --input dev_task2.xlsx --output dev_task2_long.csv \
    --input test.xlsx --output test_long.csv \
    --input test_task2.xlsx --output test_task2_long.csv

# 2. Train models for both tasks
python train.py --task 1 2 \
    --train_path train_long.csv \
    --dev_path_task1 dev_long.csv \
    --dev_path_task2 dev_task2_long.csv

# 3. Generate predictions for Task 1
python predict.py --task 1 --test_path test_long.csv

# 4. Generate predictions for Task 2
python predict.py --task 2 --test_path test_task2_long.csv
```

## Citation

Released soon...

## Support

Join the discord: <a href="https://discord.gg/G7s48MRdTq" target="_blank">HalluScoring 2026 Challenge Discord</a> for questions, discussions, and updates.

## License

This starter kit is provided for the HalluScoring 2026 challenge.

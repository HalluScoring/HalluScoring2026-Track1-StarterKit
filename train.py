"""
train.py
────────
Train CamelBERT / ArBERT / MarBERT / mBERT for hallucination detection
(HalluScoring 2026 — Track 1).

Data layout (matches the starter kit):

    <data_dir>/
    ├── task1.1_data/
    │   ├── task1.1_train.xlsx
    │   └── task1.1_dev.xlsx
    └── task1.2_data/
        ├── task1.2_train.xlsx
        └── task1.2_dev.xlsx

The .xlsx files are already in long format with columns:
    ID | Question | Gold Answer | Generator_model | Generated_answer | Label
These are mapped internally to: question / model_answer / model_name / label.

Task 1.1  →  train on task1.1_train.xlsx,  validate on task1.1_dev.xlsx
Task 1.2  →  train on task1.2_train.xlsx,  validate on task1.2_dev.xlsx

Best checkpoint is selected by AUC-ROC on the dev set. Models are saved to:
    outputs/task1.1/<bert_model>/best_model/
    outputs/task1.2/<bert_model>/best_model/

Usage
-----
# Both tasks — paths resolved automatically from --data_dir
python train.py --task 1.1 1.2 --data_dir .

# Single task
python train.py --task 1.1 --data_dir .

# Override paths explicitly (single task)
python train.py --task 1.1 \
    --train_path task1.1_data/task1.1_train.xlsx \
    --dev_path   task1.1_data/task1.1_dev.xlsx

# Train only specific BERT models
python train.py --task 1.1 --data_dir . --models camelbert marbert
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, classification_report,
)

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ──────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "camelbert": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "arbert":    "UBC-NLP/ARBERT",
    "marbert":   "UBC-NLP/MARBERTv2",
    "mbert":     "google-bert/bert-base-multilingual-cased",
}

TASKS = ["1.1", "1.2"]

# Map the starter-kit's .xlsx headers → the canonical names used below.
# (Already-canonical files are passed through unchanged.)
RAW_TO_CANON = {
    "Question":         "question",
    "Generated_answer": "model_answer",
    "Label":            "label",
    "Generator_model":  "model_name",
    "Gold Answer":      "gold_answer",
    "ID":               "id",
}

# ──────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────
class HallucinationDataset(Dataset):
    """Encodes: [CLS] question [SEP] model_answer [SEP]"""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        self.data       = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row      = self.data.iloc[idx]
        encoding = self.tokenizer(
            str(row["question"]),
            str(row["model_answer"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item


# ──────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = softmax(logits, axis=-1)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "auc_roc":  roc_auc_score(labels, probs),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "auc_pr":   average_precision_score(labels, probs),
    }


def evaluate_split(trainer, dataset, split_name: str) -> dict:
    out    = trainer.predict(dataset)
    probs  = softmax(out.predictions, axis=-1)[:, 1]
    labels = out.label_ids
    preds  = (probs >= 0.5).astype(int)

    auc_roc  = roc_auc_score(labels, probs)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    auc_pr   = average_precision_score(labels, probs)

    print(f"\n{'─'*55}")
    print(f"  {split_name}")
    print(f"{'─'*55}")
    print(f"  AUC-ROC  : {auc_roc:.4f}  ← main metric")
    print(f"  F1-Macro : {f1_macro:.4f}")
    print(f"  AUC-PR   : {auc_pr:.4f}")
    print(classification_report(
        labels, preds,
        target_names=["Not Hallucinated", "Hallucinated"],
        digits=4,
    ))
    return {"auc_roc": auc_roc, "f1_macro": f1_macro, "auc_pr": auc_pr}


# ──────────────────────────────────────────────────────────
# Train one BERT model for one task
# ──────────────────────────────────────────────────────────
def train_one(
    model_key:     str,
    task:          str,
    train_df:      pd.DataFrame,
    dev_df:        pd.DataFrame,
    output_dir:    str,
    max_length:    int   = 512,
    batch_size:    int   = 16,
    num_epochs:    int   = 5,
    learning_rate: float = 2e-5,
    warmup_ratio:  float = 0.1,
    weight_decay:  float = 0.01,
    seed:          int   = 42,
) -> dict:

    model_name = MODEL_REGISTRY[model_key]
    save_path  = os.path.join(output_dir, f"task{task}", model_key)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Task {task}  |  {model_key.upper()}  ({model_name})")
    print(f"  Device : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*65}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=2)

    train_ds = HallucinationDataset(train_df, tokenizer, max_length)
    dev_ds   = HallucinationDataset(dev_df,   tokenizer, max_length)

    args = TrainingArguments(
        output_dir                  = save_path,
        num_train_epochs            = num_epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        learning_rate               = learning_rate,
        warmup_ratio                = warmup_ratio,
        weight_decay                = weight_decay,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "auc_roc",
        greater_is_better           = True,
        logging_dir                 = os.path.join(save_path, "logs"),
        logging_steps               = 50,
        seed                        = seed,
        report_to                   = "none",
        fp16                        = torch.cuda.is_available(),
    )

    trainer = Trainer(
        model            = model,
        args             = args,
        train_dataset    = train_ds,
        eval_dataset     = dev_ds,
        processing_class = tokenizer,
        compute_metrics  = compute_metrics,
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # Evaluate best checkpoint on dev
    dev_results = evaluate_split(
        trainer, dev_ds, f"Dev — Task {task} [{model_key.upper()}]"
    )

    # Save best model
    best_path = os.path.join(save_path, "best_model")
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)
    print(f"  Saved → {best_path}")

    with open(os.path.join(save_path, "dev_results.json"), "w") as f:
        json.dump(dev_results, f, indent=2)

    return dev_results


# ──────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────
def load_table(path: str) -> pd.DataFrame:
    """Read a split, choosing the reader from the file's CONTENT (magic bytes),
    not its extension — so a mislabeled .csv/.xlsx still loads correctly."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Check --data_dir / path flags against the starter-kit layout."
        )
    with open(path, "rb") as fh:
        magic = fh.read(8)
    if magic[:4] == b"PK\x03\x04":            # .xlsx / .xlsm (a zip archive)
        return pd.read_excel(path)            # needs `openpyxl`
    if magic[:4] == b"\xd0\xcf\x11\xe0":      # legacy .xls (OLE2)
        return pd.read_excel(path)
    sep = "\t" if path.lower().endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)         # text → csv/tsv


def normalize_columns(df: pd.DataFrame, path: str = "") -> pd.DataFrame:
    """Map starter-kit headers to canonical names and require question/answer/label."""
    df = df.rename(columns={k: v for k, v in RAW_TO_CANON.items() if k in df.columns})
    required = {"question", "model_answer", "label"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns {missing} in {path}.\n"
            f"Columns found: {list(df.columns)}.\n"
            f"Expected the starter-kit headers "
            f"(Question / Generated_answer / Label) or canonical "
            f"(question / model_answer / label)."
        )
    df["label"] = df["label"].astype(int)
    return df


def load_split(path: str) -> pd.DataFrame:
    return normalize_columns(load_table(path), path)


def default_paths(data_dir: str, task: str):
    """Resolve default train/dev xlsx paths for a task from the layout."""
    folder = os.path.join(data_dir, f"task{task}_data")
    return (os.path.join(folder, f"task{task}_train.xlsx"),
            os.path.join(folder, f"task{task}_dev.xlsx"))


def resolve_task_paths(args, task: str, single_task: bool):
    d_train, d_dev = default_paths(args.data_dir, task)
    key            = task.replace(".", "_")               # "1.1" → "1_1"
    train_override = getattr(args, f"train_path_task{key}")
    dev_override   = getattr(args, f"dev_path_task{key}")
    if single_task:                                       # generic flags apply
        train_override = args.train_path or train_override
        dev_override   = args.dev_path   or dev_override
    return (train_override or d_train), (dev_override or d_dev)


# ──────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────
def print_summary(task: str, results: dict):
    print(f"\n{'='*58}")
    print(f"  FINAL SUMMARY — TASK {task} — DEV")
    print(f"{'='*58}")
    print(f"  {'Model':<12} {'AUC-ROC':>10} {'F1-Macro':>10} {'AUC-PR':>10}")
    print(f"  {'-'*46}")
    for key, r in results.items():
        print(
            f"  {key.upper():<12}"
            f"{r.get('auc_roc',  float('nan')):>10.4f}"
            f"{r.get('f1_macro', float('nan')):>10.4f}"
            f"{r.get('auc_pr',   float('nan')):>10.4f}"
        )
    print(f"{'='*58}")


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT models for hallucination detection")

    parser.add_argument("--task", nargs="+", choices=TASKS, required=True,
                        help="Which task(s) to train: 1.1, 1.2, or both")
    parser.add_argument("--data_dir", type=str, default=".",
                        help="Root dir containing task1.1_data/ and task1.2_data/")

    # Generic overrides — for a SINGLE task
    parser.add_argument("--train_path", type=str, default=None,
                        help="Train file (overrides --data_dir) when training one task")
    parser.add_argument("--dev_path",   type=str, default=None,
                        help="Dev file (overrides --data_dir) when training one task")

    # Task-specific overrides — for BOTH tasks
    parser.add_argument("--train_path_task1_1", type=str, default=None)
    parser.add_argument("--dev_path_task1_1",   type=str, default=None)
    parser.add_argument("--train_path_task1_2", type=str, default=None)
    parser.add_argument("--dev_path_task1_2",   type=str, default=None)

    parser.add_argument("--output_dir",    type=str,   default="./outputs")
    parser.add_argument("--models",        nargs="+",
                        choices=list(MODEL_REGISTRY.keys()) + ["all"],
                        default=["all"])
    parser.add_argument("--max_length",    type=int,   default=512)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--num_epochs",    type=int,   default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio",  type=float, default=0.1)
    parser.add_argument("--weight_decay",  type=float, default=0.01)
    parser.add_argument("--seed",          type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    bert_models = list(MODEL_REGISTRY.keys()) if "all" in args.models else args.models
    tasks       = args.task
    single_task = len(tasks) == 1

    # GPU check
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)\n")
    else:
        print("WARNING: No GPU detected. Training will be slow.\n")

    # ---- Pre-flight: resolve + load every selected split BEFORE any training,
    #      so a bad path/column in Task 1.2 fails fast instead of after Task 1.1.
    data = {}
    for task in tasks:
        train_path, dev_path = resolve_task_paths(args, task, single_task)
        train_df = load_split(train_path)
        dev_df   = load_split(dev_path)
        data[task] = (train_path, dev_path, train_df, dev_df)
        print(f"Task {task}")
        print(f"  Train : {train_path}  ({len(train_df)} rows)  "
              f"{train_df['label'].value_counts().to_dict()}")
        print(f"  Dev   : {dev_path}  ({len(dev_df)} rows)  "
              f"{dev_df['label'].value_counts().to_dict()}")
    print()

    # ---- Train
    all_results = {}
    for task in tasks:
        train_path, dev_path, train_df, dev_df = data[task]
        print(f"\n{'#'*65}\n  TASK {task}\n{'#'*65}")

        task_results = {}
        for model_key in bert_models:
            task_results[model_key] = train_one(
                model_key     = model_key,
                task          = task,
                train_df      = train_df,
                dev_df        = dev_df,
                output_dir    = args.output_dir,
                max_length    = args.max_length,
                batch_size    = args.batch_size,
                num_epochs    = args.num_epochs,
                learning_rate = args.learning_rate,
                warmup_ratio  = args.warmup_ratio,
                weight_decay  = args.weight_decay,
                seed          = args.seed,
            )

        print_summary(task, task_results)
        all_results[task] = task_results

    # Save all results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "all_dev_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved → {results_path}")


if __name__ == "__main__":
    main()

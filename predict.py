"""
predict.py
──────────
Run inference on a test/dev set using the best models saved by train.py.

If the file has labels → computes AUC-ROC, F1-Macro, AUC-PR + per-generator
breakdown. If no labels → saves predictions only.

Input files use the starter-kit long format (.xlsx) with columns:
    ID | Question | Gold Answer | Generator_model | Generated_answer | [Label]
mapped internally to: id / question / gold_answer / model_name / model_answer / label.
A pre-mapped .csv (question / model_answer / [label] / [model_name]) also works.

Saved model paths are expected at:
    outputs/task<1.1|1.2>/<bert_model>/best_model/

Usage
-----
# Task 1.1 test (no labels)
python predict.py --task 1.1 --test_path test_task1.1.xlsx

# Task 1.2 dev evaluation (has labels)
python predict.py --task 1.2 --test_path task1.2_data/task1.2_dev.xlsx --has_labels

# Specific model only
python predict.py --task 1.1 --test_path test.xlsx --models camelbert
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.special import softmax

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, classification_report,
)

# ──────────────────────────────────────────────────────────
# Registry / config
# ──────────────────────────────────────────────────────────
MODEL_KEYS = ["camelbert", "arbert", "marbert", "mbert"]
TASKS      = ["1.1", "1.2"]

# Map starter-kit .xlsx headers → canonical names (already-canonical passes through).
RAW_TO_CANON = {
    "Question":         "question",
    "Generated_answer": "model_answer",
    "Label":            "label",
    "Generator_model":  "model_name",
    "Gold Answer":      "gold_answer",
    "ID":               "id",
}

# ──────────────────────────────────────────────────────────
# Data loading / normalization
# ──────────────────────────────────────────────────────────
def load_table(path: str) -> pd.DataFrame:
    """Choose reader from file CONTENT (magic bytes), not extension."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test file not found: {path}")
    with open(path, "rb") as fh:
        magic = fh.read(8)
    if magic[:4] == b"PK\x03\x04":            # .xlsx / .xlsm (zip)
        return pd.read_excel(path)            # needs `openpyxl`
    if magic[:4] == b"\xd0\xcf\x11\xe0":      # legacy .xls (OLE2)
        return pd.read_excel(path)
    sep = "\t" if path.lower().endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


def normalize_columns(df: pd.DataFrame, path: str = "") -> pd.DataFrame:
    """Require question + model_answer; label/model_name are optional."""
    df = df.rename(columns={k: v for k, v in RAW_TO_CANON.items() if k in df.columns})
    required = {"question", "model_answer"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns {missing} in {path}.\n"
            f"Columns found: {list(df.columns)}.\n"
            f"Expected starter-kit headers (Question / Generated_answer) "
            f"or canonical (question / model_answer)."
        )
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    return df


# ──────────────────────────────────────────────────────────
# Dataset (no label required)
# ──────────────────────────────────────────────────────────
class InferenceDataset(Dataset):
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
        return item


# ──────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────
def run_inference(df: pd.DataFrame, model_dir: str,
                  batch_size: int = 32, max_length: int = 512) -> np.ndarray:

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    loader = DataLoader(
        InferenceDataset(df, tokenizer, max_length),
        batch_size=batch_size, shuffle=False,
    )

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            ttype = batch.get("token_type_ids")
            if ttype is not None:
                ttype = ttype.to(device)
            logits = model(input_ids=ids, attention_mask=mask,
                           token_type_ids=ttype).logits
            probs  = softmax(logits.cpu().numpy(), axis=-1)[:, 1]
            all_probs.extend(probs.tolist())

    return np.array(all_probs)


# ──────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────
def evaluate(df: pd.DataFrame, model_key: str, task: str) -> dict:
    labels = df["label"].values
    probs  = df["prob_hallucinated"].values
    preds  = df["predicted_label"].values

    auc_roc  = roc_auc_score(labels, probs)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    auc_pr   = average_precision_score(labels, probs)

    print(f"\n{'─'*58}")
    print(f"  Overall — Task {task} — {model_key.upper()}")
    print(f"{'─'*58}")
    print(f"  AUC-ROC  : {auc_roc:.4f}  ← main metric")
    print(f"  F1-Macro : {f1_macro:.4f}")
    print(f"  AUC-PR   : {auc_pr:.4f}")
    print(classification_report(
        labels, preds,
        target_names=["Not Hallucinated", "Hallucinated"],
        digits=4,
    ))

    # Per-generator breakdown (only if model_name present and has >1 class per group)
    if "model_name" in df.columns:
        print(f"  {'Generator model':<28} {'AUC-ROC':>10} {'F1-Macro':>10} {'AUC-PR':>10}")
        print(f"  {'-'*60}")
        for name, grp in df.groupby("model_name"):
            y = grp["label"].values
            p = grp["prob_hallucinated"].values
            d = grp["predicted_label"].values
            r_auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
            r_apr = average_precision_score(y, p) if len(np.unique(y)) > 1 else float("nan")
            r_f1  = f1_score(y, d, average="macro", zero_division=0)
            print(f"  {str(name):<28} {r_auc:>10.4f} {r_f1:>10.4f} {r_apr:>10.4f}")

    return {"auc_roc": auc_roc, "f1_macro": f1_macro, "auc_pr": auc_pr}


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict hallucination labels using saved best models"
    )
    parser.add_argument("--task",       type=str, choices=TASKS, required=True)
    parser.add_argument("--test_path",  type=str, required=True,
                        help="Test/dev file (.xlsx starter-kit format or .csv)")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Root dir where task<X>/<model>/best_model/ folders live")
    parser.add_argument("--models",     nargs="+",
                        choices=MODEL_KEYS + ["all"], default=["all"])
    parser.add_argument("--has_labels", action="store_true",
                        help="Force evaluation even if label detection is unsure")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()

    bert_models = MODEL_KEYS if "all" in args.models else args.models

    # Load + normalize test file
    print(f"\nLoading test file : {args.test_path}")
    test_df    = normalize_columns(load_table(args.test_path), args.test_path)
    has_labels = ("label" in test_df.columns) or args.has_labels
    print(f"  Rows           : {len(test_df)}")
    print(f"  Labels present : {has_labels}")
    if args.has_labels and "label" not in test_df.columns:
        raise ValueError("--has_labels set but no Label/label column found.")

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Device         : {device_name}")

    all_results = {}
    for model_key in bert_models:
        model_dir = os.path.join(args.output_dir, f"task{args.task}", model_key, "best_model")
        if not os.path.exists(model_dir):
            print(f"\n  [SKIP] {model_key.upper()} — model not found at {model_dir}")
            continue

        print(f"\n  Running {model_key.upper()} ...")
        probs = run_inference(test_df, model_dir, args.batch_size, args.max_length)
        preds = (probs >= 0.5).astype(int)

        result_df = test_df.copy()
        result_df["prob_hallucinated"] = probs
        result_df["predicted_label"]   = preds

        # Save per-model predictions
        out_dir = os.path.join(args.output_dir, f"task{args.task}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"predictions_{model_key}.csv")
        result_df.to_csv(out_path, index=False)
        print(f"  Predictions saved → {out_path}")

        if has_labels and "label" in result_df.columns:
            all_results[model_key] = evaluate(result_df, model_key, args.task)

    # Final comparison table (if labels available)
    if all_results:
        print(f"\n{'='*58}")
        print(f"  FINAL SUMMARY — TASK {args.task} — TEST")
        print(f"{'='*58}")
        print(f"  {'Model':<12} {'AUC-ROC':>10} {'F1-Macro':>10} {'AUC-PR':>10}")
        print(f"  {'-'*46}")
        for key, r in all_results.items():
            print(
                f"  {key.upper():<12}"
                f"{r.get('auc_roc',  float('nan')):>10.4f}"
                f"{r.get('f1_macro', float('nan')):>10.4f}"
                f"{r.get('auc_pr',   float('nan')):>10.4f}"
            )
        print(f"{'='*58}")

        out_dir = os.path.join(args.output_dir, f"task{args.task}")
        os.makedirs(out_dir, exist_ok=True)
        results_path = os.path.join(out_dir, "test_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()

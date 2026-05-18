"""
predict.py
──────────
Run inference on a test set using saved best models from train.py.

If the test file has labels → computes AUC-ROC, F1-Macro, AUC-PR + per-model breakdown.
If no labels              → saves predictions only.

Accepts both:
  - Wide XLSX  (auto-converts to long internally, no preprocess.py needed)
  - Long CSV   (already preprocessed)

Saved model paths are expected as:
    outputs/task<N>/<bert_model>/best_model/

Usage
-----
# Task 1 test (wide XLSX, no labels)
python predict.py --task 1 --test_path test.xlsx

# Task 2 test (wide XLSX, no labels)
python predict.py --task 2 --test_path test_task2.xlsx

# Task 2 dev evaluation (wide XLSX, has labels)
python predict.py --task 2 --test_path dev_task2.xlsx --has_labels

# Specific model only
python predict.py --task 1 --test_path test.xlsx --models camelbert
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
# Model registry
# ──────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "camelbert": "camelbert",
    "arbert":    "arbert",
    "marbert":   "marbert",
    "mbert":     "mbert",
}

LABEL_SUFFIX = " Factual Hallucination"
SKIP_COLS    = {"Question", "Answer", "ID"}

# ──────────────────────────────────────────────────────────
# Wide → long conversion
# ──────────────────────────────────────────────────────────
def detect_model_columns(df: pd.DataFrame) -> dict:
    cols = {}
    for col in df.columns:
        if col in SKIP_COLS or col.endswith(LABEL_SUFFIX):
            continue
        label_col = col + LABEL_SUFFIX
        # label col may or may not exist (unseen test)
        cols[col] = (col, label_col if label_col in df.columns else None)
    if not cols:
        raise ValueError("No model columns detected in the input file.")
    return cols


def wide_to_long(df: pd.DataFrame) -> tuple:
    """Returns (long_df, has_labels)."""
    model_cols = detect_model_columns(df)
    has_labels = all(lbl is not None for _, lbl in model_cols.values())
    print(f"  Detected models : {list(model_cols.keys())}")
    print(f"  Labels present  : {has_labels}")

    rows = []
    for _, row in df.iterrows():
        for model_name, (ans_col, lbl_col) in model_cols.items():
            entry = {
                "question":     row["Question"],
                "model_answer": row[ans_col],
                "model_name":   model_name,
            }
            if lbl_col:
                entry["label"] = int(row[lbl_col])
            rows.append(entry)
    return pd.DataFrame(rows), has_labels


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
def evaluate(df: pd.DataFrame, model_key: str, task: str):
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

    # Per-model breakdown
    print(f"  {'Model':<22} {'AUC-ROC':>10} {'F1-Macro':>10} {'AUC-PR':>10}")
    print(f"  {'-'*54}")
    for name, grp in df.groupby("model_name"):
        r_auc  = roc_auc_score(grp["label"], grp["prob_hallucinated"])
        r_f1   = f1_score(grp["label"], grp["predicted_label"], average="macro", zero_division=0)
        r_apr  = average_precision_score(grp["label"], grp["prob_hallucinated"])
        print(f"  {name:<22} {r_auc:>10.4f} {r_f1:>10.4f} {r_apr:>10.4f}")

    return {"auc_roc": auc_roc, "f1_macro": f1_macro, "auc_pr": auc_pr}


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict hallucination labels using saved best models"
    )
    parser.add_argument("--task",       type=str, choices=["1", "2"], required=True)
    parser.add_argument("--test_path",  type=str, required=True,
                        help="Wide XLSX or long CSV test file")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Root dir where task<N>/<model>/best_model/ folders live")
    parser.add_argument("--models",     nargs="+",
                        choices=list(MODEL_REGISTRY.keys()) + ["all"],
                        default=["all"])
    parser.add_argument("--has_labels", action="store_true",
                        help="Set if the test file has label columns (for evaluation)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()

    bert_models = list(MODEL_REGISTRY.keys()) if "all" in args.models else args.models

    # Load test file
    print(f"\nLoading test file : {args.test_path}")
    raw = pd.read_excel(args.test_path) if args.test_path.endswith(".xlsx") \
          else pd.read_csv(args.test_path)

    # Convert to long if wide
    if "model_answer" in raw.columns:
        test_df    = raw.copy()
        has_labels = "label" in test_df.columns
        print(f"  Format         : long ({len(test_df)} rows)")
        print(f"  Labels present : {has_labels}")
    else:
        print(f"  Format         : wide ({len(raw)} rows) — converting...")
        test_df, has_labels = wide_to_long(raw)

    if args.has_labels:
        has_labels = True

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

        # Save predictions
        out_path = os.path.join(
            args.output_dir, f"task{args.task}",
            f"predictions_{model_key}.csv"
        )
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

        results_path = os.path.join(args.output_dir, f"task{args.task}", "test_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()

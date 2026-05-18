"""
preprocess.py
─────────────
Convert HalluScoring wide-format files (CSV or XLSX) into long format.

Auto-detects model columns: any column that has a matching
"<col> Factual Hallucination" column is treated as a model answer column.

Output columns: question, model_answer, label, model_name

Usage
-----
# Single file
python preprocess.py --input train.csv            --output train_long.csv
python preprocess.py --input dev.xlsx             --output dev_long.csv
python preprocess.py --input dev_task2.xlsx       --output dev_task2_long.csv
python preprocess.py --input test_task2.xlsx      --output test_task2_long.csv

# Multiple files at once
python preprocess.py \
    --input train.csv           --output train_long.csv \
    --input dev.xlsx            --output dev_long.csv \
    --input dev_task2.xlsx      --output dev_task2_long.csv
"""

import argparse
import pandas as pd


LABEL_SUFFIX = " Factual Hallucination"
SKIP_COLS    = {"Question", "Answer", "ID"}


def detect_model_columns(df: pd.DataFrame) -> dict:
    """Return {model_name: (answer_col, label_col)} for all detected models."""
    cols = {}
    for col in df.columns:
        if col in SKIP_COLS or col.endswith(LABEL_SUFFIX):
            continue
        label_col = col + LABEL_SUFFIX
        if label_col in df.columns:
            cols[col] = (col, label_col)
    if not cols:
        raise ValueError(
            "No model columns detected. Every model column needs a matching "
            f"'<Model>{LABEL_SUFFIX}' column."
        )
    return cols


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    model_cols = detect_model_columns(df)
    print(f"    Detected models : {list(model_cols.keys())}")

    rows = []
    for _, row in df.iterrows():
        for model_name, (ans_col, lbl_col) in model_cols.items():
            rows.append({
                "question":     row["Question"],
                "model_answer": row[ans_col],
                "label":        int(row[lbl_col]),
                "model_name":   model_name,
            })
    return pd.DataFrame(rows)


def load_file(path: str) -> pd.DataFrame:
    return pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)


def process_file(in_path: str, out_path: str):
    print(f"\n  {in_path}  →  {out_path}")
    df      = load_file(in_path)
    long_df = wide_to_long(df)
    long_df.to_csv(out_path, index=False)
    print(f"    Rows       : {len(df)} wide  →  {len(long_df)} long")
    print(f"    Label dist :\n{long_df['label'].value_counts().to_string()}")
    print(f"    Per model  :\n{long_df['model_name'].value_counts().to_string()}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert wide HalluScoring files to long format"
    )
    parser.add_argument("--input",  action="append", dest="inputs",  metavar="PATH",
                        help="Input wide file (CSV or XLSX). Repeatable.")
    parser.add_argument("--output", action="append", dest="outputs", metavar="PATH",
                        help="Output long CSV. Must match number of --input flags.")
    args = parser.parse_args()

    if not args.inputs:
        parser.error("Provide at least one --input/--output pair.")
    if len(args.inputs) != len(args.outputs):
        parser.error("Number of --input and --output flags must match.")

    print(f"Processing {len(args.inputs)} file(s)...")
    for in_path, out_path in zip(args.inputs, args.outputs):
        process_file(in_path, out_path)
    print("\nDone.")


if __name__ == "__main__":
    main()

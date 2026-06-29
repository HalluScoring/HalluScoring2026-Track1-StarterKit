#!/usr/bin/env python3
"""
Extract ID and Label columns from _dev.xlsx files and save to CSV files.
Processes task1.1_dev.xlsx and task1.2_dev.xlsx, maintaining row order.
"""

import pandas as pd
from pathlib import Path

# Define input and output paths
base_path = Path(__file__).parent
tasks = {
    "task1.1": base_path / "raw_data" / "task1.1_dev.xlsx",
    "task1.2": base_path / "raw_data" / "task1.2_dev.xlsx",
}

output_dir = base_path / "labels"
output_dir.mkdir(exist_ok=True)

# Process each task
for task_name, input_file in tasks.items():
    if not input_file.exists():
        print(f"⚠️  File not found: {input_file}")
        continue

    # Read the Excel file
    df = pd.read_excel(input_file)

    # Create output with row numbers as ID and Label from original data
    output_df = pd.DataFrame({
        "ID": range(len(df)),
        "Label": df["Label"]
    })

    # Save to CSV
    output_file = output_dir / f"{task_name}_labels.csv"
    output_df.to_csv(output_file, index=False)

    print(f"✓ {task_name}: Extracted {len(output_df)} rows → {output_file}")

print("\nDone!")

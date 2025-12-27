# What changed:
# - Added GPU detection/device prompt and chunk size prompt with row estimation.
# - Standardized outputs under ./outputs/Check_which_column_has_Mixed_Types.
# - Added a saved report and final summary.
#
# Purpose:
# - Identify columns with mixed data types, especially string/inf issues.
# - Stream-process large CSVs in chunks.
# - Save a report of problematic columns.

import os
import pandas as pd
from collections import Counter, defaultdict
import math

# CSV file path
CSV_FILE = "../2017/final_testing_1.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Check_which_column_has_Mixed_Types")


def detect_gpu():
    gpu_available = False
    library = None
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            gpu_available = True
            library = "pytorch"
    except Exception:
        pass

    if not gpu_available:
        try:
            import tensorflow as tf  # type: ignore
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                gpu_available = True
                library = "tensorflow"
        except Exception:
            pass

    if gpu_available:
        print("GPU detected.")
    else:
        print("GPU not detected. Using CPU.")
    return gpu_available, library


def prompt_for_device(gpu_available):
    if gpu_available:
        while True:
            response = input("GPU detected. Use GPU? (y/n): ").lower().strip()
            if response in ["y", "yes"]:
                return "gpu"
            if response in ["n", "no"]:
                return "cpu"
            print("Invalid input. Please enter 'y' or 'n'.")
    return "cpu"


def prompt_for_chunk_size_mb():
    choices = {"25": 25, "100": 100, "500": 500, "1000": 1000}
    while True:
        response = input("Choose chunk size in MB (25/100/500/1000): ").strip()
        if response in choices:
            return choices[response]
        print("Invalid choice. Please enter 25, 100, 500, or 1000.")


def estimate_rows_per_chunk(file_path, chunk_mb, sample_rows=2000, default_rows=100_000):
    target_bytes = int(chunk_mb) * 1024 * 1024
    try:
        sample = pd.read_csv(file_path, nrows=sample_rows, low_memory=True)
        if sample is None or sample.empty:
            return int(default_rows)
        bytes_per_row = float(sample.memory_usage(deep=True).sum()) / float(max(1, len(sample)))
        if bytes_per_row <= 0:
            return int(default_rows)
        est = int(target_bytes / bytes_per_row)
        return max(10_000, min(2_000_000, est))
    except Exception:
        return int(default_rows)


def make_unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    counter = 2
    while True:
        candidate = f"{base}_run{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


# Helper function to classify a value's data type

def classify_value(val):
    if pd.isna(val):
        return "NaN"
    try:
        num = float(val)
        if math.isinf(num):
            return "inf"
        if num.is_integer():
            return "integer"
        return "float"
    except (ValueError, TypeError):
        return "string"


def main():
    gpu_available, _ = detect_gpu()
    device_choice = prompt_for_device(gpu_available)
    if device_choice == "gpu":
        print("GPU selected, but this script uses CPU-based pandas. Using CPU.")
    device_used = "cpu"

    if not os.path.exists(CSV_FILE):
        print(f"ERROR: File not found: {CSV_FILE}")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    chunk_mb = prompt_for_chunk_size_mb()
    chunk_rows = estimate_rows_per_chunk(CSV_FILE, chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

    col_type_counts = defaultdict(Counter)
    print("Starting to process the CSV file to find critical errors...")

    total_rows_processed = 0
    for i, chunk in enumerate(pd.read_csv(CSV_FILE, chunksize=chunk_rows, low_memory=False, dtype=str)):
        total_rows_processed += len(chunk)
        print(f"Processing chunk {i + 1} ({len(chunk):,} rows)...")
        for col in chunk.columns:
            col_type_counts[col].update(chunk[col].map(classify_value))

    print("\n--- Analysis Complete ---")

    report_lines = []
    report_lines.append("Summary of Columns with Critical Errors:")
    report_lines.append("")

    for col, counts in col_type_counts.items():
        if 'string' in counts or 'inf' in counts:
            report_lines.append(f"Column: {col} (CRITICAL - mixed types found)")
            for val_type, cnt in counts.items():
                report_lines.append(f"  {val_type} --- {cnt}")
            report_lines.append("-" * 40)

    print("\n".join(report_lines))

    report_path = make_unique_path(os.path.join(OUTPUT_FOLDER, "mixed_type_report.txt"))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved to: {report_path}")

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Device used: {device_used.upper()}")
    print(f"Chunk size: {chunk_mb}MB (~{chunk_rows:,} rows)")
    print(f"Total rows processed: {total_rows_processed:,}")
    print("Rows saved: N/A")
    print("Output paths:")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()

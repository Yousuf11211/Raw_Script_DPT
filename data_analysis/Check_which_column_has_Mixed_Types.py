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
import sys
import argparse

# Allow running this script from any working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
from collections import Counter, defaultdict
import math

from config.global_config import DEFAULT_CHUNK_SIZE_MB
from utils.chunk_utils import compute_chunk_plan, format_progress, print_chunk_plan
from utils.engine_utils import select_engine
from utils.path_utils import resolve_input_path, resolve_output_path


# CSV file path
CSV_FILE = "../2017/final_testing_1.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Check_which_column_has_Mixed_Types")

_NO_INTERACTIVE = False


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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Identify columns with mixed data types (streaming-safe).")
    p.add_argument("--input", default=CSV_FILE, help="Input CSV path")
    p.add_argument("--output-dir", default=None, help="Base output directory")
    p.add_argument("--chunk-size-mb", type=int, default=DEFAULT_CHUNK_SIZE_MB, help="Chunk size in MB")
    p.add_argument("--engine", default="pandas", choices=["pandas", "dask", "dask-gpu"], help="Execution engine")
    p.add_argument("--use-gpu", action="store_true", help="Force GPU (or fail)")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU")
    p.add_argument("--no-interactive", action="store_true", help="Disable interactive prompts")
    return p


# Replace main() with CLI-aware main(argv)

def main(argv: list[str] | None = None):
    global _NO_INTERACTIVE
    args = build_arg_parser().parse_args(argv)
    _NO_INTERACTIVE = args.no_interactive

    selection = select_engine(engine=args.engine, use_gpu_flag=args.use_gpu, no_gpu_flag=args.no_gpu)
    if selection.engine != "pandas":
        print(f"[info] --engine {selection.engine} requested; this script currently runs in pandas mode.")
    if selection.use_gpu:
        print("[info] GPU was approved, but this script uses CPU-based pandas. Using CPU.")
    device_used = "cpu"

    input_csv = resolve_input_path(args.input)
    base_output_dir = resolve_output_path(args.output_dir)
    output_folder = os.path.join(base_output_dir, "Check_which_column_has_Mixed_Types")

    if not os.path.exists(input_csv):
        print(f"ERROR: File not found: {input_csv}")
        return

    os.makedirs(output_folder, exist_ok=True)

    chunk_mb = int(args.chunk_size_mb)
    plan0 = compute_chunk_plan(input_csv, chunk_mb)
    print_chunk_plan(plan0)

    chunk_rows = estimate_rows_per_chunk(input_csv, chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

    col_type_counts = defaultdict(Counter)
    print("Starting to process the CSV file to find critical errors...")

    total_rows_processed = 0
    for chunk_idx, chunk in enumerate(pd.read_csv(input_csv, chunksize=chunk_rows, low_memory=False, dtype=str), 1):
        total_rows_processed += len(chunk)
        print(format_progress(chunk_idx, plan0.total_chunks))
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

    report_path = make_unique_path(os.path.join(output_folder, "mixed_type_report.txt"))
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

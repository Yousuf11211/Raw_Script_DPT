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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Check_which_column_has_Mixed_Types")

# Global input config for easy editing (must be before any function declaration)
GLOBAL_INPUT = {
    'folder': 'Bening',  # Default input folder (relative to project root or absolute)
    'output_dir': None,  # Default output directory (None means use default)
    'chunk_size_mb': DEFAULT_CHUNK_SIZE_MB,  # Default chunk size in MB
    'engine': 'pandas',  # Default engine
    'use_gpu': False,    # Force GPU
    'no_gpu': False,     # Force CPU
    'no_interactive': False,  # Disable interactive prompts
}

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
    p.add_argument("--folder", default=GLOBAL_INPUT['folder'], help="Input folder containing CSV files (relative to project root or absolute)")
    p.add_argument("--output-dir", default=GLOBAL_INPUT['output_dir'], help="Base output directory")
    p.add_argument("--chunk-size-mb", type=int, default=GLOBAL_INPUT['chunk_size_mb'], help="Chunk size in MB")
    p.add_argument("--engine", default=GLOBAL_INPUT['engine'], choices=["pandas", "dask", "dask-gpu"], help="Execution engine")
    p.add_argument("--use-gpu", action="store_true", default=GLOBAL_INPUT['use_gpu'], help="Force GPU (or fail)")
    p.add_argument("--no-gpu", action="store_true", default=GLOBAL_INPUT['no_gpu'], help="Force CPU")
    p.add_argument("--no-interactive", action="store_true", default=GLOBAL_INPUT['no_interactive'], help="Disable interactive prompts")
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

    # Resolve folder relative to project root if not absolute
    folder = args.folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if not os.path.isabs(folder):
        folder = os.path.join(project_root, folder)
    if not os.path.isdir(folder):
        print(f"ERROR: Folder not found: {folder}")
        return

    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in folder: {folder}")
        return

    print("\n--- CSV Files Found ---")
    for i, fname in enumerate(csv_files, 1):
        print(f"  {i}: {fname}")
    print("-----------------------")
    # Prompt user to select file
    while True:
        file_choice = input(f"Enter the number of the file to process (1-{len(csv_files)}): ").strip()
        if file_choice.isdigit() and 1 <= int(file_choice) <= len(csv_files):
            selected_file = csv_files[int(file_choice)-1]
            break
        print("Invalid selection. Please enter a valid number.")
    input_csv = os.path.join(folder, selected_file)

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

    # Identify columns (except label) with string values
    string_cols = [col for col, counts in col_type_counts.items() if col.lower() != 'label' and 'string' in counts]
    cleaned_file_path = None
    if string_cols:
        print(f"\nColumns with string values (except label): {string_cols}")
        remove_rows = input("Rows with string values found in columns (except label). Remove those rows and save a cleaned CSV? (y/n): ").strip().lower()
        if remove_rows in ('y', 'yes'):
            cleaned_file_path = make_unique_path(os.path.join(output_folder, f"{os.path.splitext(selected_file)[0]}_no_strings.csv"))
            print(f"\nSaving cleaned file without rows containing string values in columns: {string_cols}\nThis may take a while...")
            is_first_chunk = True
            rows_written = 0
            for chunk in pd.read_csv(input_csv, chunksize=chunk_rows, low_memory=False, dtype=str):
                # For each row, check if any string in any of the string_cols
                mask = chunk[string_cols].applymap(classify_value).ne('string').all(axis=1)
                cleaned_chunk = chunk[mask]
                if not cleaned_chunk.empty:
                    cleaned_chunk.to_csv(cleaned_file_path, index=False, mode='w' if is_first_chunk else 'a', header=is_first_chunk)
                    is_first_chunk = False
                    rows_written += len(cleaned_chunk)
            print(f"Cleaned file saved to: {cleaned_file_path} (rows saved: {rows_written})")

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Device used: {device_used.upper()}")
    print(f"Chunk size: {chunk_mb}MB (~{chunk_rows:,} rows)")
    print(f"Total rows processed: {total_rows_processed:,}")
    print("Rows saved: N/A" if not cleaned_file_path else f"Rows saved: see cleaned file")
    print("Output paths:")
    print(f"  - {report_path}")
    if cleaned_file_path:
        print(f"  - {cleaned_file_path}")


if __name__ == "__main__":
    main()

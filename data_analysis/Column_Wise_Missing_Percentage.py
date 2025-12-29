# What changed:
# - Added GPU detection/device prompt, chunk size prompt with row estimation, and streaming analysis.
# - Standardized outputs under ./outputs/Column_Wise_Missing_Percentage.
# - Added final summary with output path.
# - Added CLI args, engine flags, chunk plan + progress for repo consistency.
#
# Purpose:
# - Report missing and infinite values per column across CSV files.
# - Optionally save per-file reports to disk.

import os
import sys
import argparse

# Allow running this script from any working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import numpy as np

from config.global_config import DEFAULT_CHUNK_SIZE_MB
from utils.chunk_utils import compute_chunk_plan, format_progress, print_chunk_plan
from utils.engine_utils import select_engine
from utils.path_utils import resolve_input_path, resolve_output_path

# Main folder with all raw datasets
MAIN_FOLDER = "Attacks_Removed_Constant"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Column_Wise_Missing_Percentage")

# Global flag for non-interactive mode
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


def estimate_rows_per_chunk(file_path, chunk_mb, sample_rows=2000, default_rows=500_000):
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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Report missing and infinite values per column across CSV files."
    )
    p.add_argument("--input", default=MAIN_FOLDER, help="Input folder (abs or repo-root-relative)")
    p.add_argument("--output-dir", default=None, help="Base output directory")
    p.add_argument("--chunk-size-mb", type=int, default=DEFAULT_CHUNK_SIZE_MB, help="Chunk size in MB")
    p.add_argument("--engine", default="pandas", choices=["pandas", "dask", "dask-gpu"], help="Execution engine")
    p.add_argument("--use-gpu", action="store_true", help="Force GPU (or fail)")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU")
    p.add_argument("--no-interactive", action="store_true", help="Disable interactive prompts")
    p.add_argument("--save-reports", action="store_true", help="Save reports to files (non-interactive)")
    return p


def main(argv: list[str] | None = None):
    global _NO_INTERACTIVE
    args = build_arg_parser().parse_args(argv)
    _NO_INTERACTIVE = args.no_interactive

    # Engine selection
    selection = select_engine(engine=args.engine, use_gpu_flag=args.use_gpu, no_gpu_flag=args.no_gpu)
    if selection.engine != "pandas":
        print(f"[info] --engine {selection.engine} requested; this script currently runs in pandas mode.")
    device_used = "cpu"

    # Resolve paths
    input_folder = resolve_input_path(args.input)
    base_output_dir = resolve_output_path(args.output_dir)
    output_folder = os.path.join(base_output_dir, "Column_Wise_Missing_Percentage")

    if not os.path.isdir(input_folder):
        print(f"ERROR: Folder not found: {input_folder}")
        return

    # Interactive or CLI save_reports
    if _NO_INTERACTIVE:
        save_reports = args.save_reports
    else:
        gpu_available, _ = detect_gpu()
        device_choice = prompt_for_device(gpu_available)
        if device_choice == "gpu":
            print("GPU selected, but this script uses CPU-based pandas. Using CPU.")
        save_reports = input("Do you want to save reports to files? (y/n): ").strip().lower() == "y"

    os.makedirs(output_folder, exist_ok=True)

    csv_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print("No CSV files found.")
        return

    chunk_mb = int(args.chunk_size_mb)

    # Mandatory chunk plan
    plan0 = compute_chunk_plan(csv_files[0], chunk_mb)
    print_chunk_plan(plan0)

    chunk_rows = estimate_rows_per_chunk(csv_files[0], chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

    total_rows_processed = 0
    output_paths = []

    for file_path in csv_files:
        print(f"\nScanning {file_path} ...")

        # Per-file chunk plan
        file_plan = compute_chunk_plan(file_path, chunk_mb)
        print_chunk_plan(file_plan)

        missing_counts = None
        inf_counts = None
        total_rows = 0
        total_cols = 0

        try:
            for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_rows, low_memory=False), 1):
                # Standard progress
                print(format_progress(chunk_idx, file_plan.total_chunks))

                total_rows += len(chunk)
                total_cols = len(chunk.columns)
                if missing_counts is None:
                    missing_counts = chunk.isna().sum()
                    inf_counts = chunk.isin([np.inf, -np.inf]).sum()
                else:
                    missing_counts = missing_counts.add(chunk.isna().sum(), fill_value=0)
                    inf_counts = inf_counts.add(chunk.isin([np.inf, -np.inf]).sum(), fill_value=0)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        total_rows_processed += total_rows
        missing_counts = missing_counts if missing_counts is not None else pd.Series(dtype=int)
        inf_counts = inf_counts if inf_counts is not None else pd.Series(dtype=int)

        missing_perc = (missing_counts / max(1, total_rows) * 100).round(2)
        missing_report = pd.DataFrame({
            "Missing Count": missing_counts,
            "Missing %": missing_perc
        })
        missing_report = missing_report[missing_report["Missing Count"] > 0]

        inf_report = inf_counts[inf_counts > 0]

        report_lines = []
        report_lines.append(f"Report for {file_path}")
        report_lines.append("=" * 50)
        report_lines.append(f"Total rows: {total_rows}")
        report_lines.append(f"Total columns: {total_cols}")
        report_lines.append("")

        if missing_report.empty:
            report_lines.append("No missing values found in any column.")
        else:
            report_lines.append("Columns with missing values:")
            for col, row in missing_report.iterrows():
                report_lines.append(f"{col:<25}: {row['Missing Count']} missing ({row['Missing %']}%)")
        report_lines.append("")

        if inf_report.empty:
            report_lines.append("No infinite values found in any column.")
        else:
            report_lines.append("Columns with infinite values:")
            for col, cnt in inf_report.items():
                report_lines.append(f"{col:<25}: {cnt} infinite values")
        report_lines.append("")

        print("\n".join(report_lines))

        if save_reports:
            rel_path = os.path.relpath(os.path.dirname(file_path), input_folder)
            report_subfolder = os.path.join(output_folder, rel_path)
            os.makedirs(report_subfolder, exist_ok=True)

            output_file = make_unique_path(
                os.path.join(report_subfolder, f"{os.path.splitext(os.path.basename(file_path))[0]}_report.txt")
            )
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            output_paths.append(output_file)
            print(f"Report saved to {output_file}")

    if not save_reports:
        output_paths.append("(reports not saved)")

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Device used: {device_used.upper()}")
    print(f"Chunk size: {chunk_mb}MB (~{chunk_rows:,} rows)")
    print(f"Total rows processed: {total_rows_processed:,}")
    print("Rows saved: N/A")
    print("Output paths:")
    for path in output_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()

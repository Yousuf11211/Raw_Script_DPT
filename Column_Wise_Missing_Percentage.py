# What changed:
# - Added GPU detection/device prompt, chunk size prompt with row estimation, and streaming analysis.
# - Standardized outputs under ./outputs/Column_Wise_Missing_Percentage.
# - Added final summary with output path.
#
# Purpose:
# - Report missing and infinite values per column across CSV files.
# - Optionally save per-file reports to disk.

import os
import pandas as pd
import numpy as np

# Main folder with all raw datasets
MAIN_FOLDER = "Attacks_Removed_Constant"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Column_Wise_Missing_Percentage")


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


def main():
    gpu_available, _ = detect_gpu()
    device_choice = prompt_for_device(gpu_available)
    if device_choice == "gpu":
        print("GPU selected, but this script uses CPU-based pandas. Using CPU.")
    device_used = "cpu"

    if not os.path.isdir(MAIN_FOLDER):
        print(f"ERROR: Folder not found: {MAIN_FOLDER}")
        return

    save_reports = input("Do you want to save reports to files? (y/n): ").strip().lower() == "y"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    csv_files = []
    for root, _, files in os.walk(MAIN_FOLDER):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print("No CSV files found.")
        return

    chunk_mb = prompt_for_chunk_size_mb()
    chunk_rows = estimate_rows_per_chunk(csv_files[0], chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

    total_rows_processed = 0
    output_paths = []

    for file_path in csv_files:
        print(f"\nScanning {file_path} ...")

        missing_counts = None
        inf_counts = None
        total_rows = 0
        total_cols = 0

        try:
            for chunk in pd.read_csv(file_path, chunksize=chunk_rows, low_memory=False):
                total_rows += len(chunk)
                total_cols = len(chunk.columns)
                if missing_counts is None:
                    missing_counts = chunk.isna().sum()
                    inf_counts = chunk.isin([np.inf, -np.inf]).sum()
                else:
                    missing_counts = missing_counts.add(chunk.isna().sum(), fill_value=0)
                    inf_counts = inf_counts.add(chunk.isin([np.inf, -np.inf]).sum(), fill_value=0)
                if total_rows % (chunk_rows * 5) == 0:
                    print(f"  Processed {total_rows:,} rows...")
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
            rel_path = os.path.relpath(os.path.dirname(file_path), MAIN_FOLDER)
            report_subfolder = os.path.join(OUTPUT_FOLDER, rel_path)
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

# What changed:
# - Added GPU detection/device prompt and chunk size prompt with row estimation.
# - Streamed row hashing to avoid loading full CSVs into memory.
# - Standardized outputs under ./outputs/Compare_Columns_and_Rows_2CSV with a final summary.
#
# Purpose:
# - Compare column consistency between raw and processed CSV folders.
# - Compare row sets between raw and processed data using streaming hashes.
# - Save a comparison report to disk.

import os
import pandas as pd

raw_folder = "Raw_Data_2017"
processed_folder = "Processed_Data_2017"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Compare_Columns_and_Rows_2CSV")


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

    if not os.path.isdir(raw_folder) or not os.path.isdir(processed_folder):
        print("ERROR: Raw or processed folder not found.")
        return

    raw_files = sorted([f for f in os.listdir(raw_folder) if f.endswith(".csv")])
    processed_files = sorted([f for f in os.listdir(processed_folder) if f.endswith(".csv")])
    if not raw_files:
        print("No raw CSV files found.")
        return

    chunk_mb = prompt_for_chunk_size_mb()
    chunk_rows = estimate_rows_per_chunk(os.path.join(raw_folder, raw_files[0]), chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    report_path = make_unique_path(os.path.join(OUTPUT_FOLDER, "comparison_report.txt"))
    report_lines = []

    first_raw_csv = raw_files[0]
    raw_columns = list(pd.read_csv(os.path.join(raw_folder, first_raw_csv), nrows=0).columns)
    print(f"Reference columns from raw CSV ({first_raw_csv}): {raw_columns}")
    report_lines.append(f"Reference columns from raw CSV ({first_raw_csv}): {raw_columns}")

    raw_mismatches = []
    for filename in raw_files:
        df = pd.read_csv(os.path.join(raw_folder, filename), nrows=0)
        if list(df.columns) != raw_columns:
            raw_mismatches.append(filename)
            print(f"Column mismatch in raw CSV: {filename}")
            report_lines.append(f"Column mismatch in raw CSV: {filename}")

    processed_mismatches = []
    for filename in processed_files:
        df = pd.read_csv(os.path.join(processed_folder, filename), nrows=0)
        if list(df.columns) != raw_columns:
            processed_mismatches.append(filename)
            print(f"Column mismatch in processed CSV: {filename}")
            report_lines.append(f"Column mismatch in processed CSV: {filename}")

    raw_hashes = set()
    processed_hashes = set()
    extra_rows_sample = []
    total_rows_processed = 0

    print("\nCollecting row hashes from raw data...")
    for filename in raw_files:
        path = os.path.join(raw_folder, filename)
        try:
            for chunk in pd.read_csv(path, chunksize=chunk_rows, dtype=str, usecols=raw_columns):
                total_rows_processed += len(chunk)
                row_hashes = pd.util.hash_pandas_object(chunk, index=False)
                raw_hashes.update(row_hashes.astype(int).tolist())
        except Exception as e:
            print(f"  Skipping {filename} due to error: {e}")
            report_lines.append(f"Skipping {filename} due to error: {e}")
            continue

    print("Collecting row hashes from processed data...")
    for filename in processed_files:
        path = os.path.join(processed_folder, filename)
        try:
            for chunk in pd.read_csv(path, chunksize=chunk_rows, dtype=str, usecols=raw_columns):
                total_rows_processed += len(chunk)
                row_hashes = pd.util.hash_pandas_object(chunk, index=False)
                processed_hashes.update(row_hashes.astype(int).tolist())
                if len(extra_rows_sample) < 10:
                    for idx, row_hash in enumerate(row_hashes.astype(int).tolist()):
                        if row_hash not in raw_hashes and len(extra_rows_sample) < 10:
                            extra_rows_sample.append(tuple(chunk.iloc[idx].tolist()))
        except Exception as e:
            print(f"  Skipping {filename} due to error: {e}")
            report_lines.append(f"Skipping {filename} due to error: {e}")
            continue

    missing_rows_count = len(raw_hashes - processed_hashes)
    extra_rows_count = len(processed_hashes - raw_hashes)

    report_lines.append(f"Total unique rows in raw data: {len(raw_hashes)}")
    report_lines.append(f"Total unique rows in processed data: {len(processed_hashes)}")
    report_lines.append(f"Missing rows count: {missing_rows_count}")
    report_lines.append(f"Extra rows count: {extra_rows_count}")

    print(f"Total rows in raw data (unique hashes): {len(raw_hashes)}")
    print(f"Total rows in processed data (unique hashes): {len(processed_hashes)}")

    if not missing_rows_count and not extra_rows_count:
        print("All raw data rows are present in the processed folder.")
        report_lines.append("All raw data rows are present in the processed folder.")
    else:
        if missing_rows_count:
            print(f"{missing_rows_count} rows from raw data are missing in processed CSVs.")
        if extra_rows_count:
            print(f"{extra_rows_count} extra rows found in processed CSVs that were not in raw data.")

    if extra_rows_sample:
        report_lines.append("\nSample extra rows (up to 10):")
        for i, row in enumerate(extra_rows_sample, 1):
            report_lines.append(f"Row {i}: {row}")

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

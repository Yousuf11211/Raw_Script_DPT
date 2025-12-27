# What changed:
# - Added GPU detection/device prompt and chunk size prompt with row estimation.
# - Streamed invalid handshake detection and optional removal.
# - Standardized outputs under ./outputs/Checks_Which_columns_need_encoding with final summary.
#
# Purpose:
# - Identify rows where both handshake fields are invalid strings.
# - Optionally remove those rows and save a cleaned CSV.
# - Report counts of valid/invalid rows by label.

import os
import pandas as pd
import numpy as np
from collections import Counter

# --- Configuration ---
INPUT_FOLDER = "Downscale_Csv_2018"
COLUMNS_TO_CHECK = ['delta_start', 'handshake_duration', 'label']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Checks_Which_columns_need_encoding")


# --- Helpers ---

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


def prompt_for_max_rows():
    while True:
        response = input("Limit rows to save? (y/n): ").strip().lower()
        if response in ["y", "yes"]:
            while True:
                value = input("Enter max rows: ").strip()
                try:
                    max_rows = int(value)
                    if max_rows > 0:
                        return max_rows
                except ValueError:
                    pass
                print("Please enter a positive integer.")
        elif response in ["n", "no"]:
            return None
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


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


def build_invalid_mask(df):
    delta_invalid = df['delta_start'].astype(str).str.lower().str.contains("not a complete handshake", na=False)
    handshake_invalid = df['handshake_duration'].astype(str).str.lower().str.contains("not a complete handshake", na=False)
    return delta_invalid & handshake_invalid


# --- Main ---

def main():
    gpu_available, _ = detect_gpu()
    device_choice = prompt_for_device(gpu_available)
    if device_choice == "gpu":
        print("GPU selected, but this script uses CPU-based pandas. Using CPU.")
    device_used = "cpu"

    if not os.path.isdir(INPUT_FOLDER):
        print(f"ERROR: Input folder not found: {INPUT_FOLDER}")
        return

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv")]
    if not files:
        print("No CSV files found in the input folder.")
        return

    chunk_mb = prompt_for_chunk_size_mb()
    chunk_rows = estimate_rows_per_chunk(os.path.join(INPUT_FOLDER, files[0]), chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    total_rows_processed = 0
    total_rows_saved = 0
    output_paths = []

    for file in files:
        csv_file = os.path.join(INPUT_FOLDER, file)
        base_name, ext = os.path.splitext(file)
        output_csv = make_unique_path(os.path.join(OUTPUT_FOLDER, f"{base_name}_cleaned{ext}"))

        print(f"\nScanning file: {csv_file}...")

        both_valid_counter = Counter()
        both_invalid_counter = Counter()
        rows_to_remove = 0
        rows_processed = 0

        # --- Phase 1: Scan and summarize ---
        for chunk in pd.read_csv(csv_file, chunksize=chunk_rows, low_memory=False, usecols=COLUMNS_TO_CHECK):
            rows_processed += len(chunk)
            total_rows_processed += len(chunk)

            if not all(col in chunk.columns for col in COLUMNS_TO_CHECK):
                print("  Missing required columns. Skipping file.")
                rows_processed = 0
                break

            invalid_mask = build_invalid_mask(chunk)
            valid_mask = ~invalid_mask

            both_invalid_counter.update(chunk.loc[invalid_mask, 'label'].astype(str).value_counts().to_dict())
            both_valid_counter.update(chunk.loc[valid_mask, 'label'].astype(str).value_counts().to_dict())
            rows_to_remove += int(invalid_mask.sum())

        if rows_processed == 0:
            continue

        print("\nSummary of rows:")
        print(f"Both valid rows: {sum(both_valid_counter.values())}")
        for lbl, cnt in both_valid_counter.items():
            print(f"  Label '{lbl}': {cnt}")

        print(f"\nBoth invalid rows: {sum(both_invalid_counter.values())}")
        for lbl, cnt in both_invalid_counter.items():
            print(f"  Label '{lbl}': {cnt}")

        delete_confirm = input(
            f"\nDo you want to delete the {rows_to_remove} rows with invalid handshakes in '{file}'? (yes/no): ").lower()

        if delete_confirm in ['yes', 'y']:
            max_rows = prompt_for_max_rows()
            print("\nDeleting invalid rows and creating new CSV...")
            is_first_chunk = True
            rows_written = 0

            for chunk in pd.read_csv(csv_file, chunksize=chunk_rows, low_memory=False):
                if not all(col in chunk.columns for col in COLUMNS_TO_CHECK):
                    print("  Missing required columns during write. Skipping.")
                    break
                invalid_mask = build_invalid_mask(chunk)
                cleaned = chunk.loc[~invalid_mask]

                if max_rows is not None:
                    remaining = max_rows - rows_written
                    if remaining <= 0:
                        break
                    if len(cleaned) > remaining:
                        cleaned = cleaned.iloc[:remaining]

                if not cleaned.empty:
                    cleaned.to_csv(output_csv, index=False, mode='w' if is_first_chunk else 'a', header=is_first_chunk)
                    is_first_chunk = False
                    rows_written += len(cleaned)

            total_rows_saved += rows_written
            output_paths.append(output_csv)
            print(f"Cleaned CSV saved: {output_csv}")
        else:
            print("No rows deleted. Moving to next file.")

    print("\nAll files processed.")

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Device used: {device_used.upper()}")
    print(f"Chunk size: {chunk_mb}MB (~{chunk_rows:,} rows)")
    print(f"Total rows processed: {total_rows_processed:,}")
    print(f"Rows saved: {total_rows_saved:,}")
    print("Output paths:")
    for path in output_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()

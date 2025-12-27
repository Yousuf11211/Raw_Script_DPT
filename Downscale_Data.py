# What changed:
# - Added GPU detection/device prompt, chunk size prompt with row estimation, and streaming processing.
# - Standardized outputs under ./outputs/Downscale_Csv_2018 with non-overwrite paths.
# - Added optional max-rows limit for output CSVs plus a final processing summary.
#
# Purpose:
# - Downscale benign data by sampling and keep all attack rows.
# - Save separate benign and attack CSVs.
# - Report label counts for each output.

import os
import pandas as pd
from collections import Counter

# Input folder with all csv files
INPUT_FOLDER = "2018_Separated_Nomissing"

# Output folder name under ./outputs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Downscale_Csv_2018")

# Combine 10% of the benign data from each file
BENIGN_SAMPLING_FRACTION = 0.1

RANDOM_STATE = 42


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


def estimate_rows_per_chunk(file_path, chunk_mb, sample_rows=2000, default_rows=1_000_000):
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


def find_label_column(columns):
    for col in columns:
        if str(col).strip().lower() == "label":
            return col
    return None


def main():
    gpu_available, _ = detect_gpu()
    device_choice = prompt_for_device(gpu_available)
    if device_choice == "gpu":
        print("GPU selected, but this script uses CPU-based pandas. Using CPU.")
    device_used = "cpu"

    if not os.path.isdir(INPUT_FOLDER):
        print(f"ERROR: Input folder not found at '{INPUT_FOLDER}'")
        return

    csv_files = []
    for root, _, files in os.walk(INPUT_FOLDER):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print("No CSV files found in the input folder.")
        return

    chunk_mb = prompt_for_chunk_size_mb()
    chunk_rows = estimate_rows_per_chunk(csv_files[0], chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

    max_rows_limit = prompt_for_max_rows()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_benign_file = make_unique_path(os.path.join(OUTPUT_FOLDER, "benign.csv"))
    output_attacks_file = make_unique_path(os.path.join(OUTPUT_FOLDER, "attacks.csv"))

    print("Starting the downscaling and separation process...")

    benign_written = 0
    attack_written = 0
    total_rows_processed = 0
    benign_label_counts = Counter()
    attack_label_counts = Counter()

    benign_header_written = False
    attack_header_written = False

    for file_idx, file_path in enumerate(csv_files, 1):
        print(f"\nProcessing {file_path} ({file_idx}/{len(csv_files)})...")
        try:
            chunk_iter = pd.read_csv(file_path, chunksize=chunk_rows, low_memory=False)
        except Exception as e:
            print(f"  -> Error reading file: {e}. Skipping.")
            continue

        label_col_found = None
        for chunk_idx, chunk in enumerate(chunk_iter, 1):
            if label_col_found is None:
                label_col_found = find_label_column(chunk.columns)
                if not label_col_found:
                    print("  -> No label column found in this file. Skipping.")
                    break

            total_rows_processed += len(chunk)
            print(f"  Processing chunk {chunk_idx} ({len(chunk):,} rows)...")

            labels = chunk[label_col_found].astype(str)
            benign_mask = labels.str.lower().eq("benign")

            benign_chunk = chunk[benign_mask]
            attack_chunk = chunk[~benign_mask]

            if not benign_chunk.empty:
                benign_sample = benign_chunk.sample(frac=BENIGN_SAMPLING_FRACTION, random_state=RANDOM_STATE)
                if not benign_sample.empty:
                    if max_rows_limit is not None:
                        remaining = max_rows_limit - benign_written
                        if remaining <= 0:
                            benign_sample = benign_sample.iloc[:0]
                        elif len(benign_sample) > remaining:
                            benign_sample = benign_sample.iloc[:remaining]
                    if not benign_sample.empty:
                        benign_sample = benign_sample.sample(frac=1, random_state=RANDOM_STATE)
                        benign_sample.to_csv(
                            output_benign_file,
                            index=False,
                            mode="w" if not benign_header_written else "a",
                            header=not benign_header_written,
                        )
                        benign_header_written = True
                        benign_written += len(benign_sample)
                        benign_label_counts.update(benign_sample[label_col_found].astype(str))

            if not attack_chunk.empty:
                if max_rows_limit is not None:
                    remaining = max_rows_limit - attack_written
                    if remaining <= 0:
                        attack_chunk = attack_chunk.iloc[:0]
                    elif len(attack_chunk) > remaining:
                        attack_chunk = attack_chunk.iloc[:remaining]
                if not attack_chunk.empty:
                    attack_chunk = attack_chunk.sample(frac=1, random_state=RANDOM_STATE)
                    attack_chunk.to_csv(
                        output_attacks_file,
                        index=False,
                        mode="w" if not attack_header_written else "a",
                        header=not attack_header_written,
                    )
                    attack_header_written = True
                    attack_written += len(attack_chunk)
                    attack_label_counts.update(attack_chunk[label_col_found].astype(str))

            if max_rows_limit is not None and benign_written >= max_rows_limit and attack_written >= max_rows_limit:
                print("  Max rows limit reached for both outputs. Stopping early.")
                break

    print("\n" + "=" * 60)
    print(" FINAL DATASET REPORT")
    print("=" * 60)

    if benign_header_written:
        print("\n--- Counts for benign.csv ---")
        print(f"Total Rows: {benign_written:,}")
        for label, count in benign_label_counts.items():
            print(f"  {label}: {count}")
    else:
        print("\n--- No benign.csv was created ---")

    if attack_header_written:
        print("\n--- Counts for attacks.csv ---")
        print(f"Total Rows: {attack_written:,}")
        for label, count in attack_label_counts.items():
            print(f"  {label}: {count}")
    else:
        print("\n--- No attacks.csv was created ---")

    print("\n" + "=" * 60)
    print("Process finished successfully!")

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Device used: {device_used.upper()}")
    print(f"Chunk size: {chunk_mb}MB (~{chunk_rows:,} rows)")
    print(f"Total rows processed: {total_rows_processed:,}")
    print(f"Rows saved (benign): {benign_written:,}")
    print(f"Rows saved (attacks): {attack_written:,}")
    print("Output paths:")
    if benign_header_written:
        print(f"  - {output_benign_file}")
    if attack_header_written:
        print(f"  - {output_attacks_file}")


if __name__ == "__main__":
    main()

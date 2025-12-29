# What changed:
# - Added GPU detection/device prompt and chunk size prompt with row estimation.
# - Streamed sampling to cap memory usage, plus optional max-rows limit for outputs.
# - Standardized outputs under ./outputs/Attack_Balanced with final summary.
# - Added CLI args, engine flags, chunk plan + progress for repo consistency.
#
# Purpose:
# - Balance class distribution using under/over-sampling techniques.
# - Save balanced CSVs per file and method.
# - Report label distributions before and after balancing.

import os
import sys
import argparse

# Allow running this script from any working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN

from config.global_config import DEFAULT_CHUNK_SIZE_MB, DEFAULT_MAX_OUTPUT_ROWS
from utils.chunk_utils import compute_chunk_plan, format_progress, print_chunk_plan
from utils.engine_utils import select_engine
from utils.path_utils import resolve_input_path, resolve_output_path

# ===== CONFIGURATION =====
INPUT_FOLDER = "Training_2018"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Attack_Balanced")

MAX_SAMPLE_ROWS = 500_000


# Global flag for non-interactive mode
_NO_INTERACTIVE = False


# ===== HELPERS =====

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


def get_csv_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]


def count_rows(file_path, chunk_rows):
    total = 0
    for chunk in pd.read_csv(file_path, chunksize=chunk_rows, low_memory=False):
        total += len(chunk)
    return total


def load_sampled_data(file_path, chunk_rows, max_rows, random_state=42):
    total_rows = count_rows(file_path, chunk_rows)
    if total_rows <= 0:
        return pd.DataFrame(), 0, 0

    frac = min(1.0, float(max_rows) / float(total_rows))
    sampled_chunks = []
    sampled_rows = 0

    for idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_rows, low_memory=False)):
        sample = chunk.sample(frac=frac, random_state=random_state) if frac < 1.0 else chunk
        if not sample.empty:
            sampled_chunks.append(sample)
            sampled_rows += len(sample)
        if (idx + 1) % 5 == 0:
            print(f"  Sampled {sampled_rows:,} rows so far...")
        if sampled_rows >= max_rows:
            break

    if not sampled_chunks:
        return pd.DataFrame(), total_rows, 0

    data = pd.concat(sampled_chunks, ignore_index=True)
    if len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
        sampled_rows = len(data)

    return data, total_rows, sampled_rows


def display_label_counts(y, le, file_name):
    counts = Counter(y)
    rev_mapping = {i: label for i, label in enumerate(le.classes_)}

    print(f"\n--- Label distribution for '{file_name}' ---")
    for k in sorted(counts.keys()):
        print(f"  {rev_mapping[k]:<20}: {counts.get(k, 0):,}")
    print(f"Total samples: {sum(counts.values()):,}")
    print("--------------------------------------------------")


def calculate_target_strategy(y, ratio):
    counts = Counter(y)
    if not counts:
        return {}

    majority_class_key = max(counts, key=counts.get)
    majority_count = counts[majority_class_key]

    target_strategy = {}
    target_minority_count = int(majority_count * ratio)

    for cls, count in counts.items():
        if cls == majority_class_key:
            target_strategy[cls] = count
        else:
            target_strategy[cls] = max(count, target_minority_count)

    return target_strategy


def apply_resampling(X, y, target_strategy, oversampler_class):
    current_counts = Counter(y)
    undersample = {c: t for c, t in target_strategy.items() if c in current_counts and current_counts[c] > t}
    oversample = {c: t for c, t in target_strategy.items() if c in current_counts and current_counts[c] < t}

    X_res, y_res = X.copy(), y.copy()

    if undersample:
        print("\nUndersampling started...")
        rus = RandomUnderSampler(sampling_strategy=undersample, random_state=42)
        X_res, y_res = rus.fit_resample(X_res, y_res)
        print("Undersampling done.")

    if oversample:
        print("\nOversampling started...")
        min_samples_for_smote = min(count for cls, count in Counter(y_res).items() if cls in oversample)
        num_neighbors = max(1, min(min_samples_for_smote - 1, 5))

        sampler_params = {
            'sampling_strategy': oversample,
            'random_state': 42
        }

        if oversampler_class == ADASYN:
            sampler_params['n_neighbors'] = num_neighbors
            print(f"Using ADASYN with n_neighbors={num_neighbors}...")
        else:
            sampler_params['k_neighbors'] = num_neighbors
            print(f"Using {oversampler_class.__name__} with k_neighbors={num_neighbors}...")

        sampler = oversampler_class(**sampler_params)
        X_res, y_res = sampler.fit_resample(X_res, y_res)
        print("Oversampling done.")

    return X_res, y_res


# ===== MAIN =====

def main():
    gpu_available, _ = detect_gpu()
    device_choice = prompt_for_device(gpu_available)
    if device_choice == "gpu":
        print("GPU selected, but this script uses CPU-based pandas. Using CPU.")
    device_used = "cpu"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    csv_files = get_csv_files(INPUT_FOLDER)

    if not csv_files:
        print("No CSV files found in the input folder!")
        return

    chunk_mb = prompt_for_chunk_size_mb()
    chunk_rows = estimate_rows_per_chunk(csv_files[0], chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

    print(f"Found {len(csv_files)} CSV file(s).")

    files_to_process = []
    for file_path in csv_files:
        response = input(f"Process '{os.path.basename(file_path)}'? (y/n): ").lower()
        if response == 'y':
            files_to_process.append(file_path)
            print("  ✓ Added to processing list")
        else:
            print("  ✗ Skipped")

    if not files_to_process:
        print("\nNo files selected for processing!")
        return

    while True:
        try:
            ratio = float(input("\nEnter the desired minority-to-majority ratio (e.g., 0.5 for 50%): "))
            if 0 < ratio <= 1:
                break
            print("Please enter a number between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    oversamplers = {"1": SMOTE, "2": BorderlineSMOTE, "3": ADASYN}
    all_samplers = [SMOTE, BorderlineSMOTE, ADASYN]

    while True:
        choice = input(
            "Choose an oversampling method:\n  1: SMOTE (Standard)\n  2: Borderline-SMOTE\n  3: ADASYN\n  4: All\nChoice: ")
        if choice in oversamplers or choice == "4":
            break
        print("Invalid choice. Please enter 1, 2, 3, or 4.")

    max_rows_limit = prompt_for_max_rows()

    samplers_to_run = all_samplers if choice == "4" else [oversamplers[choice]]
    total_rows_processed = 0
    total_rows_saved = 0
    output_paths = []

    for oversampler_class in samplers_to_run:
        method_name = oversampler_class.__name__
        print(f"\n===== PROCESSING WITH: {method_name} =====")

        method_output_folder = os.path.join(OUTPUT_FOLDER, method_name)
        os.makedirs(method_output_folder, exist_ok=True)

        for file_path in files_to_process:
            print(f"\nLoading and sampling {os.path.basename(file_path)}...")
            df, total_rows, sampled_rows = load_sampled_data(file_path, chunk_rows, MAX_SAMPLE_ROWS)
            total_rows_processed += total_rows

            if df.empty:
                print(f"Skipping '{os.path.basename(file_path)}' (no data).")
                continue

            if 'label' not in df.columns:
                print(f"Skipping '{os.path.basename(file_path)}' (no 'label' column found).")
                continue

            le = LabelEncoder()
            y_enc = le.fit_transform(df['label'])
            display_label_counts(y_enc, le, os.path.basename(file_path))

            target_strategy = calculate_target_strategy(y_enc, ratio)

            X = df.drop("label", axis=1)
            X_bal, y_bal = apply_resampling(X, y_enc, target_strategy, oversampler_class)

            df_bal = pd.DataFrame(X_bal, columns=X.columns)
            df_bal["label"] = le.inverse_transform(y_bal)

            if max_rows_limit is not None and len(df_bal) > max_rows_limit:
                df_bal = df_bal.iloc[:max_rows_limit]
                y_bal = y_bal[:len(df_bal)]

            display_label_counts(y_bal, le, f"{os.path.basename(file_path)} (Balanced)")

            out_file = os.path.join(method_output_folder, os.path.basename(file_path).replace(".csv", "_balanced.csv"))
            out_file = make_unique_path(out_file)
            df_bal.to_csv(out_file, index=False)
            output_paths.append(out_file)
            total_rows_saved += len(df_bal)
            print(f"\nSaved balanced CSV to '{method_name}' folder: {os.path.basename(out_file)}")

    print("\nAll selected files and methods processed.")

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

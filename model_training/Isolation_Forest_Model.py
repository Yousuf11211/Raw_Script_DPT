# What changed:
# - Added GPU detection/device prompt and chunk size prompt with row estimation.
# - Streamed sampling to cap memory usage for model training.
# - Standardized outputs under ./outputs/Isolation_Forest_Model with final summary.
#
# Purpose:
# - Train an Isolation Forest model on sampled benign data.
# - Save the trained model to disk.
# - Report processing stats.

import os
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import numpy as np

# --- 1. Configuration ---
LARGE_BENIGN_FILE = "Training_isolation_model_cleaned/Benign_part_2.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Isolation_Forest_Model")
MODEL_FILENAME = os.path.join(OUTPUT_FOLDER, "isolation.joblib")

MAX_SAMPLE_ROWS = 500_000


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


def estimate_rows_per_chunk(file_path, chunk_mb, sample_rows=2000, default_rows=2_000_000):
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
        print(f"  Sampled chunk {idx + 1}, total sampled rows: {sampled_rows:,}")
        if sampled_rows >= max_rows:
            break

    if not sampled_chunks:
        return pd.DataFrame(), total_rows, 0

    data = pd.concat(sampled_chunks, ignore_index=True)
    if len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
        sampled_rows = len(data)

    return data, total_rows, sampled_rows


# --- 2. Main ---

def main():
    gpu_available, _ = detect_gpu()
    device_choice = prompt_for_device(gpu_available)
    if device_choice == "gpu":
        print("GPU selected, but IsolationForest uses CPU. Using CPU.")
    device_used = "cpu"

    if not os.path.exists(LARGE_BENIGN_FILE):
        print(f"Error: The file '{LARGE_BENIGN_FILE}' was not found.")
        return

    chunk_mb = prompt_for_chunk_size_mb()
    chunk_rows = estimate_rows_per_chunk(LARGE_BENIGN_FILE, chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"Starting to sample '{LARGE_BENIGN_FILE}'...")
    data, total_rows, sampled_rows = load_sampled_data(LARGE_BENIGN_FILE, chunk_rows, MAX_SAMPLE_ROWS)
    if data.empty:
        print("No data loaded for training.")
        return

    print(f"\nSampling complete. Total rows in sampled dataset: {sampled_rows:,}")

    print("\nSelecting numeric features for training (skipping non-numeric columns)...")
    X_train_numeric = data.select_dtypes(include=[np.number])
    print(f"Dropped {data.shape[1] - X_train_numeric.shape[1]} non-numeric columns.")

    model = IsolationForest(
        n_estimators=100,
        contamination='auto',
        random_state=42,
        n_jobs=-1
    )

    print("\nStarting model training on numeric data...")
    model.fit(X_train_numeric)
    print("Model training complete.")

    print(f"\nSaving model to '{MODEL_FILENAME}'...")
    joblib.dump(model, MODEL_FILENAME)
    print(f"Model successfully saved to {MODEL_FILENAME}")

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Device used: {device_used.upper()}")
    print(f"Chunk size: {chunk_mb}MB (~{chunk_rows:,} rows)")
    print(f"Total rows processed: {total_rows:,}")
    print("Rows saved: N/A")
    print("Output paths:")
    print(f"  - {MODEL_FILENAME}")


if __name__ == "__main__":
    main()

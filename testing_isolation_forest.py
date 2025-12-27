# What changed:
# - Added GPU detection/device prompt and chunk size prompt with row estimation.
# - Streamed predictions to avoid loading full test CSV into memory.
# - Standardized outputs under ./outputs/testing_isolation_forest with final summary.
#
# Purpose:
# - Evaluate an Isolation Forest model on a test CSV.
# - Save a report and confusion matrix.
# - Stream predictions for large files.

import os
import pandas as pd
import joblib
import numpy as np

# --- 1. Configuration ---
MODEL_FILENAME = 'Training_isolation_model_cleaned/isolation.joblib'
TEST_DATA_FILE = '../Testing_isolation_model_cleaned/Benign_part_2.csv'
LABEL_COLUMN = 'label'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "testing_isolation_forest")


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


def build_report_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    lines = []
    lines.append("Isolation Forest Evaluation Report")
    lines.append("=" * 40)
    lines.append(f"Precision: {precision:.4f}")
    lines.append(f"Recall: {recall:.4f}")
    lines.append(f"F1 Score: {f1:.4f}")
    lines.append("")
    lines.append("Confusion Matrix (labels: [Normal=1, Anomaly=-1])")
    lines.append(str(cm))
    return "\n".join(lines)


def main():
    gpu_available, _ = detect_gpu()
    device_choice = prompt_for_device(gpu_available)
    if device_choice == "gpu":
        print("GPU selected, but IsolationForest uses CPU. Using CPU.")
    device_used = "cpu"

    if not os.path.exists(MODEL_FILENAME):
        print(f"Error: The model file '{MODEL_FILENAME}' was not found.")
        return
    if not os.path.exists(TEST_DATA_FILE):
        print(f"Error: The test file '{TEST_DATA_FILE}' was not found.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    chunk_mb = prompt_for_chunk_size_mb()
    chunk_rows = estimate_rows_per_chunk(TEST_DATA_FILE, chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

    print(f"Loading model from {MODEL_FILENAME}...")
    model = joblib.load(MODEL_FILENAME)

    print(f"Loading test data from {TEST_DATA_FILE}...")

    total_rows_processed = 0
    cm = np.zeros((2, 2), dtype=int)

    for chunk_idx, chunk in enumerate(pd.read_csv(TEST_DATA_FILE, chunksize=chunk_rows, low_memory=False), 1):
        total_rows_processed += len(chunk)

        y_true_labels = chunk[LABEL_COLUMN].astype(str)
        y_true_mapped = y_true_labels.apply(lambda x: 1 if x == 'Benign' else -1).values

        X_test_numeric = chunk.select_dtypes(include=[np.number])
        y_pred = model.predict(X_test_numeric)

        # Map to confusion matrix indices: Normal=1 -> idx 0, Anomaly=-1 -> idx 1
        true_idx = np.where(y_true_mapped == 1, 0, 1)
        pred_idx = np.where(y_pred == 1, 0, 1)
        for t, p in zip(true_idx, pred_idx):
            cm[t, p] += 1

        print(f"Processed chunk {chunk_idx} ({len(chunk):,} rows)")

    print("\n--- Evaluation Results ---")
    print("\nConfusion Matrix:")
    print("         [Pred Normal] [Pred Anomaly]")
    print(f"True Normal: {cm[0]}")
    print(f"True Anomaly: {cm[1]}")

    report_text = build_report_from_cm(cm)
    report_path = os.path.join(OUTPUT_FOLDER, "isolation_forest_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    cm_path = os.path.join(OUTPUT_FOLDER, "isolation_forest_confusion_matrix.csv")
    pd.DataFrame(cm, index=["Normal", "Anomaly"], columns=["Pred Normal", "Pred Anomaly"]).to_csv(cm_path)

    print(f"\nReport saved to: {report_path}")
    print(f"Confusion matrix saved to: {cm_path}")

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Device used: {device_used.upper()}")
    print(f"Chunk size: {chunk_mb}MB (~{chunk_rows:,} rows)")
    print(f"Total rows processed: {total_rows_processed:,}")
    print("Rows saved: N/A")
    print("Output paths:")
    print(f"  - {report_path}")
    print(f"  - {cm_path}")


if __name__ == "__main__":
    main()

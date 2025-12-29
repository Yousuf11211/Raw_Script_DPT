# What changed:
# - Added GPU detection/device prompt and chunk size prompt with row estimation.
# - Streamed prediction to avoid loading full test CSV into memory.
# - Standardized outputs under ./outputs/Model_Testing with final summary.
#
# Purpose:
# - Run model inference on a test CSV.
# - Optionally save classification report, confusion matrix, predictions, and counts summary.
# - Stream predictions to handle large files.

import os
import joblib
import pandas as pd
import numpy as np
from collections import Counter

# --- CONFIG ---
MODEL_PATH = "Model_2018/full_model.pkl"
LABEL_MAPPING_PATH = "Model_2018/full_label_mapping.txt"
TEST_CSV_PATH = "Balanced_Test_2018/full.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Model_Testing")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


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


def load_label_mapping(path):
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()[2:]
        for line in lines:
            if ":" not in line:
                continue
            cls, num = line.strip().split(":")
            cls, num = cls.strip(), num.strip()
            if num.isdigit():
                mapping[int(num)] = cls
    return mapping


def build_classification_report_from_cm(cm, labels):
    rows = []
    header = f"{'label':<25} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}"
    rows.append(header)
    rows.append("-" * len(header))

    cm = cm.astype(float)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)

    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)

    for idx, label in enumerate(labels):
        rows.append(f"{label:<25} {precision[idx]:>9.2f} {recall[idx]:>9.2f} {f1[idx]:>9.2f} {int(support[idx]):>9}")

    avg_precision = precision.mean() if len(precision) else 0.0
    avg_recall = recall.mean() if len(recall) else 0.0
    avg_f1 = f1.mean() if len(f1) else 0.0
    total_support = int(support.sum())

    rows.append("")
    rows.append(f"{'macro avg':<25} {avg_precision:>9.2f} {avg_recall:>9.2f} {avg_f1:>9.2f} {total_support:>9}")
    return "\n".join(rows)


def main():
    gpu_available, _ = detect_gpu()
    device_choice = prompt_for_device(gpu_available)
    if device_choice == "gpu":
        print("GPU selected, but this script uses CPU-based pandas. Using CPU.")
    device_used = "cpu"

    save_report = input("Save classification report? (y/n): ").strip().lower() == "y"
    save_cm = input("Save confusion matrix? (y/n): ").strip().lower() == "y"
    save_preds_csv = input("Save full predictions CSV? (y/n): ").strip().lower() == "y"
    save_counts_summary = input("Save prediction counts summary? (y/n): ").strip().lower() == "y"

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    if not os.path.exists(LABEL_MAPPING_PATH):
        print(f"ERROR: Label mapping not found at {LABEL_MAPPING_PATH}")
        return
    if not os.path.exists(TEST_CSV_PATH):
        print(f"ERROR: Test CSV not found at {TEST_CSV_PATH}")
        return

    chunk_mb = prompt_for_chunk_size_mb()
    chunk_rows = estimate_rows_per_chunk(TEST_CSV_PATH, chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

    rf = joblib.load(MODEL_PATH)
    print("Model loaded.")

    mapping = load_label_mapping(LABEL_MAPPING_PATH)
    classes_by_index = [mapping[i] for i in sorted(mapping.keys())]
    inv_mapping = {v: k for k, v in mapping.items()}
    print("Label mapping loaded:", classes_by_index)

    preds_output_path = None
    rows_saved = 0

    if save_preds_csv:
        max_rows = prompt_for_max_rows()
        base_name = os.path.splitext(os.path.basename(TEST_CSV_PATH))[0]
        preds_output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_predictions.csv")
        preds_output_path = os.path.abspath(preds_output_path)

    attack_counts = Counter()
    total_rows_processed = 0

    has_labels = False
    cm = np.zeros((len(classes_by_index), len(classes_by_index)), dtype=int)

    for chunk_idx, chunk in enumerate(pd.read_csv(TEST_CSV_PATH, chunksize=chunk_rows, low_memory=False), 1):
        total_rows_processed += len(chunk)
        chunk.columns = chunk.columns.str.strip().str.lower()

        if 'label' in chunk.columns:
            has_labels = True
            y_true_raw = chunk['label'].astype(str)
            y_true = np.array([inv_mapping.get(lbl, -1) for lbl in y_true_raw])
            X_test = chunk.drop(columns=['label'])
        else:
            y_true = None
            X_test = chunk.copy()

        if hasattr(rf, 'feature_names_in_'):
            X_test = X_test.reindex(columns=rf.feature_names_in_, fill_value=0)

        y_pred = rf.predict(X_test)
        y_pred_labels = [mapping[num] for num in y_pred]
        attack_counts.update(y_pred_labels)

        if has_labels and y_true is not None:
            valid_mask = y_true >= 0
            if valid_mask.any():
                y_true_idx = y_true[valid_mask].astype(int)
                y_pred_idx = y_pred[valid_mask].astype(int)
                np.add.at(cm, (y_true_idx, y_pred_idx), 1)

        if preds_output_path:
            if max_rows is not None and rows_saved >= max_rows:
                continue
            output_df = chunk.copy()
            output_df['predicted_label'] = y_pred_labels
            if max_rows is not None:
                remaining = max_rows - rows_saved
                if remaining <= 0:
                    output_df = output_df.iloc[:0]
                elif len(output_df) > remaining:
                    output_df = output_df.iloc[:remaining]
            if not output_df.empty:
                mode = "w" if rows_saved == 0 else "a"
                header = rows_saved == 0
                output_df.to_csv(preds_output_path, index=False, mode=mode, header=header)
                rows_saved += len(output_df)

        print(f"Processed chunk {chunk_idx} ({len(chunk):,} rows)")

    print("\nPredicted attack counts:")
    for attack, count in attack_counts.items():
        print(f"{attack:<20}: {count}")

    output_paths = []

    if save_counts_summary:
        base_name = os.path.splitext(os.path.basename(TEST_CSV_PATH))[0]
        counts_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_predicted_counts.txt")
        with open(counts_path, "w", encoding="utf-8") as f:
            for attack, count in sorted(attack_counts.items()):
                f.write(f"{attack:<20}: {count}\n")
        print(f"Prediction counts summary saved -> {counts_path}")
        output_paths.append(counts_path)

    if has_labels and cm.sum() > 0:
        report = build_classification_report_from_cm(cm, classes_by_index)
        if save_report:
            report_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(os.path.basename(TEST_CSV_PATH))[0]}_report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Classification report saved -> {report_path}")
            output_paths.append(report_path)
        else:
            print("\nClassification Report:")
            print(report)

        if save_cm:
            cm_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(os.path.basename(TEST_CSV_PATH))[0]}_confusion_matrix.csv")
            cm_df = pd.DataFrame(cm, index=classes_by_index, columns=classes_by_index)
            cm_df.to_csv(cm_path)
            print(f"Confusion matrix saved -> {cm_path}")
            output_paths.append(cm_path)
        else:
            print("\nConfusion Matrix:")
            print(pd.DataFrame(cm, index=classes_by_index, columns=classes_by_index))
    else:
        print("[info] No ground-truth labels in test file, skipping report generation.")

    if preds_output_path:
        output_paths.append(preds_output_path)

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Device used: {device_used.upper()}")
    print(f"Chunk size: {chunk_mb}MB (~{chunk_rows:,} rows)")
    print(f"Total rows processed: {total_rows_processed:,}")
    if preds_output_path:
        print(f"Rows saved: {rows_saved:,}")
    else:
        print("Rows saved: N/A")
    print("Output paths:")
    for path in output_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()

# What changed:
# - Added GPU detection/device prompt and chunk size prompt with row estimation.
# - Streamed sampling to cap memory usage for model training.
# - Standardized outputs under ./outputs/Model_Random_Forest with final summary.
#
# Purpose:
# - Train RandomForest models from CSV files.
# - Save models, label mappings, and optional evaluation reports.
# - Provide a reusable training wrapper.

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import sklearn
print(f"Scikit-learn Version: {sklearn.__version__}")

# ===== CONFIGURATION =====
INPUT_FOLDER = r"C:\Users\Yousuf\Desktop\Raw_Script_DPT\Bening1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
MODEL_FOLDER = os.path.join(OUTPUT_ROOT, "Model_Random_Forest", "models")
REPORT_FOLDER = os.path.join(OUTPUT_ROOT, "Model_Random_Forest", "reports")

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

train_full_data = False
MAX_SAMPLE_ROWS = 500_000


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


# ===== REUSABLE WRAPPER FOR DJANGO =====

def train_random_forest_from_csv(
    input_csv_path: str,
    output_dir: str,
    model_filename: str = "random_forest_model.pkl",
    label_mapping_filename: str = "random_forest_label_mapping.txt",
    chunk_rows: int | None = None,
):
    os.makedirs(output_dir, exist_ok=True)

    if chunk_rows is None:
        chunk_rows = 500_000

    data, _, _ = load_sampled_data(input_csv_path, chunk_rows, MAX_SAMPLE_ROWS)
    data.columns = data.columns.str.lower()

    if "label" not in data.columns:
        raise ValueError("Input CSV must contain a 'label' column for training.")

    X = data.drop(columns=["label"])
    y_raw = data["label"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    mapping_path = os.path.join(output_dir, label_mapping_filename)
    with open(mapping_path, "w", encoding="utf-8") as f:
        f.write("Label Encoding Mapping:\n")
        f.write("=" * 40 + "\n")
        for cls, num in zip(le.classes_, range(len(le.classes_))):
            f.write(f"{cls:<30}: {num}\n")

    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    model_path = os.path.join(output_dir, model_filename)

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    joblib.dump(rf, model_path)

    details = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "classes": [str(c) for c in le.classes_],
    }

    return {
        "model_path": model_path,
        "label_mapping_path": mapping_path,
        "details": details,
    }


# ===== FUNCTIONS =====

def process_csv(file_path, chunk_rows):
    print(f"\n{'=' * 80}")
    print(f"Processing file: {os.path.basename(file_path)}")
    print(f"{'=' * 80}")

    data, total_rows, _ = load_sampled_data(file_path, chunk_rows, MAX_SAMPLE_ROWS)
    data.columns = data.columns.str.lower()

    if 'label' not in data.columns:
        print(f"Skipping {file_path} (no 'label' column found).")
        return 0, 0, []

    X = data.drop(columns=['label'])
    y_raw = data['label']

    # --- ADD THIS CLEANING BLOCK ---
    import numpy as np
    print("Cleaning Infinity and NaN values...")
    X.replace([np.inf, -np.inf], 0, inplace=True)
    X.fillna(0, inplace=True)
    # -------------------------------

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    mapping_path = os.path.join(
        MODEL_FOLDER, os.path.basename(file_path).replace(".csv", "_label_mapping.txt")
    )
    with open(mapping_path, "w", encoding="utf-8") as f:
        f.write("Label Encoding Mapping:\n")
        f.write("=" * 40 + "\n")
        for cls, num in zip(le.classes_, range(len(le.classes_))):
            f.write(f"{cls:<30}: {num}\n")
    print(f"Label mapping saved to {mapping_path}")

    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    model_name = os.path.basename(file_path).replace(".csv", "")
    model_path = os.path.join(MODEL_FOLDER, f"{model_name}_model.pkl")

    report_paths = []

    if train_full_data:
        print("Training Random Forest on sampled dataset...")
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        rf.fit(X, y)
        joblib.dump(rf, model_path)
        print(f"Model trained and saved to {model_path}")
    else:
        print("Splitting data (80% train / 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        joblib.dump(rf, model_path)
        print(f"Model trained and saved to {model_path}")

        print("Evaluating on test data...")
        y_pred = rf.predict(X_test)

        # --- FIX: Convert label names to strings so the report can format them ---
        str_classes = [str(c) for c in le.classes_]

        report = classification_report(y_test, y_pred, target_names=str_classes)
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=str_classes, columns=str_classes)

        report_path = os.path.join(REPORT_FOLDER, f"{model_name}_report.txt")
        cm_path = os.path.join(REPORT_FOLDER, f"{model_name}_confusion_matrix.csv")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Test Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)
        cm_df.to_csv(cm_path)

        report_paths.extend([report_path, cm_path])

        print("\nClassification Report:\n")
        print(report)
        print("\nConfusion Matrix:\n")
        print(cm_df)
        print(f"Report saved to {report_path}")
        print(f"Confusion matrix saved to {cm_path}")

    return total_rows, 0, [mapping_path, model_path] + report_paths


# ===== MAIN =====
if __name__ == "__main__":
    gpu_available, _ = detect_gpu()
    device_choice = prompt_for_device(gpu_available)
    if device_choice == "gpu":
        print("GPU selected, but RandomForest runs on CPU. Using CPU.")

    if not os.path.isdir(INPUT_FOLDER):
        print(f"Input folder not found: {INPUT_FOLDER}")
        raise SystemExit(1)

    csv_files = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the input folder.")
    else:
        chunk_mb = prompt_for_chunk_size_mb()
        chunk_rows = estimate_rows_per_chunk(csv_files[0], chunk_mb)
        print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

        print(f"Found {len(csv_files)} CSV file(s) in '{INPUT_FOLDER}'.")
        total_rows_processed = 0
        output_paths = []

        for file_path in csv_files:
            rows_processed, _, paths = process_csv(file_path, chunk_rows)
            total_rows_processed += rows_processed
            output_paths.extend(paths)

        print("\nAll models trained and saved successfully.")

        print("\nFinal Summary")
        print("-" * 40)
        print(f"Device used: CPU")
        print(f"Chunk size: {chunk_mb}MB (~{chunk_rows:,} rows)")
        print(f"Total rows processed: {total_rows_processed:,}")
        print("Rows saved: N/A")
        print("Output paths:")
        for path in output_paths:
            print(f"  - {path}")

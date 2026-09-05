# What changed:
# - Added GPU detection/device prompt, chunk size prompt with row estimation, and streaming sampling.
# - Standardized outputs under ./outputs/Feature_Importance_RandomForest with non-overwrite paths.
# - Added optional max-rows limit for filtered CSV saves plus a final processing summary.
#
# Purpose:
# - Compute Random Forest feature importance from CSV data.
# - Save text/CSV reports and a top-20 plot.
# - Optionally remove low-importance features and save a cleaned CSV.

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ======================
# CONFIGURATION - CHANGE THESE SETTINGS
# ======================

PROCESS_FOLDER = False  # True = process all CSVs in a folder, False = single file
# ======================
# CONFIGURATION - CHANGE THESE SETTINGS
# ======================

PROCESS_FOLDER = False  # True = process all CSVs in a folder, False = single file
FOLDER_PATH = "Bening1"

# IMPORTANT: Make sure this file is actually in the folder wh ere you are running the script!
# If it is in an outputs folder, change this to "outputs/Model.csv"
SINGLE_FILE_PATH = r"C:\Users\Yousuf\Desktop\Raw_Script_DPT\Bening1\model1_final_training_data.csv"
IMPORTANCE_THRESHOLD = 0.1  # percent

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Feature_Importance_RandomForest")

MAX_SAMPLE_ROWS = 500_000


# ======================
# HELPERS
# ======================

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


def count_rows(file_path, chunk_rows):
    total = 0
    for idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_rows, low_memory=False)):
        total += len(chunk)
        if (idx + 1) % 5 == 0:
            print(f"  Counted {total:,} rows so far in {os.path.basename(file_path)}...")
    return total


def get_csv_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]


def load_sampled_data(file_path, chunk_rows, max_rows, random_state=42):
    print(f"Counting rows in {os.path.basename(file_path)}...")
    total_rows = count_rows(file_path, chunk_rows)
    if total_rows <= 0:
        return pd.DataFrame(), 0, 0

    frac = min(1.0, float(max_rows) / float(total_rows))
    sampled_chunks = []
    sampled_rows = 0

    print(f"Sampling {os.path.basename(file_path)} (target up to {max_rows:,} rows)...")
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


def load_sampled_data_from_folder(file_paths, chunk_rows, max_rows, random_state=42):
    total_rows = 0
    per_file_counts = {}
    for path in file_paths:
        print(f"Counting rows in {os.path.basename(path)}...")
        rows = count_rows(path, chunk_rows)
        per_file_counts[path] = rows
        total_rows += rows

    if total_rows <= 0:
        return pd.DataFrame(), 0, 0

    frac = min(1.0, float(max_rows) / float(total_rows))
    sampled_chunks = []
    sampled_rows = 0

    for path in file_paths:
        if per_file_counts.get(path, 0) <= 0:
            continue
        print(f"Sampling {os.path.basename(path)}...")
        for idx, chunk in enumerate(pd.read_csv(path, chunksize=chunk_rows, low_memory=False)):
            sample = chunk.sample(frac=frac, random_state=random_state) if frac < 1.0 else chunk
            if not sample.empty:
                sampled_chunks.append(sample)
                sampled_rows += len(sample)
            if sampled_rows >= max_rows:
                break
        if sampled_rows >= max_rows:
            break

    if not sampled_chunks:
        return pd.DataFrame(), total_rows, 0

    data = pd.concat(sampled_chunks, ignore_index=True)
    if len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
        sampled_rows = len(data)

    return data, total_rows, sampled_rows


def write_filtered_csv(file_paths, output_path, columns_to_drop, chunk_rows, max_rows=None):
    rows_written = 0
    is_first_chunk = True
    for path in file_paths:
        for idx, chunk in enumerate(pd.read_csv(path, chunksize=chunk_rows, low_memory=False)):
            chunk.columns = chunk.columns.str.lower()
            chunk.drop(columns=columns_to_drop, inplace=True, errors="ignore")
            if max_rows is not None:
                remaining = max_rows - rows_written
                if remaining <= 0:
                    return rows_written
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining]
            if is_first_chunk:
                chunk.to_csv(output_path, index=False, mode="w")
                is_first_chunk = False
            else:
                chunk.to_csv(output_path, index=False, mode="a", header=False)
            rows_written += len(chunk)
            if (idx + 1) % 5 == 0:
                print(f"  Wrote {rows_written:,} rows so far...")
            if max_rows is not None and rows_written >= max_rows:
                return rows_written
    return rows_written


# ======================
# MAIN
# ======================

def main():
    gpu_available, _ = detect_gpu()
    device_choice = prompt_for_device(gpu_available)
    if device_choice == "gpu":
        print("GPU selected, but RandomForest runs on CPU. Using CPU.")
    device_used = "cpu"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if PROCESS_FOLDER:
        file_paths = get_csv_files(FOLDER_PATH)
        mode_label = "Folder"
    else:
        file_paths = [SINGLE_FILE_PATH]
        mode_label = "Single file"

    print("=== Feature Importance Analysis Tool ===")
    print(f"Processing mode: {mode_label}")
    print(f"Importance threshold: {IMPORTANCE_THRESHOLD}%")
    print("-" * 50)

    if not file_paths:
        print("ERROR: No CSV files found to process.")
        return

    missing = [p for p in file_paths if not os.path.exists(p)]
    if missing:
        print("ERROR: Missing input file(s):")
        for p in missing:
            print(f"  - {p}")
        return

    chunk_mb = prompt_for_chunk_size_mb()
    chunk_rows = estimate_rows_per_chunk(file_paths[0], chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

    print("\nLoading and sampling data...")
    if PROCESS_FOLDER:
        data, total_rows, sampled_rows = load_sampled_data_from_folder(file_paths, chunk_rows, MAX_SAMPLE_ROWS)
    else:
        data, total_rows, sampled_rows = load_sampled_data(file_paths[0], chunk_rows, MAX_SAMPLE_ROWS)

    if data.empty:
        print("ERROR: No valid CSV data loaded.")
        return

    data.columns = data.columns.str.lower()
    if "label" not in data.columns:
        print("ERROR: No 'label' column found in the data.")
        return

    print(f"\nSampled dataset shape: {data.shape}")
    print(f"Total rows scanned: {total_rows:,} | Sampled rows: {sampled_rows:,}")

    y = LabelEncoder().fit_transform(data["label"])
    X = data.drop(columns=["label"])

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    if categorical_cols:
        print(f"  Encoding {len(categorical_cols)} categorical columns...")
        for col in categorical_cols:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # --- THIS LINE IS REQUIRED TO PREVENT THE SCRIPT FROM CRASHING ---
    print("Handling missing values (NaNs)...")
    X.fillna(0, inplace=True)
    # --- REQUIRED FIX FOR MISSING DATA AND INFINITY ---
    import numpy as np

    print("Handling missing values (NaNs) and Infinity...")
    X.replace([np.inf, -np.inf], 0, inplace=True)
    X.fillna(0, inplace=True)

    print("\nTraining Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

    print("\nTraining Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    print("✓ Model training completed!")

    print("\nCalculating feature importance...")
    importances = rf.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance_pct": 100 * importances
    }).sort_values(by="Importance_pct", ascending=False)

    report_path = make_unique_path(os.path.join(OUTPUT_FOLDER, "Feature_Importance_Report.txt"))
    csv_report_path = make_unique_path(os.path.join(OUTPUT_FOLDER, "Feature_Importance.csv"))
    plot_path = make_unique_path(os.path.join(OUTPUT_FOLDER, "Top20_Feature_Importance.png"))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Feature Importance Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total features: {len(feat_imp_df)}\n")
        f.write(f"Dataset shape (sampled): {data.shape}\n")
        f.write("-" * 50 + "\n")
        for _, row in feat_imp_df.iterrows():
            f.write(f"{row['Feature']:<30}: {row['Importance_pct']:.4f}%\n")

    feat_imp_df.to_csv(csv_report_path, index=False)

    plt.figure(figsize=(12, 8))
    top_20 = feat_imp_df.head(20)
    plt.barh(range(len(top_20)), top_20["Importance_pct"])
    plt.yticks(range(len(top_20)), top_20["Feature"])
    plt.xlabel("Importance (%)")
    plt.title("Top 20 Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✓ Feature_Importance_Report.txt saved")
    print("✓ Feature_Importance.csv saved")
    print("✓ Top20_Feature_Importance.png saved")

    print(f"\nAnalyzing features with importance < {IMPORTANCE_THRESHOLD}%...")
    near_zero = feat_imp_df[feat_imp_df["Importance_pct"] < IMPORTANCE_THRESHOLD]["Feature"].tolist()

    rows_saved = 0
    filtered_output_path = None

    if near_zero:
        print(f"Found {len(near_zero)} features with very low importance.")
        for i, feature in enumerate(near_zero, 1):
            importance = feat_imp_df.loc[feat_imp_df["Feature"] == feature, "Importance_pct"].iloc[0]
            print(f"  {i}. {feature}: {importance:.4f}%")

        response = input(f"\nDo you want to remove these {len(near_zero)} low-importance features? (y/n): ")
        if response.lower() in ["y", "yes"]:
            max_rows = prompt_for_max_rows()
            if PROCESS_FOLDER:
                base_name = os.path.basename(FOLDER_PATH.rstrip(os.sep))
            else:
                base_name = os.path.splitext(os.path.basename(SINGLE_FILE_PATH))[0]
            output_filename = f"{base_name}_lessfeatures.csv"
            filtered_output_path = make_unique_path(os.path.join(OUTPUT_FOLDER, output_filename))
            print("Saving filtered dataset in chunks...")
            rows_saved = write_filtered_csv(file_paths, filtered_output_path, near_zero, chunk_rows, max_rows=max_rows)
            print(f"✓ Saved filtered dataset: {filtered_output_path}")
            print(f"  Features removed: {len(near_zero)}")
        else:
            print("No features were removed.")
    else:
        print("✓ No low-importance features found!")

    print("\n=== Analysis Complete ===")

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Device used: {device_used.upper()}")
    print(f"Chunk size: {chunk_mb}MB (~{chunk_rows:,} rows)")
    print(f"Total rows processed: {total_rows:,}")
    if filtered_output_path:
        print(f"Rows saved: {rows_saved:,}")
    else:
        print("Rows saved: N/A")
    print("Output paths:")
    for path in [report_path, csv_report_path, plot_path] + ([filtered_output_path] if filtered_output_path else []):
        print(f"  - {path}")


if __name__ == "__main__":
    main()

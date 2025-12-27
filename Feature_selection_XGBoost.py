# What changed:
# - Added GPU detection/device prompt, chunk size prompt with row estimation, and streaming reads.
# - Standardized outputs under ./outputs/Feature_Selection_Reports with non-overwrite paths.
# - Added optional max-rows limit for cleaned CSV saves plus a final processing summary.
#
# Purpose:
# - Train XGBoost to report global feature importance.
# - Optionally remove near-zero-importance features and save a cleaned CSV.
# - Optionally run per-label (one-vs-rest) feature importance analysis with plots.

import os
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 1. CONFIGURATION =====
# --- TODO: Set the path to the CSV file you want to analyze ---
INPUT_FILE_PATH = "Attack_Balanced/SMOTE/training_balanced.csv"

# --- Outputs ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Feature_Selection_Reports")

REPORT_FILENAME = "feature_importance_report.txt"
PLOT_FILENAME = "top_50_features_global.png"
INVESTIGATION_PLOT_FILENAME = "investigation_of_top_feature.png"

# Sample cap to avoid unbounded memory usage (mirrors Django max-rows default)
MAX_SAMPLE_ROWS = 500_000


# ===== 2. HELPER FUNCTIONS =====

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
            print(f"  Counted {total:,} rows so far...")
    return total


def load_sampled_data(file_path, chunk_rows, max_rows, random_state=42):
    print("Counting rows to size the sample...")
    total_rows = count_rows(file_path, chunk_rows)
    if total_rows <= 0:
        return pd.DataFrame(), 0, 0

    frac = min(1.0, float(max_rows) / float(total_rows))
    sampled_chunks = []
    sampled_rows = 0

    print(f"Sampling data (target up to {max_rows:,} rows)...")
    for idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_rows, low_memory=False)):
        if frac < 1.0:
            sample = chunk.sample(frac=frac, random_state=random_state)
        else:
            sample = chunk
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


def build_xgb_params(device):
    params = {
        "n_estimators": 100,
        "n_jobs": -1,
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
    }
    if device == "gpu":
        params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
    return params


def fit_xgb_model(X, y, device):
    params = build_xgb_params(device)
    try:
        model = xgb.XGBClassifier(**params)
        model.fit(X, y)
        return model, device
    except Exception as exc:
        if device == "gpu":
            print(f"GPU training failed, falling back to CPU. Error: {exc}")
            params = build_xgb_params("cpu")
            model = xgb.XGBClassifier(**params)
            model.fit(X, y)
            return model, "cpu"
        raise


def analyze_per_label_importance(X, y, class_names, output_folder, device):
    print("\n" + "=" * 50)
    print(" PERFORMING PER-LABEL FEATURE IMPORTANCE ANALYSIS (One-vs-Rest)")
    print("=" * 50)

    unique_labels = sorted(pd.Series(y).unique())

    for label_val in unique_labels:
        current_class_name = class_names[label_val]
        safe_class_name = "".join(c for c in current_class_name if c.isalnum() or c in (" ", "_")).rstrip()

        print(f"\n--- Analyzing features for: '{current_class_name}' ---")
        y_binary = (y == label_val).astype(int)

        model_ovr, _ = fit_xgb_model(X, y_binary, device)
        importances = model_ovr.feature_importances_

        df_importance_ovr = pd.DataFrame({
            "feature": X.columns,
            "importance": importances
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        print(f"Top 10 features for identifying '{current_class_name}':")
        print(df_importance_ovr.head(10).to_string(index=False))

        plt.figure(figsize=(10, 8))
        sns.barplot(x="importance", y="feature", data=df_importance_ovr.head(20))
        plt.title(f"Top 20 Features for Attack: {current_class_name}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plot_path = make_unique_path(os.path.join(output_folder, f"top_features_{safe_class_name}.png"))
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot for '{current_class_name}' to: {plot_path}")


def write_cleaned_csv(input_path, output_path, cols_to_drop, chunk_rows, max_rows=None):
    rows_written = 0
    is_first_chunk = True
    for idx, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_rows, low_memory=False)):
        chunk.columns = chunk.columns.str.lower()
        chunk.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        if max_rows is not None:
            remaining = max_rows - rows_written
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
        if is_first_chunk:
            chunk.to_csv(output_path, index=False, mode="w")
            is_first_chunk = False
        else:
            chunk.to_csv(output_path, index=False, mode="a", header=False)
        rows_written += len(chunk)
        print(f"  Wrote chunk {idx + 1}, total rows saved: {rows_written:,}")
        if max_rows is not None and rows_written >= max_rows:
            break
    return rows_written


# ===== 3. MAIN SCRIPT LOGIC =====

def main():
    gpu_available, gpu_library = detect_gpu()
    device_choice = prompt_for_device(gpu_available)

    if not os.path.exists(INPUT_FILE_PATH):
        print(f"ERROR: Input file not found at '{INPUT_FILE_PATH}'")
        return

    chunk_mb = prompt_for_chunk_size_mb()
    chunk_rows = estimate_rows_per_chunk(INPUT_FILE_PATH, chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    report_path = make_unique_path(os.path.join(OUTPUT_FOLDER, REPORT_FILENAME))
    plot_path = make_unique_path(os.path.join(OUTPUT_FOLDER, PLOT_FILENAME))

    print(f"Loading data from '{os.path.basename(INPUT_FILE_PATH)}'...")
    data, total_rows, sampled_rows = load_sampled_data(INPUT_FILE_PATH, chunk_rows, MAX_SAMPLE_ROWS)
    if data.empty:
        print("ERROR: No data loaded from CSV.")
        return

    data.columns = data.columns.str.lower()
    if "label" not in data.columns:
        print("ERROR: 'label' column not found in the CSV file.")
        return

    X = data.drop(columns=["label"])
    y_raw = data["label"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    print("\nTraining XGBoost model for GLOBAL feature importances...")
    model, device_used = fit_xgb_model(X, y, device_choice)
    print("Model training complete.")

    importances = model.feature_importances_
    df_importance = pd.DataFrame({
        "feature": X.columns,
        "importance_pct": (importances / sum(importances)) * 100
    }).sort_values("importance_pct", ascending=False).reset_index(drop=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("XGBoost GLOBAL Feature Importance Report\n" + "=" * 40 + "\n")
        f.write(df_importance.to_string())
    print(f"\nSuccessfully saved full feature report to: {report_path}")

    plt.figure(figsize=(12, 14))
    sns.barplot(x="importance_pct", y="feature", data=df_importance.head(50))
    plt.title("Top 50 Most Important Features (Global)")
    plt.xlabel("Importance (%)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved plot of top 50 global features to: {plot_path}")
    plt.close()

    top_feature_name = df_importance.iloc[0]["feature"]
    print(f"\nNow investigating the top feature: '{top_feature_name}'...")
    plt.figure(figsize=(15, 8))
    sns.violinplot(x=y_raw, y=X[top_feature_name])
    plt.title(f"Distribution of Top Feature \"{top_feature_name}\" Across Labels")
    plt.ylabel(f"Value of {top_feature_name}")
    plt.xlabel("Attack Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    investigation_plot_path = make_unique_path(os.path.join(OUTPUT_FOLDER, INVESTIGATION_PLOT_FILENAME))
    plt.savefig(investigation_plot_path)
    print(f"Saved investigation plot to: {investigation_plot_path}")
    plt.close()

    rows_saved = 0
    cleaned_path = None

    zero_importance_features = df_importance[df_importance["importance_pct"] < 0.0001]
    if not zero_importance_features.empty:
        print(f"\nFound {len(zero_importance_features)} features with near-zero importance.")
        response = input("Do you want to remove them and save a new CSV? (y/n): ").lower().strip()
        if response in ["y", "yes"]:
            max_rows = prompt_for_max_rows()
            cols_to_drop = zero_importance_features["feature"].tolist()
            base_name = os.path.splitext(os.path.basename(INPUT_FILE_PATH))[0]
            cleaned_path = make_unique_path(os.path.join(OUTPUT_FOLDER, f"{base_name}_top_features.csv"))
            print("Saving cleaned CSV in chunks...")
            rows_saved = write_cleaned_csv(
                INPUT_FILE_PATH,
                cleaned_path,
                cols_to_drop,
                chunk_rows,
                max_rows=max_rows,
            )
            print(f"New file saved to: {cleaned_path}")
        else:
            print("No columns removed.")
    else:
        print("\nNo features with zero importance were found.")

    if input("\nDo you want to run per-label feature importance analysis? (y/n): ").strip().lower() in ["y", "yes"]:
        analyze_per_label_importance(X, y, le.classes_, OUTPUT_FOLDER, device_used)

    print("\nProcess finished.")

    output_paths = [report_path, plot_path, investigation_plot_path]
    if cleaned_path:
        output_paths.append(cleaned_path)

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Device used: {device_used.upper()}")
    print(f"Chunk size: {chunk_mb}MB (~{chunk_rows:,} rows)")
    print(f"Total rows processed: {total_rows:,}")
    if cleaned_path:
        print(f"Rows saved: {rows_saved:,}")
    else:
        print("Rows saved: N/A")
    print("Output paths:")
    for path in output_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()

# What changed:
# - Added GPU detection/device prompt and chunk size prompt with row estimation.
# - Streamed sampling to cap memory usage for outlier analysis.
# - Standardized outputs under ./outputs/Outliner_Detection with final summary.
#
# Purpose:
# - Detect IQR-based outliers in numeric columns.
# - Save per-column plots (if matplotlib is available).
# - Report outlier counts by column.

import pandas as pd
import os

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

FOLDER = "Training_2018"
FILENAME = "training_2_validated.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUT_FOLDER = os.path.join(OUTPUT_ROOT, "Outliner_Detection")

MAX_SAMPLE_ROWS = 500_000


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


def find_iqr_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (df[column] < lower) | (df[column] > upper)
    return mask, lower, upper


def main():
    gpu_available, _ = detect_gpu()
    device_choice = prompt_for_device(gpu_available)
    if device_choice == "gpu":
        print("GPU selected, but this script uses CPU-based pandas. Using CPU.")
    device_used = "cpu"

    file_path = os.path.join(FOLDER, FILENAME)
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    chunk_mb = prompt_for_chunk_size_mb()
    chunk_rows = estimate_rows_per_chunk(file_path, chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

    print("Loading sampled data...")
    df, total_rows, sampled_rows = load_sampled_data(file_path, chunk_rows, MAX_SAMPLE_ROWS)
    if df.empty:
        print("No data loaded.")
        return

    os.makedirs(OUT_FOLDER, exist_ok=True)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found for outlier detection.")
        return

    print(f"\nFound {len(numeric_cols)} numeric columns to analyze.")
    cols_with_outliers = []

    for col in numeric_cols:
        print(f"\nProcessing column: {col}")

        df_col = df.dropna(subset=[col]).copy()
        if df_col[col].nunique() <= 1:
            print(f"  Skipping column '{col}' (not enough unique values).")
            continue

        outlier_mask, lower, upper = find_iqr_outliers(df_col, col)
        n_outliers = outlier_mask.sum()
        print(f"  IQR thresholds -> Lower: {lower:.2f}, Upper: {upper:.2f}")
        print(f"  Found {n_outliers} outliers in '{col}'.")

        if n_outliers > 0:
            cols_with_outliers.append(col)
            if 'label' in df_col.columns:
                label_counts = df_col.loc[outlier_mask, 'label'].value_counts()
                print("  Outlier label counts:")
                for label, count in label_counts.items():
                    print(f"    {label}: {count}")
        else:
            print(f"  No outliers found for '{col}'.")

        if HAS_PLOT:
            plt.figure(figsize=(10, 6))
            if df_col[col].nunique() > 50:
                df_col[col].hist(bins=50, color="steelblue", edgecolor="black")
                plt.title(f"Histogram of '{col}'")
            else:
                counts = df_col[col].value_counts().sort_index()
                plt.bar(counts.index, counts.values, color="steelblue")
                plt.title(f"Bar Chart of '{col}' Value Counts")

            plt.axvline(lower, color='red', linestyle='--', label='Lower IQR Bound')
            plt.axvline(upper, color='green', linestyle='--', label='Upper IQR Bound')
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.legend(loc='upper right')
            plt.tight_layout()

            out_plot = os.path.join(OUT_FOLDER, f"{col}_plot.png")
            plt.savefig(out_plot)
            plt.close()
            print(f"  Saved plot to: {out_plot}")
        else:
            print("  Plot skipped (matplotlib not installed).")

    print("\n===========================")
    print(f"\nPlots saved in folder: {OUT_FOLDER}")
    print(f"\nColumns with outlier values: {cols_with_outliers}")
    print(f"Number of columns with outliers: {len(cols_with_outliers)}")

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Device used: {device_used.upper()}")
    print(f"Chunk size: {chunk_mb}MB (~{chunk_rows:,} rows)")
    print(f"Total rows processed: {total_rows:,}")
    print("Rows saved: N/A")
    print("Output paths:")
    print(f"  - {OUT_FOLDER}")


if __name__ == "__main__":
    main()

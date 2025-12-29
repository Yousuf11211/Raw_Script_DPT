# What changed:
# - Added GPU detection/device prompt and chunk size prompt with row estimation.
# - Streamed sampling for training/test to cap memory usage.
# - Standardized outputs under ./outputs/Tuning_Reports_2018 with final summary.
#
# Purpose:
# - Run GridSearchCV for RandomForest.
# - Save tuning results, summary report, model, and heatmap.
# - Evaluate best model on a sampled test set.

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# ===== 1. CONFIGURATION =====
TRAIN_FILE_PATH = "Balanced_Training_2018/training_data.csv"
TEST_FILE_PATH = "Separate_Test_Data/testing_data.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
MODEL_FOLDER = os.path.join(OUTPUT_ROOT, "Tuned_Models_2018")
REPORT_FOLDER = os.path.join(OUTPUT_ROOT, "Tuning_Reports_2018")

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


# ===== MAIN =====
if __name__ == "__main__":
    gpu_available, _ = detect_gpu()
    device_choice = prompt_for_device(gpu_available)
    if device_choice == "gpu":
        print("GPU selected, but RandomForest runs on CPU. Using CPU.")
    device_used = "cpu"

    if not os.path.exists(TRAIN_FILE_PATH):
        print(f"Error: Training file not found at '{TRAIN_FILE_PATH}'")
    elif not os.path.exists(TEST_FILE_PATH):
        print(f"Error: Testing file not found at '{TEST_FILE_PATH}'")
    else:
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        os.makedirs(REPORT_FOLDER, exist_ok=True)

        chunk_mb = prompt_for_chunk_size_mb()
        chunk_rows = estimate_rows_per_chunk(TRAIN_FILE_PATH, chunk_mb)
        print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

        print("Starting the tuning process...")
        base_name = os.path.basename(TRAIN_FILE_PATH).replace(".csv", "")

        print(f"Loading training data from: {TRAIN_FILE_PATH}")
        train_data, train_total_rows, _ = load_sampled_data(TRAIN_FILE_PATH, chunk_rows, MAX_SAMPLE_ROWS)
        train_data.columns = train_data.columns.str.lower()

        X_train = train_data.drop(columns=['label'])
        y_train = train_data['label']
        print(f"Training set shape: {X_train.shape}")
        print("-" * 50)

        print("\nStarting Grid Search for Random Forest...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2']
        }

        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            scoring='f1_macro',
            cv=3,
            verbose=2,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        print("\nGrid Search Complete!")
        print("\nBest Random Forest Parameters found:")
        print(grid_search.best_params_)
        print(f"Best cross-validation F1-macro score (from training data): {grid_search.best_score_:.4f}")
        print("-" * 50)

        print(f"\nLoading separate test data from: {TEST_FILE_PATH}...")
        test_data, test_total_rows, _ = load_sampled_data(TEST_FILE_PATH, chunk_rows, MAX_SAMPLE_ROWS)
        test_data.columns = test_data.columns.str.lower()
        X_test = test_data.drop(columns=['label'])
        y_test = test_data['label']

        print("\nEvaluating the best model on the unseen test set...")
        best_rf_model = grid_search.best_estimator_
        y_pred = best_rf_model.predict(X_test)

        final_report = classification_report(y_test, y_pred)
        print("\n--- Final Classification Report ---")
        print(final_report)

        report_path = os.path.join(REPORT_FOLDER, f"{base_name}_summary_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Tuning Report for {base_name}\n")
            f.write("=" * 40 + "\n")
            f.write("Best Parameters Found:\n")
            f.write(str(grid_search.best_params_))
            f.write(f"\n\nBest Cross-Validation F1-Macro Score: {grid_search.best_score_:.4f}\n")
            f.write("\n\n--- Final Classification Report on Separate Test Set ---\n")
            f.write(final_report)
        print(f"\nSaved summary report to: {report_path}")

        model_path = os.path.join(MODEL_FOLDER, f"{base_name}_tuned_model.pkl")
        joblib.dump(best_rf_model, model_path)
        print(f"Saved tuned model to: {model_path}")

        results_df = pd.DataFrame(grid_search.cv_results_)
        results_csv_path = os.path.join(REPORT_FOLDER, f"{base_name}_full_tuning_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Saved full tuning results table to: {results_csv_path}")

        heatmap_data = results_df.groupby(['param_max_depth', 'param_n_estimators']).mean(numeric_only=True)[
            'mean_test_score'].unstack()

        plt.figure(figsize=(10, 7))
        sns.heatmap(heatmap_data, annot=True, fmt=".4g", cmap="viridis")
        plt.title(f"Hyperparameter Tuning Heatmap for {base_name}")
        plt.xlabel("Number of Estimators (n_estimators)")
        plt.ylabel("Maximum Depth (max_depth)")
        plot_path = os.path.join(REPORT_FOLDER, f"{base_name}_tuning_heatmap.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved tuning heatmap to: {plot_path}")

        print("\nFinal Summary")
        print("-" * 40)
        print(f"Device used: {device_used.upper()}")
        print(f"Chunk size: {chunk_mb}MB (~{chunk_rows:,} rows)")
        print(f"Total rows processed: {(train_total_rows + test_total_rows):,}")
        print("Rows saved: N/A")
        print("Output paths:")
        for path in [report_path, model_path, results_csv_path, plot_path]:
            print(f"  - {path}")

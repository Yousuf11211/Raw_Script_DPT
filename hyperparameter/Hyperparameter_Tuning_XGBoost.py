# What changed:
# - Added GPU detection/device prompt and chunk size prompt with row estimation.
# - Streamed sampling for training/test to cap memory usage.
# - Standardized outputs under ./outputs/Tuning_XGBoost with final summary.
#
# Purpose:
# - Run manual grid search tuning for XGBoost.
# - Save tuning results, summary report, model, and heatmap.
# - Evaluate best model on a sampled test set.

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid, cross_validate
import xgboost as xgb
from sklearn.metrics import classification_report
import joblib
import time

# ===== 1. CONFIGURATION =====
TRAIN_FILE_PATH = "After_Feature_selection/training_balanced.csv"
TEST_FILE_PATH = "Test_Ready_2018/test.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
MODEL_FOLDER = os.path.join(OUTPUT_ROOT, "Tuning_XGBoost", "Tuned_Models_2018")
REPORT_FOLDER = os.path.join(OUTPUT_ROOT, "Tuning_XGBoost", "Tuning_Reports_2018")

MAX_SAMPLE_ROWS = 500_000


# ===== Helpers =====

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


def build_xgb_params(device, extra=None):
    params = {
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "n_jobs": 1,
    }
    if device == "gpu":
        params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
    if extra:
        params.update(extra)
    return params


def fit_xgb_model(X, y, device, params):
    try:
        model = xgb.XGBClassifier(**build_xgb_params(device, params))
        model.fit(X, y)
        return model, device
    except Exception as exc:
        if device == "gpu":
            print(f"GPU training failed, falling back to CPU. Error: {exc}")
            model = xgb.XGBClassifier(**build_xgb_params("cpu", params))
            model.fit(X, y)
            return model, "cpu"
        raise


# ===== MAIN =====
if __name__ == "__main__":
    gpu_available, _ = detect_gpu()
    device_choice = prompt_for_device(gpu_available)

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

        print("Starting the XGBoost tuning process...")
        base_name = os.path.basename(TRAIN_FILE_PATH).replace(".csv", "") + "_xgboost"
        results_csv_path = os.path.join(REPORT_FOLDER, f"{base_name}_full_tuning_results.csv")

        print(f"Loading training data from: {TRAIN_FILE_PATH}")
        train_data, train_total_rows, _ = load_sampled_data(TRAIN_FILE_PATH, chunk_rows, MAX_SAMPLE_ROWS)
        train_data.columns = train_data.columns.str.lower()

        X_train = train_data.drop(columns=['label'])
        y_train = train_data['label']

        print("Remapping labels for XGBoost (to be 0-indexed)...")
        unique_labels = sorted(y_train.unique())
        label_map = {label: i for i, label in enumerate(unique_labels)}
        y_train = y_train.map(label_map)

        print(f"Label map created and applied. {len(unique_labels)} classes found.")
        print(label_map)
        print(f"Training set shape: {X_train.shape}")
        print("-" * 50)

        print("\nStarting Manual Grid Search for XGBoost...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 20],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }

        param_list = list(ParameterGrid(param_grid))
        total_iterations = len(param_list)
        print(f"Total parameter combinations to test: {total_iterations}")

        results_list = []
        best_score = -1.0
        best_params = {}

        for i, params in enumerate(param_list):
            start_time = time.time()
            print(f"\n[Iteration {i + 1}/{total_iterations}] Testing params: {params}")

            model = xgb.XGBClassifier(**build_xgb_params(device_choice, params))

            cv_results = cross_validate(
                model,
                X_train,
                y_train,
                scoring='f1_macro',
                cv=3,
                n_jobs=2,
                verbose=0
            )

            mean_score = cv_results['test_score'].mean()
            std_score = cv_results['test_score'].std()
            fit_time = cv_results['fit_time'].mean()

            print(f"  -> F1-Macro: {mean_score:.4f} (±{std_score:.4f})")
            print(f"  -> Avg. Fit Time: {fit_time:.2f}s")

            current_result = {
                'mean_test_score': mean_score,
                'std_test_score': std_score,
                'mean_fit_time': fit_time
            }
            current_result.update({f'param_{k}': v for k, v in params.items()})
            results_list.append(current_result)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                print("  -> *** New Best Score Found! ***")

            results_df = pd.DataFrame(results_list)
            results_df.to_csv(results_csv_path, index=False)
            print(f"  -> Saved {len(results_list)} results to: {results_csv_path}")

        print("\nManual Grid Search Complete!")
        print("\nBest XGBoost Parameters found:")
        print(best_params)
        print(f"Best cross-validation F1-macro score (from training data): {best_score:.4f}")
        print("-" * 50)

        print("\nRefitting the best model on the entire training set...")
        best_xgb_model, device_used = fit_xgb_model(X_train, y_train, device_choice, best_params)
        print("Refit complete.")

        print(f"\nLoading separate test data from: {TEST_FILE_PATH}...")
        test_data, test_total_rows, _ = load_sampled_data(TEST_FILE_PATH, chunk_rows, MAX_SAMPLE_ROWS)
        test_data.columns = test_data.columns.str.lower()
        X_test = test_data.drop(columns=['label'])
        y_test = test_data['label']

        if 'label_map' in locals():
            print("Applying label map to test set...")
            y_test = y_test.map(label_map)

        print("\nEvaluating the best model on the unseen test set...")
        y_pred = best_xgb_model.predict(X_test)

        final_report = classification_report(y_test, y_pred)
        print("\n--- Final Classification Report ---")
        print(final_report)

        report_path = os.path.join(REPORT_FOLDER, f"{base_name}_summary_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Tuning Report for {base_name}\n")
            f.write("=" * 40 + "\n")
            f.write("Best Parameters Found:\n")
            f.write(str(best_params))
            f.write(f"\n\nBest Cross-Validation F1-Macro Score: {best_score:.4f}\n")
            f.write("\n\n--- Final Classification Report on Separate Test Set ---\n")
            f.write(final_report)
        print(f"\nSaved summary report to: {report_path}")

        model_path = os.path.join(MODEL_FOLDER, f"{base_name}_tuned_model.pkl")
        joblib.dump(best_xgb_model, model_path)
        print(f"Saved tuned model to: {model_path}")

        print(f"Full tuning results table is saved at: {results_csv_path}")

        try:
            heatmap_data = results_df.pivot_table(
                index='param_max_depth',
                columns='param_n_estimators',
                values='mean_test_score'
            )

            plt.figure(figsize=(10, 7))
            sns.heatmap(heatmap_data, annot=True, fmt=".4g", cmap="viridis")
            plt.title(f"Hyperparameter Tuning Heatmap for {base_name}")
            plt.xlabel("Number of Estimators (n_estimators)")
            plt.ylabel("Maximum Depth (max_depth)")
            plot_path = os.path.join(REPORT_FOLDER, f"{base_name}_tuning_heatmap.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved tuning heatmap to: {plot_path}")
        except Exception as e:
            print(f"\nCould not generate heatmap. Error: {e}")
            print("This can happen if your grid search only has one value for an axis.")

        print("\nFinal Summary")
        print("-" * 40)
        print(f"Device used: {device_used.upper()}")
        print(f"Chunk size: {chunk_mb}MB (~{chunk_rows:,} rows)")
        print(f"Total rows processed: {(train_total_rows + test_total_rows):,}")
        print("Rows saved: N/A")
        print("Output paths:")
        for path in [results_csv_path, report_path, model_path]:
            print(f"  - {path}")

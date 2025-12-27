import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid, cross_validate
import xgboost as xgb  # <-- Import XGBoost
from sklearn.metrics import classification_report
import joblib
import time

# ===== 1. CONFIGURATION =====
# --- TODO: Set the exact paths to your files and folders here ---
TRAIN_FILE_PATH = "After_Feature_selection/training_balanced.csv"  # Your single training file
TEST_FILE_PATH = "Test_Ready_2018/test.csv"  # Your single testing file

MODEL_FOLDER = "Tuning_XGBoost/Tuned_Models_2018"  # Where to save the final tuned model file
REPORT_FOLDER = "Tuning_XGBoost/Tuning_Reports_2018"  # Where to save reports, results, and plots

# Create the output folders if they do not already exist
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ===== 2. MAIN EXECUTION BLOCK =====
if __name__ == "__main__":
    # --- Check if files exist before starting ---
    if not os.path.exists(TRAIN_FILE_PATH):
        print(f"Error: Training file not found at '{TRAIN_FILE_PATH}'")
    elif not os.path.exists(TEST_FILE_PATH):
        print(f"Error: Testing file not found at '{TEST_FILE_PATH}'")
    else:
        print(f"Starting the XGBoost tuning process...")

        # --- Create a unique base name for XGBoost files ---
        base_name = os.path.basename(TRAIN_FILE_PATH).replace(".csv", "") + "_xgboost"
        print(f"Using base name for output files: {base_name}")

        # Define the file path for our checkpointed results
        results_csv_path = os.path.join(REPORT_FOLDER, f"{base_name}_full_tuning_results.csv")

        # --- Load TRAINING Data ---
        print(f"Loading training data from: {TRAIN_FILE_PATH}")
        train_data = pd.read_csv(TRAIN_FILE_PATH, low_memory=False)
        train_data.columns = train_data.columns.str.lower()

        X_train = train_data.drop(columns=['label'])
        y_train = train_data['label']

        # XGBoost requires labels to be 0-indexed if they are integers
        # If your labels are strings, this is fine. If they are numbers (e.g., 1, 2, 3),
        # they might need to be remapped to (0, 1, 2).
        # We can check and apply this if needed.
        # --- THIS IS THE NEW, CORRECTED CODE ---
        print("Remapping labels for XGBoost (to be 0-indexed)...")

        # Get all unique labels (strings like 'Bot', 'DDoS_HOIC', etc.)
        unique_labels = sorted(y_train.unique())

        # Create a mapping: {'Bot': 0, 'Brute_Force_SSH': 1, ...}
        label_map = {label: i for i, label in enumerate(unique_labels)}

        # Apply this mapping to y_train
        y_train = y_train.map(label_map)

        print(f"Label map created and applied. {len(unique_labels)} classes found.")
        print(label_map)

        print(f"Training set shape: {X_train.shape}")
        print("-" * 50)

        # --- Manual Hyperparameter Tuning with Checkpointing ---
        print("\nStarting Manual Grid Search for XGBoost...")
        # --- XGBoost-specific parameter grid ---
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 20],  # RF can be deep, XGB shallower
            'learning_rate': [0.01, 0.1],  # XGB-specific
            'subsample': [0.8, 1.0]  # XGB-specific
        }

        # Create a list of all parameter combinations
        param_list = list(ParameterGrid(param_grid))
        total_iterations = len(param_list)
        print(f"Total parameter combinations to test: {total_iterations}")

        results_list = []
        best_score = -1.0  # Initialize with a low score
        best_params = {}

        # --- Start the iteration loop ---
        for i, params in enumerate(param_list):
            start_time = time.time()
            print(f"\n[Iteration {i + 1}/{total_iterations}] Testing params: {params}")

            # 1. Initialize the model with current params
            model = xgb.XGBClassifier(
                random_state=42,
                use_label_encoder=False,  # Suppress warning
                eval_metric='mlogloss',  # 'mlogloss' for multiclass
                n_jobs=1,  # Let cross_validate handle parallel jobs
                # tree_method='gpu_hist', # UNCOMMENT this if you have a GPU
                **params
            )

            # 2. Run cross-validation
            cv_results = cross_validate(
                model,
                X_train,
                y_train,
                scoring='f1_macro',
                cv=3,
                n_jobs=2,  # Run 2 folds in parallel
                verbose=0
            )

            # 3. Calculate mean score and other metrics
            mean_score = cv_results['test_score'].mean()
            std_score = cv_results['test_score'].std()
            fit_time = cv_results['fit_time'].mean()

            print(f"  -> F1-Macro: {mean_score:.4f} (±{std_score:.4f})")
            print(f"  -> Avg. Fit Time: {fit_time:.2f}s")

            # 4. Store results
            current_result = {
                'mean_test_score': mean_score,
                'std_test_score': std_score,
                'mean_fit_time': fit_time
            }
            # Add parameter values to the dictionary for easy analysis
            current_result.update({f'param_{k}': v for k, v in params.items()})
            results_list.append(current_result)

            # 5. Check if this is the new best model
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                print(f"  -> *** New Best Score Found! ***")

            # 6. CHECKPOINT: Save cumulative results to CSV
            results_df = pd.DataFrame(results_list)
            results_df.to_csv(results_csv_path, index=False)
            print(f"  -> Saved {len(results_list)} results to: {results_csv_path}")

        print("\nManual Grid Search Complete!")
        print("\nBest XGBoost Parameters found:")
        print(best_params)
        print(f"Best cross-validation F1-macro score (from training data): {best_score:.4f}")
        print("-" * 50)

        # --- Refit the Best Model on the *Entire* Training Set ---
        print("\nRefitting the best model on the entire training set...")
        best_xgb_model = xgb.XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=1,
            # tree_method='gpu_hist', # UNCOMMENT this if you have a GPU
            **best_params
        )
        best_xgb_model.fit(X_train, y_train)
        print("Refit complete.")

        # --- Final Model Evaluation on the SEPARATE Test Set ---
        print(f"\nLoading separate test data from: {TEST_FILE_PATH}...")
        test_data = pd.read_csv(TEST_FILE_PATH, low_memory=False)
        test_data.columns = test_data.columns.str.lower()
        X_test = test_data.drop(columns=['label'])
        y_test = test_data['label']

        # Apply the same label mapping to the test set
        if 'label_map' in locals():
            print("Applying label map to test set...")
            y_test = y_test.map(label_map)

        print("\nEvaluating the best model on the unseen test set...")
        y_pred = best_xgb_model.predict(X_test)

        final_report = classification_report(y_test, y_pred)
        print("\n--- Final Classification Report ---")
        print(final_report)

        # --- Save All Results and Artifacts ---
        # 1. Save the summary text report
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

        # 2. Save the final, tuned model file
        model_path = os.path.join(MODEL_FOLDER, f"{base_name}_tuned_model.pkl")
        joblib.dump(best_xgb_model, model_path)
        print(f"Saved tuned model to: {model_path}")

        # 3. The full tuning results CSV is already saved!
        print(f"Full tuning results table is saved at: {results_csv_path}")

        # 4. Save a heatmap visualization of the tuning results
        # This will create a heatmap of max_depth vs n_estimators,
        # averaging the scores across the other dimensions (learning_rate, subsample)
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

        print("\n\nProcess finished successfully.")
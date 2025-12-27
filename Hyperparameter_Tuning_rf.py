import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# ===== 1. CONFIGURATION =====
# --- TODO: Set the exact paths to your files and folders here ---
TRAIN_FILE_PATH = "Balanced_Training_2018/training_data.csv"  # Your single training file
TEST_FILE_PATH = "Separate_Test_Data/testing_data.csv"  # Your single testing file

MODEL_FOLDER = "Tuned_Models_2018"  # Where to save the final tuned model file
REPORT_FOLDER = "Tuning_Reports_2018"  # Where to save reports, results, and plots

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
        print(f"Starting the tuning process...")
        base_name = os.path.basename(TRAIN_FILE_PATH).replace(".csv", "")

        # --- Load TRAINING Data ---
        print(f"Loading training data from: {TRAIN_FILE_PATH}")
        train_data = pd.read_csv(TRAIN_FILE_PATH, low_memory=False)
        train_data.columns = train_data.columns.str.lower()

        X_train = train_data.drop(columns=['label'])
        y_train = train_data['label']
        print(f"Training set shape: {X_train.shape}")
        print("-" * 50)

        # --- Hyperparameter Tuning with Grid Search ---
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

        # --- Final Model Evaluation on the SEPARATE Test Set ---
        print(f"\nLoading separate test data from: {TEST_FILE_PATH}...")
        test_data = pd.read_csv(TEST_FILE_PATH, low_memory=False)
        test_data.columns = test_data.columns.str.lower()
        X_test = test_data.drop(columns=['label'])
        y_test = test_data['label']

        print("\nEvaluating the best model on the unseen test set...")
        best_rf_model = grid_search.best_estimator_
        y_pred = best_rf_model.predict(X_test)

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
            f.write(str(grid_search.best_params_))
            f.write(f"\n\nBest Cross-Validation F1-Macro Score: {grid_search.best_score_:.4f}\n")
            f.write("\n\n--- Final Classification Report on Separate Test Set ---\n")
            f.write(final_report)
        print(f"\nSaved summary report to: {report_path}")

        # 2. Save the final, tuned model file
        model_path = os.path.join(MODEL_FOLDER, f"{base_name}_tuned_model.pkl")
        joblib.dump(best_rf_model, model_path)
        print(f"Saved tuned model to: {model_path}")

        # 3. Save the full, detailed results of every grid search combination
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_csv_path = os.path.join(REPORT_FOLDER, f"{base_name}_full_tuning_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Saved full tuning results table to: {results_csv_path}")

        # 4. Save a heatmap visualization of the tuning results
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

        print("\n\nProcess finished successfully.")
import os
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 1. CONFIGURATION =====
# --- TODO: Set the path to the CSV file you want to analyze ---
INPUT_FILE_PATH = "Attack_Balanced/ADASYN/your_balanced_attack_file.csv"

# --- TODO: Set the folder where you want to save the output files ---
OUTPUT_FOLDER = "Feature_Selection_Reports"

# --- Output filenames (you can change these if you like) ---
REPORT_FILENAME = "feature_importance_report.txt"
PLOT_FILENAME = "top_50_features.png"
CLEANED_FILENAME = "data_top_features.csv"


# ===== 2. HELPER FUNCTION =====
def get_user_yes_no(prompt):
    """A simple function to get a 'yes' or 'no' answer from the user."""
    while True:
        response = input(f"{prompt} (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


# ===== 3. MAIN SCRIPT LOGIC =====
def main():
    """
    Main function to run the feature selection process.
    """
    # --- Setup ---
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    report_path = os.path.join(OUTPUT_FOLDER, REPORT_FILENAME)
    plot_path = os.path.join(OUTPUT_FOLDER, PLOT_FILENAME)

    if not os.path.exists(INPUT_FILE_PATH):
        print(f"ERROR: Input file not found at '{INPUT_FILE_PATH}'")
        print("Please update the INPUT_FILE_PATH variable in the script.")
        return

    # --- Load and Prepare Data ---
    print(f"Loading data from '{os.path.basename(INPUT_FILE_PATH)}'...")
    data = pd.read_csv(INPUT_FILE_PATH, low_memory=False)
    data.columns = data.columns.str.lower()

    if 'label' not in data.columns:
        print("ERROR: 'label' column not found in the CSV file.")
        return

    X = data.drop(columns=['label'])
    y_raw = data['label']

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Encode any remaining text-based feature columns
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # --- Train XGBoost to Find Feature Importances ---
    print("\nTraining XGBoost model to calculate feature importances...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X, y)
    print("Model training complete.")

    # --- Calculate Percentage Importance and Create DataFrame ---
    importances = model.feature_importances_
    total_importance = sum(importances)

    df_importance = pd.DataFrame({
        'feature': X.columns,
        'importance_pct': (importances / total_importance) * 100
    }).sort_values('importance_pct', ascending=False).reset_index(drop=True)

    # --- Save the Full Feature Importance Report ---
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("XGBoost Feature Importance Report\n")
        f.write("=" * 40 + "\n")
        for index, row in df_importance.iterrows():
            f.write(f"{row['feature']:<40}: {row['importance_pct']:.4f}%\n")
    print(f"\nSuccessfully saved full feature importance report to: {report_path}")

    # --- Create and Save a Plot of Top 50 Features ---
    plt.figure(figsize=(12, 14))
    sns.barplot(x='importance_pct', y='feature', data=df_importance.head(50))
    plt.title('Top 50 Most Important Features')
    plt.xlabel('Importance (%)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved plot of top 50 features to: {plot_path}")
    plt.close()

    # --- Analyze and Remove Zero-Importance Features ---
    zero_importance_features = df_importance[df_importance['importance_pct'] == 0]
    num_zero = len(zero_importance_features)

    if num_zero > 0:
        print(f"\nAnalysis complete: Found {num_zero} features with 0.0000% importance.")
        if get_user_yes_no("Do you want to remove these columns and save a new CSV file?"):
            cols_to_drop = zero_importance_features['feature'].tolist()
            data_cleaned = data.drop(columns=cols_to_drop)

            # Determine the output path for the cleaned file
            base_name = os.path.splitext(os.path.basename(INPUT_FILE_PATH))[0]
            cleaned_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_top_features.csv")

            data_cleaned.to_csv(cleaned_path, index=False)
            print(f"\nSuccessfully removed {num_zero} columns.")
            print(f"New file with top features saved to: {cleaned_path}")
        else:
            print("\nNo columns were removed.")
    else:
        print("\nAnalysis complete: No features with zero importance were found.")

    print("\nProcess finished.")


if __name__ == "__main__":
    main()
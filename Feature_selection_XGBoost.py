import os
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 1. CONFIGURATION =====
# --- TODO: Set the path to the CSV file you want to analyze ---
# Since your data only contains attacks, a good name might be 'all_attacks.csv'
INPUT_FILE_PATH = "Attack_Balanced/SMOTE/training_balanced.csv"

# --- TODO: Set the folder where you want to save the output files ---
OUTPUT_FOLDER = "Feature_Selection_Reports"

# --- Output filenames (you can change these if you like) ---
REPORT_FILENAME = "feature_importance_report.txt"
PLOT_FILENAME = "top_50_features_global.png"
INVESTIGATION_PLOT_FILENAME = "investigation_of_top_feature.png"


# ===== 2. HELPER FUNCTIONS =====
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


def analyze_per_label_importance(X, y, class_names, output_folder):
    """
    Trains a One-vs-Rest model for each class to find per-label feature importance.
    """
    print("\n" + "=" * 50)
    print(" PERFORMING PER-LABEL FEATURE IMPORTANCE ANALYSIS (One-vs-Rest)")
    print("=" * 50)

    unique_labels = sorted(pd.Series(y).unique())

    for label_val in unique_labels:
        current_class_name = class_names[label_val]
        # Clean the class name to be a valid filename
        safe_class_name = "".join(c for c in current_class_name if c.isalnum() or c in (' ', '_')).rstrip()

        print(f"\n--- Analyzing features for: '{current_class_name}' ---")

        y_binary = (y == label_val).astype(int)

        model_ovr = xgb.XGBClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model_ovr.fit(X, y_binary)

        importances = model_ovr.feature_importances_

        df_importance_ovr = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        print(f"Top 10 features for identifying '{current_class_name}':")
        print(df_importance_ovr.head(10).to_string(index=False))

        # Plot and save the top 20 features for this label
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=df_importance_ovr.head(20))
        plt.title(f'Top 20 Features for Attack: {current_class_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plot_path = os.path.join(output_folder, f"top_features_{safe_class_name}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot for '{current_class_name}' to: {plot_path}")


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

    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # --- Train XGBoost for GLOBAL Feature Importances ---
    print("\nTraining XGBoost model for GLOBAL feature importances...")
    model = xgb.XGBClassifier(
        n_estimators=100, n_jobs=-1, random_state=42,
        use_label_encoder=False, eval_metric='mlogloss'
    )
    model.fit(X, y)
    print("Model training complete.")

    importances = model.feature_importances_
    df_importance = pd.DataFrame({
        'feature': X.columns,
        'importance_pct': (importances / sum(importances)) * 100
    }).sort_values('importance_pct', ascending=False).reset_index(drop=True)

    # --- Save Full GLOBAL Report ---
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("XGBoost GLOBAL Feature Importance Report\n" + "=" * 40 + "\n")
        f.write(df_importance.to_string())
    print(f"\nSuccessfully saved full feature report to: {report_path}")

    # --- Create and Save GLOBAL Plot ---
    plt.figure(figsize=(12, 14))
    sns.barplot(x='importance_pct', y='feature', data=df_importance.head(50))
    plt.title('Top 50 Most Important Features (Global)')
    plt.xlabel('Importance (%)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved plot of top 50 global features to: {plot_path}")
    plt.close()

    # --- 🎯 NEW: INVESTIGATE THE TOP FEATURE ---
    top_feature_name = df_importance.iloc[0]['feature']
    print(f"\nNow investigating the top feature: '{top_feature_name}'...")
    plt.figure(figsize=(15, 8))
    sns.violinplot(x=y_raw, y=X[top_feature_name])
    plt.title(f'Distribution of Top Feature "{top_feature_name}" Across Labels')
    plt.ylabel(f'Value of {top_feature_name}')
    plt.xlabel('Attack Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    investigation_plot_path = os.path.join(OUTPUT_FOLDER, INVESTIGATION_PLOT_FILENAME)
    plt.savefig(investigation_plot_path)
    print(f"Saved investigation plot to: {investigation_plot_path}")
    plt.close()

    # --- Analyze and Remove Zero-Importance Features ---
    zero_importance_features = df_importance[df_importance['importance_pct'] < 0.0001]
    if not zero_importance_features.empty:
        print(f"\nFound {len(zero_importance_features)} features with near-zero importance.")
        if get_user_yes_no("Do you want to remove them and save a new CSV?"):
            cols_to_drop = zero_importance_features['feature'].tolist()
            data_cleaned = data.drop(columns=cols_to_drop)
            base_name = os.path.splitext(os.path.basename(INPUT_FILE_PATH))[0]
            cleaned_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_top_features.csv")
            data_cleaned.to_csv(cleaned_path, index=False)
            print(f"New file saved to: {cleaned_path}")
        else:
            print("No columns removed.")
    else:
        print("\nNo features with zero importance were found.")

    # --- 🎯 NEW: RUN PER-LABEL ANALYSIS ---
    if get_user_yes_no("\nDo you want to run per-label feature importance analysis?"):
        analyze_per_label_importance(X, y, le.classes_, OUTPUT_FOLDER)

    print("\nProcess finished.")


if __name__ == "__main__":
    main()
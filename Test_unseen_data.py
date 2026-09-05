import pandas as pd
import numpy as np
import joblib
import os

# ===== CONFIGURATION =====
# 1. Point this to your saved .pkl model file
MODEL_PATH = r"C:\Users\Yousuf\Desktop\Raw_Script_DPT\model_training\outputs\Model_Random_Forest\models\model1_final_training_data_model.pkl"

# 2. Point this to the COMPLETELY NEW CSV file (the one with extra features)
NEW_CSV_PATH = r"Benign/Benign_part_9.csv"


def predict_unseen_data():
    print("1. Loading the trained Random Forest model...")
    rf_model = joblib.load(MODEL_PATH)

    # The model remembers exactly which 49 columns it needs!
    expected_columns = rf_model.feature_names_in_
    print(f"   [+] Model is looking for exactly {len(expected_columns)} specific features.")

    print("\n2. Loading the new CSV file...")
    # low_memory=False prevents crashing on mixed data types
    df_new = pd.read_csv(NEW_CSV_PATH, low_memory=False)
    df_new.columns = df_new.columns.str.lower()
    print(f"   [+] New file loaded. It has a massive {len(df_new.columns)} total features.")

    # --- SAFETY CHECK ---
    missing_cols = [col for col in expected_columns if col not in df_new.columns]
    if missing_cols:
        print(f"\n[CRITICAL ERROR] The new CSV is missing these required columns: {missing_cols}")
        return

    print("\n3. Stripping extra columns and keeping only what the model needs...")
    # This automatically drops all the extra features!
    X_new = df_new[expected_columns].copy()

    print("4. Cleaning NaN and Infinity values in RAM...")
    X_new.replace([np.inf, -np.inf], 0, inplace=True)
    X_new.fillna(0, inplace=True)

    # Handle any potential object/string columns safely just in case
    for col in X_new.select_dtypes(include='object').columns:
        # Convert non-numeric columns to numbers, forcing errors to NaN, then to 0
        X_new[col] = pd.to_numeric(X_new[col], errors='coerce').fillna(0)

    print("\n5. Running Predictions on the unseen data...")
    predictions = rf_model.predict(X_new)

    # Attach predictions back to the dataset so you can read it
    df_new['IDS_Prediction'] = predictions

    print("\n" + "=" * 50)
    print("NETWORK TRAFFIC PREDICTION RESULTS")
    print("=" * 50)

    # Count the Attack vs Benign results
    results = df_new['IDS_Prediction'].value_counts()
    for label_num, count in results.items():
        # Referencing your label mapping (Usually 0 = Benign, 1 = Attack)
        label_name = "Attack" if label_num == 1 else "Benign"
        print(f"Detected {label_name} (Class {label_num}): {count:,} packets")

    print("=" * 50)

    # Save a final report CSV with the predictions added to the end
    output_filename = "Live_Traffic_Predictions.csv"
    df_new.to_csv(output_filename, index=False)
    print(f"\nFull report saved to: {output_filename}")


if __name__ == "__main__":
    predict_unseen_data()
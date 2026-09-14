import pandas as pd
import numpy as np
import joblib
import os

# ===== CONFIGURATION =====
MODEL_PATH = r"C:\Users\Yousuf\Desktop\Raw_Script_DPT\model_training\outputs\Model_Random_Forest\models\model2_multiclass_training_data_model.pkl"

# CHANGE THIS TO AN UNSEEN ATTACK FILE! Model 2 does not know what Benign traffic is.
NEW_CSV_PATH = r"Attacks/thursday_01_03_2018_infiltration.csv"

# Your exact Model 2 Mapping
MULTICLASS_MAPPING = {
    0: 'Bot',
    1: 'Brute_Force_FTP',
    2: 'Brute_Force_SSH',
    3: 'Brute_Force_Web',
    4: 'DDoS_HOIC',
    5: 'DDoS_LOIC_HTTP',
    6: 'DoS_Golden_Eye',
    7: 'DoS_HULK',
    8: 'DoS_SlowHTTP',
    9: 'DoS_Slowloris'
}


def predict_unseen_data():
    print("1. Loading the trained Multiclass Random Forest model...")
    rf_model = joblib.load(MODEL_PATH)

    expected_columns = rf_model.feature_names_in_
    print(f"   [+] Model is looking for exactly {len(expected_columns)} specific features.")

    print(f"\n2. Loading the new CSV file: {NEW_CSV_PATH}")
    df_new = pd.read_csv(NEW_CSV_PATH, low_memory=False)
    df_new.columns = df_new.columns.str.lower()

    missing_cols = [col for col in expected_columns if col not in df_new.columns]
    if missing_cols:
        print(f"\n[CRITICAL ERROR] The new CSV is missing these required columns: {missing_cols}")
        return

    print("\n3. Stripping extra columns...")
    X_new = df_new[expected_columns].copy()

    print("4. Cleaning NaN and Infinity values...")
    X_new.replace([np.inf, -np.inf], 0, inplace=True)
    X_new.fillna(0, inplace=True)

    for col in X_new.select_dtypes(include='object').columns:
        X_new[col] = pd.to_numeric(X_new[col], errors='coerce').fillna(0)

    print("\n5. Running Multiclass Predictions with Confidence Threshold...")
    # Get the percentage confidence for every single class
    probabilities = rf_model.predict_proba(X_new)
    max_confidence = np.max(probabilities, axis=1)
    raw_predictions = rf_model.predict(X_new)

    # SET YOUR THRESHOLD HERE (0.60 = 60% confidence)
    CONFIDENCE_THRESHOLD = 0.60

    final_predictions = []
    for i in range(len(raw_predictions)):
        # If the model is not confident enough, flag it as a Zero-Day/Unknown attack
        if max_confidence[i] < CONFIDENCE_THRESHOLD:
            final_predictions.append("Unknown_Zero_Day_Attack")
        else:
            # Otherwise, use the mapping dictionary to get the real attack name
            label_num = raw_predictions[i]
            mapped_name = MULTICLASS_MAPPING.get(label_num, f"Unknown_Class_{label_num}")
            final_predictions.append(mapped_name)

    # Save the readable string names back to the dataset
    df_new['IDS_Prediction'] = final_predictions

    print("\n" + "=" * 50)
    print("MULTICLASS TRAFFIC PREDICTION RESULTS")
    print("=" * 50)

    # Because we already mapped the names, we can just print the value counts directly!
    results = df_new['IDS_Prediction'].value_counts()
    for attack_name, count in results.items():
        print(f"Detected {attack_name}: {count:,} packets")

    print("=" * 50)

    output_filename = "Live_Multiclass_Predictions.csv"
    df_new.to_csv(output_filename, index=False)
    print(f"\nFull report saved to: {output_filename}")


if __name__ == "__main__":
    predict_unseen_data()
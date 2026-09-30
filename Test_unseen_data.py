import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# Saved trained Random Forest model
MODEL_PATH = Path(
    r"C:\Users\Yousuf\Desktop\Raw_Script_DPT\model_training\outputs"
    r"\Model_Random_Forest\models\model2_multiclass_training_data_model.pkl"
)

# Folder containing this Python script
SCRIPT_DIR = Path(__file__).resolve().parent

# test_tcp.csv must be in the same folder as this script
NEW_CSV_PATH = SCRIPT_DIR / "ssh_bruteforce.csv"

# Prediction report will also be saved in the same folder
OUTPUT_PATH = SCRIPT_DIR / "Multiclass_Live_Traffic_Predictions.csv"


# ============================================================
# COLUMN NAME NORMALIZATION
# ============================================================

def canonical_column_name(column_name):
    """
    Normalize NTLFlowLyzer column names so they match the
    feature names used when the model was trained.
    """

    # Convert to lowercase and remove accidental spaces
    name = str(column_name).strip().lower()

    # Current NTLFlowLyzer names -> training dataset names
    aliases = {
        "segment_size_mean": "avg_segment_size",
        "fwd_segment_size_mean": "fwd_avg_segment_size",
    }

    return aliases.get(name, name)


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_unseen_data():

    print("=" * 70)
    print("NIDS - UNSEEN NETWORK TRAFFIC PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check files
    # --------------------------------------------------------

    print("\n1. Checking required files...")

    if not MODEL_PATH.exists():
        print(f"\n[CRITICAL ERROR] Model file not found:")
        print(MODEL_PATH)
        return

    if not NEW_CSV_PATH.exists():
        print(f"\n[CRITICAL ERROR] CSV file not found:")
        print(NEW_CSV_PATH)
        print("\nPlace test_tcp.csv in the same folder as this script.")
        return

    print(f"   [+] Model found")
    print(f"   [+] CSV found: {NEW_CSV_PATH.name}")

    # --------------------------------------------------------
    # 2. Load trained model
    # --------------------------------------------------------

    print("\n2. Loading trained Random Forest model...")

    try:
        rf_model = joblib.load(MODEL_PATH)
        print("\nTop 20 most important features learned by the model:")
        print("-" * 70)

        feature_importance = pd.DataFrame({
            "feature": rf_model.feature_names_in_,
            "importance": rf_model.feature_importances_
        })

        feature_importance = feature_importance.sort_values(
            by="importance",
            ascending=False
        )

        print(feature_importance.head(20).to_string(index=False))
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Could not load model:")
        print(e)
        return

    # sklearn models trained using a DataFrame normally remember
    # the exact feature names used during training
    if not hasattr(rf_model, "feature_names_in_"):
        print("\n[CRITICAL ERROR]")
        print("The model does not contain 'feature_names_in_'.")
        print("The script cannot safely determine the exact training features.")
        return

    original_expected_columns = list(rf_model.feature_names_in_)

    # Normalize a copy for comparison against the new NTLFlowLyzer CSV
    expected_columns = [
        canonical_column_name(col)
        for col in original_expected_columns
    ]

    print(
        f"   [+] Model expects exactly "
        f"{len(original_expected_columns)} predictors."
    )

    # --------------------------------------------------------
    # 3. Check for target leakage
    # --------------------------------------------------------

    if "label" in expected_columns:
        print("\n" + "!" * 70)
        print("[CRITICAL ERROR: TARGET LEAKAGE]")
        print("The trained model expects 'label' as an input feature.")
        print()
        print("The label is the value the model is supposed to predict.")
        print("It must NOT be part of the model input.")
        print()
        print("This model should be retrained before live NIDS testing.")
        print("!" * 70)
        return

    print("   [+] 'label' is NOT being used as a model predictor.")

    # --------------------------------------------------------
    # 4. Load NTLFlowLyzer CSV
    # --------------------------------------------------------

    print("\n3. Loading NTLFlowLyzer CSV...")

    try:
        df_original = pd.read_csv(
            NEW_CSV_PATH,
            low_memory=False
        )
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Could not read CSV:")
        print(e)
        return

    print(f"   [+] Flows found: {len(df_original):,}")
    print(f"   [+] CSV columns: {len(df_original.columns):,}")

    if df_original.empty:
        print("\n[CRITICAL ERROR] The CSV contains no flows.")
        return

    # Keep original 347-column dataframe for final output report
    df_new = df_original.copy()

    # --------------------------------------------------------
    # 5. Normalize NTLFlowLyzer column names
    # --------------------------------------------------------

    print("\n4. Normalizing NTLFlowLyzer feature names...")

    df_new.columns = [
        canonical_column_name(col)
        for col in df_new.columns
    ]

    # Check whether normalization accidentally produced duplicates
    duplicated_columns = df_new.columns[
        df_new.columns.duplicated()
    ].tolist()

    if duplicated_columns:
        print("\n[CRITICAL ERROR]")
        print("Duplicate columns appeared after normalization:")

        for col in duplicated_columns:
            print(f"   - {col}")

        return

    print("   [+] Column names normalized successfully.")

    # --------------------------------------------------------
    # 6. Verify all required model features exist
    # --------------------------------------------------------

    print("\n5. Checking model features against the new CSV...")

    missing_columns = [
        col
        for col in expected_columns
        if col not in df_new.columns
    ]

    if missing_columns:
        print(
            f"\n[CRITICAL ERROR] "
            f"{len(missing_columns)} required feature(s) are missing:"
        )

        for col in missing_columns:
            print(f"   - {col}")

        print("\nPrediction stopped.")
        return

    print(
        f"   [+] All {len(expected_columns)} required "
        f"model predictors are available."
    )

    # --------------------------------------------------------
    # 7. Select features in exact training order
    # --------------------------------------------------------

    print("\n6. Preparing model input...")

    # Very important:
    # Use the exact feature order remembered by the trained model.
    X_new = df_new[expected_columns].copy()

    # Restore the exact original names stored in the sklearn model.
    # This prevents sklearn's feature-name validation from complaining
    # if capitalization was different during training.
    X_new.columns = original_expected_columns

    print(f"   [+] Model input shape: {X_new.shape}")

    # --------------------------------------------------------
    # 8. Convert model input to numeric values
    # --------------------------------------------------------

    print("\n7. Cleaning numeric values...")

    for col in X_new.columns:
        X_new[col] = pd.to_numeric(
            X_new[col],
            errors="coerce"
        )

    # Infinity should not be passed to sklearn
    X_new.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    missing_value_count = int(
        X_new.isna().sum().sum()
    )

    if missing_value_count > 0:
        print(
            f"   [!] Found {missing_value_count:,} "
            f"NaN/Infinity value(s)."
        )

        print(
            "   [!] Replacing them with 0 "
            "to match the current testing strategy."
        )

        X_new.fillna(0, inplace=True)

    else:
        print("   [+] No NaN or Infinity values found.")

    # --------------------------------------------------------
    # 9. Run predictions
    # --------------------------------------------------------

    print("\n8. Running Random Forest predictions...")
    # --------------------------------------------------------
    # Live SSH feature diagnostic
    # --------------------------------------------------------

    print("\nLive forward-SSH feature diagnostic:")
    print("-" * 90)

    top_features = feature_importance.head(20)["feature"].tolist()

    forward_ssh_mask = (
            (df_original["src_ip"].astype(str) == "10.10.10.10")
            & (df_original["dst_ip"].astype(str) == "10.10.20.20")
            & (pd.to_numeric(df_original["dst_port"], errors="coerce") == 22)
    )

    forward_ssh = X_new.loc[forward_ssh_mask, top_features]

    print(f"Forward SSH flows found: {len(forward_ssh)}")

    live_stats = forward_ssh.agg(
        ["min", "median", "max"]
    ).T

    live_stats["unique_values"] = forward_ssh.nunique()

    live_stats["zero_percent"] = (
            (forward_ssh == 0).mean() * 100
    )

    print(live_stats.to_string())
    print("\nChecking for abnormal backward IAT values:")
    print("-" * 100)

    iat_debug = pd.DataFrame({
        "src_ip": df_original["src_ip"],
        "src_port": df_original["src_port"],
        "dst_ip": df_original["dst_ip"],
        "dst_port": df_original["dst_port"],

        "duration": X_new["duration"],
        "bwd_packets_count": X_new["bwd_packets_count"],

        "packet_iat_total": X_new["packet_iat_total"],
        "bwd_packets_iat_total": X_new["bwd_packets_iat_total"],
        "bwd_packets_iat_mean": X_new["bwd_packets_iat_mean"],
        "bwd_packets_iat_max": X_new["bwd_packets_iat_max"],
        "bwd_packets_iat_min": X_new["bwd_packets_iat_min"]
    })

    abnormal_iat = iat_debug[
        (iat_debug["bwd_packets_iat_total"] > 1000)
        | (
                iat_debug["bwd_packets_iat_total"]
                > iat_debug["duration"] * 10
        )
        ]

    print(f"Abnormal flows found: {len(abnormal_iat)}")

    if len(abnormal_iat) > 0:
        print(abnormal_iat.to_string(index=False))
    else:
        print("No abnormal backward IAT values found.")
    try:
        predictions = rf_model.predict(X_new)
    except Exception as e:
        print("\n[CRITICAL ERROR] Prediction failed:")
        print(e)
        return

    print("   [+] Prediction completed successfully.")

    # Add prediction to original NTLFlowLyzer dataframe
    df_original["IDS_Prediction"] = predictions

    # --------------------------------------------------------
    # 10. Prediction confidence
    # --------------------------------------------------------

    if hasattr(rf_model, "predict_proba"):

        try:
            probabilities = rf_model.predict_proba(X_new)

            # ------------------------------------------------
            # Class 1 probability diagnostic
            # ------------------------------------------------

            class_1_index = list(rf_model.classes_).index(1)
            class_1_probability = probabilities[:, class_1_index]

            print("\nClass 1 probability statistics:")
            print(
                pd.Series(class_1_probability).describe(
                    percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
                )
            )

            print("\nTop 10 highest Class 1 probabilities:")
            print(
                np.sort(class_1_probability)[-10:][::-1]
            )

            # Existing confidence calculation
            max_confidence = np.max(
                probabilities,
                axis=1
            )

            probability_debug = pd.DataFrame({
                "src_ip": df_original["src_ip"],
                "src_port": df_original["src_port"],
                "dst_ip": df_original["dst_ip"],
                "dst_port": df_original["dst_port"],
                "prediction": predictions,
                "class_1_probability": class_1_probability
            })

            print("\nTop 15 flows by Class 1 probability:")
            print(
                probability_debug
                .sort_values(
                    by="class_1_probability",
                    ascending=False
                )
                .head(15)
                .to_string(index=False)
            )

            df_original["IDS_Confidence"] = max_confidence

            print(
                "   [+] Prediction confidence calculated."
            )

        except Exception as e:
            print(
                f"   [!] Could not calculate confidence: {e}"
            )

    # --------------------------------------------------------
    # 11. Display prediction results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("NETWORK TRAFFIC PREDICTION RESULTS")
    print("=" * 70)

    total_flows = len(predictions)

    print(f"\nTotal flows analyzed: {total_flows:,}")

    results = pd.Series(
        predictions
    ).value_counts()

    print("\nPrediction distribution:")

    for predicted_class, count in results.items():

        percentage = (
            (count / total_flows) * 100
            if total_flows > 0
            else 0
        )

        print(
            f"   Class {predicted_class}: "
            f"{count:,} flow(s) "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # 12. Display all classes known by the model
    # --------------------------------------------------------

    print("\nClasses known by the trained model:")

    if hasattr(rf_model, "classes_"):

        for model_class in rf_model.classes_:
            print(f"   - {model_class}")

    else:
        print("   Class information unavailable.")

    # --------------------------------------------------------
    # 13. Show individual flow predictions
    # --------------------------------------------------------

    print("\nIndividual flow predictions:")
    print("-" * 70)

    for index, prediction in enumerate(predictions):

        flow_number = index + 1

        # Get useful network information when available
        src_ip = (
            df_original.iloc[index]["src_ip"]
            if "src_ip" in df_original.columns
            else "Unknown"
        )

        dst_ip = (
            df_original.iloc[index]["dst_ip"]
            if "dst_ip" in df_original.columns
            else "Unknown"
        )

        dst_port = (
            df_original.iloc[index]["dst_port"]
            if "dst_port" in df_original.columns
            else "Unknown"
        )

        print(
            f"Flow {flow_number}: "
            f"{src_ip} -> {dst_ip}:{dst_port} "
            f"=> Class {prediction}"
        )

    # --------------------------------------------------------
    # 14. Save final report
    # --------------------------------------------------------

    print("\n9. Saving prediction report...")

    try:
        df_original.to_csv(
            OUTPUT_PATH,
            index=False
        )
    except Exception as e:
        print("\n[CRITICAL ERROR] Could not save output:")
        print(e)
        return

    print("\n" + "=" * 70)
    print("PREDICTION COMPLETE")
    print("=" * 70)

    print(f"\nFull report saved to:")
    print(OUTPUT_PATH)

    print(
        f"\nThe report contains the original "
        f"{len(df_original.columns) - 1} columns plus "
        f"the IDS prediction result."
    )

    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    predict_unseen_data()
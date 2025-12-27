import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# --- CONFIG ---
model_path = "Model_2018/full_model.pkl"
label_mapping_path = "Model_2018/full_label_mapping.txt"
test_csv_path = "Balanced_Test_2018/full.csv"
output_folder = "Test_Reports"
os.makedirs(output_folder, exist_ok=True)
# --------------

def load_label_mapping(path):
    """Loads the label mapping from a text file."""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()[2:]  # skip header + separator
        for line in lines:
            if ":" not in line:  # skip blank/invalid lines
                continue
            cls, num = line.strip().split(":")
            cls, num = cls.strip(), num.strip()
            if num.isdigit():  # only process valid numbers
                mapping[int(num)] = cls
    return mapping

def main():
    # --- USER PROMPTS ---
    save_report = input("Save classification report? (y/n): ").strip().lower() == "y"
    save_cm = input("Save confusion matrix? (y/n): ").strip().lower() == "y"
    save_preds_csv = input("Save full predictions CSV? (y/n): ").strip().lower() == "y"
    save_counts_summary = input("Save prediction counts summary? (y/n): ").strip().lower() == "y"

    # Load model
    rf = joblib.load(model_path)
    print("Model loaded.")

    # Load label mapping
    mapping = load_label_mapping(label_mapping_path)
    classes_by_index = [mapping[i] for i in sorted(mapping.keys())]
    print("Label mapping loaded:", classes_by_index)

    # Load test data
    test_df = pd.read_csv(test_csv_path)
    test_df.columns = test_df.columns.str.strip().str.lower()
    print(f"Test data loaded: {test_df.shape[0]} rows")

    # Separate features and labels
    if 'label' in test_df.columns:
        X_test = test_df.drop(columns=['label'])
        X_test = X_test.reindex(columns=rf.feature_names_in_, fill_value=0)
        y_test_raw = test_df['label'].values
        inv_mapping = {v: k for k, v in mapping.items()}
        y_test = np.array([inv_mapping.get(lbl, -1) for lbl in y_test_raw])
    else:
        X_test = test_df.copy()
        y_test = None

    # Predict
    y_pred = rf.predict(X_test)
    y_pred_labels = [mapping[num] for num in y_pred]

    # Prediction counts
    attack_counts = Counter(y_pred_labels)
    print("\nPredicted attack counts:")
    for attack, count in attack_counts.items():
        print(f"{attack:<20}: {count}")

    # Save prediction counts summary if requested
    if save_counts_summary:
        base_name = os.path.splitext(os.path.basename(test_csv_path))[0]
        counts_path = os.path.join(output_folder, f"{base_name}_predicted_counts.txt")
        with open(counts_path, "w", encoding="utf-8") as f:
            for attack, count in sorted(attack_counts.items()):
                f.write(f"{attack:<20}: {count}\n")
        print(f"Prediction counts summary saved -> {counts_path}")

    # Reports
    base_name = os.path.splitext(os.path.basename(test_csv_path))[0]

    # Classification Report & Confusion Matrix
    if y_test is not None and (y_test >= 0).all():
        report = classification_report(y_test, y_pred, target_names=classes_by_index, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=classes_by_index, columns=classes_by_index)

        if save_report:
            report_path = os.path.join(output_folder, f"{base_name}_report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Classification report saved -> {report_path}")
        else:
            print("\nClassification Report:")
            print(report)

        if save_cm:
            cm_path = os.path.join(output_folder, f"{base_name}_confusion_matrix.csv")
            cm_df.to_csv(cm_path)
            print(f"Confusion matrix saved -> {cm_path}")
        else:
            print("\nConfusion Matrix:")
            print(cm_df)
    else:
        print("[info] No ground-truth labels in test file, skipping report generation.")

    # Save full predictions to CSV
    if save_preds_csv:
        preds_path = os.path.join(output_folder, f"{base_name}_predictions.csv")
        # Create a DataFrame of predictions to save
        output_df = test_df.copy()
        output_df['predicted_label'] = y_pred_labels
        output_df.to_csv(preds_path, index=False)
        print(f"Full predictions saved -> {preds_path}")

if __name__ == "__main__":
    main()
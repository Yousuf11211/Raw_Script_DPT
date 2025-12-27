import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import sklearn
print(f"Scikit-learn Version: {sklearn.__version__}")


# ===== CONFIGURATION =====
input_folder = "Training_Attack"  # Folder containing your CSV files
model_folder = "Attack_Model"
report_folder = "Model_report"
os.makedirs(model_folder, exist_ok=True)
os.makedirs(report_folder, exist_ok=True)

train_full_data = True  # True = train on full data, False = 80/20 split

# ===== REUSABLE WRAPPER FOR DJANGO =====

def train_random_forest_from_csv(input_csv_path: str, output_dir: str,
                                 model_filename: str = "random_forest_model.pkl",
                                 label_mapping_filename: str = "random_forest_label_mapping.txt"):
    """Train a RandomForest model from a single CSV file and save model + label mapping.

    Parameters
    ----------
    input_csv_path : str
        Absolute path to the input CSV.
    output_dir : str
        Directory where model and label mapping will be saved.
    model_filename : str
        Filename for the saved model (default: "random_forest_model.pkl").
    label_mapping_filename : str
        Filename for the label encoding mapping text file.

    Returns
    -------
    dict
        Basic information about the training run (samples, features, classes).
    """
    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(input_csv_path, low_memory=False)
    data.columns = data.columns.str.lower()

    if "label" not in data.columns:
        raise ValueError("Input CSV must contain a 'label' column for training.")

    X = data.drop(columns=["label"])
    y_raw = data["label"]

    # Label encoding
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Save label mapping
    mapping_path = os.path.join(output_dir, label_mapping_filename)
    with open(mapping_path, "w", encoding="utf-8") as f:
        f.write("Label Encoding Mapping:\n")
        f.write("=" * 40 + "\n")
        for cls, num in zip(le.classes_, range(len(le.classes_))):
            f.write(f"{cls:<30}: {num}\n")

    # Encode categorical feature columns
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    model_path = os.path.join(output_dir, model_filename)

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    joblib.dump(rf, model_path)

    details = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "classes": [str(c) for c in le.classes_],
    }

    return {
        "model_path": model_path,
        "label_mapping_path": mapping_path,
        "details": details,
    }


# ===== FUNCTIONS =====
def process_csv(file_path):
    """Train and save a model for one CSV file."""
    print(f"\n{'=' * 80}")
    print(f"Processing file: {os.path.basename(file_path)}")
    print(f"{'=' * 80}")

    # --- LOAD DATA ---
    data = pd.read_csv(file_path, low_memory=False)
    data.columns = data.columns.str.lower()

    if 'label' not in data.columns:
        print(f"Skipping {file_path} (no 'label' column found).")
        return

    X = data.drop(columns=['label'])
    y_raw = data['label']

    # --- LABEL ENCODING ---
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Save label mapping
    mapping_path = os.path.join(
        model_folder, os.path.basename(file_path).replace(".csv", "_label_mapping.txt")
    )
    with open(mapping_path, "w", encoding="utf-8") as f:
        f.write("Label Encoding Mapping:\n")
        f.write("=" * 40 + "\n")
        for cls, num in zip(le.classes_, range(len(le.classes_))):
            f.write(f"{cls:<30}: {num}\n")
    print(f"Label mapping saved to {mapping_path}")

    # Encode categorical features
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # --- TRAINING ---
    model_name = os.path.basename(file_path).replace(".csv", "")
    model_path = os.path.join(model_folder, f"{model_name}_model.pkl")

    if train_full_data:
        print("Training Random Forest on full dataset...")
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        rf.fit(X, y)
        joblib.dump(rf, model_path)
        print(f"Model trained on full dataset and saved to {model_path}")

    else:
        print("Splitting data (80% train / 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        joblib.dump(rf, model_path)
        print(f"Model trained with 80/20 split and saved to {model_path}")

        # --- EVALUATION ---
        print("Evaluating on test data...")
        y_pred = rf.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=le.classes_)
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

        # Save report and confusion matrix
        report_path = os.path.join(report_folder, f"{model_name}_report.txt")
        cm_path = os.path.join(report_folder, f"{model_name}_confusion_matrix.csv")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Test Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)
        cm_df.to_csv(cm_path)

        print("\nClassification Report:\n")
        print(report)
        print("\nConfusion Matrix:\n")
        print(cm_df)
        print(f"Report saved to {report_path}")
        print(f"Confusion matrix saved to {cm_path}")


# ===== MAIN =====
if __name__ == "__main__":
    csv_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the input folder.")
    else:
        print(f"Found {len(csv_files)} CSV file(s) in '{input_folder}'.")
        for file_path in csv_files:
            process_csv(file_path)

        print("\nAll models trained and saved successfully..")

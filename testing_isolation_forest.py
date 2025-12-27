import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. Configuration ---
model_filename = 'Training_isolation_model_cleaned/isolation.joblib'
test_data_file = '../Testing_isolation_model_cleaned/Benign_part_2.csv'  # <-- IMPORTANT: Change this
label_column = 'label'  # <-- The name of your label column

# --- 2. Load Model and Test Data ---
try:
    print(f"Loading model from {model_filename}...")
    model = joblib.load(model_filename)

    print(f"Loading test data from {test_data_file}...")
    test_df = pd.read_csv(test_data_file)

    # --- 3. Prepare Data for Prediction ---

    # Store the true labels for evaluation
    # This keeps the text ('Benign', 'DDoS', etc.)
    y_true_labels = test_df[label_column]

    # Prepare the feature data for the model (numeric-only)
    # This must be the *exact same* preparation as your training data
    print("Selecting numeric features for testing...")
    X_test_numeric = test_df.select_dtypes(include=[np.number])

    print(f"Test data has {len(X_test_numeric)} samples and {X_test_numeric.shape[1]} features.")

    # --- 4. Make Predictions ---
    print("\nMaking predictions on test data...")
    # The model predicts: 1 for inlier (normal), -1 for outlier (anomaly)
    y_pred = model.predict(X_test_numeric)

    # --- 5. Evaluate the Results ---

    # We must convert our *true labels* to the same format as the model's output
    # 1 for 'Benign'
    # -1 for everything else (any attack)
    y_true_mapped = y_true_labels.apply(lambda x: 1 if x == 'Benign' else -1)

    print("\n--- Evaluation Results ---")

    # Generate and print the classification report
    print("\nClassification Report:")
    # We tell the report our class names for clarity
    report = classification_report(
        y_true_mapped,
        y_pred,
        labels=[1, -1],
        target_names=['Normal (1)', 'Anomaly (-1)']
    )
    print(report)

    # Generate and print the confusion matrix
    print("\nConfusion Matrix:")
    print("         [Pred Normal] [Pred Anomaly]")
    cm = confusion_matrix(y_true_mapped, y_pred, labels=[1, -1])
    print(f"True Normal: {cm[0]}")
    print(f"True Anomaly: {cm[1]}")


except FileNotFoundError:
    print(f"Error: The file '{test_data_file}' or '{model_filename}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
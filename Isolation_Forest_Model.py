import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import numpy as np

# --- 1. Configuration ---
large_benign_file = 'Training_isolation_model_cleaned/Benign_part_2.csv'  # <-- Your large file
model_filename = 'Training_isolation_model_cleaned/isolation.joblib'
chunk_size = 2000000  # How many rows to read at a time
sample_fraction = 0.1  # Keep 10% of the data (adjust as needed)

# --- 2. Sub-sample the Large File ---
print(f"Starting to sample '{large_benign_file}'...")

list_of_sampled_chunks = []

try:
    with pd.read_csv(large_benign_file, chunksize=chunk_size) as reader:
        for i, chunk in enumerate(reader):
            print(f"  - Processing chunk {i + 1}...")
            # Take a random sample of the chunk
            list_of_sampled_chunks.append(chunk.sample(frac=sample_fraction))

    print("All chunks processed. Concatenating samples...")
    X_train_sampled = pd.concat(list_of_sampled_chunks, ignore_index=True)

    print(f"\nSampling complete. Total rows in sampled dataset: {len(X_train_sampled)}")

    # --- 3. Train the Model ---

    # ⬇️⬇️⬇️ --- THIS IS THE FIX --- ⬇️⬇️⬇️
    print("\nSelecting numeric features for training (skipping label column)...")
    # This creates a new DataFrame containing ONLY the numeric columns.
    # It automatically skips all string columns like 'Label'.
    X_train_numeric = X_train_sampled.select_dtypes(include=[np.number])

    print(f"Dropped {X_train_sampled.shape[1] - X_train_numeric.shape[1]} non-numeric columns.")
    # ⬆️⬆️⬆️ --- END OF FIX --- ⬆️⬆️⬆️

    # Initialize the Isolation Forest
    model = IsolationForest(n_estimators=100,
                            contamination='auto',
                            random_state=42,
                            n_jobs=-1)

    print("\nStarting model training on numeric data...")

    # Train the model ONLY on the numeric features
    model.fit(X_train_numeric)

    print("Model training complete.")

    # --- 4. Save the Trained Model ---
    print(f"\nSaving model to '{model_filename}'...")
    joblib.dump(model, model_filename)
    print(f"Model successfully saved to {model_filename}")

except FileNotFoundError:
    print(f"Error: The file '{large_benign_file}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
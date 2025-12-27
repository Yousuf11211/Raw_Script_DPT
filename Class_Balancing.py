import os
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN

# ===== CONFIGURATION =====
INPUT_FOLDER = "Training_2018"
OUTPUT_FOLDER = "Attack_Balanced"


# ===== FUNCTIONS =====
def get_csv_files(folder):
    """Return all CSV files in the folder"""
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]


def display_label_counts(y, le, file_name):
    """Display label counts for a specific file"""
    counts = Counter(y)
    rev_mapping = {i: label for i, label in enumerate(le.classes_)}

    print(f"\n--- Label distribution for '{file_name}' ---")
    for k in sorted(counts.keys()):
        print(f"  {rev_mapping[k]:<20}: {counts.get(k, 0):,}")
    print(f"Total samples: {sum(counts.values()):,}")
    print("--------------------------------------------------")


def calculate_target_strategy(y, ratio):
    """Automatically calculate the target counts based on a ratio"""
    counts = Counter(y)
    if not counts:
        return {}

    majority_class_key = max(counts, key=counts.get)
    majority_count = counts[majority_class_key]

    target_strategy = {}
    target_minority_count = int(majority_count * ratio)

    for cls, count in counts.items():
        if cls == majority_class_key:
            target_strategy[cls] = count
        else:
            target_strategy[cls] = max(count, target_minority_count)

    return target_strategy


def apply_resampling(X, y, target_strategy, oversampler_class):
    """Apply undersampling and oversampling to reach target counts"""
    current_counts = Counter(y)
    undersample = {c: t for c, t in target_strategy.items() if c in current_counts and current_counts[c] > t}
    oversample = {c: t for c, t in target_strategy.items() if c in current_counts and current_counts[c] < t}

    X_res, y_res = X.copy(), y.copy()

    if undersample:
        print("\nUndersampling started...")
        rus = RandomUnderSampler(sampling_strategy=undersample, random_state=42)
        X_res, y_res = rus.fit_resample(X_res, y_res)
        print(f"Undersampling done.")

    if oversample:
        print("\nOversampling started...")
        min_samples_for_smote = min(count for cls, count in Counter(y_res).items() if cls in oversample)

        # This value will be used for either k_neighbors or n_neighbors
        num_neighbors = max(1, min(min_samples_for_smote - 1, 5))

        # --- THIS IS THE CORRECTED LOGIC ---
        # Create a dictionary for sampler parameters
        sampler_params = {
            'sampling_strategy': oversample,
            'random_state': 42
        }

        # Check the class and add the correct neighbors parameter
        if oversampler_class == ADASYN:
            sampler_params['n_neighbors'] = num_neighbors
            print(f"Using ADASYN with n_neighbors={num_neighbors}...")
        else:  # For SMOTE and BorderlineSMOTE
            sampler_params['k_neighbors'] = num_neighbors
            print(f"Using {oversampler_class.__name__} with k_neighbors={num_neighbors}...")

        # Create the sampler by unpacking the parameters dictionary
        sampler = oversampler_class(**sampler_params)
        # --- END OF CORRECTION ---

        X_res, y_res = sampler.fit_resample(X_res, y_res)
        print("Oversampling done.")

    return X_res, y_res


# ===== MAIN SCRIPT =====
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    csv_files = get_csv_files(INPUT_FOLDER)

    if not csv_files:
        print("No CSV files found in the input folder!")
        return

    print(f"Found {len(csv_files)} CSV file(s).")

    # --- Interactive file selection ---
    files_to_process = []
    for file_path in csv_files:
        response = input(f"Process '{os.path.basename(file_path)}'? (y/n): ").lower()
        if response == 'y':
            files_to_process.append(file_path)
            print(f"  ✓ Added to processing list")
        else:
            print(f"  ✗ Skipped")

    if not files_to_process:
        print("\nNo files selected for processing!")
        return

    # --- Get user settings for balancing ---
    while True:
        try:
            ratio = float(input("\nEnter the desired minority-to-majority ratio (e.g., 0.5 for 50%): "))
            if 0 < ratio <= 1:
                break
            else:
                print("Please enter a number between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    oversamplers = {"1": SMOTE, "2": BorderlineSMOTE, "3": ADASYN}
    all_samplers = [SMOTE, BorderlineSMOTE, ADASYN]

    while True:
        choice = input(
            "Choose an oversampling method:\n  1: SMOTE (Standard)\n  2: Borderline-SMOTE\n  3: ADASYN\n  4: All\nChoice: ")
        if choice in oversamplers or choice == "4":
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

    samplers_to_run = all_samplers if choice == "4" else [oversamplers[choice]]

    # --- Process each selected file with each selected method ---
    for oversampler_class in samplers_to_run:
        method_name = oversampler_class.__name__
        print(f"\n===== PROCESSING WITH: {method_name} =====")

        # Create a dedicated output folder for the method
        method_output_folder = os.path.join(OUTPUT_FOLDER, method_name)
        os.makedirs(method_output_folder, exist_ok=True)

        for file_path in files_to_process:
            df = pd.read_csv(file_path)

            if 'label' not in df.columns:
                print(f"\nSkipping '{os.path.basename(file_path)}' (no 'label' column found).")
                continue

            le = LabelEncoder()
            y_enc = le.fit_transform(df['label'])
            display_label_counts(y_enc, le, os.path.basename(file_path))

            target_strategy = calculate_target_strategy(y_enc, ratio)

            X = df.drop("label", axis=1)
            X_bal, y_bal = apply_resampling(X, y_enc, target_strategy, oversampler_class)

            df_bal = pd.DataFrame(X_bal, columns=X.columns)
            df_bal["label"] = le.inverse_transform(y_bal)

            display_label_counts(y_bal, le, f"{os.path.basename(file_path)} (Balanced)")

            out_file = os.path.join(method_output_folder, os.path.basename(file_path).replace(".csv", "_balanced.csv"))
            df_bal.to_csv(out_file, index=False)
            print(f"\nSaved balanced CSV to '{method_name}' folder: {os.path.basename(out_file)}")

    print("\nAll selected files and methods processed.")


if __name__ == "__main__":
    main()
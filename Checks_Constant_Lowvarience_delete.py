import os
import pandas as pd
from collections import defaultdict

# --- 1. Configuration ---
# Set the folder where your original CSV files are located.
INPUT_PATH = "Attacks"

# Set the folder where the new, cleaned CSV files will be saved.
OUTPUT_FOLDER = "Attacks_Removed_Constant"

# Define the number of rows to read at a time.
CHUNK_SIZE = 1_000_000


# --- 2. Helper Functions ---

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


# --- 3. Main Processing Function (No changes needed here) ---

def analyze_and_clean_csv(file_path, output_path):
    """
    Analyzes a CSV for constant/low-variance columns and optionally removes them.
    """
    print("-" * 70)
    print(f"Processing file: {os.path.basename(file_path)}")

    try:
        print("  Analyzing columns... (this may take a moment for large files)")
        col_unique_values = defaultdict(set)
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, dtype=str, low_memory=False):
            for col in chunk.columns:
                col_unique_values[col].update(chunk[col].dropna().unique())
        print("  Analysis complete.")

        columns_to_drop = []
        if get_user_yes_no("  Do you want to find constant columns?"):
            constant_cols = {col: list(vals)[0] for col, vals in col_unique_values.items() if len(vals) == 1}
            if constant_cols:
                print("\n  [RESULT] Found Constant Columns:")
                for col, val in constant_cols.items():
                    print(f"    - Column '{col}' has only one value: {val}")
                columns_to_drop.extend(constant_cols.keys())
            else:
                print("\n  [RESULT] No constant columns were found.")

        if get_user_yes_no("  Do you want to find low-variance columns?"):
            while True:
                try:
                    threshold = int(input("    Enter the maximum number of unique values (e.g., 3): "))
                    break
                except ValueError:
                    print("    That wasn't a valid number. Please enter an integer.")

            low_variance_cols = {col: list(vals) for col, vals in col_unique_values.items() if
                                 2 <= len(vals) <= threshold}
            if low_variance_cols:
                print(f"\n  [RESULT] Found Low-Variance Columns (up to {threshold} unique values):")
                for col, vals in low_variance_cols.items():
                    print(f"    - Column '{col}' has values: {vals}")
                new_cols_to_add = [col for col in low_variance_cols if col not in columns_to_drop]
                columns_to_drop.extend(new_cols_to_add)
            else:
                print(f"\n  [RESULT] No low-variance columns found with the specified threshold.")

        if not columns_to_drop:
            print("\nNo columns were selected for removal. Moving to the next file.")
            return

        final_drop_list = sorted(list(set(columns_to_drop)))
        print("\nColumns identified for removal:", final_drop_list)

        if get_user_yes_no("Do you want to remove these columns and save a new, cleaned file?"):
            print(f"  Removing {len(final_drop_list)} columns and saving new file...")
            chunk_iterator = pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False)
            first_chunk = next(chunk_iterator)
            first_chunk.drop(columns=final_drop_list, errors="ignore").to_csv(output_path, index=False)
            for chunk in chunk_iterator:
                chunk.drop(columns=final_drop_list, errors="ignore").to_csv(output_path, mode='a', header=False,
                                                                            index=False)
            print(f"  Successfully saved cleaned file to: {output_path}")
        else:
            print("  Skipping file modification as requested.")

    except Exception as e:
        print(f"ERROR: An unexpected error occurred while processing {os.path.basename(file_path)}.")
        print(f"DETAILS: {e}")


# --- 4. Script Execution (NEW and IMPROVED) ---

if __name__ == "__main__":
    # First, find all available CSV files in the input path.
    if not os.path.isdir(INPUT_PATH):
        print(f"Error: Input path '{INPUT_PATH}' is not a valid directory.")
    else:
        print(f"Searching for CSV files in '{INPUT_PATH}'...")
        csv_files = []
        for root, dirs, files in os.walk(INPUT_PATH):
            for file in files:
                if file.endswith(".csv"):
                    csv_files.append(os.path.join(root, file))

        csv_files.sort()  # Sort the list for consistent ordering.

        if not csv_files:
            print("No CSV files found in the specified directory.")
        else:
            # Display the menu of found files
            print("\n--- CSV Files Found ---")
            for i, file_path in enumerate(csv_files, 1):
                print(f"  {i}: {os.path.basename(file_path)}")
            print("-----------------------")

            # Ask the user to make a selection
            while True:
                choice = input("Enter the numbers of files to process (e.g., 1,3,5), or type 'all': ").strip().lower()
                files_to_process = []

                if choice == 'all':
                    files_to_process = csv_files
                    break

                try:
                    # Process comma-separated numbers
                    indices = [int(num.strip()) - 1 for num in choice.split(',')]
                    valid_indices = [i for i in indices if 0 <= i < len(csv_files)]

                    if len(valid_indices) != len(indices):
                        print("Warning: Some numbers were out of range and have been ignored.")

                    if not valid_indices:
                        print("Error: No valid file numbers were entered. Please try again.")
                        continue

                    files_to_process = [csv_files[i] for i in valid_indices]
                    break
                except ValueError:
                    print("Invalid input. Please enter numbers separated by commas or 'all'.")

            # Process only the selected files
            print(f"\nBeginning processing for {len(files_to_process)} selected file(s)...")
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            for file_path in files_to_process:
                output_file_path = os.path.join(OUTPUT_FOLDER, os.path.basename(file_path))
                analyze_and_clean_csv(file_path, output_file_path)

            print("\n" + "-" * 70)
            print("All selected files have been processed.")
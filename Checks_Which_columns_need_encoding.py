import os
import pandas as pd
import numpy as np
from collections import Counter

# --- Configuration ---
input_folder = "Downscale_Csv_2018"  # Folder containing CSVs
chunk_size = 2_000_000  # Adjust based on memory
columns_to_check = ['delta_start', 'handshake_duration', 'label']

# --- Process each CSV in the input folder ---
for file in os.listdir(input_folder):
    if not file.endswith(".csv"):
        continue

    csv_file = os.path.join(input_folder, file)
    base_name, ext = os.path.splitext(file)
    output_csv = os.path.join(input_folder, f"{base_name}_cleaned{ext}")

    print(f"\nScanning file: {csv_file}...")

    both_valid_counter = Counter()
    both_invalid_counter = Counter()
    rows_to_remove = []

    chunk_start_row = 0
    # --- Phase 1: Scan and summarize ---
    for chunk in pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False, usecols=columns_to_check):
        for idx, row in chunk.iterrows():
            delta = row['delta_start']
            handshake = row['handshake_duration']
            label = row['label']

            delta_invalid = isinstance(delta, str) and "not a complete handshake" in delta.lower()
            handshake_invalid = isinstance(handshake, str) and "not a complete handshake" in handshake.lower()

            if delta_invalid and handshake_invalid:
                both_invalid_counter[label] += 1
                rows_to_remove.append(chunk_start_row + idx)
            elif not delta_invalid and not handshake_invalid:
                both_valid_counter[label] += 1

        chunk_start_row += len(chunk)

    # --- Summary ---
    print("\nSummary of rows:")
    print(f"Both valid rows: {sum(both_valid_counter.values())}")
    for lbl, cnt in both_valid_counter.items():
        print(f"  Label '{lbl}': {cnt}")

    print(f"\nBoth invalid rows: {sum(both_invalid_counter.values())}")
    for lbl, cnt in both_invalid_counter.items():
        print(f"  Label '{lbl}': {cnt}")

    # --- Ask for deletion ---
    delete_confirm = input(
        f"\nDo you want to delete the {len(rows_to_remove)} rows with invalid handshakes in '{file}'? (yes/no): ").lower()

    if delete_confirm in ['yes', 'y']:
        print("\nDeleting invalid rows and creating new CSV...")
        is_first_chunk = True
        chunk_start_row = 0

        for chunk in pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False):
            chunk_indices = np.arange(chunk_start_row, chunk_start_row + len(chunk))
            mask = np.isin(chunk_indices, rows_to_remove, invert=True)
            chunk_cleaned = chunk[mask]

            if is_first_chunk:
                chunk_cleaned.to_csv(output_csv, index=False, mode='w')
                is_first_chunk = False
            else:
                chunk_cleaned.to_csv(output_csv, index=False, mode='a', header=False)

            chunk_start_row += len(chunk)

        print(f"Cleaned CSV saved: {output_csv}")

    else:
        print("No rows deleted. Moving to next file.")

print("\nAll files processed.")

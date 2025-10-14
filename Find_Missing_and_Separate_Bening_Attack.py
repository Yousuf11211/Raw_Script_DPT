import os
import pandas as pd
from collections import defaultdict
import math

# --- 1. Global Configuration ---
INPUT_FOLDER = "Downscale_Csv_2018_Cleaned"
OUTPUT_FOLDER = "Proportional_Data_V2"
CHUNK_SIZE = 1_500_000  # A larger chunk size is efficient for counting
LABEL_COLUMN_NAME = 'label'  # We will enforce lowercase internally for consistency
BENIGN_LABEL_VALUE = 'benign'  # We will enforce lowercase internally


# --- 2. Core Functions ---

def analyze_and_classify(all_files):
    """
    Reads all files ONCE to get total row counts and classify files by the labels they contain.
    This is the efficient, hybrid approach.
    """
    print("--- Phase 1: Analyzing all files for counts and classification ---")
    total_counts = defaultdict(int)
    files_by_label = defaultdict(set)  # Using a set to avoid duplicate file paths
    actual_label_col_name = None

    for file_path in all_files:
        print(f"  Scanning: {os.path.basename(file_path)}...")
        try:
            # Find the label column in the first file and reuse it
            if actual_label_col_name is None:
                header_df = pd.read_csv(file_path, nrows=0, low_memory=False)
                for col in header_df.columns:
                    if col.lower() == LABEL_COLUMN_NAME:
                        actual_label_col_name = col
                        break

            if not actual_label_col_name:
                print(f"    Warning: Label column not found in this file. Skipping.")
                continue

            # Read the file in chunks to manage memory
            for chunk in pd.read_csv(file_path, usecols=[actual_label_col_name], chunksize=CHUNK_SIZE,
                                     low_memory=False):
                chunk.columns = [col.lower() for col in chunk.columns]  # Standardize column names

                # Get counts for this chunk and add to grand total
                chunk_counts = chunk[LABEL_COLUMN_NAME].value_counts()
                for label, count in chunk_counts.items():
                    total_counts[label] += count
                    # Associate this file with this label
                    files_by_label[label].add(file_path)

        except Exception as e:
            print(f"    Error analyzing {os.path.basename(file_path)}: {e}")

    # Convert sets to lists for easier processing later
    files_by_label = {label: list(paths) for label, paths in files_by_label.items()}

    print("--- Analysis complete ---")
    return total_counts, files_by_label, actual_label_col_name


def process_and_save_proportionally(file_list, rows_per_output_file, label_name, output_base_path, should_shuffle,
                                    actual_label_col_name):
    """
    Reads data proportionally from a list of files and saves it into new CSVs.
    (This function remains the same as the previous version)
    """
    if not file_list:
        return

    num_source_files = len(file_list)
    rows_to_take_per_file = math.ceil(rows_per_output_file / num_source_files)
    print(f"\nProcessing Label: {label_name}")
    print(f"  - Using {num_source_files} source file(s).")
    print(f"  - Aiming for {rows_per_output_file:,} rows per output file.")
    print(f"  - Will attempt to take ~{rows_to_take_per_file:,} rows from each source file per output part.")

    rows_read_from_file = {path: 0 for path in file_list}
    file_part_counter = 1
    os.makedirs(output_base_path, exist_ok=True)

    while True:
        batch_dataframes = []
        total_rows_in_this_pass = 0

        for file_path in file_list:
            try:
                skip = rows_read_from_file[file_path] + 1

                df_chunk = pd.read_csv(
                    file_path,
                    skiprows=range(1, skip),
                    nrows=rows_to_take_per_file,
                    low_memory=False
                )

                if not df_chunk.empty:
                    # Ensure we only keep rows of the correct label (case-insensitive)
                    clean_chunk = df_chunk[df_chunk[actual_label_col_name].str.lower() == label_name.lower()]
                    batch_dataframes.append(clean_chunk)

                    rows_read_from_file[file_path] += len(df_chunk)
                    total_rows_in_this_pass += len(clean_chunk)

            except StopIteration:  # This can happen with some iterators, treat as end of file
                continue
            except Exception as e:
                print(
                    f"    Warning: Could not read from {os.path.basename(file_path)}. Maybe it's finished? Error: {e}")
                continue

        if not batch_dataframes:
            print(f"  - No more data to read for label '{label_name}'. Finished.")
            break

        combined_df = pd.concat(batch_dataframes, ignore_index=True)

        if should_shuffle:
            print(f"  - Shuffling {len(combined_df):,} rows...")
            combined_df = combined_df.sample(frac=1).reset_index(drop=True)

        safe_name = "".join(c for c in label_name if c.isalnum() or c in ('-', '_'))
        output_filename = os.path.join(output_base_path, f"{safe_name}_part_{file_part_counter}.csv")

        # Take the exact number of rows requested, in case we over-shot with the proportional read
        final_df = combined_df.head(rows_per_output_file)

        final_df.to_csv(output_filename, index=False)
        print(f"  -> Saved {len(final_df):,} rows to {os.path.relpath(output_filename)}")

        file_part_counter += 1


def main():
    """ Main orchestrator for the improved workflow. """
    all_csv_files = [os.path.join(root, file) for root, _, files in os.walk(INPUT_FOLDER) for file in files if
                     file.endswith(".csv")]
    if not all_csv_files:
        print(f"No CSV files found in '{INPUT_FOLDER}'. Exiting.")
        return

    # 1. Run the single, efficient analysis pass
    total_counts, files_by_label, actual_label_col = analyze_and_classify(all_csv_files)
    if not actual_label_col:
        print("Could not determine the 'Label' column from any file. Exiting.")
        return

    # --- 2. Show Report and Get User Input ---
    print("\n--- Total Row Count Report (from all files) ---")
    benign_label_in_data = None
    attack_labels_in_data = {}

    for label, count in sorted(total_counts.items()):
        print(f"  - {label}: {count:,} total rows.")
        if str(label).lower() == BENIGN_LABEL_VALUE:
            benign_label_in_data = label
        else:
            attack_labels_in_data[label] = count
    print("-------------------------------------------------")

    should_shuffle = input("Do you want to shuffle the final output files? (y/n): ").strip().lower() in ['y', 'yes']

    # --- 3. Process BENIGN Files ---
    if benign_label_in_data:
        print("\n" + "=" * 30 + " PROCESSING BENIGN DATA " + "=" * 30)
        while True:
            try:
                user_input = input(f"Enter max rows per Benign file: ").strip()
                rows_per_file = int(user_input)
                if rows_per_file > 0:
                    process_and_save_proportionally(
                        file_list=files_by_label[benign_label_in_data],
                        rows_per_output_file=rows_per_file,
                        label_name=benign_label_in_data,
                        output_base_path=os.path.join(OUTPUT_FOLDER, 'Benign'),
                        should_shuffle=should_shuffle,
                        actual_label_col_name=actual_label_col
                    )
                    break
                else:
                    print("  Please enter a positive number.")
            except ValueError:
                print("  Invalid input. Please enter a whole number.")

    # --- 4. Process ATTACK Files ---
    if attack_labels_in_data:
        print("\n" + "=" * 30 + " PROCESSING ATTACK DATA " + "=" * 30)
        while True:
            try:
                user_input = input(f"Enter max rows per Attack file (applies to each attack type): ").strip()
                rows_per_file = int(user_input)
                if rows_per_file > 0:
                    for label in attack_labels_in_data:
                        process_and_save_proportionally(
                            file_list=files_by_label[label],
                            rows_per_output_file=rows_per_file,
                            label_name=label,
                            output_base_path=os.path.join(OUTPUT_FOLDER, 'Attacks'),
                            should_shuffle=should_shuffle,
                            actual_label_col_name=actual_label_col
                        )
                    break
                else:
                    print("  Please enter a positive number.")
            except ValueError:
                print("  Invalid input. Please enter a whole number.")

    print("\n" + "=" * 80 + "\nAll processing is complete!\n" + "=" * 80)


if __name__ == "__main__":
    main()
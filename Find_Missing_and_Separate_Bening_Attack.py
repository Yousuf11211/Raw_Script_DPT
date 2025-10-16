import os
import pandas as pd
from collections import defaultdict
import math

# --- 1. Global Configuration ---
INPUT_FOLDER = "Downscale_Csv_2018_Cleaned"
OUTPUT_FOLDER = "Proportional_Data_V2"
CHUNK_SIZE = 2_500_000
LABEL_COLUMN_NAME = 'label'
BENIGN_LABEL_VALUE = 'benign'


# --- 2. Core Functions ---

def analyze_and_classify(all_files, processing_mode):
    """
    Reads files to get total row counts and classify files by the labels they contain.
    Includes an optimization to skip files based on a preview and the processing mode.
    """
    print("--- Phase 1: Analyzing all files for counts and classification ---")
    total_counts = defaultdict(int)
    files_by_label = defaultdict(set)
    actual_label_col_name = None

    for file_path in all_files:
        print(f"  Scanning: {os.path.basename(file_path)}...")
        try:
            if actual_label_col_name is None:
                header_df = pd.read_csv(file_path, nrows=0, low_memory=False)
                for col in header_df.columns:
                    if col.lower() == LABEL_COLUMN_NAME:
                        actual_label_col_name = col
                        break
            if not actual_label_col_name:
                print(f"    Warning: Label column '{LABEL_COLUMN_NAME}' not found. Skipping.")
                continue

            if processing_mode != 'both':
                try:
                    preview_df = pd.read_csv(file_path, usecols=[actual_label_col_name], nrows=20, low_memory=False)
                    unique_labels_in_preview = set(preview_df[actual_label_col_name].str.lower().unique())
                    if processing_mode == 'attacks' and unique_labels_in_preview == {BENIGN_LABEL_VALUE}:
                        print(f"    -> Optimization: Skipping file as it appears to contain only benign data.")
                        continue
                    elif processing_mode == 'benign' and BENIGN_LABEL_VALUE not in unique_labels_in_preview:
                        print(f"    -> Optimization: Skipping file as it appears to contain only attack data.")
                        continue
                except Exception as e:
                    print(f"    Warning: Could not preview file. Proceeding with full scan. Error: {e}")

            for chunk in pd.read_csv(file_path, usecols=[actual_label_col_name], chunksize=CHUNK_SIZE,
                                     low_memory=False):
                chunk.columns = [col.lower() for col in chunk.columns]
                chunk_counts = chunk[LABEL_COLUMN_NAME].value_counts()
                for label, count in chunk_counts.items():
                    total_counts[label] += count
                    files_by_label[label].add(file_path)
        except Exception as e:
            print(f"    Error analyzing {os.path.basename(file_path)}: {e}")

    files_by_label = {label: list(paths) for label, paths in files_by_label.items()}
    print("--- Analysis complete ---")
    return total_counts, files_by_label, actual_label_col_name


def process_and_save_combined(file_list, rows_per_output_file, labels_to_keep, output_group_name, output_base_path,
                              should_shuffle, actual_label_col_name):
    """
    (Memory Efficient) Dynamically reads data sequentially from multiple files to create
    evenly-sized output files. Best for combining large files of the same type (e.g., all Benign).
    """
    if not file_list or not labels_to_keep:
        return

    print(f"\nProcessing Group Sequentially: {output_group_name}")
    print(f"  - Using {len(file_list)} source file(s).")
    print(f"  - Aiming for {rows_per_output_file:,} rows per output file.")

    os.makedirs(output_base_path, exist_ok=True)
    lower_labels_to_keep = [str(lbl).lower() for lbl in labels_to_keep]
    iterators = {}
    for file_path in file_list:
        try:
            iterators[file_path] = pd.read_csv(file_path, iterator=True, chunksize=50000, low_memory=False)
        except Exception as e:
            print(f"  Warning: Could not open {os.path.basename(file_path)}. Skipping it. Error: {e}")

    file_part_counter = 1
    leftover_df = pd.DataFrame()
    while iterators:
        batch_dataframes = [leftover_df] if not leftover_df.empty else []
        rows_collected = len(leftover_df)
        while rows_collected < rows_per_output_file:
            if not iterators: break
            iterators_this_pass = list(iterators.keys())
            for file_path in iterators_this_pass:
                try:
                    chunk = next(iterators[file_path])
                    clean_chunk = chunk[chunk[actual_label_col_name].str.lower().isin(lower_labels_to_keep)]
                    if not clean_chunk.empty:
                        batch_dataframes.append(clean_chunk)
                        rows_collected += len(clean_chunk)
                        if rows_collected >= rows_per_output_file:
                            break
                except StopIteration:
                    del iterators[file_path]
                except Exception as e:
                    print(f"  Error reading chunk from {os.path.basename(file_path)}. Removing it. Error: {e}")
                    del iterators[file_path]
        if not batch_dataframes:
            break
        combined_df = pd.concat(batch_dataframes, ignore_index=True)
        if should_shuffle:
            combined_df = combined_df.sample(frac=1).reset_index(drop=True)
        final_df = combined_df.iloc[:rows_per_output_file]
        leftover_df = combined_df.iloc[rows_per_output_file:]
        output_filename = os.path.join(output_base_path, f"{output_group_name}_part_{file_part_counter}.csv")
        final_df.to_csv(output_filename, index=False)
        print(f"  -> Saved {len(final_df):,} rows to {os.path.relpath(output_filename)}")
        file_part_counter += 1
    if not leftover_df.empty:
        output_filename = os.path.join(output_base_path, f"{output_group_name}_part_{file_part_counter}.csv")
        leftover_df.to_csv(output_filename, index=False)
        print(f"  -> Saved {len(leftover_df):,} final rows to {os.path.relpath(output_filename)}")
    print(f"  - Finished processing for group '{output_group_name}'.")


def process_and_save_proportionally(file_list, rows_per_output_file, labels_to_keep, output_group_name,
                                    output_base_path, should_shuffle, actual_label_col_name):
    """
    (Memory Intensive) Creates balanced output files by loading all source data, combining it,
    shuffling, and then splitting it. This ensures each output file is a proportional mix
    of all source files. Ideal for creating representative datasets (e.g., from all Attacks).
    """
    if not file_list or not labels_to_keep:
        print("No data to process.")
        return

    print(f"\nProcessing Group Proportionally: {output_group_name}")
    print("  - Loading all relevant data from source files. This may take time and memory...")
    os.makedirs(output_base_path, exist_ok=True)
    lower_labels_to_keep = [str(lbl).lower() for lbl in labels_to_keep]

    all_samples = []
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path, low_memory=False)
            relevant_df = df[df[actual_label_col_name].str.lower().isin(lower_labels_to_keep)]
            if not relevant_df.empty:
                all_samples.append(relevant_df)
                print(f"    -> Collected {len(relevant_df):,} rows from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"    Warning: Could not read or sample {os.path.basename(file_path)}. Skipping. Error: {e}")

    if not all_samples:
        print("  - Could not gather any data. Aborting.")
        return

    print("  - Combining all data into a master DataFrame...")
    master_df = pd.concat(all_samples, ignore_index=True)

    if should_shuffle:
        print(f"  - Shuffling the master DataFrame of {len(master_df):,} total rows...")
        master_df = master_df.sample(frac=1).reset_index(drop=True)

    num_output_files = math.ceil(len(master_df) / rows_per_output_file)
    print(f"  - Splitting into {num_output_files} file(s) of up to {rows_per_output_file:,} rows each.")

    for i in range(num_output_files):
        start_index = i * rows_per_output_file
        end_index = start_index + rows_per_output_file
        output_df = master_df.iloc[start_index:end_index]
        output_filename = os.path.join(output_base_path, f"{output_group_name}_part_{i + 1}.csv")
        output_df.to_csv(output_filename, index=False)
        print(f"  -> Saved {len(output_df):,} rows to {os.path.relpath(output_filename)}")

    print(f"  - Finished proportional processing for group '{output_group_name}'.")


def main():
    """ Main orchestrator for the improved workflow. """
    all_csv_files = [os.path.join(root, file) for root, _, files in os.walk(INPUT_FOLDER) for file in files if
                     file.endswith(".csv")]
    if not all_csv_files:
        print(f"No CSV files found in '{INPUT_FOLDER}'. Exiting.")
        return

    while True:
        print("\nPlease choose which data group to process:")
        print("  1: Benign Only")
        print("  2: Attacks Only")
        print("  3: Both Benign and Attacks")
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        if choice in ['1', '2', '3']:
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    processing_mode = 'both'
    if choice == '1':
        processing_mode = 'benign'
    elif choice == '2':
        processing_mode = 'attacks'

    total_counts, files_by_label, actual_label_col = analyze_and_classify(all_csv_files, processing_mode)
    if not actual_label_col:
        print("Could not determine the 'Label' column from any file. Exiting.")
        return

    print("\n--- Total Row Count Report (from analyzed files) ---")
    benign_label_in_data = None
    attack_labels_in_data = {}
    for label, count in sorted(total_counts.items()):
        print(f"  - {label}: {count:,} total rows.")
        if str(label).lower() == BENIGN_LABEL_VALUE:
            benign_label_in_data = label
        else:
            attack_labels_in_data[label] = count
    print("-------------------------------------------------")

    process_benign = choice in ['1', '3']
    process_attacks = choice in ['2', '3']

    should_shuffle = input("Do you want to shuffle the final output files? (y/n): ").strip().lower() in ['y', 'yes']

    if process_benign and benign_label_in_data:
        print("\n" + "=" * 30 + " PROCESSING BENIGN DATA " + "=" * 30)
        while True:
            try:
                rows_per_file = int(input(f"Enter max rows per Benign file: ").strip())
                if rows_per_file > 0:
                    # Use the original memory-efficient method for benign data
                    process_and_save_combined(
                        file_list=files_by_label[benign_label_in_data],
                        rows_per_output_file=rows_per_file,
                        labels_to_keep=[benign_label_in_data],
                        output_group_name='Benign',
                        output_base_path=os.path.join(OUTPUT_FOLDER, 'Benign'),
                        should_shuffle=should_shuffle,
                        actual_label_col_name=actual_label_col
                    )
                    break
                else:
                    print("  Please enter a positive number.")
            except ValueError:
                print("  Invalid input. Please enter a whole number.")
    elif process_benign:
        print("\nSkipping Benign processing: No 'Benign' labels found in the analyzed data.")

    if process_attacks and attack_labels_in_data:
        print("\n" + "=" * 30 + " PROCESSING ATTACK DATA " + "=" * 30)
        all_attack_files = sorted(list(set(f for lbl in attack_labels_in_data for f in files_by_label.get(lbl, []))))
        all_attack_labels = list(attack_labels_in_data.keys())
        total_attack_rows = sum(attack_labels_in_data.values())

        while True:
            try:
                rows_per_file = int(
                    input(f"Enter max rows per Attack file ({total_attack_rows:,} total available): ").strip())
                if rows_per_file > 0:
                    # Use the new proportional method for attack data
                    process_and_save_proportionally(
                        file_list=all_attack_files,
                        rows_per_output_file=rows_per_file,
                        labels_to_keep=all_attack_labels,
                        output_group_name='Attacks',
                        output_base_path=os.path.join(OUTPUT_FOLDER, 'Attacks'),
                        should_shuffle=should_shuffle,
                        actual_label_col_name=actual_label_col
                    )
                    break
                else:
                    print("  Please enter a positive number.")
            except ValueError:
                print("  Invalid input. Please enter a whole number.")
    elif process_attacks:
        print("\nSkipping Attack processing: No attack labels found in the analyzed data.")

    print("\n" + "=" * 80 + "\nAll processing is complete!\n" + "=" * 80)


if __name__ == "__main__":
    main()
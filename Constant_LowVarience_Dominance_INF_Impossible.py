import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# --- 1. GLOBAL CONFIGURATION ---
INPUT_FOLDER = "Downscale_Csv_2018"
OUTPUT_FOLDER = "Cleaned_Files_2018"
CHUNK_SIZE = 1_000_000

# --- Task 1 Config ---
DOMINANCE_RANGES = [
    (0.95, 1.01, "95-100%"), (0.90, 0.95, "90-95%"),
    (0.80, 0.90, "80-90%"), (0.70, 0.80, "70-80%"),
    (0.60, 0.70, "60-70%"), (0.50, 0.60, "50-60%"),
]

# --- Task 2 Config ---
NEVER_NEGATIVE_KEYWORDS = [
    'port', 'duration', 'count', 'bytes', 'size', 'rate', 'percentage',
    'variance', 'std', 'total', 'max', 'min', 'median', 'mode', 'mean',
    'iat', 'active', 'idle', 'bulk', 'handshake', 'subflow'
]
CAN_BE_NEGATIVE_KEYWORDS = ['skew', 'cov', 'delta']
PORT_COLUMNS = ['src_port', 'dst_port']

# --- Task 3 Config ---
INF_THRESHOLD = 0.30  # 30% threshold for removing 'inf' columns


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
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


# ==============================================================================
# TASK 1: STATIC DOMINANCE REPORT LOGIC
# ==============================================================================
# ==============================================================================
# TASK 1: DOMINANCE REPORT LOGIC (MODIFIED FOR SUMMARIZED OUTPUT)
# ==============================================================================
def generate_dominance_report(file_path):
    """
    Analyzes a CSV for value dominance and creates a summarized report.
    - Shows the top dominant value in detail.
    - Summarizes all other remaining values into a single line.
    """
    print(f"\n--- [Task 1] Generating Dominance Report for: {os.path.basename(file_path)} ---")
    col_counters = defaultdict(Counter)
    total_counts = Counter()
    label_counter = Counter()
    col_value_label_counter = defaultdict(lambda: defaultdict(Counter))
    try:
        # (Analysis phase is unchanged)
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, dtype=str, low_memory=False):
            labels = chunk.get("Label") or chunk.get("label")
            if labels is not None:
                label_counter.update(labels.dropna())
            for col in chunk.columns:
                values = chunk[col].dropna()
                col_counters[col].update(values)
                total_counts[col] += len(values)
                if labels is not None and col.lower() != "label":
                    for v, lbl in zip(chunk[col], labels):
                        if pd.notna(v) and pd.notna(lbl):
                            col_value_label_counter[col][v][lbl] += 1

        bucketed = {label: [] for _, _, label in DOMINANCE_RANGES}
        for col, counts in col_counters.items():
            if not counts: continue
            _, most_common_count = counts.most_common(1)[0]
            ratio = most_common_count / total_counts[col]
            for low, high, label in DOMINANCE_RANGES:
                if low <= ratio < high:
                    bucketed[label].append((col, counts, total_counts[col]))
                    break

        report_path = os.path.join(OUTPUT_FOLDER,
                                   f"{os.path.splitext(os.path.basename(file_path))[0]}_dominance_report.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            # (Header and Label distribution part is unchanged)
            header_text = f"Dominance Report for {os.path.basename(file_path)}"
            f.write(header_text + "\n" + "=" * 60 + "\n\n")
            print("\n" + header_text)
            if label_counter:
                total_labels = sum(label_counter.values())
                label_header = "Global Label Distribution:\n" + "-" * 40
                f.write(label_header + "\n")
                print("\n" + label_header)
                for lbl, count in label_counter.most_common():
                    line_text = f"  {lbl}: {count:,} ({(count / total_labels) * 100:.2f}%)"
                    f.write(line_text + "\n")
                    print(line_text)
                f.write("\n")

            # --- MODIFIED REPORTING LOGIC STARTS HERE ---
            for label in bucketed:
                bucket_header = f"\nColumns in {label} range:\n" + "-" * 40
                f.write(bucket_header + "\n")
                print(bucket_header)
                if not bucketed[label]:
                    f.write("  None\n")
                    print("  None")
                else:
                    for col, counts, total in bucketed[label]:
                        col_header = f"\nColumn: {col}"
                        f.write(col_header + "\n")
                        print(col_header)

                        # 1. Get and display the dominant value
                        dominant_val, dominant_count = counts.most_common(1)[0]
                        dominant_ratio = dominant_count / total

                        dominant_line = f"  Value '{dominant_val}': {dominant_count:,} ({dominant_ratio * 100:.2f}%)"
                        if dominant_val in col_value_label_counter.get(col, {}):
                            lbl_counts = col_value_label_counter[col][dominant_val]
                            breakdown = ", ".join(f"{lbl}: {c:,}" for lbl, c in lbl_counts.most_common())
                            dominant_line += f" -> Labels: [{breakdown}]"

                        f.write(dominant_line + "\n")
                        print(dominant_line)

                        # 2. Summarize all other values
                        num_remaining_unique = len(counts) - 1
                        if num_remaining_unique > 0:
                            total_remaining_count = total - dominant_count
                            remaining_ratio = total_remaining_count / total

                            # Aggregate label counts for all remaining values
                            remaining_label_counts = Counter()
                            for val, count in counts.items():
                                if val != dominant_val:
                                    # Get the label breakdown for this specific other value
                                    labels_for_this_val = col_value_label_counter[col].get(val, {})
                                    remaining_label_counts.update(labels_for_this_val)

                            remaining_breakdown = ", ".join(
                                f"{lbl}: {c:,}" for lbl, c in remaining_label_counts.most_common())

                            summary_line = f"  Remaining {remaining_ratio * 100:.2f}%: in {num_remaining_unique} other unique values -> Labels: [{remaining_breakdown}]"
                            f.write(summary_line + "\n")
                            print(summary_line)

        print(f"\nReport saved to: {report_path}")
    except Exception as e:
        print(f"ERROR during dominance report: {e}")

# ==============================================================================
# TASK 2: DATA VALIDATION & CLEANING LOGIC (ROW REMOVAL)
# ==============================================================================
def run_data_validation(file_path):
    """Loads a CSV and runs the full validation and cleaning pipeline."""
    print(f"\n--- [Task 2] Validating and Cleaning: {os.path.basename(file_path)} ---")
    try:
        # Load the entire file into memory for this task
        df = pd.concat([chunk for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE)])
        print(f"Loaded {len(df):,} rows.")
        results = {'negative_issues': {}, 'port_issues': {}}

        # Standardize label column name
        if 'Label' in df.columns and 'label' not in df.columns:
            df = df.rename(columns={'Label': 'label'})
        if 'label' not in df.columns:
            print("Warning: 'label' column not found. Creating a placeholder.")
            df['label'] = 'Unknown'

        # Check for negative values in specific columns
        for col in df.columns:
            if any(kw in col.lower() for kw in CAN_BE_NEGATIVE_KEYWORDS):
                continue
            if any(kw in col.lower() for kw in NEVER_NEGATIVE_KEYWORDS):
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if (numeric_col < 0).any():
                    mask = numeric_col < 0
                    results['negative_issues'][col] = {'count': mask.sum(), 'rows': list(df[mask].index)}

        # Check for invalid port numbers
        for col in PORT_COLUMNS:
            if col in df.columns:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if (~numeric_col.between(0, 65535)).any():
                    mask = ~numeric_col.between(0, 65535)
                    results['port_issues'][col] = {'count': mask.sum(), 'rows': list(df[mask].index)}

        # Aggregate all invalid row indices
        invalid_indices = set()
        for group in results.values():
            for info in group.values():
                invalid_indices.update(info['rows'])

        if not invalid_indices:
            print("\n[RESULT] No invalid rows found based on the rules.")
            return

        print(f"\n[RESULT] Found {len(invalid_indices):,} unique rows with invalid values.")
        if get_user_yes_no("Do you want to remove these invalid rows and save a new file?"):
            df_clean = df.drop(index=list(invalid_indices)).copy()
            clean_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_validated.csv"
            output_path = os.path.join(OUTPUT_FOLDER, clean_filename)
            df_clean.to_csv(output_path, index=False)
            print(f"  Successfully saved clean data ({len(df_clean):,} rows) to: {output_path}")
        else:
            print("  Skipping data cleaning.")
    except Exception as e:
        print(f"ERROR during data validation: {e}")

# ==============================================================================
# TASK 3: 'INF' COLUMN REMOVAL & IMPUTATION LOGIC
# ==============================================================================
def run_inf_column_removal(file_path):
    """Analyzes and removes columns with a high percentage of 'inf' values."""
    print(f"\n--- [Task 3] Processing for 'inf' columns: {os.path.basename(file_path)} ---")
    print(f"Phase 1: Analyzing columns (Threshold: {INF_THRESHOLD:.0%})...")
    inf_counts = pd.Series(dtype=int)
    total_rows = 0
    try:
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
            total_rows += len(chunk)
            inf_counts = inf_counts.add(chunk.apply(pd.to_numeric, errors='coerce').pipe(np.isinf).sum(), fill_value=0)
        if total_rows == 0:
            print("File is empty. Skipping.")
            return
        inf_percentages = inf_counts / total_rows
        columns_to_delete = inf_percentages[inf_percentages > INF_THRESHOLD].index.tolist()
    except Exception as e:
        print(f"ERROR during analysis: {e}")
        return

    if not columns_to_delete:
        print("\n[RESULT] No columns exceeded the 'inf' threshold.")
        if (inf_counts > 0).any():
            if get_user_yes_no("  Some 'inf' values were found below the threshold. Handle them with imputation?"):
                run_inf_imputation(file_path)
        return

    print(f"\n[RESULT] Found {len(columns_to_delete)} columns to remove:")
    for col in columns_to_delete:
        print(f"  - '{col}' ({inf_percentages[col]:.2%} inf)")

    if not get_user_yes_no("\nDo you want to permanently delete these columns?"):
        print("Operation cancelled.")
        return

    print("\nPhase 2: Deleting columns and creating new file...")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"{base_name}_inf_cleaned.csv"
    output_csv_path = os.path.join(OUTPUT_FOLDER, output_filename)
    try:
        is_first_chunk = True
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
            chunk.drop(columns=columns_to_delete, inplace=True, errors='ignore')
            if is_first_chunk:
                chunk.to_csv(output_csv_path, index=False, mode='w')
                is_first_chunk = False
            else:
                chunk.to_csv(output_csv_path, index=False, mode='a', header=False)
        print(f"  Successfully created '{output_filename}'")
        print("\n--- Next Steps for the Cleaned File ---")
        print("  1: Re-analyze the cleaned file for remaining 'inf' values")
        print("  2: Handle remaining 'inf' values with median imputation")
        print("  3: Do nothing / Continue")
        choice = input("Enter your choice (1, 2, or 3): ")
        if choice == '1':
            report_remaining_inf(output_csv_path)
        elif choice == '2':
            run_inf_imputation(output_csv_path)
        else:
            print("Continuing.")
    except Exception as e:
        print(f"ERROR during file creation: {e}")

def report_remaining_inf(file_path):
    """A simple analysis pass to report, but not act on, 'inf' values."""
    print(f"\n--- Re-analyzing for remaining 'inf' in {os.path.basename(file_path)} ---")
    inf_counts = pd.Series(dtype=int)
    total_rows = 0
    try:
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
            total_rows += len(chunk)
            inf_counts = inf_counts.add(chunk.apply(pd.to_numeric, errors='coerce').pipe(np.isinf).sum(), fill_value=0)
        if total_rows == 0: return
        inf_percentages = inf_counts / total_rows
        remaining_inf_cols = inf_percentages[inf_percentages > 0].index.tolist()
        if not remaining_inf_cols:
            print("[RESULT] No remaining 'inf' values found.")
        else:
            print("[RESULT] Found remaining 'inf' values in the following columns:")
            for col in remaining_inf_cols:
                print(f"  - '{col}': {inf_counts[col]} values ({inf_percentages[col]:.4f}%)")
    except Exception as e:
        print(f"ERROR during re-analysis: {e}")

def run_inf_imputation(file_path):
    """Finds all 'inf' values and replaces them with the column median."""
    print(f"\n--- Imputing 'inf' values in {os.path.basename(file_path)} ---")
    medians = {}
    try:
        print("Phase 1: Calculating medians for columns with 'inf' values...")
        inf_counts = pd.Series(dtype=int)
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
            inf_counts = inf_counts.add(chunk.apply(pd.to_numeric, errors='coerce').pipe(np.isinf).sum(), fill_value=0)
        cols_to_process = inf_counts[inf_counts > 0].index.tolist()
        if not cols_to_process:
            print("No 'inf' values found to impute.")
            return
        for col in cols_to_process:
            series = pd.read_csv(file_path, usecols=[col], low_memory=False).squeeze("columns")
            median_val = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).median()
            medians[col] = median_val
            print(f"  - Column '{col}': Median is {median_val}")
        print("\nPhase 2: Replacing 'inf' values and saving new file...")
        base_name = os.path.splitext(os.path.basename(file_path))[0].replace('_inf_cleaned', '')
        output_filename = f"{base_name}_imputed.csv"
        output_csv_path = os.path.join(OUTPUT_FOLDER, output_filename)
        is_first_chunk = True
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
            for col, median_val in medians.items():
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce').replace([np.inf, -np.inf], median_val)
            if is_first_chunk:
                chunk.to_csv(output_csv_path, index=False, mode='w')
                is_first_chunk = False
            else:
                chunk.to_csv(output_csv_path, index=False, mode='a', header=False)
        print(f"  Successfully created '{output_filename}'")
    except Exception as e:
        print(f"ERROR during imputation: {e}")

# ==============================================================================
# TASK 4: REMOVE CONSTANT OR LOW-VARIANCE COLUMNS
# ==============================================================================
def run_variance_analysis(file_path):
    """
    Analyzes a CSV for constant/low-variance columns and optionally removes them.
    """
    print(f"\n--- [Task 4] Analyzing for Low-Variance Columns: {os.path.basename(file_path)} ---")
    try:
        print("  Analyzing columns... (this may take a moment for large files)")
        col_unique_values = defaultdict(set)
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, dtype=str, low_memory=False):
            for col in chunk.columns:
                col_unique_values[col].update(chunk[col].dropna().unique())
        print("  Analysis complete.")

        columns_to_drop = []
        if get_user_yes_no("  Do you want to find constant columns (1 unique value)?"):
            constant_cols = {col: list(vals)[0] for col, vals in col_unique_values.items() if len(vals) == 1}
            if constant_cols:
                print("\n  [RESULT] Found Constant Columns:")
                for col, val in constant_cols.items():
                    print(f"    - {col}: (value is '{val}')")
                columns_to_drop.extend(constant_cols.keys())
            else:
                print("\n  [RESULT] No constant columns were found.")

        if get_user_yes_no("  Do you want to find low-variance columns (2+ unique values)?"):
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
                    print(f"    - {col}: (values are {vals})")
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
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_variance_cleaned.csv")
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
        print(f"ERROR: An unexpected error occurred: {e}")

# ==============================================================================
# TASK 5: INTERACTIVE DOMINANCE & VARIANCE ANALYSIS (NEW)
# ==============================================================================
def run_interactive_dominance_analysis(file_path):
    """
    Interactively finds columns with a user-defined dominance percentage,
    then allows for a follow-up low-variance check and optional deletion.
    """
    print(f"\n--- [Task 5] Interactive Dominance Analysis: {os.path.basename(file_path)} ---")
    try:
        # Step 1: Perform a full analysis of the file once
        print("  Analyzing file to gather column statistics... (this may take a moment)")
        col_counters = defaultdict(Counter)
        total_counts = Counter()
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, dtype=str, low_memory=False):
            for col in chunk.columns:
                values = chunk[col].dropna()
                col_counters[col].update(values)
                total_counts[col] += len(values)
        print("  Analysis complete.")

        # Step 2: Enter the interactive loop for this file
        while True:
            print("\n--- Dominance Range Selection ---")
            try:
                min_perc = float(input("  Enter the MINIMUM dominance percentage (e.g., 90): "))
                max_perc = float(input("  Enter the MAXIMUM dominance percentage (e.g., 100): "))
                if not (0 <= min_perc <= 100 and 0 <= max_perc <= 100 and min_perc <= max_perc):
                    print("  Error: Please enter valid percentages between 0 and 100.")
                    continue
            except ValueError:
                print("  Error: Invalid input. Please enter numbers.")
                continue

            # Step 3: Find and display columns within the user-defined range
            columns_in_range = {}
            for col, counts in col_counters.items():
                if total_counts[col] == 0: continue
                most_common_val, most_common_count = counts.most_common(1)[0]
                ratio = most_common_count / total_counts[col]

                if (min_perc / 100) <= ratio <= (max_perc / 100):
                    columns_in_range[col] = (most_common_val, ratio)

            if not columns_in_range:
                print(f"\n[RESULT] No columns found with a dominance between {min_perc}% and {max_perc}%.")
                if not get_user_yes_no("Do you want to try a different percentage range for this file?"):
                    break  # Exit loop for this file
                else:
                    continue  # Restart loop for this file

            print(f"\n[RESULT] Found {len(columns_in_range)} columns with dominance between {min_perc}% and {max_perc}%:")
            for col, (val, ratio) in columns_in_range.items():
                print(f"  - Column '{col}': Value '{val}' dominates with {ratio:.2%}")

            # Step 4: Ask for follow-up low-variance analysis
            if get_user_yes_no("\nDo you want to see the unique value counts for these specific columns?"):
                print("\n--- Low-Variance Detail ---")
                for col in columns_in_range.keys():
                    unique_count = len(col_counters[col])
                    print(f"  - Column '{col}' has {unique_count} unique value(s).")

            # Step 5: Ask for the final action
            print("\n--- Action Menu ---")
            action = input("Enter 'd' to delete these columns, 'r' to re-analyze, or 'n' for the next file: ").lower().strip()

            if action == 'd':
                cols_to_delete = list(columns_in_range.keys())
                print(f"\nPreparing to delete {len(cols_to_delete)} columns...")
                if get_user_yes_no("Are you sure you want to permanently remove these columns?"):
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    output_filename = f"{base_name}_interactive_cleaned.csv"
                    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

                    print(f"  Deleting columns and saving to {output_path}...")
                    is_first_chunk = True
                    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
                        chunk.drop(columns=cols_to_delete, inplace=True, errors='ignore')
                        if is_first_chunk:
                            chunk.to_csv(output_path, index=False, mode='w')
                            is_first_chunk = False
                        else:
                            chunk.to_csv(output_path, index=False, mode='a', header=False)
                    print("  Deletion successful.")
                else:
                    print("  Deletion cancelled.")
                break  # Deletion is a final action, so we exit the loop for this file

            elif action == 'r':
                print("\nRestarting analysis for this file...")
                continue  # Go to the top of the while loop

            else:  # 'n' or any other key
                print("  Moving to the next file.")
                break  # Exit the while loop for this file

    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")

# ==============================================================================
# MAIN DRIVER: INTEGRATES TASK AND FILE SELECTION
# ==============================================================================
def main():
    """Main function to prompt user for a task and which files to process."""
    print("--- Data Analysis and Validation Tool ---")
    print(f"Searching for CSVs in: '{INPUT_FOLDER}'")

    # --- Task Selection ---
    print("\nPlease choose a task to perform on the files:")
    print("  1: Generate Static Dominance Report (.txt file)")
    print("  2: Validate Data and Remove Invalid Rows")
    print("  3: Handle Columns with High 'inf' Values")
    print("  4: Remove Constant or Low-Variance Columns")
    print("  5: Interactive Dominance & Variance Analysis (NEW!)")

    task_choice = input("Enter your choice (1, 2, 3, 4, or 5): ").strip()
    if task_choice not in ['1', '2', '3', '4', '5']:
        print("Invalid choice. Exiting.")
        return

    # --- File Discovery and Selection ---
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Error: Input folder not found at '{INPUT_FOLDER}'")
        return

    csv_files = sorted([os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv")])

    if not csv_files:
        print("No CSV files found in the specified directory.")
        return

    print("\n--- CSV Files Found ---")
    for i, file_path in enumerate(csv_files, 1):
        print(f"  {i}: {os.path.basename(file_path)}")
    print("-----------------------")

    while True:
        file_choice = input("Enter the numbers of files to process (e.g., 1,3,5), or type 'all': ").strip().lower()
        files_to_process = []
        if file_choice == 'all':
            files_to_process = csv_files
            break
        try:
            indices = [int(num.strip()) - 1 for num in file_choice.split(',')]
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

    # --- Process only the selected files with the chosen task ---
    print(f"\nBeginning processing for {len(files_to_process)} selected file(s)...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for file_path in files_to_process:
        if task_choice == '1':
            generate_dominance_report(file_path)
        elif task_choice == '2':
            run_data_validation(file_path)
        elif task_choice == '3':
            run_inf_column_removal(file_path)
        elif task_choice == '4':
            run_variance_analysis(file_path)
        elif task_choice == '5':
            run_interactive_dominance_analysis(file_path)
        print("-" * 70)

    print("\nAll selected files have been processed.")


if __name__ == "__main__":
    main()
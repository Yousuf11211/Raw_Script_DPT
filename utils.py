import os
import pandas as pd
from typing import Dict, Tuple, Optional, List


# Function to count the number of Columns, and duplicate columns(Remove if any) print them optionally save the results
def count_column_duplicate(
        input_folder: str,
        output_folder: str,
        txtfile_name: str,
        save_results: bool = False,
        chunksize: Optional[int] = None  #  chunk size for reading a full file if any duplicates were found
) -> Dict[str, Tuple[int, Optional[List[str]]]]:
    print("Counting Columns and Duplicate Columns...")
    column_data = {}

    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"The input folder '{input_folder}' does not exist.")
        return {}

    # Make sure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    print(f"Scanning {input_folder} for CSV files...")

    found_csv = False                               # No Csv files found

    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".csv"):
                found_csv = True                    # Update once CSV is found
                file_path = os.path.join(root, file_name)

                # Reads only header for column count and check duplicate
                try:
                    df_header = pd.read_csv(
                        file_path,
                        nrows=0,
                        sep=',',                                # Comma-separated values
                        mangle_dupe_cols=False                  # To make sure pandas doesn't rename duplicate columns
                    )

                    columns = list(df_header.columns)
                    count = len(columns)

                    # Duplicate column name Check
                    seen_cols = set()
                    duplicate_cols = []

                    # Checking duplicates Columns from the headers
                    for col in columns:
                        if col in seen_cols and col not in duplicate_cols:
                            duplicate_cols.append(col)
                        seen_cols.add(col)

                    duplicate_to_report = sorted(duplicate_cols) if duplicate_cols else None
                    column_data[file_path] = (count, duplicate_to_report)

                    # Printing results
                    if duplicate_to_report:
                        print(
                            f"Result: {count} columns in: {file_path} | Duplicates Found: {', '.join(duplicate_to_report)}")

                        # If Duplicates are found, Clean and Save the new CSV using Chunk size (Memory efficient)
                        output_file_name = f"CLEANED_{file_name}"
                        output_file_path = os.path.join(output_folder, output_file_name)

                        # Identify indices to keep (based on first occurrence)
                        keep_indices = []
                        cleaned_cols = []
                        for i, col in enumerate(df_header.columns):
                            if col not in cleaned_cols:
                                cleaned_cols.append(col)
                                keep_indices.append(i)

                        is_first_chunk = True

                        # Use iterator=True to enable chunking or full read
                        reader = pd.read_csv(
                            file_path,
                            sep=',',
                            mangle_dupe_cols=False,
                            iterator=True,
                            chunksize=chunksize
                        )

                        # Process file chunk-by-chunk
                        for chunk in reader:
                            # iloc selects columns by the calculated index list, removing later duplicates
                            df_cleaned_chunk = chunk.iloc[:, keep_indices]

                            # Write the chunk: 'w' for the first chunk, 'a' for later chunks
                            df_cleaned_chunk.to_csv(
                                output_file_path,
                                mode='w' if is_first_chunk else 'a',
                                header=is_first_chunk,
                                index=False
                            )
                            is_first_chunk = False

                        print(f"Cleaned CSV saved to: {output_file_path} (Chunksize: {chunksize if chunksize else 'Full Load'})")

                    else:
                        print(f"Result: {count} columns in: {file_path} | No Duplicates Found.")

                except pd.errors.EmptyDataError:
                    column_data[file_path] = (0, None)
                    print(f" Error reading {file_path}: File is empty.")

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    # Check for no files found
    if not found_csv:
        print(f"No CSV files found in the {input_folder} folder.")

    # Optional save results
    if save_results:
        output_path = os.path.join(output_folder, txtfile_name)
        with open(output_path, 'w') as outfile:
            outfile.write("Result Summary (Columns and Duplicate Columns)\n\n")

            if not column_data:
                outfile.write(f"No CSV files found in the {input_folder} folder.\n")

            for file_path, (count, duplicate_cols) in column_data.items():
                relative_path = os.path.relpath(file_path, input_folder)
                outfile.write(f"File: {relative_path}\n")
                outfile.write(f"Columns: {count}\n")

                if duplicate_cols:
                    outfile.write(f"Duplicate Columns: {', '.join(duplicate_cols)}\n")
                    outfile.write(f"ACTION: Cleaned CSV saved to output folder.\n")
                else:
                    outfile.write("Duplicate Columns: None\n")
                outfile.write("---\n")

        if column_data:
            print(f"\nResults saved to: {output_path}")

    return column_data



# Function to count the number of Rows and Duplicate Rows, then Delete Duplicate Rows save new CSV
def count_row_duplicate(
        input_folder: str,
        output_folder: str,
        txtfile_name: str,
        save_results: bool = False,
        chunksize: Optional[int] = None
) -> Dict[str, Tuple[int, int]]:  # Returns a Dictionary mapping files to tuple for total rows and duplicate rows

    print("Counting Rows and Duplicate Rows...")
    row_data = {}

    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"The input folder '{input_folder}' does not exist.")
        return {}

    # Make sure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    found_csv = False  # No Csv files found

    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".csv"):
                found_csv = True  # Found CSV
                file_path = os.path.join(root, file_name)

                # Start processing
                total_rows = 0
                duplicate_rows_removed = 0
                is_first_chunk = True

                # FIX 3: Added prefix to prevent confusion/overwriting from other cleaning steps
                output_file_name = f"Cleaned_{file_name}"
                output_file_path = os.path.join(output_folder, output_file_name)

                try:  # Reading Full file in chunks
                    reader = pd.read_csv(
                        file_path,
                        sep=',',
                        iterator=True,
                        chunksize=chunksize
                    )

                    # Process file chunk-by-chunk
                    for chunk in reader:
                        initial_chunk_size = len(chunk)

                        # Row Counting
                        total_rows += initial_chunk_size

                        # Duplicate check - Keep the first occurrence (default)
                        df_clean_chunk = chunk.drop_duplicates(keep='first')

                        # Count Duplicates
                        rows_removed_in_chunk = initial_chunk_size - len(df_clean_chunk)
                        duplicate_rows_removed += rows_removed_in_chunk

                        # Save Cleaned Chunk
                        df_clean_chunk.to_csv(
                            output_file_path,
                            mode='w' if is_first_chunk else 'a',
                            header=is_first_chunk,
                            index=False
                        )
                        is_first_chunk = False

                except pd.errors.EmptyDataError:
                    row_data[file_path] = (0, 0)
                    print(f" Warning: File is empty (0 rows): {file_path}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                else:
                    # FIX 2: Store results and print statements moved outside the chunk loop (using 'else' after try/except)
                    # Store results ONLY after the file is completely processed
                    row_data[file_path] = (total_rows, duplicate_rows_removed)

                    print(f"File: {file_name} processed.")
                    print(f"    Total Rows: {total_rows}, Duplicates Removed: {duplicate_rows_removed}")
                    print(f"    Saved file to {output_file_path}")

    # Check for no files found
    if not found_csv:
        print(f"No CSV files found in the {input_folder} folder.")

    # Optional save results (Report summary)
    if save_results:
        output_path = os.path.join(output_folder, txtfile_name)
        with open(output_path, 'w') as outfile:
            outfile.write("--- Row Count and Duplication Removal Summary ---\n\n")

            if not row_data:
                outfile.write(f"No CSV files found in the {input_folder} folder.\n")

            for file_path, (total, removed) in row_data.items():
                relative_path = os.path.relpath(file_path, input_folder)
                outfile.write(f"File: {relative_path}\n")
                outfile.write(f"Total Rows (Initial): {total}\n")
                outfile.write(f"Duplicate Rows Removed: {removed}\n")
                outfile.write(f"Final Clean Rows: {total - removed}\n")
                outfile.write("---\n")

        if row_data:
            print(f"\nResults saved to: {output_path}")

    return row_data


# TODO add missing value handling



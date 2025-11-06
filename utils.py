import os
import pandas as pd
from typing import Dict, Tuple, Optional, List


# Function to count the number of Columns, and duplicate columns print them optionally save the results
def count_column_duplicate(
    input_folder: str,
    output_folder: str,
    txtfile_name: str,
    save_results: bool = False
) -> Dict[str, Tuple[int, Optional[List[str]]]]:


    column_data = {}

    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"The input folder '{input_folder}' does not exist.")

        # return empty dictionary
        return {}

    # Make sure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    print(f"Scanning {input_folder} for CSV files...")

    found_csv = False           # To track if any CSV were found

    for root, _, files in os.walk(input_folder):                # _ means ignore the subfolders inside root
        for file_name in files:
            if file_name.endswith(".csv"):
                found_csv = True                                # Update to ture for finding CSV files
                file_path = os.path.join(root, file_name)

                try:
                    df = pd.read_csv(
                        file_path,
                        nrows=0,            # Read only the first row to get the column names
                        sep=','             # Comma-separated values
                    )

                    columns = list(df.columns)          #
                    count = len(columns)

                    # Duplicate column name Check
                    duplicate_cols = [
                        col for col in set(columns)
                        if columns.count(col) > 1
                    ]

                    duplicate_to_report = sorted(duplicate_cols) if duplicate_cols else None

                    column_data[file_path] = (count, duplicate_to_report)


                    # Printing results
                    if duplicate_to_report:
                        print(f"Result: {count} columns in: {file_path} | Duplicates Found: {', '.join(duplicate_to_report)}")
                    else:
                        print(f"Result: {count} columns in: {file_path} | No Duplicates Found.")

                except pd.errors.EmptyDataError:            # If files are empty
                    column_data[file_path] = (0, None)
                    print(f" Error reading {file_path}: File is empty.")

                except Exception as e:                      # General errors
                    print(f"Error reading {file_path}: {e}")

    # Check for no files found
    if not found_csv:
        print(f"No CSV files found in the {input_folder} folder.")

    # Optional save results
    if save_results:
        output_path = os.path.join(output_folder, txtfile_name)
        with open(output_path, 'w') as outfile:
            outfile.write("Result summary (Columns, Duplicate Columns Count)")

            if not column_data:
                outfile.write(f"No CSV files found in the {input_folder} folder.")

            for file_path, (count, duplicate_cols) in column_data.items():
                # Display path
                relative_path = os.path.relpath(file_path, input_folder)
                outfile.write(f"File: {relative_path}\n")
                outfile.write(f"Columns: {count}\n")

                if duplicate_cols:
                    outfile.write(f"Duplicate Columns: {', '.join(duplicate_cols)}\n")
                else:
                    outfile.write("No duplicate columns found.\n")

        if column_data:
            print(f"Results saved to: {output_path}")

    return column_data
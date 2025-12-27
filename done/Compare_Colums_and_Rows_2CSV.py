import os
import pandas as pd

# Folders
raw_folder = "Raw_Data_2017"
processed_folder = "Processed_Data_2017"

# Step 1: Get columns from the first raw CSV as reference
first_raw_csv = next(f for f in os.listdir(raw_folder) if f.endswith(".csv"))
raw_columns = list(pd.read_csv(os.path.join(raw_folder, first_raw_csv), nrows=0).columns)
print(f"Reference columns from raw CSV ({first_raw_csv}): {raw_columns}")

# Step 2: Check that all raw CSVs have the same columns
for filename in os.listdir(raw_folder):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(raw_folder, filename), nrows=0)
        if list(df.columns) != raw_columns:
            print(f"Column mismatch in raw CSV: {filename}")
            print("Columns:", list(df.columns))

# Step 3: Check that all processed CSVs have the same columns and order
for filename in os.listdir(processed_folder):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(processed_folder, filename), nrows=0)
        if list(df.columns) != raw_columns:
            print(f"Column mismatch in processed CSV: {filename}")
            print("Columns:", list(df.columns))

# Step 4: Collect all rows from raw CSVs
raw_rows = set()
for filename in os.listdir(raw_folder):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(raw_folder, filename), dtype=str)
        raw_rows.update(tuple(row) for row in df.values)

print(f"Total rows in raw data: {len(raw_rows)}")

# Step 5: Collect all rows from processed CSVs
processed_rows = set()
for filename in os.listdir(processed_folder):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(processed_folder, filename), dtype=str)
        processed_rows.update(tuple(row) for row in df.values)

print(f"Total rows in processed data: {len(processed_rows)}")

# Step 6: Compare rows
missing_rows = raw_rows - processed_rows
extra_rows = processed_rows - raw_rows
# Convert extra_rows set to list to access elements
extra_rows_list = list(extra_rows)

print("Showing 10 extra rows in processed CSVs:")
for i, row in enumerate(extra_rows_list[:10]):
    print(f"Row {i+1}: {row}")


if not missing_rows and not extra_rows:
    print("All raw data rows are present in the processed folder.")
else:
    if missing_rows:
        print(f"{len(missing_rows)} rows from raw data are missing in processed CSVs.")
    if extra_rows:
        print(f"{len(extra_rows)} extra rows found in processed CSVs that were not in raw data.")

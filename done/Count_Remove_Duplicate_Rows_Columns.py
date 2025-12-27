import os
import pandas as pd

# ========= CONFIG =========
INPUT_FOLDER = "uploads/raw"
CHUNK_SIZE = 500_000  # For big files

# ======= ASK USER WHAT TO DO ========
print("What do you want to check/do for each file? Answer 'y' or 'n'.")

do_col_count      = input("Show column count? (y/n): ").lower() == 'y'
do_row_count      = input("Show row count? (y/n): ").lower() == 'y'
do_dup_colnames   = input("Check for duplicate column names? (y/n): ").lower() == 'y'
do_dup_cols_remove= False
do_dup_rows       = input("Check for duplicate rows? (y/n): ").lower() == 'y'
do_dup_rows_remove= False
do_missing        = input("Check for missing values? (y/n): ").lower() == 'y'

# If you want to *automatically* ask about removals only if relevant:
ask_remove_dup_cols = do_dup_colnames
ask_remove_dup_rows = do_dup_rows

# ========== MAIN SCRIPT ===========
for filename in os.listdir(INPUT_FOLDER):
    if not filename.endswith(".csv"):
        continue

    file_path = os.path.join(INPUT_FOLDER, filename)
    print(f"\nProcessing file: {filename}")

    # Load CSV in chunks if large
    dfs = []
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, dtype=str):
        dfs.append(chunk)
    df = pd.concat(dfs, ignore_index=True)
    del dfs  # free RAM

    # Column count
    if do_col_count:
        print(f"Number of columns: {df.shape[1]}")

    # Row count
    if do_row_count:
        print(f"Number of rows: {df.shape[0]}")

    # Detect duplicate and auto-renamed columns (like .1, .2)
    if do_dup_colnames:
        base_names = [c.split('.')[0] for c in df.columns]  # remove .1, .2 suffixes
        col_counts = pd.Series(base_names).value_counts()
        duplicate_bases = col_counts[col_counts > 1]

        if not duplicate_bases.empty:
            print(f"Duplicate or renamed duplicate columns detected (like .1, .2): {list(duplicate_bases.index)}")

            # Optionally remove the duplicates
            if ask_remove_dup_cols:
                remove_cols = input("Remove duplicates keeping only first occurrence? (y/n): ")
                if remove_cols.lower() == "y":
                    seen = set()
                    unique_cols = []
                    for c in df.columns:
                        base = c.split('.')[0]
                        if base not in seen:
                            unique_cols.append(c)
                            seen.add(base)
                    df = df[unique_cols]
                    out_path = os.path.join(INPUT_FOLDER, f"{os.path.splitext(filename)[0]}_nodupcol.csv")
                    df.to_csv(out_path, index=False)
                    print(f"Saved file without duplicate columns: {out_path}")
        else:
            print("No duplicate or renamed duplicate column names.")


    # Duplicate rows
    if do_dup_rows:
        dup_rows = df.duplicated().sum()
        print(f"Duplicate rows: {dup_rows}")
        if dup_rows > 0 and ask_remove_dup_rows:
            remove_rows = input("Remove duplicate rows and save updated CSV? (y/n): ")
            if remove_rows.lower() == "y":
                rows_before = df.shape[0]
                df = df.drop_duplicates()
                out_path = os.path.join(INPUT_FOLDER, f"{os.path.splitext(filename)[0]}_nodup.csv")
                df.to_csv(out_path, index=False)
                print(f"Saved file without duplicate rows: {out_path}")

    # Missing values
    if do_missing:
        missing = df.isnull().sum()
        missing_dict = {col: n for col, n in missing.items() if n > 0}
        if missing_dict:
            print("Missing values per column:")
            for col, count in missing_dict.items():
                print(f"  {col}: {count}")
        else:
            print("No missing values found.")

import os
import pandas as pd
from collections import Counter

# ---------------- Configuration ----------------
DATASET1_PATH = "Combined_DATA"
DATASET2_PATH = "Combined_DATA_1"
OUTPUT_PATH = "Normalized_SET"

chunk_size = 500_000  # adjust for memory
log_file = os.path.join(OUTPUT_PATH, "label_change_log.txt")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ---------------- Label Mapping ----------------
mapping = {
    "Botnet_ARES": "Bot",
    "DDoS_LOIT": "DDoS_LOIC_HTTP",
    "DoS_GoldenEye": "DoS_Golden_Eye",
    "DoS_Hulk": "DoS_HULK",
    "DoS_Slowhttptest": "DoS_SlowHTTP",
    "DoS_Slowloris": "DoS_Slowloris",
    "Benign": "Benign",
    "FTP-Patator": "Brute_Force_FTP",
    "Heartbleed": "Heartbleed",
    "Port_Scan": "Port_Scan",
    "SSH-Patator": "Brute_Force_SSH",
    "Web_Brute_Force": "Brute_Force_Web",
    "Web_SQL_Injection": "SQL_Injection",
    "Web_XSS": "Brute_Force_XSS"
}

if os.path.exists(log_file):
    os.remove(log_file)

# ---------------- Helper Functions ----------------
def detect_label_column(columns):
    """Find label column name ignoring case/spaces"""
    for col in columns:
        if col.strip().lower() == "label":
            return col
    return None

def get_first_csv_columns(folder):
    """Get columns from the first CSV in folder"""
    for f in os.listdir(folder):
        if f.endswith(".csv"):
            path = os.path.join(folder, f)
            try:
                cols = pd.read_csv(path, nrows=0).columns.tolist()
                return set(cols)
            except Exception as e:
                print(f"Error reading {f}: {e}")
    return set()

def normalize_folder(folder_path, drop_columns):
    """Normalize all CSVs in a folder"""
    for file in os.listdir(folder_path):
        if not file.endswith(".csv"):
            continue

        input_path = os.path.join(folder_path, file)
        output_path = os.path.join(OUTPUT_PATH, file)

        print(f"Processing {file} from {folder_path}...")

        if os.path.exists(output_path):
            os.remove(output_path)

        change_counter = Counter()

        for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
            label_col = detect_label_column(chunk.columns)
            if not label_col:
                print(f"No 'Label' column found in {file}, skipping chunk {i}.")
                continue

            # Drop extra columns if any
            if drop_columns:
                existing = [c for c in chunk.columns if c not in drop_columns]
                chunk = chunk[existing]

            # Normalize labels
            original_labels = chunk[label_col].copy()
            chunk.loc[:, label_col] = chunk[label_col].map(mapping).fillna(chunk[label_col])

            changed = original_labels[original_labels != chunk[label_col]]
            for old_label, count in changed.value_counts().items():
                new_label = mapping.get(old_label, old_label)
                change_counter[(old_label, new_label)] += count

            # Save chunk
            chunk.to_csv(output_path, mode="a", index=False, header=not os.path.exists(output_path))

        # Log results
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"\n{file}\n")
            if change_counter:
                for (old, new), count in change_counter.items():
                    log.write(f"  {old:<20} -> {new:<20} {count}\n")
            else:
                log.write("  No label changes detected.\n")

        print(f"Finished {file}\n")


# ---------------- Column Comparison ----------------
print("Checking column differences between Dataset1 and Dataset2...")

cols1 = get_first_csv_columns(DATASET1_PATH)
cols2 = get_first_csv_columns(DATASET2_PATH)

extra_in_1 = cols1 - cols2
extra_in_2 = cols2 - cols1

drop_from_dataset1 = set()
drop_from_dataset2 = set()

if extra_in_1:
    print(f"\nDataset1 has {len(extra_in_1)} columns not in Dataset2:")
    print(", ".join(sorted(extra_in_1)))
    ans = input("Remove them before saving? (y/n): ").strip().lower()
    if ans == "y":
        drop_from_dataset1 = extra_in_1

if extra_in_2:
    print(f"\nDataset2 has {len(extra_in_2)} columns not in Dataset1:")
    print(", ".join(sorted(extra_in_2)))
    ans = input("Remove them before saving? (y/n): ").strip().lower()
    if ans == "y":
        drop_from_dataset2 = extra_in_2

# ---------------- Run Normalization ----------------
normalize_folder(DATASET1_PATH, drop_from_dataset1)
normalize_folder(DATASET2_PATH, drop_from_dataset2)

print("\nAll files processed successfully.")
print(f"Normalized CSVs saved in: {OUTPUT_PATH}")
print(f"Label change log saved as: {log_file}")

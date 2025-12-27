import os
import pandas as pd
from collections import Counter, defaultdict

PARENT_FOLDER = "Attacks_Cleaned"   # which has the csv
REPORTS_FOLDER = "a"        # saves the reports here
TRAIN_FOLDER = "Training_Attack"             #savves training spli
TEST_FOLDER = "Testing_Attack"                  #saves testing split
TRAIN_CSV_NAME = "training.csv"          #file name for training split
TEST_CSV_NAME = "test.csv"               #file name for testing split
CHUNK_SIZE = 1_500_000
LABEL_COLUMN = "label"
OUTPUT_ENCODING = "utf-8"
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

def safe_lower(x):
    try:
        return str(x).strip().lower()
    except Exception:
        return str(x)

def count_labels_first(file_path):
    label_counts = Counter()
    total = 0
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=True):
        label_col = next((c for c in chunk.columns if c.lower() == LABEL_COLUMN.lower()), None)
        if not label_col:
            print(f"No '{LABEL_COLUMN}' column found in {file_path}. Skipping.")
            return Counter(), 0
        s = chunk[label_col].dropna()
        label_counts.update(s)
        total += len(chunk)
    return label_counts, total

def plan_stratified_split(counts, train_ratio=0.6):
    train_needed = {}
    test_needed = {}
    for label, cnt in counts.items():
        t = int(round(cnt * train_ratio))
        t = max(0, min(cnt, t))
        train_needed[label] = t
        test_needed[label] = cnt - t
    return train_needed, test_needed

def build_report(file_path, total_samples, label_counts, train_counts, test_counts):
    benign_key_variants = {k for k in label_counts.keys() if safe_lower(k) == "benign"}
    benign_total = sum(label_counts[k] for k in benign_key_variants) if benign_key_variants else 0
    attack_total = total_samples - benign_total
    lines = []
    lines.append(f"Report for {file_path}")
    lines.append("=" * 60)
    lines.append(f"Total samples: {total_samples}")
    lines.append(f"Benign: {benign_total}")
    lines.append(f"Attacks: {attack_total}")
    lines.append("")
    lines.append("Breakdown by label (full dataset):")
    lines.append("-" * 40)
    for label, cnt in label_counts.items():
        lines.append(f"{label:<25}: {cnt}")
    lines.append("")
    lines.append("Training split label counts (60% target):")
    lines.append("-" * 40)
    for label, cnt in sorted(train_counts.items(), key=lambda x: str(x[0]).lower()):
        lines.append(f"{label:<25}: {cnt}")
    lines.append("")
    lines.append("Test split label counts (40% target):")
    lines.append("-" * 40)
    for label, cnt in sorted(test_counts.items(), key=lambda x: str(x[0]).lower()):
        lines.append(f"{label:<25}: {cnt}")
    return "\n".join(lines)

def write_report_text(report_text, file_path):
    rel = os.path.relpath(file_path, PARENT_FOLDER)
    rel_name = rel.replace(os.sep, "_")
    out_path = os.path.join(REPORTS_FOLDER, f"{os.path.splitext(rel_name)[0]}.txt")
    with open(out_path, "w", encoding=OUTPUT_ENCODING) as f:
        f.write(report_text)
    print(f"Saved report to {out_path}")

def split_and_write(file_path):
    # Pass 1: counts
    try:
        label_counts, total_rows = count_labels_first(file_path)
    except ValueError as e:
        print(f"Error: {e} in {file_path}. Maybe no '{LABEL_COLUMN}' column.")
        return
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    if total_rows == 0:
        print(f"No rows found in {file_path}, skipping.")
        return

    train_needed, test_needed = plan_stratified_split(label_counts, train_ratio=0.6)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    train_path = os.path.join(TRAIN_FOLDER, TRAIN_CSV_NAME)
    test_path = os.path.join(TEST_FOLDER, TEST_CSV_NAME)

    if os.path.exists(train_path):
        os.remove(train_path)
    if os.path.exists(test_path):
        os.remove(test_path)

    written_train = defaultdict(int)
    written_test = defaultdict(int)

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        low_memory=True
    ):
        if LABEL_COLUMN not in chunk.columns:
            print(f"'{LABEL_COLUMN}' column not found in {file_path}, skipping this file.")
            return

        train_rows = []
        test_rows = []

        for idx, row in chunk.iterrows():
            label_col = next((c for c in chunk.columns if c.lower() == LABEL_COLUMN.lower()), LABEL_COLUMN)
            label = row[label_col]
            if label not in train_needed:
                train_needed[label] = 0
                test_needed[label] = 0

            if written_train[label] < train_needed[label]:
                train_rows.append(row)
                written_train[label] += 1
            elif written_test[label] < test_needed[label]:
                test_rows.append(row)
                written_test[label] += 1
            else:
                if written_train[label] < written_test[label]:
                    train_rows.append(row)
                    written_train[label] += 1
                else:
                    test_rows.append(row)
                    written_test[label] += 1

        if train_rows:
            train_df = pd.DataFrame(train_rows)
            header = not os.path.exists(train_path)
            train_df.to_csv(train_path, mode="a", index=False, header=header)
        if test_rows:
            test_df = pd.DataFrame(test_rows)
            header = not os.path.exists(test_path)
            test_df.to_csv(test_path, mode="a", index=False, header=header)

    final_train_counts = dict(written_train)
    final_test_counts = dict(written_test)

    # Optional: write a report per input file
    report_text = build_report(
        file_path=file_path,
        total_samples=total_rows,
        label_counts=label_counts,
        train_counts=final_train_counts,
        test_counts=final_test_counts
    )
    write_report_text(report_text, file_path)

    print(f"Done: {file_path}")
    print("Train counts per label:", final_train_counts)
    print("Test counts per label:", final_test_counts)

def label_report(file_path):
    label_counts, total_rows = count_labels_first(file_path)
    if total_rows == 0:
        print(f"No rows found in {file_path}, skipping.")
        return
    print(f"\nLabel count for {file_path}:")
    for label, cnt in label_counts.items():
        print(f"  {label}: {cnt}")
    print(f"Total samples: {total_rows}")
    save = input("Do you want to save this label report? (y/n): ").strip().lower()
    if save == 'y':
        report_text = f"Label count report for {file_path}\n\nTotal samples: {total_rows}\n"
        for label, cnt in label_counts.items():
            report_text += f"{label}: {cnt}\n"
        write_report_text(report_text, file_path)
    else:
        print("Label report not saved.")

def main():
    for root, dirs, files in os.walk(PARENT_FOLDER):
        for file in files:
            if not file.endswith(".csv"):
                continue
            file_path = os.path.join(root, file)
            print(f"\nProcessing file: {file_path}")
            choice = input("Label test only (shows and/or saves label counts, no split)? (y/n): ").strip().lower()
            if choice == 'y':
                label_report(file_path)
                continue
            do_split = input("Do you want to perform train-test split on this file? (y/n): ").strip().lower()
            if do_split == 'y':
                split_and_write(file_path)
            else:
                print("Skipping train-test split for this file.")

if __name__ == "__main__":
    main()

import os
import pandas as pd
import matplotlib.pyplot as plt

# Parent folder containing all CSVs
parent_folder = "Raw_Data_2017"

# Collect global label counts
overall_counts = {}

# Walk through all CSV files
for root, dirs, files in os.walk(parent_folder):
    for file in files:
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(root, file)
        print(f"Processing {file_path}...")

        try:
            # Load just the first row to detect the "Label" column (case-insensitive)
            header_df = pd.read_csv(file_path, nrows=0)
            label_col = None
            for col in header_df.columns:
                if col.lower() == "label":
                    label_col = col
                    break

            if label_col is None:
                print(f"No 'Label' column in {file_path}, skipping.")
                continue

            # Now load only the label column, in chunks (to save memory)
            for chunk in pd.read_csv(file_path, usecols=[label_col], chunksize=100000):
                file_counts = chunk[label_col].value_counts().to_dict()
                for lbl, cnt in file_counts.items():
                    overall_counts[lbl] = overall_counts.get(lbl, 0) + cnt

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

# Convert to DataFrame for saving
summary_df = pd.DataFrame(list(overall_counts.items()), columns=["Label", "Count"])
summary_df.to_csv("Overall_Label_Distribution.csv", index=False)

# --- Visualization ---
plt.figure(figsize=(10, 6))
summary_df.set_index("Label")["Count"].plot(kind="bar")
plt.title("Overall Class Distribution (Benign vs. Attack Types)")
plt.ylabel("Number of Samples")
plt.xlabel("Label")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Overall_Class_Distribution.png", dpi=300)
plt.show()

print("Saved: Overall_Label_Distribution.csv and Overall_Class_Distribution.png")

import os
import glob
import pandas as pd
import dask.dataframe as dd

# ==========================================
# 1. CONFIGURATION
# ==========================================
BENIGN_FOLDER = "./Benign"
ATTACK_FOLDER = "./Attacks"
OUTPUT_CSV = "model1_final_training_data.csv" # Single final file name

LABEL_COL = "label"         # Make sure this matches your CSV column header (case-sensitive)
MIN_ATTACK_SAMPLES = 500    # Drop attacks with fewer rows than this
MAX_ATTACK_SAMPLES = 30000  # Cap massive attacks at this number
BENIGN_RATIO = 2            # Keep 2 Benign rows for every 1 Attack row

print("--- STARTING DASK PIPELINE ---")

# ==========================================
# 2. LAZY LOAD MULTIPLE CSVs & DEDUPLICATE
# ==========================================
print("Loading and deduplicating Benign folder...")
df_benign = dd.read_csv(
    os.path.join(BENIGN_FOLDER, "*.csv"),
    assume_missing=True,
    na_values=['not a complete handshake']  # <-- Fix for the mismatched dtype error
)
df_benign[LABEL_COL] = 'Benign'
df_benign = df_benign.drop_duplicates()

print("Loading and deduplicating Attack folder...")
df_attack = dd.read_csv(
    os.path.join(ATTACK_FOLDER, "*.csv"),
    assume_missing=True,
    na_values=['not a complete handshake']  # <-- Fix for the mismatched dtype error
)
df_attack = df_attack.drop_duplicates()

# ==========================================
# 3. COMPUTE ATTACK COUNTS & FILTER/CAP
# ==========================================
print("\nComputing attack distributions...")
attack_counts = df_attack[LABEL_COL].value_counts().compute()
print("Original Attack Counts:\n", attack_counts)

processed_attack_dfs = []
total_balanced_attacks = 0

for attack_name, count in attack_counts.items():
    if attack_name == 'Benign':
        continue

    sub_df = df_attack[df_attack[LABEL_COL] == attack_name]

    if count < MIN_ATTACK_SAMPLES:
        print(f"  -> Dropping '{attack_name}': {count} rows (Below minimum)")
        continue

    if count > MAX_ATTACK_SAMPLES:
        print(f"  -> Capping '{attack_name}': Downsampling to {MAX_ATTACK_SAMPLES}")
        fraction = MAX_ATTACK_SAMPLES / count
        sub_df = sub_df.sample(frac=fraction, random_state=42)
        total_balanced_attacks += MAX_ATTACK_SAMPLES
    else:
        print(f"  -> Keeping '{attack_name}': {count} rows")
        total_balanced_attacks += count

    processed_attack_dfs.append(sub_df)

# Combine processed attacks lazily
df_attack_final = dd.concat(processed_attack_dfs)

# ==========================================
# 4. SAMPLE BENIGN PROPORTIONALLY
# ==========================================
print("\n--- PROCESSING BENIGN ---")
target_benign_count = total_balanced_attacks * BENIGN_RATIO
total_benign_count = len(df_benign) # Triggers computation

if total_benign_count > target_benign_count:
    fraction = target_benign_count / total_benign_count
    print(f"Sampling Benign down to ~{target_benign_count} rows...")
    df_benign_final = df_benign.sample(frac=fraction, random_state=42)
else:
    print("Keeping all available Benign rows.")
    df_benign_final = df_benign

# ==========================================
# 5. FIND COMMON COLUMNS, MERGE, LABEL, AND EXPORT
# ==========================================
print("\n--- MERGING AND EXPORTING SINGLE CSV ---")

# 1. Find the exact overlapping columns (The Intersection Fix)
common_cols = list(set(df_benign_final.columns).intersection(set(df_attack_final.columns)))

# Make sure the label column is included just in case
if LABEL_COL not in common_cols:
    common_cols.append(LABEL_COL)

print(f"Attack dataset had {len(df_attack_final.columns)} columns.")
print(f"Benign dataset had {len(df_benign_final.columns)} columns.")
print(f"Keeping only the {len(common_cols)} shared columns to prevent Data Leakage...")

# 2. Slice both datasets to only include the shared columns
df_benign_final = df_benign_final[common_cols]
df_attack_final = df_attack_final[common_cols]

# 3. Now merge them fairly
df_final = dd.concat([df_benign_final, df_attack_final])

# Map to Binary (0 = Benign, 1 = Attack) directly in the existing column
df_final[LABEL_COL] = df_final[LABEL_COL].map(lambda x: 0 if x == 'Benign' else 1, meta=(LABEL_COL, 'int64'))

# Shuffle the final distributed dataset
df_final = df_final.sample(frac=1.0, random_state=42)

# Export to a single unified CSV file using Dask's compute() to standard pandas
print(f"Combining partitions and saving to single file: {OUTPUT_CSV}...")
df_pd = df_final.compute()
df_pd.to_csv(OUTPUT_CSV, index=False)

print(f"\nPipeline Complete! Saved final clean dataset to {OUTPUT_CSV}")
print(f"Final dataset shape: {df_pd.shape}")
print("Final Binary Label Distribution:")
print(df_pd[LABEL_COL].value_counts())
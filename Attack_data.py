import os
import glob
import pandas as pd
import dask.dataframe as dd

# ==========================================
# 1. CONFIGURATION
# ==========================================
ATTACK_FOLDER = "./Attacks"
OUTPUT_CSV = "model2_multiclass_training_data.csv"
MAPPING_TXT = "model2_label_mapping.txt"

LABEL_COL = "label"
MIN_ATTACK_SAMPLES = 500
MAX_ATTACK_SAMPLES = 30000

# Your exact list of approved features from Model 1 (including 'label')
TARGET_COLUMNS = [
    "subflow_bwd_packets", "fwd_payload_bytes_max", "bytes_rate", "bwd_packets_iat_min",
    "packets_iat_mean", "packets_rate", "payload_bytes_max", "bwd_bulk_duration",
    "ece_flag_counts", "fwd_fin_flag_counts", "bwd_packets_iat_mean", "bwd_packets_iat_max",
    "ack_flag_counts", "fwd_psh_flag_counts", "std_header_bytes", "bwd_syn_flag_counts",
    "bwd_packets_rate", "fwd_mean_header_bytes", "total_payload_bytes", "fwd_payload_bytes_std",
    "dst_port", "bwd_rst_flag_counts", "fin_flag_counts", "duration", "avg_bwd_bulk_rate",
    "fwd_syn_flag_counts", "fwd_total_payload_bytes", "fwd_packets_rate", "fwd_packets_iat_total",
    "fwd_packets_count", "bwd_payload_bytes_std", "bwd_total_payload_bytes", "fwd_ack_flag_counts",
    "packet_iat_total", "packets_count", "subflow_bwd_bytes", "payload_bytes_variance",
    "bwd_ack_flag_counts", "fwd_packets_iat_max", "bwd_fin_flag_counts", "packet_iat_std",
    "syn_flag_counts", "bwd_bulk_state_count", "subflow_fwd_packets", "bwd_init_win_bytes",
    "fwd_packets_iat_mean", "fwd_bytes_rate", "bwd_payload_bytes_max", "fwd_packets_iat_std",
    "mean_header_bytes", "fwd_rst_flag_counts", "bwd_packets_iat_total", "packet_iat_min",
    "bwd_total_header_bytes", "rst_flag_counts", "subflow_fwd_bytes", "fwd_init_win_bytes",
    "payload_bytes_std", "cwr_flag_counts", "label", "avg_segment_size", "bwd_mean_header_bytes",
    "bwd_bytes_rate", "fwd_packets_iat_min", "bwd_packets_count", "down_up_rate",
    "fwd_avg_segment_size", "packet_iat_max", "total_header_bytes", "payload_bytes_mean",
    "psh_flag_counts", "bwd_packets_iat_std", "fwd_total_header_bytes"
]

print("--- STARTING DASK PIPELINE FOR MODEL 2 ---")

# ==========================================
# 2. LAZY LOAD ATTACK CSVs & DEDUPLICATE
# ==========================================
print("Loading and deduplicating Attack folder...")
df_attack = dd.read_csv(
    os.path.join(ATTACK_FOLDER, "*.csv"),
    assume_missing=True,
    na_values=['not a complete handshake']
)
df_attack = df_attack.drop_duplicates()

# ==========================================
# 3. COMPUTE ATTACK COUNTS & FILTER/CAP
# ==========================================
print("\nComputing attack distributions...")
attack_counts = df_attack[LABEL_COL].value_counts().compute()
print("Original Attack Counts:\n", attack_counts)

processed_attack_dfs = []

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
    else:
        print(f"  -> Keeping '{attack_name}': {count} rows")

    processed_attack_dfs.append(sub_df)

df_attack_final = dd.concat(processed_attack_dfs)

# ==========================================
# 4. SHUFFLE AND COMPUTE TO PANDAS
# ==========================================
print("\n--- SHUFFLING AND COMPUTING ---")
df_attack_final = df_attack_final.sample(frac=1.0, random_state=42)

print(f"Pulling into memory...")
df_pd = df_attack_final.compute()

# ==========================================
# 5. FILTER COLUMNS, CREATE MAPPING, AND EXPORT
# ==========================================
print(f"\n--- FILTERING TO EXACT MODEL 1 FEATURES ---")
# Keep only the columns specified in TARGET_COLUMNS
# Using errors='ignore' ensures it doesn't crash if a column has a slight typo
df_pd = df_pd[TARGET_COLUMNS]

print("\n--- CREATING LABEL MAPPING ---")
unique_attacks = sorted(df_pd[LABEL_COL].unique().tolist())
label_mapping = {attack: idx for idx, attack in enumerate(unique_attacks)}

with open(MAPPING_TXT, "w") as f:
    f.write("Multiclass Label Mapping:\n")
    f.write("-" * 25 + "\n")
    for attack, idx in label_mapping.items():
        f.write(f"{idx}: {attack}\n")
print(f"Saved mapping reference to {MAPPING_TXT}")

# Apply the mapping
df_pd[LABEL_COL] = df_pd[LABEL_COL].map(label_mapping)

print(f"\nSaving final multiclass dataset to {OUTPUT_CSV}...")
df_pd.to_csv(OUTPUT_CSV, index=False)

print("\nPipeline Complete!")
print(f"Final dataset shape: {df_pd.shape} (Matches Model 1 features exactly)")
print("Final Label Distribution (Mapped):")
print(df_pd[LABEL_COL].value_counts())
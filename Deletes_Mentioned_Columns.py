import os
import pandas as pd

# --- 1. Configuration ---
input_folder = "No_Missing"       # Folder with original CSVs
output_folder = "Downscale_Csv_2018_Cleaned"  # Folder to save cleaned CSVs
columns_to_remove = [
    'flow_id','src_ip','dst_ip','timestamp','active_cov','active_max','active_mean','active_median',
    'active_min','active_mode','active_skewness','active_std','active_variance','bwd_cwr_flag_counts',
    'bwd_cwr_flag_percentage_in_bwd_packets','bwd_cwr_flag_percentage_in_total','bwd_payload_bytes_min',
    'bwd_urg_flag_counts','bwd_urg_flag_percentage_in_bwd_packets','bwd_urg_flag_percentage_in_total',
    'fwd_payload_bytes_min','fwd_urg_flag_counts','fwd_urg_flag_percentage_in_fwd_packets',
    'fwd_urg_flag_percentage_in_total','handshake_state','idle_cov','idle_max','idle_mean','idle_median',
    'idle_min','idle_mode','idle_skewness','idle_std','idle_variance','median_bwd_header_bytes_delta_len',
    'median_fwd_header_bytes_delta_len','median_header_bytes_delta_len','mode_bwd_header_bytes_delta_len',
    'mode_fwd_header_bytes_delta_len','payload_bytes_min','urg_flag_counts','protocol',
    'urg_flag_percentage_in_total','cov_bwd_payload_bytes_delta_len','cov_fwd_header_bytes_delta_len',
    'cov_fwd_packets_delta_len','cov_fwd_payload_bytes_delta_len','cov_header_bytes_delta_len',
    'cov_packets_delta_len','cov_payload_bytes_delta_len'
]

chunk_size = 1_000_000  # Adjust based on memory

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

print(f"Searching for CSV files in: {input_folder}\n")

# --- 2. Process each file ---
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        input_csv_path = os.path.join(input_folder, filename)
        output_csv_path = os.path.join(output_folder, filename)  # same name in new folder

        print(f"Processing '{filename}'...")

        # Read first row to detect which columns exist
        try:
            df_head = pd.read_csv(input_csv_path, nrows=0)
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            continue

        # Determine which columns will be deleted
        cols_found = [col for col in df_head.columns if col in columns_to_remove]
        print(f"  Number of columns to delete: {len(cols_found)}")
        print(f"  Columns found for deletion: {cols_found}")
        print("  Moving to deletion and saving...")

        # Process file in chunks
        is_first_chunk = True
        for chunk in pd.read_csv(input_csv_path, chunksize=chunk_size, low_memory=False):
            chunk.drop(columns=cols_found, inplace=True, errors='ignore')
            if is_first_chunk:
                chunk.to_csv(output_csv_path, index=False, mode='w')
                is_first_chunk = False
            else:
                chunk.to_csv(output_csv_path, index=False, mode='a', header=False)

        print(f"----> Successfully created cleaned file in '{output_folder}'\n")

print("All files processed successfully.")

import os
import pandas as pd

# --- 1. Configuration ---
input_folder = "Raw_Data_2018/BCCC-CIC-CSE-IDS2018"
output_folder = "Column_Cleaned"

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
    'cov_packets_delta_len','cov_payload_bytes_delta_len',

    'mean_payload_bytes_delta_len','fwd_payload_bytes_mode','mode_header_bytes_delta_len', 'payload_bytes_mode',
    'min_header_bytes_delta_len', 'bwd_ece_flag_percentage_in_bwd_packets', 'avg_fwd_bytes_per_bulk','avg_fwd_packets_per_bulk',
    'packets_IAT_mode','fwd_syn_flag_counts','fwd_bulk_per_packet','fwd_bulk_total_size','bwd_rst_flag_counts',
    'bwd_syn_flag_counts','fwd_bulk_state_count','bwd_ece_flag_percentage_in_total','bwd_ece_flag_counts',
    'mean_fwd_payload_bytes_delta_len','mode_fwd_payload_bytes_delta_len','mode_payload_bytes_delta_len','std_bwd_packets_delta_time',
    'cov_packets_delta_time', 'cov_fwd_packets_delta_time', 'mean_packets_delta_time', 'variance_bwd_packets_delta_time',
    'fwd_payload_bytes_cov', 'mode_packets_delta_len','min_fwd_packets_delta_time', 'avg_fwd_bulk_rate', 'mean_bwd_packets_delta_time',
    'fwd_packets_IAT_min', 'fwd_payload_bytes_median', 'rst_flag_counts', 'skewness_packets_delta_time','skewness_fwd_packets_delta_time',
    'variance_packets_delta_time', 'bwd_rst_flag_percentage_in_bwd_packets','fwd_variance_header_bytes', 'bwd_fin_flag_counts',
    'bwd_fin_flag_percentage_in_total', 'rst_flag_percentage_in_total', 'handshake_duration', 'std_packets_delta_time', 'fwd_packets_IAT_mode',
    'psh_flag_percentage_in_total', 'payload_bytes_median', 'variance_fwd_packets_delta_time', 'bwd_fin_flag_percentage_in_bwd_packets',
    'fwd_std_header_bytes', 'max_bwd_packets_delta_time', 'mode_fwd_packets_delta_time','skewness_bwd_header_bytes_delta_len',
    'mode_bwd_packets_delta_time', 'bwd_packets_IAT_mode', 'max_bwd_header_bytes_delta_len', 'mean_bwd_header_bytes_delta_len',
    'skewness_fwd_payload_bytes_delta_len', 'skewness_payload_bytes_delta_len', 'median_bwd_packets_delta_len', 'packet_IAT_min',
    'bwd_variance_header_bytes', 'std_fwd_packets_delta_time', 'mean_fwd_packets_delta_time', 'median_fwd_packets_delta_len',
    'median_bwd_payload_bytes_delta_len', 'min_fwd_header_bytes_delta_len', 'fwd_psh_flag_percentage_in_fwd_packets', 'cov_bwd_packets_delta_time',
    'fwd_packets_count', 'fwd_ack_flag_counts', 'mean_fwd_packets_delta_len', 'bwd_rst_flag_percentage_in_total', 'avg_bwd_bulk_rate',
    'fwd_cov_header_bytes', 'fwd_psh_flag_counts', 'fwd_payload_bytes_skewness', 'fwd_rst_flag_percentage_in_fwd_packets',
    'fwd_syn_flag_percentage_in_total',
]

chunk_size = 1_000_000
os.makedirs(output_folder, exist_ok=True)

print(f"Total columns planned for deletion: {len(columns_to_remove)}\n")
print(f"Searching for CSV files in: {input_folder}\n")

# --- 2. Process each file ---
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        input_csv_path = os.path.join(input_folder, filename)
        output_csv_path = os.path.join(output_folder, filename)

        print(f"Processing '{filename}'...")

        # Read first row to detect all columns
        try:
            df_head = pd.read_csv(input_csv_path, nrows=0)
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            continue

        all_columns = df_head.columns.tolist()

        # Determine which columns exist in the file
        cols_found = [col for col in columns_to_remove if col in all_columns]
        cols_not_found = [col for col in columns_to_remove if col not in all_columns]

        print(f"  Columns found for deletion: {len(cols_found)}")
        if cols_found:
            print(f"    Found: {cols_found}")
        print(f"  Columns not found in file: {len(cols_not_found)}")
        if cols_not_found:
            print(f"    Not found: {cols_not_found}")

        # Process file in chunks and delete columns
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

import os
import pandas as pd

# --- 1. Configuration ---
input_folder = "Attacks"
output_folder = "Attacks_Cleaned"

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
    'cov_packets_delta_len','cov_payload_bytes_delta_len', 'protocol',

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

# # To match the 2017 dataset
# columns_to_remove = ['flow_id','src_ip','dst_ip','timestamp','std_packets_delta_time', 'min_packets_delta_len', 'ece_flag_percentage_in_total', 'bwd_syn_flag_percentage_in_total', 'bwd_rst_flag_percentage_in_total', 'fwd_psh_flag_percentage_in_total', 'median_bwd_header_bytes_delta_len', 'variance_payload_bytes_delta_len', 'median_fwd_packets_delta_len', 'fwd_ack_flag_percentage_in_fwd_packets', 'cov_fwd_header_bytes_delta_len', 'fwd_packets_iat_mode', 'fwd_payload_bytes_median', 'mean_payload_bytes_delta_len', 'min_payload_bytes_delta_len', 'cov_fwd_packets_delta_len', 'median_payload_bytes_delta_len', 'std_payload_bytes_delta_len', 'payload_bytes_median', 'fwd_packets_iat_cov', 'cov_bwd_header_bytes_delta_len', 'rst_flag_percentage_in_total', 'bwd_payload_bytes_cov', 'variance_bwd_header_bytes_delta_len', 'variance_packets_delta_len', 'fwd_mode_header_bytes', 'fwd_packets_iat_median', 'syn_flag_percentage_in_total', 'skewness_fwd_header_bytes_delta_len', 'max_header_bytes_delta_len', 'std_fwd_header_bytes_delta_len', 'variance_bwd_packets_delta_time', 'fwd_rst_flag_percentage_in_fwd_packets', 'min_fwd_payload_bytes_delta_len', 'bwd_variance_header_bytes', 'variance_fwd_packets_delta_time', 'mean_bwd_payload_bytes_delta_len', 'skewness_packets_delta_len', 'active_skewness', 'cov_bwd_payload_bytes_delta_len', 'delta_start', 'cov_fwd_payload_bytes_delta_len', 'mean_bwd_packets_delta_len', 'max_bwd_payload_bytes_delta_len', 'min_bwd_header_bytes_delta_len', 'bwd_skewness_header_bytes', 'median_fwd_header_bytes_delta_len', 'payload_bytes_skewness', 'bwd_payload_bytes_mode', 'idle_skewness', 'payload_bytes_mode', 'skewness_payload_bytes_delta_len', 'fin_flag_percentage_in_total', 'min_bwd_packets_delta_time', 'std_bwd_header_bytes_delta_len', 'active_median', 'handshake_state', 'mode_packets_delta_len', 'skewness_bwd_packets_delta_len', 'payload_bytes_cov', 'fwd_packets_iat_variance', 'mode_bwd_packets_delta_time', 'variance_header_bytes', 'cov_header_bytes', 'bwd_syn_flag_percentage_in_bwd_packets', 'min_fwd_packets_delta_time', 'std_fwd_packets_delta_len', 'bwd_mode_header_bytes', 'bwd_urg_flag_percentage_in_bwd_packets', 'bwd_cov_header_bytes', 'bwd_payload_bytes_median', 'median_bwd_packets_delta_len', 'skewness_header_bytes_delta_len', 'packets_iat_skewness', 'mode_payload_bytes_delta_len', 'std_packets_delta_len', 'max_fwd_payload_bytes_delta_len', 'mean_fwd_header_bytes_delta_len', 'max_packets_delta_len', 'fwd_payload_bytes_cov', 'fwd_psh_flag_percentage_in_fwd_packets', 'skewness_bwd_packets_delta_time', 'bwd_ack_flag_percentage_in_total', 'min_fwd_packets_delta_len', 'std_header_bytes_delta_len', 'fwd_urg_flag_percentage_in_total', 'bwd_cwr_flag_percentage_in_total', 'idle_mode', 'fwd_urg_flag_percentage_in_fwd_packets', 'std_bwd_payload_bytes_delta_len', 'fwd_fin_flag_percentage_in_total', 'max_fwd_packets_delta_time', 'bwd_packets_iat_skewness', 'idle_cov', 'skewness_bwd_header_bytes_delta_len', 'mode_packets_delta_time', 'cov_bwd_packets_delta_len', 'packets_iat_median', 'mode_fwd_packets_delta_time', 'min_bwd_payload_bytes_delta_len', 'median_packets_delta_len', 'skewness_fwd_packets_delta_time', 'skewness_bwd_payload_bytes_delta_len', 'max_fwd_header_bytes_delta_len', 'idle_variance', 'cov_packets_delta_len', 'active_cov', 'median_bwd_payload_bytes_delta_len', 'fwd_rst_flag_percentage_in_total', 'active_variance', 'fwd_payload_bytes_mode', 'mode_bwd_header_bytes_delta_len', 'ack_flag_percentage_in_total', 'bwd_rst_flag_percentage_in_bwd_packets', 'bwd_psh_flag_percentage_in_bwd_packets', 'handshake_duration', 'fwd_cwr_flag_percentage_in_fwd_packets', 'min_fwd_header_bytes_delta_len', 'fwd_ece_flag_percentage_in_total', 'packets_iat_variance', 'variance_fwd_packets_delta_len', 'mode_bwd_packets_delta_len', 'max_bwd_packets_delta_len', 'bwd_median_header_bytes', 'fwd_median_header_bytes', 'mode_fwd_payload_bytes_delta_len', 'psh_flag_percentage_in_total', 'fwd_syn_flag_percentage_in_total', 'max_bwd_packets_delta_time', 'fwd_cov_header_bytes', 'median_fwd_packets_delta_time', 'mean_bwd_header_bytes_delta_len', 'variance_fwd_header_bytes_delta_len', 'median_header_bytes', 'variance_bwd_payload_bytes_delta_len', 'median_bwd_packets_delta_time', 'skewness_packets_delta_time', 'packets_iat_mode', 'mean_packets_delta_time', 'mean_header_bytes_delta_len', 'std_bwd_packets_delta_time', 'mean_bwd_packets_delta_time', 'max_fwd_packets_delta_len', 'bwd_packets_iat_variance', 'bwd_urg_flag_percentage_in_total', 'fwd_variance_header_bytes', 'median_header_bytes_delta_len', 'mode_fwd_packets_delta_len', 'fwd_fin_flag_percentage_in_fwd_packets', 'fwd_ack_flag_percentage_in_total', 'variance_packets_delta_time', 'bwd_packets_iat_mode', 'cov_header_bytes_delta_len', 'variance_header_bytes_delta_len', 'std_fwd_payload_bytes_delta_len', 'median_packets_delta_time', 'bwd_fin_flag_percentage_in_bwd_packets', 'min_bwd_packets_delta_len', 'mean_packets_delta_len', 'fwd_ece_flag_percentage_in_fwd_packets', 'bwd_packets_iat_median', 'fwd_cwr_flag_percentage_in_total', 'bwd_psh_flag_percentage_in_total', 'skewness_fwd_packets_delta_len', 'median_fwd_payload_bytes_delta_len', 'max_payload_bytes_delta_len', 'variance_bwd_packets_delta_len', 'fwd_syn_flag_percentage_in_fwd_packets', 'fwd_packets_iat_skewness', 'mean_fwd_packets_delta_len', 'std_bwd_packets_delta_len', 'mean_fwd_packets_delta_time', 'bwd_ece_flag_percentage_in_total', 'mode_header_bytes_delta_len', 'urg_flag_percentage_in_total', 'bwd_fin_flag_percentage_in_total', 'std_fwd_packets_delta_time', 'skewness_fwd_payload_bytes_delta_len', 'active_mode', 'bwd_packets_iat_cov', 'mean_fwd_payload_bytes_delta_len', 'cov_bwd_packets_delta_time', 'bwd_ece_flag_percentage_in_bwd_packets', 'mode_bwd_payload_bytes_delta_len', 'cwr_flag_percentage_in_total', 'bwd_cwr_flag_percentage_in_bwd_packets', 'mode_fwd_header_bytes_delta_len', 'max_bwd_header_bytes_delta_len', 'mode_header_bytes', 'fwd_payload_bytes_skewness', 'bwd_payload_bytes_skewness', 'cov_payload_bytes_delta_len', 'packets_iat_cov', 'bwd_ack_flag_percentage_in_bwd_packets', 'variance_fwd_payload_bytes_delta_len', 'cov_fwd_packets_delta_time', 'idle_median', 'skewness_header_bytes', 'fwd_skewness_header_bytes', 'min_header_bytes_delta_len', 'cov_packets_delta_time'
#
# ]

chunk_size = 100000
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
            df_head = pd.read_csv(input_csv_path, nrows=0, encoding='utf-8-sig')
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            continue

        # --- Normalize column names to avoid mismatch issues ---
        df_head.columns = df_head.columns.str.strip().str.lower()
        all_columns = df_head.columns.tolist()

        # Normalize the list of columns to remove
        normalized_remove = [c.strip().lower() for c in columns_to_remove]

        # Determine which columns exist in the file
        cols_found = [col for col in normalized_remove if col in all_columns]
        cols_not_found = [col for col in normalized_remove if col not in all_columns]

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

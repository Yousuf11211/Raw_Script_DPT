import os
import sys
import csv
import argparse
import pandas as pd
from typing import List, Tuple, Dict

# --- Default configuration (can be overridden by CLI) ---
DEFAULT_INPUT_FOLDER = "Attacks"
DEFAULT_OUTPUT_FOLDER = "Attacks_Cleaned"
DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_SUMMARY_NAME = "deletion_summary.csv"

# Base columns to remove (will be merged/edited at runtime)
BASE_COLUMNS_TO_REMOVE = [
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
    'cov_packets_delta_len','cov_payload_bytes_delta_len','protocol',  # duplicate in original list
    'mean_payload_bytes_delta_len','fwd_payload_bytes_mode','mode_header_bytes_delta_len','payload_bytes_mode',
    'min_header_bytes_delta_len','bwd_ece_flag_percentage_in_bwd_packets','avg_fwd_bytes_per_bulk','avg_fwd_packets_per_bulk',
    'packets_IAT_mode','fwd_syn_flag_counts','fwd_bulk_per_packet','fwd_bulk_total_size','bwd_rst_flag_counts',
    'bwd_syn_flag_counts','fwd_bulk_state_count','bwd_ece_flag_percentage_in_total','bwd_ece_flag_counts',
    'mean_fwd_payload_bytes_delta_len','mode_fwd_payload_bytes_delta_len','mode_payload_bytes_delta_len','std_bwd_packets_delta_time',
    'cov_packets_delta_time','cov_fwd_packets_delta_time','mean_packets_delta_time','variance_bwd_packets_delta_time',
    'fwd_payload_bytes_cov','mode_packets_delta_len','min_fwd_packets_delta_time','avg_fwd_bulk_rate','mean_bwd_packets_delta_time',
    'fwd_packets_IAT_min','fwd_payload_bytes_median','rst_flag_counts','skewness_packets_delta_time','skewness_fwd_packets_delta_time',
    'variance_packets_delta_time','bwd_rst_flag_percentage_in_bwd_packets','fwd_variance_header_bytes','bwd_fin_flag_counts',
    'bwd_fin_flag_percentage_in_total','rst_flag_percentage_in_total','handshake_duration','std_packets_delta_time','fwd_packets_IAT_mode',
    'psh_flag_percentage_in_total','payload_bytes_median','variance_fwd_packets_delta_time','bwd_fin_flag_percentage_in_bwd_packets',
    'fwd_std_header_bytes','max_bwd_packets_delta_time','mode_fwd_packets_delta_time','skewness_bwd_header_bytes_delta_len',
    'mode_bwd_packets_delta_time','bwd_packets_IAT_mode','max_bwd_header_bytes_delta_len','mean_bwd_header_bytes_delta_len',
    'skewness_fwd_payload_bytes_delta_len','skewness_payload_bytes_delta_len','median_bwd_packets_delta_len','packet_IAT_min',
    'bwd_variance_header_bytes','std_fwd_packets_delta_time','mean_fwd_packets_delta_time','median_fwd_packets_delta_len',
    'median_bwd_payload_bytes_delta_len','min_fwd_header_bytes_delta_len','fwd_psh_flag_percentage_in_fwd_packets','cov_bwd_packets_delta_time',
    'fwd_packets_count','fwd_ack_flag_counts','mean_fwd_packets_delta_len','bwd_rst_flag_percentage_in_total','avg_bwd_bulk_rate',
    'fwd_cov_header_bytes','fwd_psh_flag_counts','fwd_payload_bytes_skewness','fwd_rst_flag_percentage_in_fwd_packets',
    'fwd_syn_flag_percentage_in_total'
]

# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------

def load_columns_file(path: str) -> List[str]:
    """Load extra columns from file, one per line; ignores blanks and lines starting with #."""
    if not path or not os.path.isfile(path):
        return []
    cols: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):  # comment / blank
                continue
            # allow comma-separated in same line
            parts = [p.strip() for p in line.split(',') if p.strip()]
            cols.extend(parts)
    return cols

def normalize_list(values: List[str]) -> List[str]:
    return [v.strip().lower() for v in values]


def build_final_deletion_list(base: List[str], add: List[str], remove: List[str]) -> Tuple[List[str], List[str]]:
    """Return final deletion list (case-insensitive) and list of duplicates eliminated."""
    base_norm = normalize_list(base)
    add_norm = normalize_list(add)
    remove_norm = set(normalize_list(remove)) if remove else set()
    combined = base_norm + add_norm
    # eliminate duplicates while preserving order
    seen = set()
    final_list: List[str] = []
    duplicates: List[str] = []
    for col in combined:
        if col in remove_norm:
            continue
        if col not in seen:
            final_list.append(col)
            seen.add(col)
        else:
            duplicates.append(col)
    return final_list, duplicates


def write_summary(summary_rows: List[Dict[str, str]], output_folder: str, summary_name: str) -> None:
    path = os.path.join(output_folder, summary_name)
    if not summary_rows:
        return
    fieldnames = list(summary_rows[0].keys())
    try:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary written: {path}")
    except Exception as e:
        print(f"Failed to write summary CSV: {e}")

# ----------------------------------------------------------------------------
# Core processing
# ----------------------------------------------------------------------------

def process_file(input_csv_path: str,
                 output_csv_path: str,
                 columns_to_remove_norm: List[str],
                 chunk_size: int,
                 dry_run: bool) -> Tuple[int, int, List[str], List[str]]:
    """Process a single CSV: determine removable columns, then stream delete.
    Returns: (found_count, not_found_count, found_list, not_found_list)."""
    try:
        df_head = pd.read_csv(input_csv_path, nrows=0, encoding='utf-8-sig')
    except Exception as e:
        print(f"  Error reading header: {e}")
        return 0, 0, [], []

    df_head.columns = df_head.columns.str.strip().str.lower()
    all_columns = df_head.columns.tolist()

    cols_found = [col for col in columns_to_remove_norm if col in all_columns]
    cols_not_found = [col for col in columns_to_remove_norm if col not in all_columns]

    print(f"  Columns found for deletion: {len(cols_found)}")
    if cols_found:
        print(f"    Found: {cols_found}")
    print(f"  Columns not found in file: {len(cols_not_found)}")
    if cols_not_found:
        print(f"    Not found: {cols_not_found}")

    if dry_run:
        print("  Dry-run enabled; skipping write.")
        return len(cols_found), len(cols_not_found), cols_found, cols_not_found

    is_first_chunk = True
    for chunk in pd.read_csv(input_csv_path, chunksize=chunk_size, low_memory=False):
        # drop only the columns that exist as they appear in original header (already normalized)
        chunk.columns = chunk.columns.str.strip().str.lower()
        chunk.drop(columns=[c for c in cols_found if c in chunk.columns], inplace=True, errors='ignore')
        if is_first_chunk:
            chunk.to_csv(output_csv_path, index=False, mode='w')
            is_first_chunk = False
        else:
            chunk.to_csv(output_csv_path, index=False, mode='a', header=False)
    return len(cols_found), len(cols_not_found), cols_found, cols_not_found

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch delete specified columns from all CSV files in a folder.")
    p.add_argument('-i','--input-folder', default=DEFAULT_INPUT_FOLDER, help='Folder containing source CSV files')
    p.add_argument('-o','--output-folder', default=DEFAULT_OUTPUT_FOLDER, help='Folder to write cleaned CSV files')
    p.add_argument('-c','--columns-file', default='', help='Optional file listing additional columns to delete (one per line)')
    p.add_argument('--add-cols', default='', help='Comma-separated extra columns to add to deletion list')
    p.add_argument('--remove-cols', default='', help='Comma-separated columns to remove from the deletion list (i.e., keep)')
    p.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE, help='Rows per chunk (streaming)')
    p.add_argument('--dry-run', action='store_true', help='Analyze only; do not create output files')
    p.add_argument('--summary-name', default=DEFAULT_SUMMARY_NAME, help='Name of summary CSV file')
    return p.parse_args(argv)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    args = parse_args(sys.argv[1:])
    input_folder = args.input_folder
    output_folder = args.output_folder
    chunk_size = args.chunk_size
    dry_run = args.dry_run

    os.makedirs(output_folder, exist_ok=True)

    # Load external columns file + add/remove modifiers
    file_cols = load_columns_file(args.columns_file)
    add_cols = [c for c in args.add_cols.split(',') if c.strip()] if args.add_cols else []
    remove_cols = [c for c in args.remove_cols.split(',') if c.strip()] if args.remove_cols else []

    final_list, duplicates = build_final_deletion_list(BASE_COLUMNS_TO_REMOVE, file_cols + add_cols, remove_cols)

    print("======================================")
    print(" Dynamic Column Deletion Tool")
    print("======================================")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Dry-run: {dry_run}")
    print(f"Chunk size: {chunk_size}")
    print(f"Base columns: {len(BASE_COLUMNS_TO_REMOVE)} | External file: {len(file_cols)} | CLI add: {len(add_cols)} | CLI remove: {len(remove_cols)}")
    if duplicates:
        print(f"Duplicate entries removed (after normalization): {duplicates}")
    print(f"Final unique columns scheduled for deletion: {len(final_list)}")
    print("--------------------------------------")

    if not os.path.isdir(input_folder):
        print(f"ERROR: Input folder does not exist: {input_folder}")
        sys.exit(1)

    summary_rows: List[Dict[str,str]] = []

    for filename in sorted(os.listdir(input_folder)):
        if not filename.lower().endswith('.csv'):
            continue
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        print(f"Processing '{filename}'...")
        found_cnt, not_found_cnt, found_cols, not_found_cols = process_file(
            input_csv_path=input_path,
            output_csv_path=output_path,
            columns_to_remove_norm=final_list,
            chunk_size=chunk_size,
            dry_run=dry_run
        )
        summary_rows.append({
            'file': filename,
            'found_count': str(found_cnt),
            'not_found_count': str(not_found_cnt),
            'removed_columns': ';'.join(found_cols),
            'requested_missing': ';'.join(not_found_cols)
        })
        print(f"----> Done '{filename}'\n")

    write_summary(summary_rows, output_folder, args.summary_name)
    print("All files processed.")
    if dry_run:
        print("Dry-run complete. No files were written.")

if __name__ == '__main__':
    main()

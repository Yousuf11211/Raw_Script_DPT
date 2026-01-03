# What changed:
# - Added GPU detection/device prompt, chunk size prompt with row estimation, and streaming limits.
# - Standardized outputs under ./outputs with non-overwrite paths and optional max-rows saving.
# - Added final processing summary.
#
# Purpose:
# - Delete specified columns from CSV files in a folder.
# - Stream-process large CSVs without loading full files into memory.
# - Save a summary CSV of deletions.

import os
import sys
import csv
import argparse
import pandas as pd
from typing import List, Tuple, Dict
from itertools import chain

# --- Default configuration (can be overridden by CLI) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
DEFAULT_INPUT_FOLDER = "Bening"
DEFAULT_OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Attacks_Cleaned")
DEFAULT_SUMMARY_NAME = "deletion_summary.csv"

# -------------------
# Column Deletion Lists (by reason)
# -------------------
CONSTANT_COLUMNS = [
    'flow_id','src_ip','dst_ip','timestamp','protocol', 'payload_bytes_min','fwd_payload_bytes_min','bwd_payload_bytes_min','urg_flag_counts',
    'fwd_urg_flag_counts','bwd_urg_flag_counts','urg_flag_percentage_in_total','fwd_urg_flag_percentage_in_total','bwd_urg_flag_percentage_in_total','fwd_urg_flag_percentage_in_fwd_packets',
    'bwd_urg_flag_percentage_in_bwd_packets'
]
LOW_VARIANCE_COLUMNS_95_99 = [
    'payload_bytes_mode','bwd_payload_bytes_mode','max_header_bytes','min_header_bytes','median_header_bytes', 'mode_header_bytes','fwd_max_header_bytes',
    'fwd_min_header_bytes','fwd_median_header_bytes','fwd_mode_header_bytes','bwd_min_header_bytes','fwd_init_win_bytes','active_min',
    'active_max','active_mean','active_std','active_median','active_skewness','active_cov','active_mode','active_variance',
    'idle_min','idle_max','idle_mean','idle_std','idle_median','idle_skewness','idle_cov','idle_mode','idle_variance','avg_fwd_bytes_per_bulk',
    'avg_fwd_packets_per_bulk','avg_fwd_bulk_rate','fwd_bulk_state_count','fwd_bulk_total_size','fwd_bulk_per_packet','fwd_bulk_duration',
    'syn_flag_counts','fwd_syn_flag_counts','bwd_syn_flag_counts','bwd_cwr_flag_counts','bwd_cwr_flag_percentage_in_total','bwd_ack_flag_percentage_in_bwd_packets',
    'bwd_cwr_flag_percentage_in_bwd_packets','handshake_state','mode_header_bytes_delta_len','median_header_bytes_delta_len','mode_bwd_header_bytes_delta_len',
    'median_bwd_header_bytes_delta_len','min_fwd_header_bytes_delta_len','max_fwd_header_bytes_delta_len','mode_fwd_header_bytes_delta_len','median_fwd_header_bytes_delta_len',
    'mean_payload_bytes_delta_len','cov_payload_bytes_delta_len'
]

LOW_VARIANCE_COLUMNS_90_95 = [
    'fwd_payload_bytes_mode','bwd_max_header_bytes','bwd_median_header_bytes','bwd_mode_header_bytes','min_header_bytes_delta_len','min_bwd_header_bytes_delta_len',
    'mean_fwd_payload_bytes_delta_len','cov_fwd_payload_bytes_delta_len'
]
#
# Master list of list names to use for deletion
DELETION_LISTS = [
    'CONSTANT_COLUMNS',
    'LOW_VARIANCE_COLUMNS_95_99',
    'LOW_VARIANCE_COLUMNS_90_95',
    # Add new list names here to include them in deletion
]

# Utility to collect all columns from the named lists
def get_columns_to_remove(list_names: list[str]) -> list[str]:
    cols = []
    for name in list_names:
        if name in globals():
            cols.extend(globals()[name])
        else:
            print(f"Warning: List '{name}' not defined.")
    return cols

# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------

def detect_gpu():
    gpu_available = False
    library = None
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            gpu_available = True
            library = "pytorch"
    except Exception:
        pass

    if not gpu_available:
        try:
            import tensorflow as tf  # type: ignore
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                gpu_available = True
                library = "tensorflow"
        except Exception:
            pass

    if gpu_available:
        print("GPU detected.")
    else:
        print("GPU not detected. Using CPU.")
    return gpu_available, library


def prompt_for_device(gpu_available):
    if gpu_available:
        while True:
            response = input("GPU detected. Use GPU? (y/n): ").lower().strip()
            if response in ["y", "yes"]:
                return "gpu"
            if response in ["n", "no"]:
                return "cpu"
            print("Invalid input. Please enter 'y' or 'n'.")
    return "cpu"


def prompt_for_chunk_size_mb():
    choices = {"25": 25, "100": 100, "500": 500, "1000": 1000}
    while True:
        response = input("Choose chunk size in MB (25/100/500/1000): ").strip()
        if response in choices:
            return choices[response]
        print("Invalid choice. Please enter 25, 100, 500, or 1000.")


def estimate_rows_per_chunk(file_path, chunk_mb, sample_rows=2000, default_rows=100_000):
    target_bytes = int(chunk_mb) * 1024 * 1024
    try:
        sample = pd.read_csv(file_path, nrows=sample_rows, low_memory=True)
        if sample is None or sample.empty:
            return int(default_rows)
        bytes_per_row = float(sample.memory_usage(deep=True).sum()) / float(max(1, len(sample)))
        if bytes_per_row <= 0:
            return int(default_rows)
        est = int(target_bytes / bytes_per_row)
        return max(10_000, min(2_000_000, est))
    except Exception:
        return int(default_rows)


def prompt_for_max_rows():
    while True:
        response = input("Limit rows to save? (y/n): ").strip().lower()
        if response in ["y", "yes"]:
            while True:
                value = input("Enter max rows: ").strip()
                try:
                    max_rows = int(value)
                    if max_rows > 0:
                        return max_rows
                except ValueError:
                    pass
                print("Please enter a positive integer.")
        elif response in ["n", "no"]:
            return None
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def load_columns_file(path: str) -> List[str]:
    if not path or not os.path.isfile(path):
        return []
    cols: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',') if p.strip()]
            cols.extend(parts)
    return cols


def normalize_list(values: List[str]) -> List[str]:
    return [v.strip().lower() for v in values]


def build_final_deletion_list(base: List[str], add: List[str], remove: List[str]) -> Tuple[List[str], List[str]]:
    base_norm = normalize_list(base)
    add_norm = normalize_list(add)
    remove_norm = set(normalize_list(remove)) if remove else set()
    combined = base_norm + add_norm
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


def write_summary(summary_rows: List[Dict[str, str]], output_folder: str, summary_name: str) -> str | None:
    if not summary_rows:
        return None
    path = os.path.join(output_folder, summary_name)
    fieldnames = list(summary_rows[0].keys())
    try:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary written: {path}")
        return path
    except Exception as e:
        print(f"Failed to write summary CSV: {e}")
        return None


def make_unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    counter = 2
    while True:
        candidate = f"{base}_run{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1

# ----------------------------------------------------------------------------
# core processing
# ----------------------------------------------------------------------------

def process_file(
    input_csv_path: str,
    output_csv_path: str,
    columns_to_remove_norm: List[str],
    chunk_size: int,
    dry_run: bool,
    max_rows: int | None,
) -> Tuple[int, int, List[str], List[str], int, int]:
    try:
        df_head = pd.read_csv(input_csv_path, nrows=0, encoding='utf-8-sig')
    except Exception as e:
        print(f"  Error reading header: {e}")
        return 0, 0, [], [], 0

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
        return len(cols_found), len(cols_not_found), cols_found, cols_not_found, 0, 0

    rows_written = 0
    rows_processed = 0
    is_first_chunk = True
    for idx, chunk in enumerate(pd.read_csv(input_csv_path, chunksize=chunk_size, low_memory=False)):
        rows_processed += len(chunk)
        chunk.columns = chunk.columns.str.strip().str.lower()
        chunk.drop(columns=[c for c in cols_found if c in chunk.columns], inplace=True, errors='ignore')

        if max_rows is not None:
            remaining = max_rows - rows_written
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]

        if is_first_chunk:
            chunk.to_csv(output_csv_path, index=False, mode='w')
            is_first_chunk = False
        else:
            chunk.to_csv(output_csv_path, index=False, mode='a', header=False)

        rows_written += len(chunk)
        if (idx + 1) % 5 == 0:
            print(f"  Processed {rows_written:,} rows so far...")

        if max_rows is not None and rows_written >= max_rows:
            break

    return len(cols_found), len(cols_not_found), cols_found, cols_not_found, rows_written, rows_processed

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
    p.add_argument('--dry-run', action='store_true', help='Analyze only; do not create output files')
    p.add_argument('--summary-name', default=DEFAULT_SUMMARY_NAME, help='Name of summary CSV file')
    return p.parse_args(argv)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    gpu_available, _ = detect_gpu()
    device_choice = prompt_for_device(gpu_available)
    if device_choice == "gpu":
        print("GPU selected, but this script uses CPU-based pandas. Using CPU.")
    device_used = "cpu"

    args = parse_args(sys.argv[1:])
    input_folder = args.input_folder
    output_folder = args.output_folder
    dry_run = args.dry_run

    # Resolve input folder relative to project root if not absolute
    if not os.path.isabs(input_folder):
        # Project root is parent of script directory
        project_root = os.path.dirname(SCRIPT_DIR)
        input_folder = os.path.join(project_root, input_folder)

    if not os.path.isabs(output_folder):
        output_folder = os.path.join(OUTPUT_ROOT, output_folder)

    if not os.path.isdir(input_folder):
        print(f"ERROR: Input folder does not exist: {input_folder}")
        sys.exit(1)

    csv_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith('.csv')])
    if not csv_files:
        print("ERROR: No CSV files found in the input folder.")
        sys.exit(1)

    chunk_mb = prompt_for_chunk_size_mb()
    first_file = os.path.join(input_folder, csv_files[0])
    chunk_size = estimate_rows_per_chunk(first_file, chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_size:,} rows per chunk)")

    max_rows = None if dry_run else prompt_for_max_rows()

    os.makedirs(output_folder, exist_ok=True)

    file_cols = load_columns_file(args.columns_file)
    add_cols = [c for c in args.add_cols.split(',') if c.strip()] if args.add_cols else []
    remove_cols = [c for c in args.remove_cols.split(',') if c.strip()] if args.remove_cols else []

    # Instead of BASE_COLUMNS_TO_REMOVE, use the scalable system
    # Collect columns from all lists in DELETION_LISTS
    base_columns = get_columns_to_remove(DELETION_LISTS)
    final_list, duplicates = build_final_deletion_list(base_columns, file_cols + add_cols, remove_cols)

    print("======================================")
    print(" Dynamic Column Deletion Tool")
    print("======================================")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Dry-run: {dry_run}")
    print(f"Chunk size: {chunk_size}")
    print(f"Base columns (from DELETION_LISTS): {len(base_columns)} | External file: {len(file_cols)} | CLI add: {len(add_cols)} | CLI remove: {len(remove_cols)}")
    if duplicates:
        print(f"Duplicate entries removed (after normalization): {duplicates}")
    print(f"Final unique columns scheduled for deletion: {len(final_list)}")
    print("--------------------------------------")

    summary_rows: List[Dict[str,str]] = []
    total_rows_processed = 0
    total_rows_saved = 0
    output_paths = []

    for filename in csv_files:
        input_path = os.path.join(input_folder, filename)
        output_path = make_unique_path(os.path.join(output_folder, filename))
        print(f"Processing '{filename}'...")
        found_cnt, not_found_cnt, found_cols, not_found_cols, rows_written, rows_processed = process_file(
            input_csv_path=input_path,
            output_csv_path=output_path,
            columns_to_remove_norm=final_list,
            chunk_size=chunk_size,
            dry_run=dry_run,
            max_rows=max_rows,
        )
        total_rows_processed += rows_processed
        if not dry_run:
            output_paths.append(output_path)
        total_rows_saved += rows_written

        summary_rows.append({
            'file': filename,
            'found_count': str(found_cnt),
            'not_found_count': str(not_found_cnt),
            'removed_columns': ';'.join(found_cols),
            'requested_missing': ';'.join(not_found_cols)
        })
        print(f"----> Done '{filename}'\n")

    summary_path = None
    if not dry_run:
        summary_path = write_summary(summary_rows, output_folder, args.summary_name)

    print("All files processed.")
    if dry_run:
        print("Dry-run complete. No files were written.")

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Device used: {device_used.upper()}")
    print(f"Chunk size: {chunk_mb}MB (~{chunk_size:,} rows)")
    print(f"Total rows processed: {total_rows_processed:,}")
    print(f"Rows saved: {total_rows_saved:,}")
    print("Output paths:")
    if output_paths:
        for path in output_paths:
            print(f"  - {path}")
    if summary_path:
        print(f"  - {summary_path}")


if __name__ == '__main__':
    main()

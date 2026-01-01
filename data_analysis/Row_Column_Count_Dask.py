"""
Row and Column Counter for CSV files using Dask (and Dask-CUDA if available).
- Prints column count (from header) and row count (chunked, memory safe) for each CSV in a folder.
- Uses GPU if Dask-CUDA is available and GPU is detected, otherwise falls back to CPU Dask.
- Chunk size is set from global config.
"""
import os
import sys
import dask
import dask.dataframe as dd
from config.global_config import DEFAULT_CHUNK_SIZE_MB

try:
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

import glob

# --- CONFIG ---
INPUT_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, "output_missing")  # Corrected to parent folder
CHUNK_SIZE_MB = DEFAULT_CHUNK_SIZE_MB


def print_info(msg):
    print(f"[INFO] {msg}")

def print_warn(msg):
    print(f"[WARN] {msg}")

def count_columns(file_path):
    import pandas as pd
    try:
        df = pd.read_csv(file_path, nrows=0)
        return len(df.columns), list(df.columns)
    except Exception as e:
        print_warn(f"Failed to read columns for {file_path}: {e}")
        return 0, []

def count_rows_dask(file_path, chunk_size_bytes):
    try:
        ddf = dd.read_csv(file_path, blocksize=chunk_size_bytes, assume_missing=True, dtype="object")
        return ddf.shape[0].compute()
    except Exception as e:
        print_warn(f"Failed to count rows for {file_path}: {e}")
        return 0

def main():
    folder = os.path.abspath(INPUT_FOLDER)
    if not os.path.isdir(folder):
        print_warn(f"INPUT_FOLDER not found: {folder}")
        return

    print_info(f"Chunk size: {CHUNK_SIZE_MB} MB")
    chunk_size_bytes = CHUNK_SIZE_MB * 1024 * 1024

    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not csv_files:
        print_warn(f"No CSV files found in {folder}")
        return

    print_info(f"Total CSV files found: {len(csv_files)}")

    if GPU_AVAILABLE:
        print_info("Dask-CUDA detected. Using GPU for Dask cluster.")
        cluster = LocalCUDACluster()
        client = Client(cluster)
    else:
        print_info("Using CPU Dask.")
        client = None

    for file_path in csv_files:
        fname = os.path.basename(file_path)
        col_count, col_names = count_columns(file_path)
        print(f"{fname}: Columns = {col_count}")
        row_count = count_rows_dask(file_path, chunk_size_bytes)
        print(f"{fname}: Rows = {row_count}")

    if client is not None:
        client.close()

if __name__ == "__main__":
    main()


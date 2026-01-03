# What changed:
# - Added GPU detection/device prompt, chunk size prompt with row estimation, and streaming analysis.
# - Standardized outputs under ./outputs/Count_Remove_Duplicate_Rows_Columns.
# - Added optional max-rows limit for saved outputs, a report file, and a final summary.
#
# Purpose:
# - Count rows/columns and detect duplicate columns/rows.
# - Optionally remove duplicate columns and/or duplicate rows and save cleaned CSVs.
# - Report missing values per column.

import os
import sys

# Allow running this script from any working directory.
# We add the repo root (parent of this file's folder) to sys.path so imports like
# `import config.global_config` and `from utils...` work reliably.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
from collections import Counter
import argparse

from config.global_config import DEFAULT_CHUNK_SIZE_MB
from utils.chunk_utils import compute_chunk_plan, format_progress, print_chunk_plan
from utils.dedup_utils import SQLiteHashStore
from utils.engine_utils import select_engine
from utils.path_utils import resolve_input_path, resolve_output_path

# ========= CONFIG =========
# Legacy defaults (kept for backward compatibility)
INPUT_FOLDER = "Bening"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Count_Remove_Duplicate_Rows_Columns")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Count rows/columns, detect duplicate columns/rows, and optionally remove duplicates. "
            "Streaming-safe for large CSVs."
        )
    )
    p.add_argument("--input", default=INPUT_FOLDER, help="Input folder containing CSVs (abs or repo-root-relative)")
    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Base output directory (abs or repo-root-relative). "
            "Defaults to repo-root outputs/."
        ),
    )
    p.add_argument("--engine", default="pandas", choices=["pandas", "dask", "dask-gpu"], help="Execution engine")
    p.add_argument("--chunk-size-mb", type=int, default=DEFAULT_CHUNK_SIZE_MB, help="Chunk size in MB")
    p.add_argument("--use-gpu", action="store_true", help="Force GPU (or fail)")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU")
    p.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive prompts; runs only enabled checks and does not modify files.",
    )
    return p


# ======= HELPERS ========

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


def estimate_rows_per_chunk(file_path, chunk_mb, sample_rows=2000, default_rows=500_000):
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


def get_unique_columns(columns):
    seen = set()
    unique_cols = []
    for col in columns:
        base = str(col).split(".")[0]
        if base not in seen:
            unique_cols.append(col)
            seen.add(base)
    return unique_cols


# ======= MAIN SCRIPT ========

def main(argv: list[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Engine selection (forward-compatible). This script currently performs pandas streaming.
    selection = select_engine(engine=args.engine, use_gpu_flag=args.use_gpu, no_gpu_flag=args.no_gpu)
    if selection.engine != "pandas":
        # Keep behavior stable for now; full Dask migration will be done repo-wide next.
        print(f"[info] --engine {selection.engine} requested; this script currently runs in pandas mode for safety.")
    if selection.use_gpu:
        print("[info] GPU was approved, but this script uses CPU-based pandas. Using CPU.")
    device_used = "cpu"

    input_folder = resolve_input_path(args.input)
    base_output_dir = resolve_output_path(args.output_dir)
    output_folder = os.path.join(base_output_dir, "Count_Remove_Duplicate_Rows_Columns")

    if not os.path.isdir(input_folder):
        print(f"ERROR: Input folder not found: {input_folder}")
        return

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(".csv")]
    if not files:
        print("No CSV files found in the input folder.")
        return

    # Chunk planning from file size (mandatory). We still need rows-per-chunk for pandas;
    # we re-use the existing estimator but also print the mandatory plan.
    first_file = os.path.join(input_folder, files[0])
    chunk_mb = int(args.chunk_size_mb)
    plan = compute_chunk_plan(first_file, chunk_mb)
    print_chunk_plan(plan)

    chunk_rows = estimate_rows_per_chunk(first_file, chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

    # Interactive options (kept). In no-interactive mode, default to analysis-only.
    if args.no_interactive:
        do_col_count = True
        do_row_count = True
        do_dup_colnames = True
        do_dup_rows = True
        do_missing = True
        remove_dup_cols = False
        remove_dup_rows = False
        max_rows_limit = None
    else:
        print("What do you want to check/do for each file? Answer 'y' or 'n'.")
        do_col_count = input("Show column count? (y/n): ").lower() == 'y'
        do_row_count = input("Show row count? (y/n): ").lower() == 'y'
        do_dup_colnames = input("Check for duplicate column names? (y/n): ").lower() == 'y'
        do_dup_rows = input("Check for duplicate rows? (y/n): ").lower() == 'y'
        do_missing = input("Check for missing values? (y/n): ").lower() == 'y'

        remove_dup_cols = False
        remove_dup_rows = False
        if do_dup_colnames:
            remove_dup_cols = input("Remove duplicates keeping only first occurrence? (y/n): ").lower() == 'y'
        if do_dup_rows:
            remove_dup_rows = input("Remove duplicate rows and save updated CSV? (y/n): ").lower() == 'y'

        max_rows_limit = prompt_for_max_rows() if (remove_dup_cols or remove_dup_rows) else None

    os.makedirs(output_folder, exist_ok=True)
    report_path = make_unique_path(os.path.join(output_folder, "duplicate_check_report.txt"))

    total_rows_processed = 0
    total_rows_saved = 0
    output_paths = []

    report_lines = []

    for filename in files:
        file_path = os.path.join(input_folder, filename)
        print(f"\nProcessing file: {filename}")
        report_lines.append(f"File: {filename}")

        # Mandatory chunk plan per file for correct progress.
        file_plan = compute_chunk_plan(file_path, chunk_mb)
        print_chunk_plan(file_plan)

        try:
            header_df = pd.read_csv(file_path, nrows=0, dtype=str)
        except Exception as e:
            print(f"  ERROR reading header: {e}")
            report_lines.append(f"  ERROR reading header: {e}")
            continue

        columns = list(header_df.columns)
        unique_cols = get_unique_columns(columns)

        if do_col_count:
            print(f"Number of columns: {len(columns)}")
            report_lines.append(f"  Column count: {len(columns)}")

        if do_dup_colnames:
            base_names = [c.split('.')[0] for c in columns]
            col_counts = pd.Series(base_names).value_counts()
            duplicate_bases = col_counts[col_counts > 1].index.tolist()
            if duplicate_bases:
                print(f"Duplicate or renamed duplicate columns detected: {duplicate_bases}")
                report_lines.append(f"  Duplicate columns: {duplicate_bases}")
            else:
                print("No duplicate or renamed duplicate column names.")
                report_lines.append("  Duplicate columns: None")

        dup_rows_count = 0
        missing_counts = Counter()
        duplicate_row_examples = []  # Collect up to 10 duplicate rows as dicts
        duplicate_row_indices = []   # Collect their original indices (row numbers in file)
        row_hash_counter = Counter()  # Count occurrences of each row hash
        row_hash_to_row = {}         # Map row hash to row content (first occurrence)

        output_path = None
        header_written = False

        # Cross-chunk dedup MUST be safe: use a disk-backed store.
        dedup_db_path = None
        store_ctx = None
        if do_dup_rows or remove_dup_rows:
            dedup_db_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_dedup.sqlite")

        if remove_dup_cols or remove_dup_rows:
            stem = os.path.splitext(filename)[0]
            suffix = "nodupcol" if remove_dup_cols else ""
            if remove_dup_rows:
                suffix = f"{suffix}_noduprows" if suffix else "noduprows"
            output_filename = f"{stem}_{suffix}.csv" if suffix else f"{stem}_cleaned.csv"
            output_path = make_unique_path(os.path.join(output_folder, output_filename))

        file_rows_processed = 0

        if dedup_db_path:
            store_ctx = SQLiteHashStore(dedup_db_path)

        try:
            if store_ctx:
                store = store_ctx.__enter__()
            else:
                store = None

            for chunk_idx, chunk in enumerate(
                pd.read_csv(file_path, chunksize=chunk_rows, dtype=str, low_memory=False), 1
            ):
                # Progress based on precomputed file chunk plan.
                print(format_progress(chunk_idx, file_plan.total_chunks))

                file_rows_processed += len(chunk)
                total_rows_processed += len(chunk)

                if do_missing:
                    missing_counts.update(chunk.isna().sum().to_dict())

                if do_dup_rows or remove_dup_rows:
                    row_hashes = pd.util.hash_pandas_object(chunk, index=False).astype("int64")
                    keep_mask = store.keep_mask(row_hashes.tolist()) if store is not None else [True] * len(chunk)
                    # Count all row hashes for top duplicates
                    for i, h in enumerate(row_hashes):
                        row_hash_counter[h] += 1
                        if h not in row_hash_to_row:
                            row_hash_to_row[h] = chunk.iloc[i].to_dict()
                    if do_dup_rows:
                        dup_rows_count += int(len(keep_mask) - sum(keep_mask))
                    if remove_dup_rows:
                        chunk = chunk.loc[keep_mask]

                if remove_dup_cols:
                    chunk = chunk[unique_cols]

                if output_path and not chunk.empty:
                    if max_rows_limit is not None:
                        remaining = max_rows_limit - total_rows_saved
                        if remaining <= 0:
                            break
                        if len(chunk) > remaining:
                            chunk = chunk.iloc[:remaining]
                    if not chunk.empty:
                        chunk.to_csv(
                            output_path,
                            index=False,
                            mode="w" if not header_written else "a",
                            header=not header_written,
                        )
                        header_written = True
                        total_rows_saved += len(chunk)

                if output_path and max_rows_limit is not None and total_rows_saved >= max_rows_limit:
                    break

        finally:
            if store_ctx is not None:
                store_ctx.__exit__(None, None, None)

        if do_row_count:
            print(f"Number of rows: {file_rows_processed}")
            report_lines.append(f"  Row count: {file_rows_processed}")

        if do_dup_rows:
            print(f"Duplicate rows: {dup_rows_count}")
            report_lines.append(f"  Duplicate rows: {dup_rows_count}")
            # Print top 10 most frequent duplicate rows
            top_dupes = [(h, c) for h, c in row_hash_counter.items() if c > 1]
            top_dupes.sort(key=lambda x: -x[1])
            top10 = top_dupes[:10]
            if top10:
                report_lines.append(f"  Top 10 most frequent duplicate rows:")
                print("Top 10 most frequent duplicate rows:")
                for i, (h, count) in enumerate(top10, 1):
                    row = row_hash_to_row[h]
                    row_str = ', '.join(f'{k}={repr(v)}' for k, v in row.items())
                    report_lines.append(f"    [{i}] (count={count}) {row_str}")
                    print(f"    [{i}] (count={count}) {row_str}")

        if do_missing:
            missing_dict = {col: count for col, count in missing_counts.items() if count > 0}
            if missing_dict:
                print("Missing values per column:")
                for col, count in missing_dict.items():
                    print(f"  {col}: {count}")
                report_lines.append(f"  Missing values: {missing_dict}")
            else:
                print("No missing values found.")
                report_lines.append("  Missing values: None")

        if output_path and header_written:
            print(f"Saved cleaned file: {output_path}")
            output_paths.append(output_path)

        # Keep the sqlite file path in outputs for reproducibility/debug.
        if dedup_db_path:
            output_paths.append(dedup_db_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    output_paths.append(report_path)

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Device used: {device_used.upper()}")
    print(f"Chunk size: {chunk_mb}MB (~{chunk_rows:,} rows)")
    print(f"Total rows processed: {total_rows_processed:,}")
    print(f"Rows saved: {total_rows_saved:,}")
    print("Output paths:")
    for path in output_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()

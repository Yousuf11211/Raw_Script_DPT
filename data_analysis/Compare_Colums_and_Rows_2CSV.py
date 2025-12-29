# What changed:
# - Added GPU detection/device prompt and chunk size prompt with row estimation.
# - Streamed row hashing to avoid loading full CSVs into memory.
# - Standardized outputs under ./outputs/Compare_Columns_and_Rows_2CSV with a final summary.
#
# Purpose:
# - Compare column consistency between raw and processed CSV folders.
# - Compare row sets between raw and processed data using streaming hashes.
# - Save a comparison report to disk.

import os
import sys
import argparse

# Allow running this script from any working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from config.global_config import DEFAULT_CHUNK_SIZE_MB
from utils.chunk_utils import compute_chunk_plan, format_progress, print_chunk_plan
from utils.dedup_utils import SQLiteHashStore
from utils.engine_utils import select_engine
from utils.gpu_utils import gpu_available as dask_cuda_gpu_available
from utils.path_utils import resolve_input_path, resolve_output_path


raw_folder = "Raw_Data_2017"
processed_folder = "Processed_Data_2017"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def detect_gpu():
    """Best-effort GPU detection used for legacy prompts.

    Prefer lightweight Dask-CUDA detection, fall back to torch/tensorflow.
    """
    if dask_cuda_gpu_available():
        print("GPU detected.")
        return True, "dask_cuda"

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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compare column consistency between raw and processed folders, and compare row sets "
            "using streaming hashes (memory safe)."
        )
    )
    p.add_argument("--input-raw", default=raw_folder, help="Raw CSV folder (abs or repo-root-relative)")
    p.add_argument("--input-processed", default=processed_folder, help="Processed CSV folder (abs or repo-root-relative)")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Base output directory (abs or repo-root-relative). Defaults to repo outputs/",
    )
    p.add_argument("--engine", default="pandas", choices=["pandas", "dask", "dask-gpu"], help="Execution engine")
    p.add_argument("--chunk-size-mb", type=int, default=DEFAULT_CHUNK_SIZE_MB, help="Chunk size in MB")
    p.add_argument("--use-gpu", action="store_true", help="Force GPU (or fail)")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU")
    p.add_argument("--no-interactive", action="store_true", help="Disable interactive prompts")
    return p


def _count_rows_in_sqlite(db_path: str) -> int:
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT COUNT(*) FROM seen")
        return int(cur.fetchone()[0])
    finally:
        conn.close()


def _count_missing_from_raw(raw_db: str, processed_db: str) -> int:
    """Count hashes present in raw but not in processed."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(f"ATTACH DATABASE '{raw_db}' AS raw")
        conn.execute(f"ATTACH DATABASE '{processed_db}' AS proc")
        cur = conn.execute("SELECT COUNT(*) FROM raw.seen r LEFT JOIN proc.seen p ON r.h = p.h WHERE p.h IS NULL")
        return int(cur.fetchone()[0])
    finally:
        conn.close()


def _count_extra_in_processed(raw_db: str, processed_db: str) -> int:
    """Count hashes present in processed but not in raw."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(f"ATTACH DATABASE '{raw_db}' AS raw")
        conn.execute(f"ATTACH DATABASE '{processed_db}' AS proc")
        cur = conn.execute("SELECT COUNT(*) FROM proc.seen p LEFT JOIN raw.seen r ON p.h = r.h WHERE r.h IS NULL")
        return int(cur.fetchone()[0])
    finally:
        conn.close()


def main(argv: list[str] | None = None):
    args = build_arg_parser().parse_args(argv)

    selection = select_engine(engine=args.engine, use_gpu_flag=args.use_gpu, no_gpu_flag=args.no_gpu)
    if selection.engine != "pandas":
        print(f"[info] --engine {selection.engine} requested; this script currently runs in pandas mode for safety.")
    if selection.use_gpu:
        print("[info] GPU was approved, but this script uses CPU-based pandas. Using CPU.")
    device_used = "cpu"

    # Keep legacy prompt behavior when interactive.
    if not args.no_interactive:
        gpu_available, _ = detect_gpu()
        device_choice = prompt_for_device(gpu_available)
        if device_choice == "gpu":
            print("GPU selected, but this script uses CPU-based pandas. Using CPU.")

    raw_path = resolve_input_path(args.input_raw)
    processed_path = resolve_input_path(args.input_processed)

    if not os.path.isdir(raw_path) or not os.path.isdir(processed_path):
        print("ERROR: Raw or processed folder not found.")
        return

    raw_files = sorted([f for f in os.listdir(raw_path) if f.endswith(".csv")])
    processed_files = sorted([f for f in os.listdir(processed_path) if f.endswith(".csv")])
    if not raw_files:
        print("No raw CSV files found.")
        return

    chunk_mb = int(args.chunk_size_mb)

    # Mandatory chunk plan pre-calc (use first raw file).
    first_raw_file_path = os.path.join(raw_path, raw_files[0])
    plan0 = compute_chunk_plan(first_raw_file_path, chunk_mb)
    print_chunk_plan(plan0)

    chunk_rows = estimate_rows_per_chunk(first_raw_file_path, chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

    base_output_dir = resolve_output_path(args.output_dir)
    output_folder = os.path.join(base_output_dir, "Compare_Columns_and_Rows_2CSV")
    os.makedirs(output_folder, exist_ok=True)

    report_path = make_unique_path(os.path.join(output_folder, "comparison_report.txt"))

    # Disk-backed hash stores (no RAM growth).
    raw_db = os.path.join(output_folder, "raw_hashes.sqlite")
    processed_db = os.path.join(output_folder, "processed_hashes.sqlite")

    report_lines: list[str] = []

    raw_columns = list(pd.read_csv(first_raw_file_path, nrows=0).columns)
    print(f"Reference columns from raw CSV ({raw_files[0]}): {raw_columns}")
    report_lines.append(f"Reference columns from raw CSV ({raw_files[0]}): {raw_columns}")

    raw_mismatches = []
    for filename in raw_files:
        df = pd.read_csv(os.path.join(raw_path, filename), nrows=0)
        if list(df.columns) != raw_columns:
            raw_mismatches.append(filename)
            print(f"Column mismatch in raw CSV: {filename}")
            report_lines.append(f"Column mismatch in raw CSV: {filename}")

    processed_mismatches = []
    for filename in processed_files:
        df = pd.read_csv(os.path.join(processed_path, filename), nrows=0)
        if list(df.columns) != raw_columns:
            processed_mismatches.append(filename)
            print(f"Column mismatch in processed CSV: {filename}")
            report_lines.append(f"Column mismatch in processed CSV: {filename}")

    extra_rows_sample: list[tuple] = []
    total_rows_processed = 0

    print("\nCollecting row hashes from raw data...")
    with SQLiteHashStore(raw_db) as raw_store:
        for file_idx, filename in enumerate(raw_files, 1):
            path = os.path.join(raw_path, filename)
            file_plan = compute_chunk_plan(path, chunk_mb)
            print_chunk_plan(file_plan)

            try:
                for chunk_idx, chunk in enumerate(
                    pd.read_csv(path, chunksize=chunk_rows, dtype=str, usecols=raw_columns, low_memory=False), 1
                ):
                    print(format_progress(chunk_idx, file_plan.total_chunks))
                    total_rows_processed += len(chunk)
                    row_hashes = pd.util.hash_pandas_object(chunk, index=False).astype("int64")
                    # Insert hashes to disk-backed store. We don't need the keep-mask here.
                    raw_store.keep_mask(row_hashes.tolist())
            except Exception as e:
                print(f"  Skipping {filename} due to error: {e}")
                report_lines.append(f"Skipping {filename} due to error: {e}")
                continue

    raw_unique = _count_rows_in_sqlite(raw_db)

    print("Collecting row hashes from processed data...")
    with SQLiteHashStore(processed_db) as proc_store:
        for filename in processed_files:
            path = os.path.join(processed_path, filename)
            file_plan = compute_chunk_plan(path, chunk_mb)
            print_chunk_plan(file_plan)

            try:
                for chunk_idx, chunk in enumerate(
                    pd.read_csv(path, chunksize=chunk_rows, dtype=str, usecols=raw_columns, low_memory=False), 1
                ):
                    print(format_progress(chunk_idx, file_plan.total_chunks))
                    total_rows_processed += len(chunk)
                    row_hashes = pd.util.hash_pandas_object(chunk, index=False).astype("int64")
                    keep_mask = proc_store.keep_mask(row_hashes.tolist())

                    # Sample extra rows: those that are new for processed AND not found in raw.
                    # To avoid per-row DB lookups, we do a small bounded check using SQLite join.
                    if len(extra_rows_sample) < 10:
                        # We'll check only the newly inserted hashes from this chunk.
                        new_hashes = [h for h, keep in zip(row_hashes.tolist(), keep_mask) if keep]
                        if new_hashes:
                            import sqlite3

                            conn = sqlite3.connect(":memory:")
                            try:
                                conn.execute(f"ATTACH DATABASE '{raw_db}' AS raw")
                                # Create a temp table of just-new hashes.
                                conn.execute("CREATE TABLE temp.new(h INTEGER PRIMARY KEY)")
                                conn.executemany("INSERT OR IGNORE INTO temp.new(h) VALUES (?)", [(int(h),) for h in new_hashes])
                                missing = set(
                                    r[0]
                                    for r in conn.execute(
                                        "SELECT n.h FROM temp.new n LEFT JOIN raw.seen r ON n.h = r.h WHERE r.h IS NULL"
                                    )
                                )
                            finally:
                                conn.close()

                            if missing:
                                for idx, h in enumerate(row_hashes.tolist()):
                                    if h in missing and len(extra_rows_sample) < 10:
                                        extra_rows_sample.append(tuple(chunk.iloc[idx].tolist()))

            except Exception as e:
                print(f"  Skipping {filename} due to error: {e}")
                report_lines.append(f"Skipping {filename} due to error: {e}")
                continue

    processed_unique = _count_rows_in_sqlite(processed_db)

    missing_rows_count = _count_missing_from_raw(raw_db, processed_db)
    extra_rows_count = _count_extra_in_processed(raw_db, processed_db)

    report_lines.append(f"Total unique rows in raw data: {raw_unique}")
    report_lines.append(f"Total unique rows in processed data: {processed_unique}")
    report_lines.append(f"Missing rows count: {missing_rows_count}")
    report_lines.append(f"Extra rows count: {extra_rows_count}")

    print(f"Total unique rows in raw data: {raw_unique}")
    print(f"Total unique rows in processed data: {processed_unique}")

    if not missing_rows_count and not extra_rows_count:
        print("All raw data rows are present in the processed folder.")
        report_lines.append("All raw data rows are present in the processed folder.")
    else:
        if missing_rows_count:
            print(f"{missing_rows_count} rows from raw data are missing in processed CSVs.")
        if extra_rows_count:
            print(f"{extra_rows_count} extra rows found in processed CSVs that were not in raw data.")

    if extra_rows_sample:
        report_lines.append("\nSample extra rows (up to 10):")
        for i, row in enumerate(extra_rows_sample, 1):
            report_lines.append(f"Row {i}: {row}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved to: {report_path}")

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Device used: {device_used.upper()}")
    print(f"Chunk size: {chunk_mb}MB (~{chunk_rows:,} rows)")
    print(f"Total rows processed: {total_rows_processed:,}")
    print("Rows saved: N/A")
    print("Output paths:")
    print(f"  - {report_path}")
    print(f"  - {raw_db}")
    print(f"  - {processed_db}")


if __name__ == "__main__":
    main()

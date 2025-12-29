# What changed:
# - Added GPU detection/device prompt, chunk size prompt with row estimation, and streaming processing.
# - Standardized outputs under ./outputs/Downscale_Csv_2018 with non-overwrite paths.
# - Added optional max-rows limit for output CSVs plus a final processing summary.
#
# Purpose:
# - Downscale benign data by sampling and keep all attack rows.
# - Save separate benign and attack CSVs.
# - Report label counts for each output.

import os
import sys

# Allow running this script from any working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import pandas as pd
from collections import Counter

from config.global_config import (
    DEFAULT_CHUNK_SIZE_MB,
    DEFAULT_MAX_OUTPUT_ROWS,
)
from utils.path_utils import resolve_input_path, resolve_output_path
from utils.gpu_utils import gpu_available as dask_cuda_gpu_available
from utils.chunk_utils import compute_chunk_plan, format_progress, print_chunk_plan
from utils.engine_utils import select_engine


# Input folder with all csv files
INPUT_FOLDER = "2018_Separated_Nomissing"

# Output folder name under ./outputs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Downscale_Csv_2018")

# Combine 10% of the benign data from each file
BENIGN_SAMPLING_FRACTION = 0.1

RANDOM_STATE = 42


def detect_gpu():
    """Best-effort GPU detection used for legacy prompts.

    We prefer a lightweight Dask-CUDA check (no heavy framework imports).
    If that fails, we fall back to torch/tensorflow checks.
    """
    if dask_cuda_gpu_available():
        print("GPU detected.")
        return True, "dask_cuda"

    # --- legacy fallbacks ---
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


def estimate_rows_per_chunk(file_path, chunk_mb, sample_rows=2000, default_rows=1_000_000):
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


def find_label_column(columns):
    for col in columns:
        if str(col).strip().lower() == "label":
            return col
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Downscale benign rows (sampling) and keep all attack rows from CSV files. "
            "Writes benign.csv and attacks.csv to an output directory."
        )
    )

    parser.add_argument(
        "--engine",
        default="pandas",
        choices=["pandas", "dask", "dask-gpu"],
        help="Execution engine (dask support will be added repo-wide; pandas is used today).",
    )
    parser.add_argument("--use-gpu", action="store_true", help="Force GPU (or fail)")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU")

    parser.add_argument(
        "--input",
        default=INPUT_FOLDER,
        help="Input folder containing CSV files (absolute or repo-root-relative).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory (absolute or repo-root-relative). "
            "Defaults to './outputs/Downscale_Csv_2018' under the repo root."
        ),
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=DEFAULT_CHUNK_SIZE_MB,
        help=f"Chunk size in MB (default: {DEFAULT_CHUNK_SIZE_MB}).",
    )
    parser.add_argument(
        "--max-output-rows",
        type=int,
        default=DEFAULT_MAX_OUTPUT_ROWS,
        help=(
            "Maximum rows to save per output file (benign and attacks). "
            f"Default: {DEFAULT_MAX_OUTPUT_ROWS}."
        ),
    )
    parser.add_argument(
        "--benign-fraction",
        type=float,
        default=BENIGN_SAMPLING_FRACTION,
        help=f"Fraction of benign rows kept per chunk (default: {BENIGN_SAMPLING_FRACTION}).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help=f"Random seed for sampling/shuffling (default: {RANDOM_STATE}).",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive prompts; requires --max-output-rows if you want a cap.",
    )

    return parser


def main(argv: list[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Engine selection (forward-compatible). This script currently performs pandas streaming.
    selection = select_engine(engine=args.engine, use_gpu_flag=args.use_gpu, no_gpu_flag=args.no_gpu)
    if selection.engine != "pandas":
        print(f"[info] --engine {selection.engine} requested; this script currently runs in pandas mode for safety.")
    if selection.use_gpu:
        print("[info] GPU was approved, but this script uses CPU-based pandas. Using CPU.")
    device_used = "cpu"

    # Keep the legacy prompt behavior (but avoid heavy framework imports when possible).
    if not args.no_interactive:
        gpu_present, _ = detect_gpu()
        device_choice = prompt_for_device(gpu_present)
        if device_choice == "gpu":
            print("GPU selected, but this script uses CPU-based pandas. Using CPU.")

    input_folder = resolve_input_path(args.input)

    # Output: if user passes --output-dir, use it; else default to <repo_root>/outputs/Downscale_Csv_2018
    base_output_dir = resolve_output_path(args.output_dir)
    output_folder = os.path.join(base_output_dir, "Downscale_Csv_2018")

    if not os.path.isdir(input_folder):
        print(f"ERROR: Input folder not found at '{input_folder}'")
        return

    csv_files: list[str] = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print("No CSV files found in the input folder.")
        return

    chunk_mb = int(args.chunk_size_mb)

    # Mandatory chunk pre-calculation (from file size) for progress tracking.
    plan0 = compute_chunk_plan(csv_files[0], chunk_mb)
    print_chunk_plan(plan0)

    chunk_rows = estimate_rows_per_chunk(csv_files[0], chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

    max_rows_limit = args.max_output_rows

    benign_fraction = float(args.benign_fraction)
    random_state = int(args.random_state)

    os.makedirs(output_folder, exist_ok=True)
    output_benign_file = make_unique_path(os.path.join(output_folder, "benign.csv"))
    output_attacks_file = make_unique_path(os.path.join(output_folder, "attacks.csv"))

    print("Starting the downscaling and separation process...")

    benign_written = 0
    attack_written = 0
    total_rows_processed = 0
    benign_label_counts = Counter()
    attack_label_counts = Counter()

    benign_header_written = False
    attack_header_written = False

    for file_idx, file_path in enumerate(csv_files, 1):
        print(f"\nProcessing {file_path} ({file_idx}/{len(csv_files)})...")

        file_plan = compute_chunk_plan(file_path, chunk_mb)
        print_chunk_plan(file_plan)

        try:
            chunk_iter = pd.read_csv(file_path, chunksize=chunk_rows, low_memory=False)
        except Exception as e:
            print(f"  -> Error reading file: {e}. Skipping.")
            continue

        label_col_found = None
        for chunk_idx, chunk in enumerate(chunk_iter, 1):
            # Standard progress format (mandatory).
            print(format_progress(chunk_idx, file_plan.total_chunks))

            if label_col_found is None:
                label_col_found = find_label_column(chunk.columns)
                if not label_col_found:
                    print("  -> No label column found in this file. Skipping.")
                    break

            total_rows_processed += len(chunk)

            labels = chunk[label_col_found].astype(str)
            benign_mask = labels.str.lower().eq("benign")

            benign_chunk = chunk[benign_mask]
            attack_chunk = chunk[~benign_mask]

            if not benign_chunk.empty:
                benign_sample = benign_chunk.sample(frac=benign_fraction, random_state=random_state)
                if not benign_sample.empty:
                    if max_rows_limit is not None:
                        remaining = max_rows_limit - benign_written
                        if remaining <= 0:
                            benign_sample = benign_sample.iloc[:0]
                        elif len(benign_sample) > remaining:
                            benign_sample = benign_sample.iloc[:remaining]
                    if not benign_sample.empty:
                        benign_sample = benign_sample.sample(frac=1, random_state=random_state)
                        benign_sample.to_csv(
                            output_benign_file,
                            index=False,
                            mode="w" if not benign_header_written else "a",
                            header=not benign_header_written,
                        )
                        benign_header_written = True
                        benign_written += len(benign_sample)
                        benign_label_counts.update(benign_sample[label_col_found].astype(str))

            if not attack_chunk.empty:
                if max_rows_limit is not None:
                    remaining = max_rows_limit - attack_written
                    if remaining <= 0:
                        attack_chunk = attack_chunk.iloc[:0]
                    elif len(attack_chunk) > remaining:
                        attack_chunk = attack_chunk.iloc[:remaining]
                if not attack_chunk.empty:
                    attack_chunk = attack_chunk.sample(frac=1, random_state=random_state)
                    attack_chunk.to_csv(
                        output_attacks_file,
                        index=False,
                        mode="w" if not attack_header_written else "a",
                        header=not attack_header_written,
                    )
                    attack_header_written = True
                    attack_written += len(attack_chunk)
                    attack_label_counts.update(attack_chunk[label_col_found].astype(str))

            if max_rows_limit is not None and benign_written >= max_rows_limit and attack_written >= max_rows_limit:
                print("  Max rows limit reached for both outputs. Stopping early.")
                break

    print("\n" + "=" * 60)
    print(" FINAL DATASET REPORT")
    print("=" * 60)

    if benign_header_written:
        print("\n--- Counts for benign.csv ---")
        print(f"Total Rows: {benign_written:,}")
        for label, count in benign_label_counts.items():
            print(f"  {label}: {count}")
    else:
        print("\n--- No benign.csv was created ---")

    if attack_header_written:
        print("\n--- Counts for attacks.csv ---")
        print(f"Total Rows: {attack_written:,}")
        for label, count in attack_label_counts.items():
            print(f"  {label}: {count}")
    else:
        print("\n--- No attacks.csv was created ---")

    print("\n" + "=" * 60)
    print("Process finished successfully!")

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Device used: {device_used.upper()}")
    print(f"Chunk size: {chunk_mb}MB (~{chunk_rows:,} rows)")
    print(f"Total rows processed: {total_rows_processed:,}")
    print(f"Rows saved (benign): {benign_written:,}")
    print(f"Rows saved (attacks): {attack_written:,}")
    print("Output paths:")
    if benign_header_written:
        print(f"  - {output_benign_file}")
    if attack_header_written:
        print(f"  - {output_attacks_file}")


if __name__ == "__main__":
    main()

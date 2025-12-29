# What changed:
# - Added GPU detection/device prompt and chunk size prompt with row estimation.
# - Streamed splitting to avoid full in-memory loads; added optional max-rows limits.
# - Standardized outputs under ./outputs/Separated_Model_Data with final summary.
#
# Purpose:
# - Analyze label distribution across CSVs.
# - Separate benign and attack rows into output files.
# - Save chunked outputs with progress reporting.

import os
import sys
import argparse

# Allow running this script from any working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
from collections import defaultdict

from config.global_config import DEFAULT_CHUNK_SIZE_MB, DEFAULT_MAX_OUTPUT_ROWS
from utils.chunk_utils import compute_chunk_plan, format_progress, print_chunk_plan
from utils.engine_utils import select_engine
from utils.path_utils import resolve_input_path, resolve_output_path

# --- 1. Global Configuration ---
INPUT_FOLDER = "Normalized_SET"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Separated_Model_Data")

LABEL_COLUMN_NAME = 'label'
BENIGN_LABEL_VALUE = 'benign'

CHUNK_ROWS = 500_000

SUMMARY = {
    "total_rows_processed": 0,
    "rows_saved": 0,
    "output_paths": [],
}


# --- Helpers ---

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


def record_output(path, rows_saved=0, rows_processed=0):
    if path:
        SUMMARY["output_paths"].append(path)
    SUMMARY["rows_saved"] += int(rows_saved)
    SUMMARY["total_rows_processed"] += int(rows_processed)


# --- 2. Core Functions ---

def analyze_and_classify(all_files, processing_mode):
    print("--- Phase 1: Analyzing all files for counts and classification ---")
    total_counts = defaultdict(int)
    files_by_label = defaultdict(set)
    actual_label_col_name = None

    for file_path in all_files:
        print(f"  Scanning: {os.path.basename(file_path)}...")
        try:
            if actual_label_col_name is None:
                header_df = pd.read_csv(file_path, nrows=0, low_memory=False)
                for col in header_df.columns:
                    if col.lower() == LABEL_COLUMN_NAME:
                        actual_label_col_name = col
                        break
            if not actual_label_col_name:
                print(f"    Warning: Label column '{LABEL_COLUMN_NAME}' not found. Skipping.")
                continue

            if processing_mode != 'both':
                try:
                    preview_df = pd.read_csv(file_path, usecols=[actual_label_col_name], nrows=20, low_memory=False)
                    unique_labels_in_preview = set(preview_df[actual_label_col_name].str.lower().unique())
                    if processing_mode == 'attacks' and unique_labels_in_preview == {BENIGN_LABEL_VALUE}:
                        print("    -> Optimization: Skipping file as it appears to contain only benign data.")
                        continue
                    if processing_mode == 'benign' and BENIGN_LABEL_VALUE not in unique_labels_in_preview:
                        print("    -> Optimization: Skipping file as it appears to contain only attack data.")
                        continue
                except Exception as e:
                    print(f"    Warning: Could not preview file. Proceeding with full scan. Error: {e}")

            for chunk in pd.read_csv(file_path, usecols=[actual_label_col_name], chunksize=CHUNK_ROWS, low_memory=False):
                chunk.columns = [col.lower() for col in chunk.columns]
                chunk_counts = chunk[LABEL_COLUMN_NAME].value_counts()
                for label, count in chunk_counts.items():
                    total_counts[label] += count
                    files_by_label[label].add(file_path)
        except Exception as e:
            print(f"    Error analyzing {os.path.basename(file_path)}: {e}")

    files_by_label = {label: list(paths) for label, paths in files_by_label.items()}
    print("--- Analysis complete ---")
    return total_counts, files_by_label, actual_label_col_name


def process_and_save_combined(
    file_list,
    rows_per_output_file,
    labels_to_keep,
    output_group_name,
    output_base_path,
    should_shuffle,
    actual_label_col_name,
    max_rows_limit=None,
):
    if not file_list or not labels_to_keep:
        return

    print(f"\nProcessing Group Sequentially: {output_group_name}")
    print(f"  - Using {len(file_list)} source file(s).")
    print(f"  - Aiming for {rows_per_output_file:,} rows per output file.")

    os.makedirs(output_base_path, exist_ok=True)
    lower_labels_to_keep = [str(lbl).lower() for lbl in labels_to_keep]
    iterators = {}
    for file_path in file_list:
        try:
            iterators[file_path] = pd.read_csv(file_path, iterator=True, chunksize=CHUNK_ROWS, low_memory=False)
        except Exception as e:
            print(f"  Warning: Could not open {os.path.basename(file_path)}. Skipping it. Error: {e}")

    file_part_counter = 1
    leftover_df = pd.DataFrame()
    total_saved = 0

    while iterators:
        if max_rows_limit is not None and total_saved >= max_rows_limit:
            break

        batch_dataframes = [leftover_df] if not leftover_df.empty else []
        rows_collected = len(leftover_df)

        while rows_collected < rows_per_output_file and iterators:
            iterators_this_pass = list(iterators.keys())
            for file_path in iterators_this_pass:
                try:
                    chunk = next(iterators[file_path])
                    SUMMARY["total_rows_processed"] += len(chunk)
                    clean_chunk = chunk[chunk[actual_label_col_name].str.lower().isin(lower_labels_to_keep)]
                    if not clean_chunk.empty:
                        batch_dataframes.append(clean_chunk)
                        rows_collected += len(clean_chunk)
                        if rows_collected >= rows_per_output_file:
                            break
                except StopIteration:
                    del iterators[file_path]
                except Exception as e:
                    print(f"  Error reading chunk from {os.path.basename(file_path)}. Removing it. Error: {e}")
                    del iterators[file_path]

        if not batch_dataframes:
            break

        combined_df = pd.concat(batch_dataframes, ignore_index=True)
        if should_shuffle:
            combined_df = combined_df.sample(frac=1).reset_index(drop=True)

        if max_rows_limit is not None:
            remaining = max_rows_limit - total_saved
            if remaining <= 0:
                break
            if len(combined_df) > remaining:
                combined_df = combined_df.iloc[:remaining]

        final_df = combined_df.iloc[:rows_per_output_file]
        leftover_df = combined_df.iloc[rows_per_output_file:]

        output_filename = os.path.join(output_base_path, f"{output_group_name}_part_{file_part_counter}.csv")
        output_filename = make_unique_path(output_filename)
        final_df.to_csv(output_filename, index=False)
        print(f"  -> Saved {len(final_df):,} rows to {os.path.relpath(output_filename)}")
        record_output(output_filename, rows_saved=len(final_df))
        total_saved += len(final_df)
        file_part_counter += 1

    if not leftover_df.empty and (max_rows_limit is None or total_saved < max_rows_limit):
        if max_rows_limit is not None:
            remaining = max_rows_limit - total_saved
            leftover_df = leftover_df.iloc[:remaining]
        output_filename = os.path.join(output_base_path, f"{output_group_name}_part_{file_part_counter}.csv")
        output_filename = make_unique_path(output_filename)
        leftover_df.to_csv(output_filename, index=False)
        print(f"  -> Saved {len(leftover_df):,} final rows to {os.path.relpath(output_filename)}")
        record_output(output_filename, rows_saved=len(leftover_df))

    print(f"  - Finished processing for group '{output_group_name}'.")


# --- 3. Main ---

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Analyze label distributions and separate benign/attack rows into output files (streaming-safe)."
        )
    )
    p.add_argument("--input", default=INPUT_FOLDER, help="Input folder containing CSVs")
    p.add_argument("--output-dir", default=None, help="Base output directory")
    p.add_argument("--chunk-size-mb", type=int, default=DEFAULT_CHUNK_SIZE_MB, help="Chunk size in MB")
    p.add_argument("--engine", default="pandas", choices=["pandas", "dask", "dask-gpu"], help="Execution engine")
    p.add_argument("--use-gpu", action="store_true", help="Force GPU (or fail)")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU")
    p.add_argument("--no-interactive", action="store_true", help="Disable interactive prompts")
    p.add_argument(
        "--processing-mode",
        choices=["benign", "attacks", "both"],
        default=None,
        help="Override interactive group selection",
    )
    p.add_argument(
        "--rows-per-file",
        type=int,
        default=None,
        help="Max rows per output file (non-interactive override)",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle output rows before saving (non-interactive)",
    )
    p.add_argument(
        "--max-output-rows",
        type=int,
        default=DEFAULT_MAX_OUTPUT_ROWS,
        help="Max total rows to write per group (non-interactive default)",
    )
    return p


def get_user_choice(prompt: str, *, default: str | None = None) -> str:
    if _NO_INTERACTIVE:
        if default is None:
            raise ValueError(f"Non-interactive mode requires default for prompt: {prompt}")
        print(f"{prompt} [auto: {default}]")
        return str(default)
    return input(prompt)


def get_yes_no(prompt: str, *, default: bool = False) -> bool:
    if _NO_INTERACTIVE:
        print(f"{prompt} (y/n): [auto: {'y' if default else 'n'}]")
        return bool(default)
    return input(f"{prompt} (y/n): ").strip().lower() in {"y", "yes"}


def main(argv: list[str] | None = None):
    global _NO_INTERACTIVE, CHUNK_ROWS, OUTPUT_FOLDER
    args = build_arg_parser().parse_args(argv)
    _NO_INTERACTIVE = args.no_interactive

    # Engine selection (pandas streaming today)
    selection = select_engine(engine=args.engine, use_gpu_flag=args.use_gpu, no_gpu_flag=args.no_gpu)
    if selection.engine != "pandas":
        print(f"[info] --engine {selection.engine} requested; this script currently runs in pandas mode.")
    if selection.use_gpu:
        print("[info] GPU was approved, but this script uses CPU-based pandas. Using CPU.")
    device_used = "cpu"

    input_folder = resolve_input_path(args.input)
    base_output_dir = resolve_output_path(args.output_dir)
    OUTPUT_FOLDER = os.path.join(base_output_dir, "Separated_Model_Data")

    if not os.path.isdir(input_folder):
        print(f"No CSV files found in '{input_folder}'. Exiting.")
        return

    all_csv_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_folder)
        for file in files if file.endswith(".csv")
    ]
    if not all_csv_files:
        print(f"No CSV files found in '{input_folder}'. Exiting.")
        return

    chunk_mb = int(args.chunk_size_mb)
    plan0 = compute_chunk_plan(all_csv_files[0], chunk_mb)
    print_chunk_plan(plan0)

    CHUNK_ROWS = estimate_rows_per_chunk(all_csv_files[0], chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{CHUNK_ROWS:,} rows per chunk)")

    # Choose processing mode
    if args.processing_mode is not None:
        processing_mode = args.processing_mode
    else:
        if _NO_INTERACTIVE:
            processing_mode = "both"
        else:
            while True:
                print("\nPlease choose which data group to process:")
                print("  1: Benign Only")
                print("  2: Attacks Only")
                print("  3: Both Benign and Attacks")
                choice = get_user_choice("Enter your choice (1, 2, or 3): ").strip()
                if choice in ['1', '2', '3']:
                    break
                print("Invalid choice. Please enter 1, 2, or 3.")
            processing_mode = 'both'
            if choice == '1':
                processing_mode = 'benign'
            elif choice == '2':
                processing_mode = 'attacks'

    total_counts, files_by_label, actual_label_col = analyze_and_classify(all_csv_files, processing_mode)
    if not actual_label_col:
        print("Could not determine the 'Label' column from any file. Exiting.")
        return

    print("\n--- Total Row Count Report (from analyzed files) ---")
    benign_label_in_data = None
    attack_labels_in_data = {}
    for label, count in sorted(total_counts.items()):
        print(f"  - {label}: {count:,} total rows.")
        if str(label).lower() == BENIGN_LABEL_VALUE:
            benign_label_in_data = label
        else:
            attack_labels_in_data[label] = count
    print("-------------------------------------------------")

    process_benign = processing_mode in ['benign', 'both']
    process_attacks = processing_mode in ['attacks', 'both']

    should_shuffle = bool(args.shuffle) if _NO_INTERACTIVE else get_yes_no("Do you want to shuffle the final output files?", default=False)
    max_rows_limit = int(args.max_output_rows) if _NO_INTERACTIVE else prompt_for_max_rows()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if args.rows_per_file is not None:
        rows_per_file = int(args.rows_per_file)
    else:
        rows_per_file = None

    if process_benign and benign_label_in_data:
        print("\n" + "=" * 30 + " PROCESSING BENIGN DATA " + "=" * 30)
        if rows_per_file is None:
            if _NO_INTERACTIVE:
                rows_per_file = 500_000
                print(f"Enter max rows per Benign file: [auto: {rows_per_file}]")
            else:
                while True:
                    try:
                        rows_per_file = int(input("Enter max rows per Benign file: ").strip())
                        if rows_per_file > 0:
                            break
                        print("  Please enter a positive number.")
                    except ValueError:
                        print("  Invalid input. Please enter a whole number.")

        process_and_save_combined(
            file_list=files_by_label.get(benign_label_in_data, []),
            rows_per_output_file=int(rows_per_file),
            labels_to_keep=[benign_label_in_data],
            output_group_name='Benign',
            output_base_path=os.path.join(OUTPUT_FOLDER, 'Benign'),
            should_shuffle=should_shuffle,
            actual_label_col_name=actual_label_col,
            max_rows_limit=max_rows_limit,
        )
    elif process_benign:
        print("\nSkipping Benign processing: No 'Benign' labels found in the analyzed data.")

    if process_attacks and attack_labels_in_data:
        print("\n" + "=" * 30 + " PROCESSING ATTACK DATA " + "=" * 30)
        all_attack_files = sorted(list(set(f for lbl in attack_labels_in_data for f in files_by_label.get(lbl, []))))
        all_attack_labels = list(attack_labels_in_data.keys())
        total_attack_rows = sum(attack_labels_in_data.values())

        if rows_per_file is None:
            if _NO_INTERACTIVE:
                rows_per_file = 500_000
                print(f"Enter max rows per Attack file ({total_attack_rows:,} total available): [auto: {rows_per_file}]")
            else:
                while True:
                    try:
                        rows_per_file = int(input(f"Enter max rows per Attack file ({total_attack_rows:,} total available): ").strip())
                        if rows_per_file > 0:
                            break
                        print("  Please enter a positive number.")
                    except ValueError:
                        print("  Invalid input. Please enter a whole number.")

        process_and_save_combined(
            file_list=all_attack_files,
            rows_per_output_file=int(rows_per_file),
            labels_to_keep=all_attack_labels,
            output_group_name='Attacks',
            output_base_path=os.path.join(OUTPUT_FOLDER, 'Attacks'),
            should_shuffle=should_shuffle,
            actual_label_col_name=actual_label_col,
            max_rows_limit=max_rows_limit,
        )
    elif process_attacks:
        print("\nSkipping Attack processing: No attack labels found in the analyzed data.")

    print("\n" + "=" * 80 + "\nAll processing is complete!\n" + "=" * 80)

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Device used: {device_used.upper()}")
    print(f"Chunk size: {chunk_mb}MB (~{CHUNK_ROWS:,} rows)")
    print(f"Total rows processed: {SUMMARY['total_rows_processed']:,}")
    print(f"Rows saved: {SUMMARY['rows_saved']:,}")
    print("Output paths:")
    for path in SUMMARY["output_paths"]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()

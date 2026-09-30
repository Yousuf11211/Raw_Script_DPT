# What changed:
# - Proportional sampling: Extracts an equal quota of rows from all input files per batch.
# - Added prompt for max output files to generate (prevents unwanted extra files).
# - Streamed splitting to avoid full in-memory loads; added optional max-rows limits.
# - Standardized outputs under ./outputs/Separated_Model_Data with final summary.

import os
import sys
import argparse
import math
from collections import defaultdict

# Allow running this script from any working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from config.global_config import DEFAULT_CHUNK_SIZE_MB, DEFAULT_MAX_OUTPUT_ROWS
from utils.chunk_utils import compute_chunk_plan, format_progress, print_chunk_plan
from utils.engine_utils import select_engine
from utils.path_utils import resolve_input_path, resolve_output_path

# --- 1. Global Configuration ---
INPUT_FOLDER = "IDS2018"

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
        response = input("Limit total rows to save? (y/n): ").strip().lower()
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


def prompt_for_max_files(group_name=""):
    prompt_label = f"How many {group_name} files do you want to generate? (e.g. 1 for quick test, or press Enter for 'all'): "
    while True:
        response = input(prompt_label).strip().lower()
        if response in ["", "all"]:
            return None
        try:
            val = int(response)
            if val > 0:
                return val
        except ValueError:
            pass
        print("  Please enter a positive whole number or press Enter for 'all'.")


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
                    unique_labels_in_preview = set(preview_df[actual_label_col_name].astype(str).str.lower().unique())
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
    max_files_limit=None,
):
    if not file_list or not labels_to_keep:
        return

    print(f"\nProcessing Group Sequentially: {output_group_name}")
    print(f"  - Using {len(file_list)} source file(s).")
    print(f"  - Aiming for {rows_per_output_file:,} rows per output file.")
    if max_files_limit:
        print(f"  - File limit: Max {max_files_limit} file(s) will be generated.")

    os.makedirs(output_base_path, exist_ok=True)
    lower_labels_to_keep = [str(lbl).lower() for lbl in labels_to_keep]

    # Initialize streaming iterators for every source file
    iterators = {}
    for file_path in file_list:
        try:
            iterators[file_path] = pd.read_csv(file_path, iterator=True, chunksize=CHUNK_ROWS, low_memory=False)
        except Exception as e:
            print(f"  Warning: Could not open {os.path.basename(file_path)}. Skipping it. Error: {e}")

    # Per-file buffers so extra rows read from a chunk aren't lost
    file_buffers = {fp: pd.DataFrame() for fp in iterators.keys()}
    file_part_counter = 1
    total_saved = 0

    while iterators or any(not buf.empty for buf in file_buffers.values()):
        # Stop if user-defined file count is reached
        if max_files_limit is not None and file_part_counter > max_files_limit:
            print(f"  -> Reached file limit of {max_files_limit}. Stopping.")
            break

        # Stop if user-defined total row count is reached
        if max_rows_limit is not None and total_saved >= max_rows_limit:
            break

        batch_dataframes = []
        rows_collected = 0

        # Round-robin: sample equally across all active source files
        while rows_collected < rows_per_output_file:
            active_sources = [fp for fp in list(file_buffers.keys()) if (fp in iterators or not file_buffers[fp].empty)]
            if not active_sources:
                break

            remaining_needed = rows_per_output_file - rows_collected
            per_file_target = max(1, math.ceil(remaining_needed / len(active_sources)))

            progress_made = False
            for fp in active_sources:
                # Top up buffer from iterator if below quota
                while len(file_buffers[fp]) < per_file_target and fp in iterators:
                    try:
                        chunk = next(iterators[fp])
                        SUMMARY["total_rows_processed"] += len(chunk)
                        chunk_clean = chunk[chunk[actual_label_col_name].astype(str).str.lower().isin(lower_labels_to_keep)]
                        if not chunk_clean.empty:
                            file_buffers[fp] = pd.concat([file_buffers[fp], chunk_clean], ignore_index=True)
                    except StopIteration:
                        del iterators[fp]
                    except Exception as e:
                        print(f"  Error reading from {os.path.basename(fp)}: {e}")
                        if fp in iterators:
                            del iterators[fp]

                # Pull proportional rows from this file's buffer
                if not file_buffers[fp].empty:
                    pull_count = min(len(file_buffers[fp]), per_file_target, rows_per_output_file - rows_collected)
                    if pull_count > 0:
                        slice_df = file_buffers[fp].iloc[:pull_count]
                        file_buffers[fp] = file_buffers[fp].iloc[pull_count:]
                        batch_dataframes.append(slice_df)
                        rows_collected += pull_count
                        progress_made = True

                if rows_collected >= rows_per_output_file:
                    break

            if not progress_made:
                break

        if not batch_dataframes:
            break

        # Combine all proportional slices
        combined_df = pd.concat(batch_dataframes, ignore_index=True)
        if should_shuffle:
            combined_df = combined_df.sample(frac=1).reset_index(drop=True)

        if max_rows_limit is not None:
            remaining = max_rows_limit - total_saved
            if remaining <= 0:
                break
            if len(combined_df) > remaining:
                combined_df = combined_df.iloc[:remaining]

        output_filename = os.path.join(output_base_path, f"{output_group_name}_part_{file_part_counter}.csv")
        output_filename = make_unique_path(output_filename)
        combined_df.to_csv(output_filename, index=False)
        print(f"  -> Saved {len(combined_df):,} rows to {os.path.relpath(output_filename)} (sampled evenly across sources)")
        record_output(output_filename, rows_saved=len(combined_df))
        total_saved += len(combined_df)
        file_part_counter += 1

    print(f"  - Finished processing for group '{output_group_name}'.")


# --- 3. Main ---

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze label distributions and separate benign/attack rows into output files (streaming-safe)."
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
        "--max-files",
        type=int,
        default=None,
        help="Max number of output files to write per group",
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

    should_shuffle = bool(args.shuffle) if _NO_INTERACTIVE else get_yes_no("Do you want to shuffle the final output files?", default=True)
    max_rows_limit = int(args.max_output_rows) if _NO_INTERACTIVE else prompt_for_max_rows()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    rows_per_file = int(args.rows_per_file) if args.rows_per_file is not None else None

    # --- Process Benign ---
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

        max_files_benign = args.max_files if _NO_INTERACTIVE else prompt_for_max_files("Benign")

        process_and_save_combined(
            file_list=files_by_label.get(benign_label_in_data, []),
            rows_per_output_file=int(rows_per_file),
            labels_to_keep=[benign_label_in_data],
            output_group_name='Benign',
            output_base_path=os.path.join(OUTPUT_FOLDER, 'Benign'),
            should_shuffle=should_shuffle,
            actual_label_col_name=actual_label_col,
            max_rows_limit=max_rows_limit,
            max_files_limit=max_files_benign,
        )
    elif process_benign:
        print("\nSkipping Benign processing: No 'Benign' labels found in the analyzed data.")

    # --- Process Attacks ---
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

        max_files_attacks = args.max_files if _NO_INTERACTIVE else prompt_for_max_files("Attack")

        process_and_save_combined(
            file_list=all_attack_files,
            rows_per_output_file=int(rows_per_file),
            labels_to_keep=all_attack_labels,
            output_group_name='Attacks',
            output_base_path=os.path.join(OUTPUT_FOLDER, 'Attacks'),
            should_shuffle=should_shuffle,
            actual_label_col_name=actual_label_col,
            max_rows_limit=max_rows_limit,
            max_files_limit=max_files_attacks,
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
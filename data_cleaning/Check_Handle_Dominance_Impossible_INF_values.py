# What changed:
# - Added GPU detection/device prompt and chunk size prompt with row estimation.
# - Standardized outputs under ./outputs/Training_isolation_model with non-overwrite paths.
# - Added optional max-rows limit for saved CSVs and a final summary.
#
# Purpose:
# - Generate dominance reports, validate data, and handle inf values.
# - Optionally clean CSVs based on detected issues.
# - Provide helper functions for dominance analysis and column deletion.

import os
import sys

# Allow running this script from any working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import argparse

from config.global_config import DEFAULT_CHUNK_SIZE_MB, DEFAULT_MAX_OUTPUT_ROWS
from utils.path_utils import resolve_input_path, resolve_output_path
from utils.gpu_utils import gpu_available as dask_cuda_gpu_available

# --- GLOBAL CONFIGURATION VARIABLES ---
INPUT_FOLDER = "Bening"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Training_isolation_model")

DOMINANCE_RANGES = [
    (1.0, 1.01, "100%"),
    (0.95, 1.0, "95-100%"),
    (0.90, 0.95, "90-95%"),
    (0.80, 0.90, "80-90%"),
    (0.70, 0.80, "70-80%"),
    (0.60, 0.70, "60-70%"),
    (0.50, 0.60, "50-60%"),
]
NEVER_NEGATIVE_KEYWORDS = [
    'port', 'duration', 'count', 'bytes', 'size', 'rate', 'percentage',
    'variance', 'std', 'total', 'max', 'min', 'median', 'mode', 'mean',
    'iat', 'active', 'idle', 'bulk', 'handshake', 'subflow'
]
CAN_BE_NEGATIVE_KEYWORDS = ['skew', 'cov', 'delta']
PORT_COLUMNS = ['src_port', 'dst_port']
INF_THRESHOLD = 0

CHUNK_ROWS = 1_000_000

SUMMARY = {
    "total_rows_processed": 0,
    "rows_saved": 0,
    "output_paths": [],
}


# ============================================================================== 
# HELPERS
# ============================================================================== 

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dominance reporting + data validation + inf handling for CSV files in a folder. "
            "Supports interactive selection, but also accepts CLI defaults for automation."
        )
    )
    parser.add_argument(
        "--input",
        default=INPUT_FOLDER,
        help="Input folder (absolute or repo-root-relative).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base output directory (absolute or repo-root-relative). Defaults to ./outputs.",
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
        help=f"Max rows to write for any produced CSV (default: {DEFAULT_MAX_OUTPUT_ROWS}).",
    )
    parser.add_argument(
        "--task",
        choices=["1", "2", "3"],
        default=None,
        help="Task to run: 1=dominance report, 2=validate/remove invalid rows, 3=inf handling.",
    )
    parser.add_argument(
        "--files",
        default=None,
        help="Comma-separated file indices to process (e.g. '1,3'), or 'all'. If omitted, prompt.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive prompts; requires --task and --files (or --files=all).",
    )
    return parser


# ==============================================================================
# TASK 1: DOMINANCE REPORT LOGIC
# ============================================================================== 

def generate_dominance_report(file_path):
    print(f"\nGenerating Dominance Report for: {os.path.basename(file_path)}")
    col_counters = defaultdict(Counter)
    total_counts = Counter()
    label_counter = Counter()
    col_value_label_counter = defaultdict(lambda: defaultdict(Counter))
    rows_processed = 0

    try:
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_ROWS, dtype=str, low_memory=False):
            rows_processed += len(chunk)
            labels = chunk.get("Label") or chunk.get("label")
            if labels is not None:
                label_counter.update(labels.dropna())
            for col in chunk.columns:
                values = chunk[col].dropna()
                col_counters[col].update(values)
                total_counts[col] += len(values)
                if labels is not None and col.lower() != "label":
                    for v, lbl in zip(chunk[col], labels):
                        if pd.notna(v) and pd.notna(lbl):
                            col_value_label_counter[col][v][lbl] += 1

        bucketed = {label: [] for _, _, label in DOMINANCE_RANGES}
        for col, counts in col_counters.items():
            if total_counts[col] == 0:
                continue
            _, most_common_count = counts.most_common(1)[0]
            ratio = most_common_count / total_counts[col]
            for low, high, label in DOMINANCE_RANGES:
                if low <= ratio < high:
                    bucketed[label].append((col, counts, total_counts[col]))
                    break

        report_path = make_unique_path(
            os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(os.path.basename(file_path))[0]}_dominance_report.txt")
        )
        with open(report_path, "w", encoding="utf-8") as f:
            header_text = f"Dominance Report for {os.path.basename(file_path)}"
            f.write(header_text + "\n")
            f.write("=" * 60 + "\n\n")

            if label_counter:
                total_labels = sum(label_counter.values())
                label_header = "Global Label Distribution:\n" + "-" * 40
                f.write(label_header + "\n")
                print("\n" + label_header)
                for lbl, count in label_counter.most_common():
                    line_text = f"  {lbl}: {count:,} ({(count / total_labels) * 100:.2f}%)"
                    f.write(line_text + "\n")
                    print(line_text)
                f.write("\n")

            for label in bucketed:
                bucket_header = f"\nColumns in {label} range:\n" + "-" * 40
                f.write(bucket_header + "\n")
                print(bucket_header)

                if not bucketed[label]:
                    f.write("  None\n")
                    print("  None")
                else:
                    for col, counts, total in bucketed[label]:
                        col_header = f"\nColumn: {col}"
                        f.write(col_header + "\n")
                        print(col_header)

                        # Only show top 5 values for each column
                        for val, count in counts.most_common(5):
                            ratio = count / total
                            line_to_output = f"  Value '{val}': {count:,} ({ratio * 100:.2f}%)"
                            if val in col_value_label_counter.get(col, {}):
                                lbl_counts = col_value_label_counter[col][val]
                                breakdown = ", ".join(f"{lbl}: {c:,}" for lbl, c in lbl_counts.most_common())
                                line_to_output += f" -> Labels: [{breakdown}]"

                            f.write(line_to_output + "\n")
                            print(line_to_output)

        print(f"\nReport also saved to {report_path}")
        record_output(report_path, rows_processed=rows_processed)

    except Exception as e:
        print(f"Error during dominance report: {e}")


# ============================================================================== 
# TASK 2: DATA VALIDATION & CLEANING LOGIC
# ============================================================================== 

def _build_invalid_mask(chunk):
    invalid_mask = pd.Series(False, index=chunk.index)
    for col in chunk.columns:
        if any(kw in col.lower() for kw in CAN_BE_NEGATIVE_KEYWORDS):
            continue
        if any(kw in col.lower() for kw in NEVER_NEGATIVE_KEYWORDS):
            numeric_col = pd.to_numeric(chunk[col], errors='coerce')
            invalid_mask |= numeric_col < 0
    for col in PORT_COLUMNS:
        if col in chunk.columns:
            numeric_col = pd.to_numeric(chunk[col], errors='coerce')
            invalid_mask |= ~numeric_col.between(0, 65535)
    return invalid_mask


def run_data_validation(file_path):
    print(f"\nValidating and Cleaning: {os.path.basename(file_path)}")
    rows_processed = 0
    invalid_total = 0
    try:
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_ROWS, low_memory=False):
            rows_processed += len(chunk)
            if 'Label' in chunk.columns and 'label' not in chunk.columns:
                chunk = chunk.rename(columns={'Label': 'label'})
            invalid_mask = _build_invalid_mask(chunk)
            invalid_total += int(invalid_mask.sum())

        if invalid_total == 0:
            print("\nNo invalid rows to clean.")
            record_output(None, rows_processed=rows_processed)
            return

        print(f"\nFound {invalid_total} invalid rows.")
        if input("Remove invalid rows and save new file? (y/n): ").lower() == 'y':
            max_rows = prompt_for_max_rows()
            clean_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_validated.csv"
            output_path = make_unique_path(os.path.join(OUTPUT_FOLDER, clean_filename))

            is_first_chunk = True
            rows_written = 0
            for chunk in pd.read_csv(file_path, chunksize=CHUNK_ROWS, low_memory=False):
                if 'Label' in chunk.columns and 'label' not in chunk.columns:
                    chunk = chunk.rename(columns={'Label': 'label'})
                invalid_mask = _build_invalid_mask(chunk)
                cleaned = chunk.loc[~invalid_mask].copy()
                if max_rows is not None:
                    remaining = max_rows - rows_written
                    if remaining <= 0:
                        break
                    if len(cleaned) > remaining:
                        cleaned = cleaned.iloc[:remaining]
                if not cleaned.empty:
                    cleaned.to_csv(output_path, index=False, mode='w' if is_first_chunk else 'a', header=is_first_chunk)
                    is_first_chunk = False
                    rows_written += len(cleaned)
            print(f"Saved clean data to: {output_path}")
            record_output(output_path, rows_saved=rows_written, rows_processed=rows_processed)
        else:
            print("Skipping data cleaning.")
            record_output(None, rows_processed=rows_processed)
    except Exception as e:
        print(f"Error during data validation: {e}")


# ============================================================================== 
# TASK 3: 'INF' COLUMN REMOVAL LOGIC
# ============================================================================== 

def run_inf_column_removal(file_path):
    print(f"\n--- Processing file for 'inf' columns: {os.path.basename(file_path)} ---")
    print(f"Phase 1: Analyzing columns (Threshold: {INF_THRESHOLD:.0%})...")
    inf_counts = pd.Series(dtype=int)
    total_rows = 0
    try:
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_ROWS, low_memory=False):
            total_rows += len(chunk)
            inf_counts = inf_counts.add(chunk.apply(pd.to_numeric, errors='coerce').pipe(np.isinf).sum(), fill_value=0)
        if total_rows == 0:
            print("File is empty. Skipping.")
            return
        inf_percentages = inf_counts / total_rows
        columns_with_inf = inf_counts[inf_counts > 0].index.tolist()
    except Exception as e:
        print(f"Error during analysis: {e}")
        return

    if not columns_with_inf:
        print("No columns with 'inf' values found.")
        record_output(None, rows_processed=total_rows)
        return

    print(f"\nColumns with 'inf' values:")
    for idx, col in enumerate(columns_with_inf, 1):
        print(f"  {idx}: '{col}' ({inf_counts[col]} inf values, {inf_percentages[col]:.4%})")

    delete_input = input("Enter numbers of columns to DELETE (comma-separated, e.g. 1,3): ").strip()
    handle_input = input("Enter numbers of columns to HANDLE (replace inf with 0, comma-separated, e.g. 2,4): ").strip()
    to_delete = []
    to_handle = []
    if delete_input:
        to_delete = [columns_with_inf[int(i)-1] for i in delete_input.split(',') if i.strip().isdigit() and 1 <= int(i) <= len(columns_with_inf)]
    if handle_input:
        to_handle = [columns_with_inf[int(i)-1] for i in handle_input.split(',') if i.strip().isdigit() and 1 <= int(i) <= len(columns_with_inf)]

    print(f"\nColumns to delete: {to_delete}")
    print(f"Columns to handle (replace inf with 0): {to_handle}")

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"{base_name}_inf_handled.csv"
    output_csv_path = make_unique_path(os.path.join(OUTPUT_FOLDER, output_filename))
    max_rows = prompt_for_max_rows()

    try:
        is_first_chunk = True
        rows_written = 0
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_ROWS, low_memory=False):
            # Drop columns to delete
            chunk.drop(columns=to_delete, inplace=True, errors='ignore')
            # Replace inf with 0 in columns to handle
            for col in to_handle:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce').replace([np.inf, -np.inf], 0)
            if max_rows is not None:
                remaining = max_rows - rows_written
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining]
            if not chunk.empty:
                chunk.to_csv(output_csv_path, index=False, mode='w' if is_first_chunk else 'a', header=is_first_chunk)
                is_first_chunk = False
                rows_written += len(chunk)
        print(f"Successfully created '{output_filename}' with inf handling.")
        record_output(output_csv_path, rows_saved=rows_written, rows_processed=total_rows)
    except Exception as e:
        print(f"Error during file creation: {e}")



# ============================================================================== 
# NEW: REUSABLE DOMINANCE ANALYSIS HELPERS FOR DJANGO VIEWS
# ============================================================================== 

def analyze_dominance_for_web(file_path, dominance_label=None):
    col_counters = defaultdict(Counter)
    total_counts = Counter()
    label_counter = Counter()
    col_value_label_counter = defaultdict(lambda: defaultdict(Counter))

    for chunk in pd.read_csv(file_path, chunksize=CHUNK_ROWS, dtype=str, low_memory=False):
        labels = chunk.get("Label") or chunk.get("label")
        if labels is not None:
            label_counter.update(labels.dropna())
        for col in chunk.columns:
            values = chunk[col].dropna()
            col_counters[col].update(values)
            total_counts[col] += len(values)
            if labels is not None and col.lower() != "label":
                for v, lbl in zip(chunk[col], labels):
                    if pd.notna(v) and pd.notna(lbl):
                        col_value_label_counter[col][v][lbl] += 1

    bucketed = {label: [] for _, _, label in DOMINANCE_RANGES}
    for col, counts in col_counters.items():
        if total_counts[col] == 0:
            continue
        most_common_val, most_common_count = counts.most_common(1)[0]
        ratio = most_common_count / total_counts[col]
        for low, high, label in DOMINANCE_RANGES:
            if low <= ratio < high:
                bucketed[label].append((col, most_common_val, most_common_count, ratio, counts, total_counts[col]))
                break

    label_distribution = []
    if label_counter:
        total_labels = sum(label_counter.values())
        for lbl, count in label_counter.most_common():
            label_distribution.append({
                "label": str(lbl),
                "count": int(count),
                "percentage": float((count / total_labels) * 100.0),
            })

    columns = []
    wanted_labels = [dominance_label] if dominance_label else list(bucketed.keys())
    for bucket_label in wanted_labels:
        for col, dom_val, dom_count, ratio, counts, total in bucketed.get(bucket_label, []):
            lbl_breakdown_raw = col_value_label_counter.get(col, {}).get(dom_val, {})
            label_breakdown = [
                {"label": str(lbl), "count": int(c)}
                for lbl, c in lbl_breakdown_raw.most_common()
            ]
            columns.append({
                "column": col,
                "dominant_value": dom_val,
                "dominant_count": int(dom_count),
                "dominant_ratio": float(ratio),
                "total_count": int(total),
                "dominance_bucket": bucket_label,
                "label_breakdown": label_breakdown,
            })

    columns.sort(key=lambda c: c.get("dominant_ratio", 0.0), reverse=True)

    return {"label_distribution": label_distribution, "columns": columns}


def delete_columns_for_web(file_path, output_path, columns_to_delete):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cols_to_drop = list(dict.fromkeys(columns_to_delete or []))
    if not cols_to_drop:
        is_first_chunk = True
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_ROWS, low_memory=False):
            mode = "w" if is_first_chunk else "a"
            chunk.to_csv(output_path, index=False, mode=mode, header=is_first_chunk)
            is_first_chunk = False
        return {
            "output_path": output_path,
            "removed_columns": [],
        }

    is_first_chunk = True
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_ROWS, low_memory=False):
        chunk.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        mode = "w" if is_first_chunk else "a"
        chunk.to_csv(output_path, index=False, mode=mode, header=is_first_chunk)
        is_first_chunk = False

    return {
        "output_path": output_path,
        "removed_columns": cols_to_drop,
    }


# ============================================================================== 
# MAIN
# ============================================================================== 

def main(argv: list[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    gpu_available, _ = detect_gpu()
    device_choice = prompt_for_device(gpu_available)
    if device_choice == "gpu":
        print("GPU selected, but this script uses CPU-based pandas. Using CPU.")
    device_used = "cpu"

    input_folder = resolve_input_path(args.input)
    base_output_dir = resolve_output_path(args.output_dir)

    # Keep original subfolder name to avoid behavior change
    global OUTPUT_FOLDER
    OUTPUT_FOLDER = os.path.join(base_output_dir, "Training_isolation_model")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder not found at '{input_folder}'")
        return

    csv_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".csv")])
    if not csv_files:
        print("No CSV files found in the specified directory.")
        return

    chunk_mb = int(args.chunk_size_mb)
    global CHUNK_ROWS
    CHUNK_ROWS = estimate_rows_per_chunk(csv_files[0], chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{CHUNK_ROWS:,} rows per chunk)")

    if args.no_interactive:
        if args.task is None or args.files is None:
            print("ERROR: --no-interactive requires --task and --files (or --files=all).")
            return
        task_choice = args.task
        file_choice = args.files.strip().lower()
    else:
        print("\nPlease choose a task to perform on the files:")
        print("  1: Generate Dominance Report")
        print("  2: Validate Data and Remove Invalid Rows")
        print("  3: Handle Columns with High 'inf' Values")
        task_choice = input("Enter your choice (1, 2, or 3): ").strip()
        if task_choice not in ['1', '2', '3']:
            print("Invalid choice. Exiting.")
            return

        print("\n--- CSV Files Found ---")
        for i, file_path in enumerate(csv_files, 1):
            print(f"  {i}: {os.path.basename(file_path)}")
        print("-----------------------")
        file_choice = input("Enter the numbers of files to process (e.g., 1,3,5), or type 'all': ").strip().lower()

    if file_choice == 'all':
        files_to_process = csv_files
    else:
        try:
            indices = [int(num.strip()) - 1 for num in file_choice.split(',') if num.strip()]
            valid_indices = [i for i in indices if 0 <= i < len(csv_files)]
            if not valid_indices:
                print("Error: No valid file numbers were entered.")
                return
            files_to_process = [csv_files[i] for i in valid_indices]
        except ValueError:
            print("Invalid file list.")
            return

    # Wire CLI max-rows into existing prompts by short-circuiting prompt usage
    # (we keep interactive behavior by default)
    if args.no_interactive:
        def _no_prompt_max_rows():
            return int(args.max_output_rows) if args.max_output_rows is not None else None
        globals()['prompt_for_max_rows'] = _no_prompt_max_rows

    for file_path in files_to_process:
        if task_choice == '1':
            generate_dominance_report(file_path)
        elif task_choice == '2':
            run_data_validation(file_path)
        elif task_choice == '3':
            run_inf_column_removal(file_path)
        print("-" * 70)

    print("\nAll selected operations are complete.")

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

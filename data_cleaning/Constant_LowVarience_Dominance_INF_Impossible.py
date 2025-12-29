# What changed:
# - Added GPU detection/device prompt, chunk size prompt with row estimation, and streaming limits.
# - Standardized outputs under ./outputs/Normalized_Constant_Handled with non-overwrite paths.
# - Added optional max-rows limit for saved CSVs plus a final summary.
#
# Purpose:
# - Analyze dominance, validate data, handle inf values, and detect low-variance columns.
# - Optionally clean and save updated CSVs based on selected tasks.
# - Compare dominance across two files and optionally remove common dominant columns.

import os
import sys
import argparse

# Allow running this script from any working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import numpy as np
from collections import Counter, defaultdict

from config.global_config import DEFAULT_CHUNK_SIZE_MB, DEFAULT_MAX_OUTPUT_ROWS
from utils.chunk_utils import compute_chunk_plan, format_progress, print_chunk_plan
from utils.engine_utils import select_engine
from utils.path_utils import resolve_input_path, resolve_output_path

# --- 1. GLOBAL CONFIGURATION ---
INPUT_FOLDER = "Normalized_SET"

# Legacy output location (kept for compatibility if --output-dir is not given)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Normalized_Constant_Handled")

# --- Task 1 Config ---
DOMINANCE_RANGES = [
    (1.0, 1.01, "100%"),
    (0.95, 1.0, "95-100%"),
    (0.90, 0.95, "90-95%"),
    (0.80, 0.90, "80-90%"),
    (0.70, 0.80, "70-80%"),
    (0.60, 0.70, "60-70%"),
    (0.50, 0.60, "50-60%"),
]

# --- Task 2 Config ---
NEVER_NEGATIVE_KEYWORDS = [
    'port', 'duration', 'count', 'bytes', 'size', 'rate', 'percentage',
    'variance', 'std', 'total', 'max', 'min', 'median', 'mode', 'mean',
    'iat', 'active', 'idle', 'bulk', 'handshake', 'subflow'
]
CAN_BE_NEGATIVE_KEYWORDS = ['skew', 'cov', 'delta']
PORT_COLUMNS = ['src_port', 'dst_port']

# --- Task 3 Config ---
INF_THRESHOLD = 0.30  # 30% threshold for removing 'inf' columns

# Chunk size rows (set in main after prompt)
CHUNK_ROWS = 1_000_000

SUMMARY = {
    "total_rows_processed": 0,
    "rows_saved": 0,
    "output_paths": [],
}

# Global flag set by main() when --no-interactive is passed.
# Used by helper functions to skip prompts and use sensible defaults.
_NO_INTERACTIVE = False


# ============================================================================== 
# HELPERS
# ============================================================================== 

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


def get_user_yes_no(prompt, default=True):
    """Prompt user for yes/no. In non-interactive mode, returns `default`."""
    if _NO_INTERACTIVE:
        print(f"{prompt} (y/n): [auto: {'y' if default else 'n'}]")
        return default
    while True:
        response = input(f"{prompt} (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print("Invalid input. Please enter 'y' or 'n'.")


# ============================================================================== 
# TASK 1: STATIC DOMINANCE REPORT LOGIC
# ============================================================================== 

def generate_dominance_report(file_path):
    print(f"\n--- [Task 1] Generating Dominance Report for: {os.path.basename(file_path)} ---")
    col_counters = defaultdict(Counter)
    total_counts = Counter()
    label_counter = Counter()
    col_value_label_counter = defaultdict(lambda: defaultdict(Counter))
    rows_processed = 0

    try:
        for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=CHUNK_ROWS, dtype=str, low_memory=False), 1):
            rows_processed += len(chunk)
            if chunk_idx % 5 == 0:
                print(f"  Processed {rows_processed:,} rows...")

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
            if not counts:
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
            f.write(header_text + "\n" + "=" * 60 + "\n\n")
            print("\n" + header_text)
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

                        dominant_val, dominant_count = counts.most_common(1)[0]
                        dominant_ratio = dominant_count / total
                        dominant_line = f"  Value '{dominant_val}': {dominant_count:,} ({dominant_ratio * 100:.2f}%)"
                        if dominant_val in col_value_label_counter.get(col, {}):
                            lbl_counts = col_value_label_counter[col][dominant_val]
                            breakdown = ", ".join(f"{lbl}: {c:,}" for lbl, c in lbl_counts.most_common())
                            dominant_line += f" -> Labels: [{breakdown}]"
                        f.write(dominant_line + "\n")
                        print(dominant_line)

                        num_remaining_unique = len(counts) - 1
                        if num_remaining_unique > 0:
                            total_remaining_count = total - dominant_count
                            remaining_ratio = total_remaining_count / total
                            remaining_label_counts = Counter()
                            for val, count in counts.items():
                                if val != dominant_val:
                                    labels_for_this_val = col_value_label_counter[col].get(val, {})
                                    remaining_label_counts.update(labels_for_this_val)
                            remaining_breakdown = ", ".join(
                                f"{lbl}: {c:,}" for lbl, c in remaining_label_counts.most_common())
                            summary_line = (
                                f"  Remaining {remaining_ratio * 100:.2f}%: in {num_remaining_unique} other unique values"
                                f" -> Labels: [{remaining_breakdown}]"
                            )
                            f.write(summary_line + "\n")
                            print(summary_line)

        print(f"\nReport saved to: {report_path}")
        record_output(report_path, rows_processed=rows_processed)
    except Exception as e:
        print(f"ERROR during dominance report: {e}")


# ============================================================================== 
# TASK 2: DATA VALIDATION & CLEANING LOGIC (ROW REMOVAL)
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
    print(f"\n--- [Task 2] Validating and Cleaning: {os.path.basename(file_path)} ---")
    rows_processed = 0
    invalid_total = 0
    try:
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_ROWS, low_memory=False):
            rows_processed += len(chunk)
            if 'Label' in chunk.columns and 'label' not in chunk.columns:
                chunk = chunk.rename(columns={'Label': 'label'})
            invalid_mask = _build_invalid_mask(chunk)
            invalid_total += int(invalid_mask.sum())
        if rows_processed == 0:
            print("No rows found.")
            return

        print(f"\n[RESULT] Found {invalid_total:,} invalid rows based on the rules.")
        if invalid_total == 0:
            record_output(None, rows_processed=rows_processed)
            return

        if get_user_yes_no("Do you want to remove these invalid rows and save a new file?"):
            max_rows = prompt_for_max_rows()
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = make_unique_path(os.path.join(OUTPUT_FOLDER, f"{base_name}_validated.csv"))
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
                    cleaned.to_csv(output_path, index=False, mode="w" if is_first_chunk else "a", header=is_first_chunk)
                    is_first_chunk = False
                    rows_written += len(cleaned)
            print(f"  Successfully saved clean data ({rows_written:,} rows) to: {output_path}")
            record_output(output_path, rows_saved=rows_written, rows_processed=rows_processed)
        else:
            print("  Skipping data cleaning.")
            record_output(None, rows_processed=rows_processed)
    except Exception as e:
        print(f"ERROR during data validation: {e}")


# ============================================================================== 
# TASK 3: 'INF' COLUMN REMOVAL & IMPUTATION LOGIC
# ============================================================================== 

def run_inf_column_removal(file_path):
    print(f"\n--- [Task 3] Processing for 'inf' columns: {os.path.basename(file_path)} ---")
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
        columns_to_delete = inf_percentages[inf_percentages > INF_THRESHOLD].index.tolist()
    except Exception as e:
        print(f"ERROR during analysis: {e}")
        return

    if not columns_to_delete:
        print("\n[RESULT] No columns exceeded the 'inf' threshold.")
        if (inf_counts > 0).any():
            if get_user_yes_no("  Some 'inf' values were found below the threshold. Handle them with imputation?"):
                run_inf_imputation(file_path)
        record_output(None, rows_processed=total_rows)
        return

    print(f"\n[RESULT] Found {len(columns_to_delete)} columns to remove:")
    for col in columns_to_delete:
        print(f"  - '{col}' ({inf_percentages[col]:.2%} inf)")

    if not get_user_yes_no("\nDo you want to permanently delete these columns?"):
        print("Operation cancelled.")
        record_output(None, rows_processed=total_rows)
        return

    print("\nPhase 2: Deleting columns and creating new file...")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"{base_name}_inf_cleaned.csv"
    output_csv_path = make_unique_path(os.path.join(OUTPUT_FOLDER, output_filename))
    max_rows = prompt_for_max_rows()

    try:
        is_first_chunk = True
        rows_written = 0
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_ROWS, low_memory=False):
            chunk.drop(columns=columns_to_delete, inplace=True, errors='ignore')
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
        print(f"  Successfully created '{output_filename}'")
        record_output(output_csv_path, rows_saved=rows_written, rows_processed=total_rows)

        print("\n--- Next Steps for the Cleaned File ---")
        print("  1: Re-analyze the cleaned file for remaining 'inf' values")
        print("  2: Handle remaining 'inf' values with median imputation")
        print("  3: Do nothing / Continue")
        choice = input("Enter your choice (1, 2, or 3): ")
        if choice == '1':
            report_remaining_inf(output_csv_path)
        elif choice == '2':
            run_inf_imputation(output_csv_path)
        else:
            print("Continuing.")
    except Exception as e:
        print(f"ERROR during file creation: {e}")


def report_remaining_inf(file_path):
    print(f"\n--- Re-analyzing for remaining 'inf' in {os.path.basename(file_path)} ---")
    inf_counts = pd.Series(dtype=int)
    total_rows = 0
    try:
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_ROWS, low_memory=False):
            total_rows += len(chunk)
            inf_counts = inf_counts.add(chunk.apply(pd.to_numeric, errors='coerce').pipe(np.isinf).sum(), fill_value=0)
        if total_rows == 0:
            return
        inf_percentages = inf_counts / total_rows
        remaining_inf_cols = inf_percentages[inf_percentages > 0].index.tolist()
        if not remaining_inf_cols:
            print("[RESULT] No remaining 'inf' values found.")
        else:
            print("[RESULT] Found remaining 'inf' values in the following columns:")
            for col in remaining_inf_cols:
                print(f"  - '{col}': {inf_counts[col]} values ({inf_percentages[col]:.4f}%)")
        record_output(None, rows_processed=total_rows)
    except Exception as e:
        print(f"ERROR during re-analysis: {e}")


def run_inf_imputation(file_path):
    print(f"\n--- Imputing 'inf' values in {os.path.basename(file_path)} ---")
    medians = {}
    try:
        print("Phase 1: Calculating medians for columns with 'inf' values...")
        inf_counts = pd.Series(dtype=int)
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_ROWS, low_memory=False):
            inf_counts = inf_counts.add(chunk.apply(pd.to_numeric, errors='coerce').pipe(np.isinf).sum(), fill_value=0)
        cols_to_process = inf_counts[inf_counts > 0].index.tolist()
        if not cols_to_process:
            print("No 'inf' values found to impute.")
            return
        for col in cols_to_process:
            series = pd.read_csv(file_path, usecols=[col], low_memory=False).squeeze("columns")
            median_val = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).median()
            medians[col] = median_val
            print(f"  - Column '{col}': Median is {median_val}")

        print("\nPhase 2: Replacing 'inf' values and saving new file...")
        base_name = os.path.splitext(os.path.basename(file_path))[0].replace('_inf_cleaned', '')
        output_filename = f"{base_name}_imputed.csv"
        output_csv_path = make_unique_path(os.path.join(OUTPUT_FOLDER, output_filename))
        max_rows = prompt_for_max_rows()

        is_first_chunk = True
        rows_written = 0
        rows_processed = 0
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_ROWS, low_memory=False):
            rows_processed += len(chunk)
            for col, median_val in medians.items():
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce').replace([np.inf, -np.inf], median_val)
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
        print(f"  Successfully created '{output_filename}'")
        record_output(output_csv_path, rows_saved=rows_written, rows_processed=rows_processed)
    except Exception as e:
        print(f"ERROR during imputation: {e}")


# ============================================================================== 
# TASK 4: REMOVE CONSTANT OR LOW-VARIANCE COLUMNS
# ============================================================================== 

# Memory-safe unique tracking: store up to a small cap and mark columns as "too many uniques".
class _CappedUniques:
    __slots__ = ("cap", "values", "overflow")

    def __init__(self, cap: int):
        self.cap = int(cap)
        self.values: set[str] = set()
        self.overflow = False

    def add_many(self, items) -> None:
        if self.overflow:
            return
        for x in items:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                continue
            self.values.add(str(x))
            if len(self.values) > self.cap:
                self.overflow = True
                # Free memory aggressively once we've exceeded the cap.
                self.values.clear()
                return

    def count(self) -> int:
        return self.cap + 1 if self.overflow else len(self.values)


def run_variance_analysis(file_path, *, chunk_mb: int | None = None):
    print(f"\n--- [Task 4] Analyzing for Low-Variance Columns: {os.path.basename(file_path)} ---")
    rows_processed = 0
    try:
        print("  Analyzing columns... (memory-safe; stops tracking uniques once a cap is exceeded)")

        # In non-interactive mode, default to checking both constant and low-variance columns.
        want_constant = get_user_yes_no("  Do you want to find constant columns (1 unique value)?", default=True)
        want_low_var = get_user_yes_no("  Do you want to find low-variance columns (2+ unique values)?", default=True)
        if not want_constant and not want_low_var:
            print("  No variance checks selected.")
            record_output(None, rows_processed=0)
            return

        threshold = None
        if want_low_var:
            if _NO_INTERACTIVE:
                # Default threshold for non-interactive mode.
                threshold = 3
                print(f"    Using default threshold: {threshold} unique values")
            else:
                while True:
                    try:
                        threshold = int(input("    Enter the maximum number of unique values (e.g., 3): "))
                        if threshold < 2:
                            print("    Please enter an integer >= 2.")
                            continue
                        break
                    except ValueError:
                        print("    That wasn't a valid number. Please enter an integer.")

        cap = int(threshold) if threshold is not None else 1
        trackers: dict[str, _CappedUniques] = {}

        # Mandatory chunk plan & progress (use actual chunk_mb if available).
        file_plan = None
        if chunk_mb is not None:
            try:
                file_plan = compute_chunk_plan(file_path, int(chunk_mb))
                print_chunk_plan(file_plan)
            except Exception:
                file_plan = None

        for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=CHUNK_ROWS, dtype=str, low_memory=False), 1):
            rows_processed += len(chunk)

            if file_plan is not None:
                print(format_progress(chunk_idx, file_plan.total_chunks))
            elif chunk_idx % 5 == 0:
                print(f"  Processed {rows_processed:,} rows...")

            for col in chunk.columns:
                tr = trackers.get(col)
                if tr is None:
                    tr = trackers[col] = _CappedUniques(cap=cap)
                # Drop NAs and add. This stays bounded.
                tr.add_many(chunk[col].dropna().tolist())

        print("  Analysis complete.")

        columns_to_drop = []

        if want_constant:
            constant_cols = {col: next(iter(tr.values)) for col, tr in trackers.items() if tr.count() == 1 and not tr.overflow}
            if constant_cols:
                print("\n  [RESULT] Found Constant Columns:")
                for col, val in constant_cols.items():
                    print(f"    - {col}: (value is '{val}')")
                columns_to_drop.extend(constant_cols.keys())
            else:
                print("\n  [RESULT] No constant columns were found.")

        if want_low_var and threshold is not None:
            low_variance_cols = {
                col: sorted(list(tr.values))
                for col, tr in trackers.items()
                if (not tr.overflow) and 2 <= tr.count() <= threshold
            }
            if low_variance_cols:
                print(f"\n  [RESULT] Found Low-Variance Columns (up to {threshold} unique values):")
                for col, vals in low_variance_cols.items():
                    print(f"    - {col}: (values are {vals})")
                new_cols_to_add = [col for col in low_variance_cols if col not in columns_to_drop]
                columns_to_drop.extend(new_cols_to_add)
            else:
                print(f"\n  [RESULT] No low-variance columns found with the specified threshold.")

        if not columns_to_drop:
            print("\nNo columns were selected for removal. Moving to the next file.")
            record_output(None, rows_processed=rows_processed)
            return

        final_drop_list = sorted(list(set(columns_to_drop)))
        print("\nColumns identified for removal:", final_drop_list)
        if get_user_yes_no("Do you want to remove these columns and save a new, cleaned file?"):
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = make_unique_path(os.path.join(OUTPUT_FOLDER, f"{base_name}_variance_cleaned.csv"))
            max_rows = prompt_for_max_rows()
            print(f"  Removing {len(final_drop_list)} columns and saving new file...")
            chunk_iterator = pd.read_csv(file_path, chunksize=CHUNK_ROWS, low_memory=False)
            rows_written = 0
            is_first_chunk = True
            for chunk in chunk_iterator:
                chunk.drop(columns=final_drop_list, errors="ignore", inplace=True)
                if max_rows is not None:
                    remaining = max_rows - rows_written
                    if remaining <= 0:
                        break
                    if len(chunk) > remaining:
                        chunk = chunk.iloc[:remaining]
                if not chunk.empty:
                    chunk.to_csv(output_path, index=False, mode='w' if is_first_chunk else 'a', header=is_first_chunk)
                    is_first_chunk = False
                    rows_written += len(chunk)
            print(f"  Successfully saved cleaned file to: {output_path}")
            record_output(output_path, rows_saved=rows_written, rows_processed=rows_processed)
        else:
            print("  Skipping file modification as requested.")
            record_output(None, rows_processed=rows_processed)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")


# ============================================================================== 
# TASK 5: INTERACTIVE DOMINANCE COMPARISON
# ============================================================================== 

def _analyze_file_for_dominance(file_path):
    print(f"  Analyzing file: {os.path.basename(file_path)}...")
    col_counters = defaultdict(Counter)
    total_counts = Counter()
    rows_processed = 0
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_ROWS, dtype=str, low_memory=False):
        rows_processed += len(chunk)
        for col in chunk.columns:
            values = chunk[col].dropna()
            col_counters[col].update(values)
            total_counts[col] += len(values)
    print(f"  Analysis of {os.path.basename(file_path)} complete.")
    return col_counters, total_counts, rows_processed


def _get_dominant_columns(col_counters, total_counts, min_r, max_r):
    dominant_cols = {}
    for col, counts in col_counters.items():
        if total_counts[col] > 0:
            _, most_common_count = counts.most_common(1)[0]
            ratio = most_common_count / total_counts[col]
            if min_r <= ratio <= max_r:
                dominant_cols[col] = ratio
    return dominant_cols


def _delete_columns_from_file(input_path, output_path, cols_to_delete, max_rows=None):
    print(f"  Removing {len(cols_to_delete)} columns from {os.path.basename(input_path)}...")
    is_first_chunk = True
    rows_written = 0
    for chunk in pd.read_csv(input_path, chunksize=CHUNK_ROWS, low_memory=False):
        chunk.drop(columns=cols_to_delete, inplace=True, errors='ignore')
        if max_rows is not None:
            remaining = max_rows - rows_written
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
        if not chunk.empty:
            chunk.to_csv(output_path, index=False, mode='w' if is_first_chunk else 'a', header=is_first_chunk)
            is_first_chunk = False
            rows_written += len(chunk)
    print(f"  Saved new file to: {output_path}")
    return rows_written


def run_interactive_dominance_comparison(file1_path, file2_path):
    print(f"\n--- [Task 5] Comparing Dominance for '{os.path.basename(file1_path)}' and '{os.path.basename(file2_path)}' ---")
    try:
        stats1 = _analyze_file_for_dominance(file1_path)
        stats2 = _analyze_file_for_dominance(file2_path)
        rows_processed = stats1[2] + stats2[2]
        while True:
            try:
                min_perc = float(input("\nEnter the MINIMUM dominance percentage (e.g., 99): "))
                max_perc = float(input("Enter the MAXIMUM dominance percentage (e.g., 100): "))
                if not (0 <= min_perc <= 100 and 0 <= max_perc <= 100 and min_perc <= max_perc):
                    print("Error: Please enter valid percentages between 0 and 100.")
                    continue
            except ValueError:
                print("Error: Invalid input. Please enter numbers.")
                continue

            dominant1 = _get_dominant_columns(stats1[0], stats1[1], min_perc / 100, max_perc / 100)
            dominant2 = _get_dominant_columns(stats2[0], stats2[1], min_perc / 100, max_perc / 100)
            common_cols = set(dominant1.keys()).intersection(set(dominant2.keys()))
            unique_to_1 = set(dominant1.keys()) - common_cols
            unique_to_2 = set(dominant2.keys()) - common_cols

            if not common_cols and not unique_to_1 and not unique_to_2:
                print(f"\n[RESULT] No columns found in the {min_perc}%-{max_perc}% range for either file.")
                if not get_user_yes_no("Try a different percentage range?"):
                    break
                continue

            print(f"\n--- Comparison Results ({min_perc}% - {max_perc}%) ---")
            if common_cols:
                print(f"\n[COMMON] {len(common_cols)} columns are dominant in BOTH files (candidates for removal):")
                for col in sorted(list(common_cols)):
                    print(f"  - '{col}' (Dominance: {dominant1[col]:.2%} in file 1, {dominant2[col]:.2%} in file 2)")
            if unique_to_1:
                print(f"\n[UNIQUE] {len(unique_to_1)} columns are dominant ONLY in {os.path.basename(file1_path)}:")
                for col in sorted(list(unique_to_1)):
                    print(f"  - '{col}' ({dominant1[col]:.2%})")
            if unique_to_2:
                print(f"\n[UNIQUE] {len(unique_to_2)} columns are dominant ONLY in {os.path.basename(file2_path)}:")
                for col in sorted(list(unique_to_2)):
                    print(f"  - '{col}' ({dominant2[col]:.2%})")

            print("\n--- Action Menu ---")
            if not common_cols:
                print("No common columns to delete.")
                if not get_user_yes_no("Do you want to re-analyze with a different range?"):
                    break
                continue

            action = input("Enter 'd' to delete the COMMON columns, 'r' to re-analyze, or 'n' to quit: ").lower().strip()
            if action == 'd':
                cols_to_delete = sorted(list(common_cols))
                if get_user_yes_no(f"Are you sure you want to remove these {len(cols_to_delete)} columns from BOTH files?"):
                    max_rows = prompt_for_max_rows()
                    base1 = os.path.splitext(os.path.basename(file1_path))[0]
                    base2 = os.path.splitext(os.path.basename(file2_path))[0]
                    output1_path = make_unique_path(os.path.join(OUTPUT_FOLDER, f"{base1}_cleaned.csv"))
                    output2_path = make_unique_path(os.path.join(OUTPUT_FOLDER, f"{base2}_cleaned.csv"))
                    rows_written_1 = _delete_columns_from_file(file1_path, output1_path, cols_to_delete, max_rows=max_rows)
                    rows_written_2 = _delete_columns_from_file(file2_path, output2_path, cols_to_delete, max_rows=max_rows)
                    record_output(output1_path, rows_saved=rows_written_1)
                    record_output(output2_path, rows_saved=rows_written_2)
                    print("\nDeletion complete for both files.")
                else:
                    print("Deletion cancelled.")
                break
            elif action == 'r':
                print("\nRestarting analysis...")
                continue
            else:
                print("Exiting analysis.")
                break
        record_output(None, rows_processed=rows_processed)
    except Exception as e:
        print(f"ERROR during dominance comparison: {e}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Analyze dominance, validate data, handle inf values, and detect constant/low-variance columns. "
            "Designed for large CSVs (streaming)."
        )
    )
    p.add_argument("--input", default=INPUT_FOLDER, help="Input folder (abs or repo-root-relative)")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Base output directory (abs or repo-root-relative). Defaults to ./outputs",
    )
    p.add_argument("--chunk-size-mb", type=int, default=DEFAULT_CHUNK_SIZE_MB, help="Chunk size in MB")
    p.add_argument("--max-output-rows", type=int, default=DEFAULT_MAX_OUTPUT_ROWS, help="Max rows to write")

    p.add_argument("--engine", default="pandas", choices=["pandas", "dask", "dask-gpu"], help="Execution engine")
    p.add_argument("--use-gpu", action="store_true", help="Force GPU (or fail)")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU")

    p.add_argument("--task", choices=["1", "2", "3", "4", "5"], default=None, help="Task to run")
    p.add_argument(
        "--files",
        default=None,
        help="Comma-separated indices (1-based) or 'all'. Used for tasks 1-4.",
    )
    p.add_argument("--no-interactive", action="store_true", help="Disable interactive prompts")
    return p


# ==============================================================================
# MAIN DRIVER
# ============================================================================== 

def main(argv: list[str] | None = None):
    args = build_arg_parser().parse_args(argv)

    # Engine selection (forward-compatible). This script currently performs pandas streaming.
    selection = select_engine(engine=args.engine, use_gpu_flag=args.use_gpu, no_gpu_flag=args.no_gpu)
    if selection.engine != "pandas":
        print(f"[info] --engine {selection.engine} requested; this script currently runs in pandas mode for safety.")
    if selection.use_gpu:
        print("[info] GPU was approved, but this script uses CPU-based pandas. Using CPU.")
    device_used = "cpu"

    # Resolve paths safely.
    input_folder = resolve_input_path(args.input)
    base_output_dir = resolve_output_path(args.output_dir)

    global OUTPUT_FOLDER
    OUTPUT_FOLDER = os.path.join(base_output_dir, "Normalized_Constant_Handled")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("--- Data Analysis and Validation Tool ---")
    print(f"Searching for CSVs in: '{input_folder}'")
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

    # Mandatory chunk plan pre-calc.
    plan0 = compute_chunk_plan(csv_files[0], chunk_mb)
    print_chunk_plan(plan0)

    print(f"Using chunk size: {chunk_mb}MB (~{CHUNK_ROWS:,} rows per chunk)")

    # Set global flag so helper functions skip prompts.
    global _NO_INTERACTIVE
    _NO_INTERACTIVE = args.no_interactive

    # Keep interactive behavior unless --no-interactive.
    if args.no_interactive:
        if args.task is None:
            print("ERROR: --no-interactive requires --task")
            return
        task_choice = args.task
        file_choice = (args.files or "").strip().lower() if task_choice != "5" else None
        if task_choice != "5" and not file_choice:
            print("ERROR: --no-interactive for tasks 1-4 requires --files or --files=all")
            return
    else:
        # legacy GPU prompt
        gpu_available, _ = detect_gpu()
        device_choice = prompt_for_device(gpu_available)
        if device_choice == "gpu":
            print("GPU selected, but this script uses CPU-based pandas. Using CPU.")

        print("\nPlease choose a task to perform on the files:")
        print("  1: Generate Static Dominance Report (.txt file)")
        print("  2: Validate Data and Remove Invalid Rows")
        print("  3: Handle Columns with High 'inf' Values")
        print("  4: Remove Constant or Low-Variance Columns")
        print("  5: Interactive Dominance COMPARISON (select 2 files)")
        task_choice = input("Enter your choice (1, 2, 3, 4, or 5): ").strip()
        if task_choice not in ['1', '2', '3', '4', '5']:
            print("Invalid choice. Exiting.")
            return

        print("\n--- CSV Files Found ---")
        for i, file_path in enumerate(csv_files, 1):
            print(f"  {i}: {os.path.basename(file_path)}")
        print("-----------------------")

        if task_choice == '5':
            file_choice = None
        else:
            file_choice = input("Enter the numbers of files to process (e.g., 1,3,5), or type 'all': ").strip().lower()

    # Wire CLI max-rows into existing prompts for non-interactive runs.
    if args.no_interactive:
        def _no_prompt_max_rows():
            return int(args.max_output_rows) if args.max_output_rows is not None else None
        globals()['prompt_for_max_rows'] = _no_prompt_max_rows

    if task_choice == '5':
        if len(csv_files) < 2:
            print("Error: Task 5 requires at least two CSV files in the input folder.")
            return
        if args.no_interactive:
            print("ERROR: Task 5 is interactive only.")
            return
        try:
            idx1 = int(input("Enter the number of the FIRST file (e.g., your benign set): ")) - 1
            idx2 = int(input("Enter the number of the SECOND file (e.g., your attack set): ")) - 1
            if not (0 <= idx1 < len(csv_files) and 0 <= idx2 < len(csv_files) and idx1 != idx2):
                print("Error: Invalid selection. Please select two different, valid file numbers.")
                return
            run_interactive_dominance_comparison(csv_files[idx1], csv_files[idx2])
        except (ValueError, IndexError):
            print("Invalid input. Please enter valid numbers.")
    else:
        if file_choice == 'all':
            files_to_process = csv_files
        else:
            indices = [int(num.strip()) - 1 for num in file_choice.split(',') if num.strip()]
            valid_indices = [i for i in indices if 0 <= i < len(csv_files)]
            if not valid_indices:
                print("Error: No valid file numbers were entered.")
                return
            files_to_process = [csv_files[i] for i in valid_indices]

        print(f"\nBeginning processing for {len(files_to_process)} selected file(s)...")
        for file_path in files_to_process:
            if task_choice == '1':
                generate_dominance_report(file_path)
            elif task_choice == '2':
                run_data_validation(file_path)
            elif task_choice == '3':
                run_inf_column_removal(file_path)
            elif task_choice == '4':
                run_variance_analysis(file_path, chunk_mb=chunk_mb)
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

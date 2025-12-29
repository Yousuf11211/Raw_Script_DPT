# What changed:
# - Added GPU detection/device prompt and chunk size prompt with row estimation.
# - Streamed stratified split with chunked writes and optional max-rows limit.
# - Standardized outputs under ./outputs/Label_Detection_and_Splitting with final summary.
#
# Purpose:
# - Count labels in CSV files.
# - Split data into train/test with approximate stratification.
# - Save per-file reports plus train/test CSVs.

import os

import sys
import argparse
import pandas as pd
from collections import Counter, defaultdict

# Allow running this script from any working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config.global_config import DEFAULT_CHUNK_SIZE_MB, DEFAULT_MAX_OUTPUT_ROWS
from utils.chunk_utils import compute_chunk_plan, format_progress, print_chunk_plan
from utils.engine_utils import select_engine
from utils.path_utils import resolve_input_path, resolve_output_path

PARENT_FOLDER = "Attacks_Cleaned"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
REPORTS_FOLDER = os.path.join(OUTPUT_ROOT, "Label_Detection_and_Splitting", "reports")
TRAIN_FOLDER = os.path.join(OUTPUT_ROOT, "Label_Detection_and_Splitting", "train")
TEST_FOLDER = os.path.join(OUTPUT_ROOT, "Label_Detection_and_Splitting", "test")

TRAIN_CSV_NAME = "training.csv"
TEST_CSV_NAME = "test.csv"
LABEL_COLUMN = "label"
OUTPUT_ENCODING = "utf-8"

CHUNK_ROWS = 1_500_000

os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

_NO_INTERACTIVE = False


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


def safe_lower(x):
    try:
        return str(x).strip().lower()
    except Exception:
        return str(x)


def count_labels_first(file_path):
    label_counts = Counter()
    total = 0
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_ROWS, low_memory=True):
        label_col = next((c for c in chunk.columns if c.lower() == LABEL_COLUMN.lower()), None)
        if not label_col:
            print(f"No '{LABEL_COLUMN}' column found in {file_path}. Skipping.")
            return Counter(), 0
        s = chunk[label_col].dropna()
        label_counts.update(s)
        total += len(chunk)
    return label_counts, total


def plan_stratified_split(counts, train_ratio=0.6):
    train_needed = {}
    test_needed = {}
    for label, cnt in counts.items():
        t = int(round(cnt * train_ratio))
        t = max(0, min(cnt, t))
        train_needed[label] = t
        test_needed[label] = cnt - t
    return train_needed, test_needed


def build_report(file_path, total_samples, label_counts, train_counts, test_counts):
    benign_key_variants = {k for k in label_counts.keys() if safe_lower(k) == "benign"}
    benign_total = sum(label_counts[k] for k in benign_key_variants) if benign_key_variants else 0
    attack_total = total_samples - benign_total
    lines = []
    lines.append(f"Report for {file_path}")
    lines.append("=" * 60)
    lines.append(f"Total samples: {total_samples}")
    lines.append(f"Benign: {benign_total}")
    lines.append(f"Attacks: {attack_total}")
    lines.append("")
    lines.append("Breakdown by label (full dataset):")
    lines.append("-" * 40)
    for label, cnt in label_counts.items():
        lines.append(f"{label:<25}: {cnt}")
    lines.append("")
    lines.append("Training split label counts (60% target):")
    lines.append("-" * 40)
    for label, cnt in sorted(train_counts.items(), key=lambda x: str(x[0]).lower()):
        lines.append(f"{label:<25}: {cnt}")
    lines.append("")
    lines.append("Test split label counts (40% target):")
    lines.append("-" * 40)
    for label, cnt in sorted(test_counts.items(), key=lambda x: str(x[0]).lower()):
        lines.append(f"{label:<25}: {cnt}")
    return "\n".join(lines)


def write_report_text(report_text, file_path, reports_folder):
    rel = os.path.relpath(file_path, PARENT_FOLDER)
    rel_name = rel.replace(os.sep, "_")
    out_path = make_unique_path(os.path.join(reports_folder, f"{os.path.splitext(rel_name)[0]}.txt"))
    with open(out_path, "w", encoding=OUTPUT_ENCODING) as f:
        f.write(report_text)
    print(f"Saved report to {out_path}")
    return out_path


def split_and_write(file_path, *, train_folder, test_folder, reports_folder, max_rows=None, chunk_mb: int | None = None):
    if chunk_mb is not None:
        file_plan = compute_chunk_plan(file_path, chunk_mb)
        print_chunk_plan(file_plan)
    else:
        file_plan = None

    label_counts, total_rows = count_labels_first(file_path)

    if total_rows == 0:
        print(f"No rows found in {file_path}, skipping.")
        return 0, 0, None, None, None

    train_needed, test_needed = plan_stratified_split(label_counts, train_ratio=0.6)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    train_path = make_unique_path(os.path.join(train_folder, f"{base_name}_{TRAIN_CSV_NAME}"))
    test_path = make_unique_path(os.path.join(test_folder, f"{base_name}_{TEST_CSV_NAME}"))

    written_train = defaultdict(int)
    written_test = defaultdict(int)

    train_header_written = False
    test_header_written = False

    for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=CHUNK_ROWS, low_memory=True), 1):
        if file_plan is not None:
            print(format_progress(chunk_idx, file_plan.total_chunks))
        label_col = next((c for c in chunk.columns if c.lower() == LABEL_COLUMN.lower()), LABEL_COLUMN)
        if label_col not in chunk.columns:
            print(f"'{LABEL_COLUMN}' column not found in {file_path}, skipping this file.")
            return 0, 0, None, None, None

        label_series = chunk[label_col]
        train_idx = []
        test_idx = []

        for label, idxs in label_series.groupby(label_series).groups.items():
            idxs = list(idxs)
            if label not in train_needed:
                train_needed[label] = 0
                test_needed[label] = 0

            available = len(idxs)
            t_take = max(0, min(train_needed[label] - written_train[label], available))
            train_sel = idxs[:t_take]
            remaining = idxs[t_take:]
            te_take = max(0, min(test_needed[label] - written_test[label], len(remaining)))
            test_sel = remaining[:te_take]
            leftover = remaining[te_take:]

            train_idx.extend(train_sel)
            test_idx.extend(test_sel)
            written_train[label] += len(train_sel)
            written_test[label] += len(test_sel)

            for idx in leftover:
                if written_train[label] <= written_test[label]:
                    train_idx.append(idx)
                    written_train[label] += 1
                else:
                    test_idx.append(idx)
                    written_test[label] += 1

        if train_idx:
            train_df = chunk.loc[train_idx]
            if max_rows is not None:
                remaining = max_rows - sum(written_train.values())
                if remaining <= 0:
                    train_df = train_df.iloc[:0]
                elif len(train_df) > remaining:
                    train_df = train_df.iloc[:remaining]
            if not train_df.empty:
                train_df.to_csv(train_path, mode="w" if not train_header_written else "a", index=False, header=not train_header_written)
                train_header_written = True

        if test_idx:
            test_df = chunk.loc[test_idx]
            if max_rows is not None:
                remaining = max_rows - sum(written_test.values())
                if remaining <= 0:
                    test_df = test_df.iloc[:0]
                elif len(test_df) > remaining:
                    test_df = test_df.iloc[:remaining]
            if not test_df.empty:
                test_df.to_csv(test_path, mode="w" if not test_header_written else "a", index=False, header=not test_header_written)
                test_header_written = True

    report_text = build_report(
        file_path=file_path,
        total_samples=total_rows,
        label_counts=label_counts,
        train_counts=dict(written_train),
        test_counts=dict(written_test)
    )
    report_path = write_report_text(report_text, file_path, reports_folder)

    print(f"Done: {file_path}")
    print("Train counts per label:", dict(written_train))
    print("Test counts per label:", dict(written_test))

    return sum(written_train.values()), sum(written_test.values()), train_path, test_path, report_path


def label_report(file_path, *, reports_folder, chunk_mb: int | None = None):
    if chunk_mb is not None:
        file_plan = compute_chunk_plan(file_path, chunk_mb)
        print_chunk_plan(file_plan)
    label_counts, total_rows = count_labels_first(file_path)
    if total_rows == 0:
        print(f"No rows found in {file_path}, skipping.")
        return None
    print(f"\nLabel count for {file_path}:")
    for label, cnt in label_counts.items():
        print(f"  {label}: {cnt}")
    print(f"Total samples: {total_rows}")

    if _NO_INTERACTIVE:
        save = False
        print("Do you want to save this label report? [auto: n]")
    else:
        save = input("Do you want to save this label report? (y/n): ").strip().lower() == 'y'

    if save:
        report_text = f"Label count report for {file_path}\n\nTotal samples: {total_rows}\n"
        for label, cnt in label_counts.items():
            report_text += f"{label}: {cnt}\n"
        return write_report_text(report_text, file_path, reports_folder)

    print("Label report not saved.")
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Count labels and split CSVs into train/test (streaming-safe).")
    p.add_argument("--input", default=PARENT_FOLDER, help="Input folder with CSVs")
    p.add_argument("--output-dir", default=None, help="Base output directory")
    p.add_argument("--chunk-size-mb", type=int, default=DEFAULT_CHUNK_SIZE_MB, help="Chunk size in MB")
    p.add_argument("--max-output-rows", type=int, default=DEFAULT_MAX_OUTPUT_ROWS, help="Max rows to write per split")
    p.add_argument("--engine", default="pandas", choices=["pandas", "dask", "dask-gpu"], help="Execution engine")
    p.add_argument("--use-gpu", action="store_true", help="Force GPU (or fail)")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU")
    p.add_argument("--no-interactive", action="store_true", help="Disable interactive prompts")
    p.add_argument("--label-only", action="store_true", help="Only count labels (no split)")
    p.add_argument("--do-split", action="store_true", help="Force split (non-interactive)")
    return p


def main(argv: list[str] | None = None):
    global _NO_INTERACTIVE, CHUNK_ROWS, PARENT_FOLDER
    args = build_arg_parser().parse_args(argv)
    _NO_INTERACTIVE = args.no_interactive

    selection = select_engine(engine=args.engine, use_gpu_flag=args.use_gpu, no_gpu_flag=args.no_gpu)
    if selection.engine != "pandas":
        print(f"[info] --engine {selection.engine} requested; this script currently runs in pandas mode.")
    if selection.use_gpu:
        print("[info] GPU was approved, but this script uses CPU-based pandas. Using CPU.")
    device_used = "cpu"

    PARENT_FOLDER = resolve_input_path(args.input)
    base_output_dir = resolve_output_path(args.output_dir)

    reports_folder = os.path.join(base_output_dir, "Label_Detection_and_Splitting", "reports")
    train_folder = os.path.join(base_output_dir, "Label_Detection_and_Splitting", "train")
    test_folder = os.path.join(base_output_dir, "Label_Detection_and_Splitting", "test")

    os.makedirs(reports_folder, exist_ok=True)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    if not os.path.isdir(PARENT_FOLDER):
        print(f"ERROR: Input folder not found: {PARENT_FOLDER}")
        return

    csv_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(PARENT_FOLDER)
        for file in files
        if file.endswith(".csv")
    ]
    if not csv_files:
        print("No CSV files found.")
        return

    chunk_mb = int(args.chunk_size_mb)
    plan0 = compute_chunk_plan(csv_files[0], chunk_mb)
    print_chunk_plan(plan0)

    CHUNK_ROWS = estimate_rows_per_chunk(csv_files[0], chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{CHUNK_ROWS:,} rows per chunk)")

    max_rows = int(args.max_output_rows) if args.max_output_rows is not None else None

    total_rows_processed = 0
    total_rows_saved = 0
    output_paths: list[str] = []

    for file_path in csv_files:
        print(f"\nProcessing file: {file_path}")

        if args.label_only:
            report_path = label_report(file_path, reports_folder=reports_folder, chunk_mb=chunk_mb)
            if report_path:
                output_paths.append(report_path)
            # Still count rows processed for summary.
            _, rows = count_labels_first(file_path)
            total_rows_processed += rows
            continue

        if _NO_INTERACTIVE:
            do_split = True
        else:
            do_label_only = input(
                "Label test only (shows and/or saves label counts, no split)? (y/n): "
            ).strip().lower() == 'y'
            if do_label_only:
                report_path = label_report(file_path, reports_folder=reports_folder, chunk_mb=chunk_mb)
                if report_path:
                    output_paths.append(report_path)
                _, rows = count_labels_first(file_path)
                total_rows_processed += rows
                continue
            do_split = input("Do you want to perform train-test split on this file? (y/n): ").strip().lower() == 'y'

        if do_split:
            train_rows, test_rows, train_path, test_path, report_path = split_and_write(
                file_path,
                train_folder=train_folder,
                test_folder=test_folder,
                reports_folder=reports_folder,
                max_rows=max_rows,
                chunk_mb=chunk_mb,
            )
            total_rows_saved += train_rows + test_rows
            output_paths.extend([p for p in [train_path, test_path, report_path] if p])
        else:
            print("Skipping train-test split for this file.")

        _, rows = count_labels_first(file_path)
        total_rows_processed += rows

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Device used: {device_used.upper()}")
    print(f"Chunk size: {chunk_mb}MB (~{CHUNK_ROWS:,} rows)")
    print(f"Total rows processed: {total_rows_processed:,}")
    print(f"Rows saved: {total_rows_saved:,}")
    print("Output paths:")
    for path in output_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()


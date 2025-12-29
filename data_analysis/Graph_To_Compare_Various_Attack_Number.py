# What changed:
# - Added GPU detection/device prompt and chunk size prompt with row estimation.
# - Standardized outputs under ./outputs/Graph_To_Compare_Various_Attack_Number.
# - Added final summary with output paths.
#
# Purpose:
# - Count label distribution across CSV files in a folder.
# - Save a summary CSV and bar chart.
# - Stream-process large files in chunks.

import os
import sys
import argparse

# Allow running this script from any working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import matplotlib.pyplot as plt

from config.global_config import DEFAULT_CHUNK_SIZE_MB
from utils.chunk_utils import compute_chunk_plan, format_progress, print_chunk_plan
from utils.engine_utils import select_engine
from utils.path_utils import resolve_input_path, resolve_output_path


# Parent folder containing all CSVs
PARENT_FOLDER = "Raw_Data_2017"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "Graph_To_Compare_Various_Attack_Number")


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
    p = argparse.ArgumentParser(description="Count label distribution across CSV files and save chart/report.")
    p.add_argument("--input", default=PARENT_FOLDER, help="Input folder containing CSV files")
    p.add_argument("--output-dir", default=None, help="Base output directory")
    p.add_argument("--chunk-size-mb", type=int, default=DEFAULT_CHUNK_SIZE_MB, help="Chunk size in MB")
    p.add_argument("--engine", default="pandas", choices=["pandas", "dask", "dask-gpu"], help="Execution engine")
    p.add_argument("--use-gpu", action="store_true", help="Force GPU (or fail)")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU")
    p.add_argument("--no-interactive", action="store_true", help="Disable interactive prompts")
    return p


def main(argv: list[str] | None = None):
    args = build_arg_parser().parse_args(argv)

    selection = select_engine(engine=args.engine, use_gpu_flag=args.use_gpu, no_gpu_flag=args.no_gpu)
    if selection.engine != "pandas":
        print(f"[info] --engine {selection.engine} requested; this script currently runs in pandas mode.")
    if selection.use_gpu:
        print("[info] GPU was approved, but this script uses CPU-based pandas. Using CPU.")
    device_used = "cpu"

    parent_folder = resolve_input_path(args.input)
    base_output_dir = resolve_output_path(args.output_dir)
    output_folder = os.path.join(base_output_dir, "Graph_To_Compare_Various_Attack_Number")

    if not os.path.isdir(parent_folder):
        print(f"ERROR: Folder not found: {parent_folder}")
        return

    csv_files = [os.path.join(root, file)
                 for root, _, files in os.walk(parent_folder)
                 for file in files if file.endswith(".csv")]
    if not csv_files:
        print("No CSV files found.")
        return

    chunk_mb = int(args.chunk_size_mb)
    plan0 = compute_chunk_plan(csv_files[0], chunk_mb)
    print_chunk_plan(plan0)

    chunk_rows = estimate_rows_per_chunk(csv_files[0], chunk_mb)
    print(f"Using chunk size: {chunk_mb}MB (~{chunk_rows:,} rows per chunk)")

    os.makedirs(output_folder, exist_ok=True)

    overall_counts = {}
    total_rows_processed = 0

    for file_path in csv_files:
        print(f"Processing {file_path}...")
        file_plan = compute_chunk_plan(file_path, chunk_mb)
        print_chunk_plan(file_plan)
        try:
            header_df = pd.read_csv(file_path, nrows=0)
            label_col = None
            for col in header_df.columns:
                if col.lower() == "label":
                    label_col = col
                    break

            if label_col is None:
                print(f"No 'Label' column in {file_path}, skipping.")
                continue

            for chunk_idx, chunk in enumerate(pd.read_csv(file_path, usecols=[label_col], chunksize=chunk_rows), 1):
                print(format_progress(chunk_idx, file_plan.total_chunks))
                total_rows_processed += len(chunk)
                file_counts = chunk[label_col].value_counts().to_dict()
                for lbl, cnt in file_counts.items():
                    overall_counts[lbl] = overall_counts.get(lbl, 0) + cnt

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    summary_df = pd.DataFrame(list(overall_counts.items()), columns=["Label", "Count"])
    csv_path = make_unique_path(os.path.join(output_folder, "Overall_Label_Distribution.csv"))
    summary_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    summary_df.set_index("Label")["Count"].plot(kind="bar")
    plt.title("Overall Class Distribution (Benign vs. Attack Types)")
    plt.ylabel("Number of Samples")
    plt.xlabel("Label")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = make_unique_path(os.path.join(output_folder, "Overall_Class_Distribution.png"))
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("Saved: Overall_Label_Distribution.csv and Overall_Class_Distribution.png")

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Device used: {device_used.upper()}")
    print(f"Chunk size: {chunk_mb}MB (~{chunk_rows:,} rows)")
    print(f"Total rows processed: {total_rows_processed:,}")
    print("Rows saved: N/A")
    print("Output paths:")
    print(f"  - {csv_path}")
    print(f"  - {plot_path}")


if __name__ == "__main__":
    main()

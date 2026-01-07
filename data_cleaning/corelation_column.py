import os
import sys
import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict

# =========================
# USER CONFIG (edit here or use CLI)
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # always project root
INPUT_FOLDER = "Bening"  # default input folder (relative to project root)
OUTPUT_FOLDER = "correlation_output"
CORR_THRESHOLD = 0.85
CHUNK_SIZE_MB = 500  # default chunk size in MB
MAX_ROWS = None  # set to an int to limit rows saved, or None for all

# =========================
# PATH SETUP
# =========================
def resolve_path(folder):
    if os.path.isabs(folder):
        return folder
    return os.path.join(PROJECT_ROOT, folder)

input_folder = resolve_path(INPUT_FOLDER)
output_path = resolve_path(OUTPUT_FOLDER)
os.makedirs(output_path, exist_ok=True)

# =========================
# GPU DETECTION & ENGINE SELECTION
# =========================
def detect_gpu():
    try:
        import dask_cuda
        from dask_cuda import LocalCUDACluster
        import dask
        import cupy
        return True, "dask-cuda"
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return True, "torch"
    except Exception:
        pass
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices("GPU"):
            return True, "tensorflow"
    except Exception:
        pass
    return False, None

gpu_available, gpu_lib = detect_gpu()
engine = "pandas"
if gpu_available:
    print("[INFO] GPU detected.")
    use_gpu = input("Use GPU for correlation (Dask-cuDF)? (y/n): ").strip().lower() in ("y", "yes")
    if use_gpu:
        engine = "dask-gpu"
        print("[INFO] Using Dask-cuDF (GPU) for correlation.")
    else:
        print("[INFO] Using CPU (pandas/Dask).")
else:
    print("[INFO] GPU not detected. Using CPU (pandas/Dask).")

# =========================
# FILE SELECTION
# =========================
csv_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv')]
if not csv_files:
    print(f"[ERROR] No CSV files found in {input_folder}")
    sys.exit(1)

print("\n--- CSV Files Found ---")
for i, fname in enumerate(csv_files, 1):
    print(f"  {i}: {fname}")
print("-----------------------")
while True:
    file_choice = input(f"Enter the number of the file to process (1-{len(csv_files)}): ").strip()
    if file_choice.isdigit() and 1 <= int(file_choice) <= len(csv_files):
        selected_file = csv_files[int(file_choice)-1]
        break
    print("Invalid selection. Please enter a valid number.")
input_csv = os.path.join(input_folder, selected_file)

# =========================
# CHUNK SIZE ESTIMATION
# =========================
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

chunk_rows = estimate_rows_per_chunk(input_csv, CHUNK_SIZE_MB)
print(f"Using chunk size: {CHUNK_SIZE_MB}MB (~{chunk_rows:,} rows per chunk)")

# =========================
# LOAD DATA (CHUNKED, NUMERIC ONLY)
# =========================
excluded_cols = []
if engine == "dask-gpu":
    import dask_cudf
    import cudf
    ddf = dask_cudf.read_csv(input_csv)
    all_cols = list(ddf.columns)
    numeric_cols = ddf.select_dtypes(include=[np.number]).columns
    numeric_cols = list(numeric_cols)
    excluded_cols = [col for col in all_cols if col not in numeric_cols]
    ddf = ddf[numeric_cols]
    if MAX_ROWS is not None:
        ddf = ddf.head(MAX_ROWS, compute=False)
    print(f"[INFO] Numeric columns used: {len(numeric_cols)} (Dask-cuDF)")
    if excluded_cols:
        print(f"[INFO] Excluded non-numeric columns: {excluded_cols}")
    corr_matrix = ddf.corr().compute().abs()
    df = ddf.compute()
else:
    print(f"\n[INFO] Loading numeric columns from: {input_csv}\n")
    reader = pd.read_csv(input_csv, chunksize=chunk_rows, low_memory=False)
    numeric_cols = None
    all_cols = None
    chunks = []
    total_rows = 0
    for chunk in reader:
        if numeric_cols is None:
            all_cols = list(chunk.columns)
            numeric_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
            excluded_cols = [col for col in all_cols if col not in numeric_cols]
            if not numeric_cols:
                print("[ERROR] No numeric columns found. Exiting.")
                sys.exit(1)
            print(f"[INFO] Numeric columns used: {len(numeric_cols)} (pandas)")
            if excluded_cols:
                print(f"[INFO] Excluded non-numeric columns: {excluded_cols}")
        chunk_numeric = chunk[numeric_cols]
        chunks.append(chunk_numeric)
        total_rows += len(chunk_numeric)
        if MAX_ROWS is not None and total_rows >= MAX_ROWS:
            break
    if MAX_ROWS is not None:
        # Truncate to max rows
        rows_needed = MAX_ROWS
        for i, c in enumerate(chunks):
            if len(c) >= rows_needed:
                chunks[i] = c.iloc[:rows_needed]
                chunks = chunks[:i+1]
                break
            else:
                rows_needed -= len(c)
    df = pd.concat(chunks, ignore_index=True)
    corr_matrix = df.corr(method="pearson").abs()

# =========================
# BUILD CORRELATION GRAPH
# =========================
graph = defaultdict(set)
for col1, col2 in combinations(corr_matrix.columns, 2):
    corr_val = corr_matrix.loc[col1, col2]
    if corr_val >= CORR_THRESHOLD:
        graph[col1].add(col2)
        graph[col2].add(col1)

# =========================
# FIND CONNECTED COMPONENTS
# =========================
visited = set()
groups = []
def dfs(node, group):
    for neighbor in graph[node]:
        if neighbor not in visited:
            visited.add(neighbor)
            group.add(neighbor)
            dfs(neighbor, group)
for col in graph:
    if col not in visited:
        visited.add(col)
        group = {col}
        dfs(col, group)
        groups.append(sorted(group))

# =========================
# SAVE CORRELATION REPORT
# =========================
report_rows = []
for group in groups:
    if len(group) > 1:
        for c1, c2 in combinations(group, 2):
            report_rows.append({
                "feature_1": c1,
                "feature_2": c2,
                "correlation": corr_matrix.loc[c1, c2]
            })
report_df = pd.DataFrame(report_rows)
report_csv = os.path.join(output_path, f"{os.path.splitext(selected_file)[0]}_correlation_report.csv")
report_df.to_csv(report_csv, index=False)
# Text report
group_txt = os.path.join(output_path, f"{os.path.splitext(selected_file)[0]}_correlation_groups.txt")
with open(group_txt, "w") as f:
    for i, group in enumerate(groups, 1):
        if len(group) > 1:
            f.write(f"{i}) {group[0]} ---> {group[1:]}\n")
print(f"\n[INFO] Correlation reports saved to: {output_path}")

# =========================
# INTERACTIVE DELETION
# =========================
print("\n========== CORRELATED FEATURE GROUPS ==========")
indexed_groups = {}
idx = 1
for group in groups:
    if len(group) > 1:
        print(f"{idx}) {group[0]}  --->  {group[1:]}")
        indexed_groups[idx] = group
        idx += 1
if not indexed_groups:
    print("\n[INFO] No correlated groups found above threshold.")
    sys.exit(0)
print("\n==============================================")
to_delete = set()
while True:
    choice = input("\nEnter group number to delete columns from (or 'q' to finish): ")
    if choice.lower() == "q":
        break
    if not choice.isdigit() or int(choice) not in indexed_groups:
        print("[ERROR] Invalid choice.")
        continue
    group = indexed_groups[int(choice)]
    print(f"\nSelected group: {group}")
    cols = input("Enter column names to DELETE (comma-separated): ")
    selected = [c.strip() for c in cols.split(",") if c.strip() in group]
    if not selected:
        print("[ERROR] No valid columns selected.")
        continue
    to_delete.update(selected)
    print(f"[INFO] Marked for deletion: {selected}")
# =========================
# APPLY DELETION
# =========================
if to_delete:
    cleaned_df = df.drop(columns=list(to_delete), errors="ignore")
    cleaned_path = os.path.join(output_path, f"{os.path.splitext(selected_file)[0]}_cleaned.csv")
    cleaned_df.to_csv(cleaned_path, index=False)
    print("\n[INFO] Columns deleted:")
    for c in to_delete:
        print(f" - {c}")
    print(f"\n[INFO] Cleaned dataset saved to: {cleaned_path}")
else:
    print("\n[INFO] No columns deleted.")
print("\n[DONE] Correlation pruning completed successfully.\n")

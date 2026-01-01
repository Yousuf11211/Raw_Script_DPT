"""Dask-based label cleaner (out-of-core).

This script is intentionally a standalone utility (no CLI) to match the repo's
script-style UX. It:

- Scans CSV files under INPUT_FOLDER.
- Detects a label column ("label" or "LABEL").
- Filters rows based on TARGET_LABEL_MODE (benign vs attack).
- Splits rows into:
    * Cleaned rows (no missing values, globally deduplicated)
    * Rows removed due to missing values (saved separately)
- Writes outputs in multiple CSV part files capped by DEFAULT_MAX_OUTPUT_ROWS.
- Prints per-file and overall progress based on a file-size chunk estimate.

Constraints:
- Uses Dask for out-of-core processing.
- Never materializes a full file or full Dask dataframe in RAM.

Important notes:
- Global deduplication uses Dask's drop_duplicates() which may trigger a shuffle.
- Output splitting is done deterministically by partition + row slicing; we only
  compute one output block at a time.
"""

from __future__ import annotations

import math
import os
import sys
import time
import sqlite3
from typing import Optional, Tuple

# Allow running this script from any working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

try:
    import dask
    import dask.dataframe as dd
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "This script requires 'dask[dataframe]'. Install it before running."
    ) from e

# Use a predictable scheduler unless the user overrides via DASK_SCHEDULER env.
# This makes progress printing more consistent on Windows.
if not os.environ.get("DASK_SCHEDULER"):
    dask.config.set(scheduler="single-threaded")


from config.global_config import (
    DEFAULT_CHUNK_SIZE_MB,
    DEFAULT_MAX_OUTPUT_ROWS,
)


# ==============================================================================
# 0. Hardcoded defaults (MANDATORY)
# ==============================================================================

# 0 → benign rows
# 1 → attack rows (anything NOT benign)
TARGET_LABEL_MODE = 0

BENIGN_LABEL_VALUE = "benign"
LABEL_COLUMNS = ["label", "LABEL"]

INPUT_FOLDER = "IDS2018"

OUTPUT_CLEAN_FOLDER = "output_cleaned"
OUTPUT_MISSING_FOLDER = "output_missing"

OUTPUT_BASE_NAME = "bening"

# SQLite file used to guarantee cross-chunk dedup without loading all keys into RAM.
DEDUP_DB_PATH = "dedup_seen.sqlite"


# ==============================================================================
# 1. Normalization + label logic (MANDATORY)
# ==============================================================================

def normalize_label(value):
    return str(value).strip().lower()


def _row_matches_target(label_value) -> bool:
    """Return True if a label value matches the configured TARGET_LABEL_MODE."""
    is_benign = normalize_label(label_value) == normalize_label(BENIGN_LABEL_VALUE)
    if TARGET_LABEL_MODE == 0:
        return bool(is_benign)
    if TARGET_LABEL_MODE == 1:
        return bool(not is_benign)
    raise ValueError(f"Unsupported TARGET_LABEL_MODE: {TARGET_LABEL_MODE}")


def _label_mask(ddf: dd.DataFrame, label_col: str) -> dd.Series:
    """Vectorized Dask mask for target label selection.

    - Case-insensitive
    - Whitespace-insensitive
    - Missing labels are treated as non-match for both modes
    """
    benign_norm = normalize_label(BENIGN_LABEL_VALUE)

    # Normalize labels per-partition without turning the entire column into python objects.
    def _norm_partition(part):
        # Ensure we don't error on NaN: fill with empty string first.
        return part.astype("object").fillna("").map(normalize_label)

    s = ddf[label_col].map_partitions(_norm_partition, meta=(label_col, "object"))

    if TARGET_LABEL_MODE == 0:
        return s == benign_norm
    if TARGET_LABEL_MODE == 1:
        return s != benign_norm
    raise ValueError(f"Unsupported TARGET_LABEL_MODE: {TARGET_LABEL_MODE}")


# ==============================================================================
# 2. Pre-scan helpers (MANDATORY)
# ==============================================================================

def _detect_label_column_from_header(columns) -> Optional[str]:
    lower_to_actual = {str(c).strip().lower(): str(c) for c in columns}
    for candidate in LABEL_COLUMNS:
        key = str(candidate).strip().lower()
        if key in lower_to_actual:
            return lower_to_actual[key]
    return None


# Debug toggle for prescan label prints (kept minimal for large folders)
DEBUG_PRESCAN_LABELS = False


def _prescan_file_for_target_rows(path: str, *, nrows: int = 10) -> Tuple[bool, bool, Optional[str]]:
    """Return (has_label_column, matches_target, label_col)."""
    try:
        preview = pd.read_csv(
            path,
            nrows=nrows,
            low_memory=True,
            engine="c",
        )
    except Exception:
        return False, False, None

    if preview is None or preview.empty:
        return False, False, None

    label_col = _detect_label_column_from_header(preview.columns)
    if not label_col:
        return False, False, None

    try:
        s = preview[label_col]
        matches = bool(s.map(_row_matches_target).any())
        if DEBUG_PRESCAN_LABELS and not matches:
            try:
                uniq = sorted({normalize_label(v) for v in s.dropna().tolist()})
                print(f"[DEBUG] {os.path.basename(path)} preview labels (normalized): {uniq}")
            except Exception:
                pass
        return True, matches, str(label_col)
    except Exception:
        return True, False, str(label_col)


# ==============================================================================
# 3. Output writing (deterministic splitting, memory-safe)
# ==============================================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _part_path(folder: str, base: str, part_index: int) -> str:
    # Naming rules:
    # bening_csv.csv
    # bening_csv_1.csv
    # bening_csv_2.csv
    if part_index == 0:
        return os.path.join(folder, f"{base}.csv")
    return os.path.join(folder, f"{base}_{part_index}.csv")


def _append_partition_to_split_csvs(
    pdf: "pd.DataFrame",
    *,
    out_folder: str,
    base_name: str,
    max_rows_per_file: int,
    part_index: int,
    rows_in_current_file: int,
    header_written: bool,
) -> Tuple[int, int, bool]:
    """Append a pandas DataFrame to split CSV outputs respecting max_rows_per_file.

    Returns:
        (new_part_index, new_rows_in_current_file, new_header_written)
    """
    if pdf is None or pdf.empty:
        return part_index, rows_in_current_file, header_written

    start = 0
    while start < len(pdf):
        remaining_in_file = max_rows_per_file - rows_in_current_file
        if remaining_in_file <= 0:
            part_index += 1
            rows_in_current_file = 0
            header_written = False
            remaining_in_file = max_rows_per_file

        take = min(remaining_in_file, len(pdf) - start)
        slice_df = pdf.iloc[start : start + take]

        out_path = _part_path(out_folder, base_name, part_index)
        mode = "w" if not header_written else "a"
        slice_df.to_csv(out_path, index=False, mode=mode, header=not header_written)
        header_written = True

        rows_in_current_file += int(len(slice_df))
        start += int(take)

    return part_index, rows_in_current_file, header_written


def _stable_row_hash(pdf: "pd.DataFrame") -> "pd.Series":
    """Return a stable hash per row based on all column values.

    Uses pandas hashing (fast) then converts to hex strings for SQLite.
    """
    # hash_pandas_object is stable within a run; combined with dtype='object' it is deterministic.
    import pandas as _pd
    h = _pd.util.hash_pandas_object(pdf, index=False)
    return h.astype("uint64").map(lambda x: format(int(x), "016x"))


def _dedup_filter_sqlite(pdf: "pd.DataFrame", *, db_path: str, table: str = "seen") -> "pd.DataFrame":
    """Filter out rows already seen (cross-chunk) using an on-disk SQLite set.

    This avoids Dask global shuffle and starts producing output immediately.

    Note: still out-of-core because we only process one pandas partition at a time.
    """
    if pdf is None or pdf.empty:
        return pdf

    keys = _stable_row_hash(pdf)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table} (k TEXT PRIMARY KEY)")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        cur = conn.cursor()
        # Find which keys already exist.
        # Use executemany in chunks to reduce overhead.
        existing = set()
        chunk = 5000
        key_list = list(keys)
        for i in range(0, len(key_list), chunk):
            sub = key_list[i : i + chunk]
            qs = ",".join(["(?)"] * len(sub))
            rows = cur.execute(f"SELECT k FROM {table} WHERE k IN ({qs})", sub).fetchall()
            existing.update(r[0] for r in rows)

        keep_mask = [k not in existing for k in key_list]
        out = pdf.loc[keep_mask].copy()

        # Insert new keys.
        new_keys = [k for k in key_list if k not in existing]
        for i in range(0, len(new_keys), chunk):
            sub = [(k,) for k in new_keys[i : i + chunk]]
            cur.executemany(f"INSERT OR IGNORE INTO {table}(k) VALUES (?)", sub)
        conn.commit()

        return out
    finally:
        conn.close()


def _write_ddf_split_streaming(
    ddf: dd.DataFrame,
    *,
    out_folder: str,
    base_name: str,
    max_rows_per_file: int,
    start_part_index: int,
    start_rows_in_current: int,
    start_header_written: bool,
    file_name: str,
    file_chunks_estimate: int,
    overall_done_ref: list,
    overall_total_chunks: int,
    progress_prefix: str,
    dedup_db_path: str,
) -> Tuple[int, int, bool]:
    """Write a Dask DataFrame incrementally, computing one delayed partition at a time.

    Per-file progress requirement:
      - Print sequential chunk counters of the form `chunk 12 / 386`.

    Timing:
      - Prints per-chunk elapsed wall time (compute + write).
      - Prints a per-stream total at the end (clean vs missing).

    Because Dask partitions and our file-size chunk estimate may not match 1:1,
    we map partition index -> an estimated chunk counter. We still print a
    sequential counter so users see forward progress.

    Important:
      - overall_done_ref is incremented by the *caller* once per source file chunk estimate.
        This avoids double counting when we write both CLEAN and MISSING streams.
    """

    _ensure_dir(out_folder)

    delayed_parts = ddf.to_delayed()
    n_partitions = len(delayed_parts)

    part_index = int(start_part_index)
    rows_in_current = int(start_rows_in_current)
    header_written = bool(start_header_written)

    # Ensure we start printing at 1 / estimate.
    last_est_chunk = 0

    stream_start = time.perf_counter()

    for p_idx, d in enumerate(delayed_parts, 1):
        chunk_start = time.perf_counter()

        pdf = dask.compute(d)[0]
        pdf = _dedup_filter_sqlite(pdf, db_path=dedup_db_path)

        # Compute requested display chunk number (1..estimate)
        est_chunk = int(math.ceil((p_idx / max(1, n_partitions)) * int(file_chunks_estimate)))
        est_chunk = max(1, min(int(file_chunks_estimate), int(est_chunk)))

        # Force sequential display without skipping 1.
        if est_chunk <= last_est_chunk:
            est_chunk = min(int(file_chunks_estimate), last_est_chunk + 1)
        last_est_chunk = est_chunk

        part_index, rows_in_current, header_written = _append_partition_to_split_csvs(
            pdf,
            out_folder=out_folder,
            base_name=base_name,
            max_rows_per_file=max_rows_per_file,
            part_index=part_index,
            rows_in_current_file=rows_in_current,
            header_written=header_written,
        )

        del pdf

        # Overall progress: increment once per displayed chunk tick (bounded).
        overall_done_ref[0] = int(min(int(overall_total_chunks), int(overall_done_ref[0]) + 1))

        elapsed = time.perf_counter() - chunk_start
        overall_now = int(overall_done_ref[0])
        print(
            f"{file_name} -> chunk {est_chunk} / {file_chunks_estimate} ({progress_prefix}) "
            f"— {elapsed:.2f}s"
        )
        print(f"Overall progress: {overall_now} / {overall_total_chunks} chunks")

    stream_elapsed = time.perf_counter() - stream_start
    print(f"[INFO] {file_name} {progress_prefix} stream finished in {stream_elapsed:.2f}s")

    return part_index, rows_in_current, header_written


# ==============================================================================
# 4. Main processing
# ==============================================================================

def main() -> None:
    chunk_size_bytes = int(DEFAULT_CHUNK_SIZE_MB) * 1024 * 1024

    # Mandatory prints
    print(f"[INFO] Chunk size: {DEFAULT_CHUNK_SIZE_MB} MB")
    print(f"[INFO] Max output rows per file: {DEFAULT_MAX_OUTPUT_ROWS}")

    # Resolve INPUT_FOLDER relative to repo root (matches other scripts' UX)
    input_root = INPUT_FOLDER
    if not os.path.isabs(input_root):
        input_root = os.path.join(_REPO_ROOT, input_root)
    input_root = os.path.abspath(input_root)
    print(f"[INFO] Input folder resolved to: {input_root}")

    if not os.path.isdir(input_root):
        print(f"[WARN] INPUT_FOLDER not found: {input_root}")
        return

    # Recursively find CSV files under the input folder
    all_csv_files: list[str] = []
    for root, _, files in os.walk(input_root):
        for f in files:
            if f.lower().endswith(".csv"):
                all_csv_files.append(os.path.join(root, f))
    all_csv_files = sorted(all_csv_files)

    print(f"[INFO] Total CSV files found: {len(all_csv_files)}")

    # --- Pre-scan ---
    files_with_label = 0
    files_matching_target: list[Tuple[str, str]] = []
    files_skipped = 0

    for path in all_csv_files:
        has_label, matches, label_col = _prescan_file_for_target_rows(path, nrows=10)

        if not has_label:
            files_skipped += 1
            print(f"[WARN] Skipping (no label column): {os.path.basename(path)}")
            continue

        files_with_label += 1

        if not matches:
            files_skipped += 1
            print(f"[INFO] Skipping (no target rows): {os.path.basename(path)}")
            continue

        # label_col is already resolved to the actual header
        files_matching_target.append((path, label_col or ""))

    print(f"[INFO] Files with label column: {files_with_label}")
    print(f"[INFO] Files matching target label mode: {len(files_matching_target)}")
    print(f"[INFO] Files skipped: {files_skipped}")

    if not files_matching_target:
        print("[DONE] Files processed matching target mode: 0")
        print(f"[DONE] Cleaned data saved to {OUTPUT_CLEAN_FOLDER}/")
        print(f"[DONE] Missing-value rows saved to {OUTPUT_MISSING_FOLDER}/")
        return

    # --- Chunk estimation across eligible files (file-size based) ---
    # NOTE: this estimate is required by the spec; we will use it for UX.
    per_file_chunk_est: dict[str, int] = {}
    total_chunks_all_files = 0
    for path, _label_col in files_matching_target:
        file_size_bytes = os.path.getsize(path)
        file_chunks = int(math.ceil(file_size_bytes / float(max(1, chunk_size_bytes))))
        print(f"{os.path.basename(path)} -> {file_chunks} chunks")
        per_file_chunk_est[path] = file_chunks
        total_chunks_all_files += file_chunks

    print(f"[INFO] Total chunks to process: {total_chunks_all_files}")

    # --- Output dirs ---
    _ensure_dir(OUTPUT_CLEAN_FOLDER)
    _ensure_dir(OUTPUT_MISSING_FOLDER)

    overall_chunks_done = [0]  # mutable cell shared between writers

    # Deterministic split state across all files for each stream.
    clean_part_index = 0
    clean_rows_in_current = 0
    clean_header_written = False

    missing_part_index = 0
    missing_rows_in_current = 0
    missing_header_written = False

    for (path, _label_col_from_prescan) in files_matching_target:
        file_name = os.path.basename(path)
        file_chunks_estimate = per_file_chunk_est.get(path, 1)

        print("-" * 80)
        print(f"[INFO] Processing: {file_name} (estimated {file_chunks_estimate} chunks)")
        print("[INFO] Building Dask task graph (lazy)...")

        # Read with Dask (MANDATORY settings)
        ddf = dd.read_csv(
            path,
            blocksize=chunk_size_bytes,
            assume_missing=True,
            dtype="object",
        )

        # Capture raw column order (MANDATORY: do not change output column sequence)
        raw_columns = list(ddf.columns)

        # Detect label column (skip file if missing)
        actual_label_col = _detect_label_column_from_header(ddf.columns)
        if not actual_label_col:
            print(f"[WARN] Skipping (no label column at runtime): {file_name}")
            continue

        # Force a tiny, cheap read so users see activity immediately and we fail fast on parse errors.
        try:
            print("[INFO] Warming up: reading first partition header/rows (this should be quick)...")
            _ = ddf.head(5)
        except Exception as e:
            print(f"[ERROR] Failed to read CSV with Dask: {file_name} | {e}")
            continue

        # 9A) Label filtering (skip non-matching rows entirely)
        ddf = ddf[_label_mask(ddf, actual_label_col)]

        # 9B) Missing-values split
        missing_mask = ddf.isna().any(axis=1)
        ddf_missing = ddf[missing_mask]
        ddf_clean = ddf[~missing_mask]

        # NOTE: We intentionally do NOT call ddf.drop_duplicates() here.
        # Dask global drop_duplicates triggers a shuffle and can look "stuck" for a long time.
        # Instead we do disk-backed dedup per computed partition in the writer.

        # Preserve original raw column order exactly
        ddf_clean = ddf_clean[raw_columns]
        ddf_missing = ddf_missing[raw_columns]

        # IMPORTANT: progress increments are handled inside the streaming writer.
        # Do NOT pre-increment overall progress here.

        # Write CLEAN (streaming)
        clean_part_index, clean_rows_in_current, clean_header_written = _write_ddf_split_streaming(
            ddf_clean,
            out_folder=OUTPUT_CLEAN_FOLDER,
            base_name=OUTPUT_BASE_NAME,
            max_rows_per_file=int(DEFAULT_MAX_OUTPUT_ROWS),
            start_part_index=clean_part_index,
            start_rows_in_current=clean_rows_in_current,
            start_header_written=clean_header_written,
            file_name=file_name,
            file_chunks_estimate=file_chunks_estimate,
            overall_done_ref=overall_chunks_done,
            overall_total_chunks=total_chunks_all_files,
            progress_prefix="clean",
            dedup_db_path=os.path.join(OUTPUT_CLEAN_FOLDER, DEDUP_DB_PATH),
        )

        # Write MISSING (streaming)
        missing_part_index, missing_rows_in_current, missing_header_written = _write_ddf_split_streaming(
            ddf_missing,
            out_folder=OUTPUT_MISSING_FOLDER,
            base_name=OUTPUT_BASE_NAME,
            max_rows_per_file=int(DEFAULT_MAX_OUTPUT_ROWS),
            start_part_index=missing_part_index,
            start_rows_in_current=missing_rows_in_current,
            start_header_written=missing_header_written,
            file_name=file_name,
            file_chunks_estimate=file_chunks_estimate,
            overall_done_ref=overall_chunks_done,
            overall_total_chunks=total_chunks_all_files,
            progress_prefix="missing",
            dedup_db_path=os.path.join(OUTPUT_MISSING_FOLDER, DEDUP_DB_PATH),
        )

        # Release references between files (MANDATORY)
        del ddf
        del ddf_clean
        del ddf_missing

    print("-" * 80)
    print(f"[DONE] Files processed matching target mode: {len(files_matching_target)}")
    print(f"[DONE] Cleaned data saved to {OUTPUT_CLEAN_FOLDER}/")
    print(f"[DONE] Missing-value rows saved to {OUTPUT_MISSING_FOLDER}/")


if __name__ == "__main__":
    main()


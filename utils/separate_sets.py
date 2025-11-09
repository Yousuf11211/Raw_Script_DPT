import os
import math
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

DEFAULT_INPUT_FOLDER = "Normalized_SET"
DEFAULT_OUTPUT_FOLDER = "Separated_Model_Data"
DEFAULT_CHUNK_SIZE = 500_000
DEFAULT_LABEL_COLUMN_NAME = 'label'
DEFAULT_BENIGN_LABEL_VALUE = 'benign'


def analyze_and_classify(all_files: List[str],
                         processing_mode: str,
                         label_column_name: str = DEFAULT_LABEL_COLUMN_NAME,
                         benign_label_value: str = DEFAULT_BENIGN_LABEL_VALUE,
                         chunk_size: int = DEFAULT_CHUNK_SIZE) -> Tuple[Dict[str, int], Dict[str, List[str]], Optional[str], List[str]]:
    """
    Scan files to count labels and classify files by labels present.
    Returns: (total_counts, files_by_label, actual_label_col_name, skipped_files)
    """
    total_counts: Dict[str, int] = defaultdict(int)
    files_by_label: Dict[str, set] = defaultdict(set)
    actual_label_col_name: Optional[str] = None
    skipped: List[str] = []

    for file_path in all_files:
        try:
            if actual_label_col_name is None:
                header_df = pd.read_csv(file_path, nrows=0, low_memory=False)
                for col in header_df.columns:
                    if col.lower() == label_column_name:
                        actual_label_col_name = col
                        break
            if not actual_label_col_name:
                skipped.append(file_path)
                continue

            if processing_mode != 'both':
                try:
                    preview_df = pd.read_csv(file_path, usecols=[actual_label_col_name], nrows=20, low_memory=False)
                    unique_labels_preview = set(preview_df[actual_label_col_name].astype(str).str.lower().unique())
                    if processing_mode == 'attacks' and unique_labels_preview == {benign_label_value}:
                        skipped.append(file_path)
                        continue
                    elif processing_mode == 'benign' and benign_label_value not in unique_labels_preview:
                        skipped.append(file_path)
                        continue
                except Exception:
                    pass

            for chunk in pd.read_csv(file_path, usecols=[actual_label_col_name], chunksize=chunk_size, low_memory=False):
                chunk.columns = [c.lower() for c in chunk.columns]
                counts = chunk[label_column_name].astype(str).str.lower().value_counts()
                for label, cnt in counts.items():
                    total_counts[label] += int(cnt)
                    files_by_label[label].add(file_path)
        except Exception:
            skipped.append(file_path)
            continue

    files_by_label_out: Dict[str, List[str]] = {lbl: sorted(list(paths)) for lbl, paths in files_by_label.items()}
    return dict(total_counts), files_by_label_out, actual_label_col_name, skipped


def process_and_save_combined(file_list: List[str],
                              rows_per_output_file: int,
                              labels_to_keep: List[str],
                              output_group_name: str,
                              output_base_path: str,
                              should_shuffle: bool,
                              actual_label_col_name: str) -> Dict[str, Any]:
    """
    Memory-efficient sequential combiner. Returns {'created_files': [...], 'total_rows': int}
    """
    created: List[str] = []
    total_rows_written = 0
    if not file_list or not labels_to_keep:
        return {"created_files": created, "total_rows": total_rows_written}

    os.makedirs(output_base_path, exist_ok=True)
    lower_keep = [str(lbl).lower() for lbl in labels_to_keep]
    iterators: Dict[str, Any] = {}
    for file_path in file_list:
        try:
            iterators[file_path] = pd.read_csv(file_path, iterator=True, chunksize=50_000, low_memory=False)
        except Exception:
            continue

    file_part = 1
    leftover_df = pd.DataFrame()
    while iterators:
        batch_frames = [leftover_df] if not leftover_df.empty else []
        rows_collected = len(leftover_df)
        while rows_collected < rows_per_output_file:
            if not iterators:
                break
            for file_path in list(iterators.keys()):
                try:
                    chunk = next(iterators[file_path])
                    clean_chunk = chunk[chunk[actual_label_col_name].astype(str).str.lower().isin(lower_keep)]
                    if not clean_chunk.empty:
                        batch_frames.append(clean_chunk)
                        rows_collected += len(clean_chunk)
                        if rows_collected >= rows_per_output_file:
                            break
                except StopIteration:
                    del iterators[file_path]
                except Exception:
                    del iterators[file_path]
        if not batch_frames:
            break
        combined_df = pd.concat(batch_frames, ignore_index=True)
        if should_shuffle:
            combined_df = combined_df.sample(frac=1).reset_index(drop=True)
        final_df = combined_df.iloc[:rows_per_output_file]
        leftover_df = combined_df.iloc[rows_per_output_file:]
        out_path = os.path.join(output_base_path, f"{output_group_name}_part_{file_part}.csv")
        final_df.to_csv(out_path, index=False)
        created.append(out_path)
        total_rows_written += len(final_df)
        file_part += 1
    if not leftover_df.empty:
        out_path = os.path.join(output_base_path, f"{output_group_name}_part_{file_part}.csv")
        leftover_df.to_csv(out_path, index=False)
        created.append(out_path)
        total_rows_written += len(leftover_df)
    return {"created_files": created, "total_rows": total_rows_written}


def process_and_save_proportionally(file_list: List[str],
                                    rows_per_output_file: int,
                                    labels_to_keep: List[str],
                                    output_group_name: str,
                                    output_base_path: str,
                                    should_shuffle: bool,
                                    actual_label_col_name: str) -> Dict[str, Any]:
    """Memory-intensive proportional splitter. Returns {'created_files': [...], 'total_rows': int}
    """
    created: List[str] = []
    total_rows_written = 0
    if not file_list or not labels_to_keep:
        return {"created_files": created, "total_rows": total_rows_written}

    os.makedirs(output_base_path, exist_ok=True)
    lower_keep = [str(lbl).lower() for lbl in labels_to_keep]

    samples: List[pd.DataFrame] = []
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path, low_memory=False)
            relevant = df[df[actual_label_col_name].astype(str).str.lower().isin(lower_keep)]
            if not relevant.empty:
                samples.append(relevant)
        except Exception:
            continue

    if not samples:
        return {"created_files": created, "total_rows": total_rows_written}

    master_df = pd.concat(samples, ignore_index=True)
    if should_shuffle:
        master_df = master_df.sample(frac=1).reset_index(drop=True)

    num_files = math.ceil(len(master_df) / rows_per_output_file)
    for i in range(num_files):
        start = i * rows_per_output_file
        end = start + rows_per_output_file
        out_df = master_df.iloc[start:end]
        out_path = os.path.join(output_base_path, f"{output_group_name}_part_{i+1}.csv")
        out_df.to_csv(out_path, index=False)
        created.append(out_path)
        total_rows_written += len(out_df)

    return {"created_files": created, "total_rows": total_rows_written}


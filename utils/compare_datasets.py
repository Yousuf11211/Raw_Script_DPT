import os
import hashlib
import pandas as pd
from typing import List, Dict, Any, Optional, Set, Tuple

CHUNK_SIZE = 200_000  # adjust for memory; used when reading large CSVs in row comparison
SUPPORTED_EXT = ('.csv',)


def list_csv_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith('.csv')]


def get_reference_columns(folder: str) -> Dict[str, Any]:
    """Return reference column order from first CSV and a list of mismatches in other files."""
    files = list_csv_files(folder)
    if not files:
        return {"reference": None, "mismatches": [], "error": "No CSV files found"}
    ref_file = files[0]
    try:
        ref_cols = list(pd.read_csv(ref_file, nrows=0).columns)
    except Exception as e:
        return {"reference": None, "mismatches": [], "error": f"Failed to read {ref_file}: {e}"}
    mismatches = []
    for f in files[1:]:
        try:
            cols = list(pd.read_csv(f, nrows=0).columns)
            if cols != ref_cols:
                mismatches.append({"file": f, "columns": cols})
        except Exception as e:
            mismatches.append({"file": f, "error": str(e)})
    return {"reference": ref_cols, "mismatches": mismatches, "file_count": len(files)}


def hash_row(values: Tuple[str, ...], method: str = 'md5') -> str:
    h = hashlib.new(method)
    h.update('\x1f'.join(values).encode('utf-8', errors='ignore'))
    return h.hexdigest()


def iterate_rows(filepath: str, hash_mode: bool, method: str, collect_samples: bool, max_samples: int, row_hashes: Set[str], hash_to_row: Dict[str, Tuple[str, ...]]):
    """Populate row_hashes with row signatures; optionally store sample rows in hash_to_row."""
    try:
        for chunk in pd.read_csv(filepath, dtype=str, chunksize=CHUNK_SIZE, low_memory=False):
            for row in chunk.itertuples(index=False, name=None):
                if hash_mode:
                    sig = hash_row(tuple('' if v is None else v for v in row), method)
                else:
                    sig = tuple('' if v is None else v for v in row)  # type: ignore
                if sig not in row_hashes:
                    row_hashes.add(sig)
                    if collect_samples and len(hash_to_row) < max_samples:
                        hash_to_row[sig] = row if isinstance(sig, tuple) else row
    except Exception as e:
        # Record a synthetic error signature if needed
        pass


def compare_rows_between_folders(raw_folder: str,
                                  processed_folder: str,
                                  mode: str = 'hash',  # 'hash' or 'full'
                                  hash_method: str = 'md5',
                                  sample_limit: int = 10) -> Dict[str, Any]:
    """Compare union of rows across all CSVs in two folders.
    mode='hash' stores row hashes only (memory efficient) with a few samples.
    mode='full' stores full tuples (memory heavy).
    Returns counts and sample missing/extra rows.
    """
    raw_files = list_csv_files(raw_folder)
    proc_files = list_csv_files(processed_folder)
    if not raw_files:
        return {"error": f"No CSV files in raw folder {raw_folder}"}
    if not proc_files:
        return {"error": f"No CSV files in processed folder {processed_folder}"}

    hash_mode = mode == 'hash'
    raw_set: Set[Any] = set()
    proc_set: Set[Any] = set()
    sample_raw_map: Dict[Any, Tuple[str, ...]] = {}
    sample_proc_map: Dict[Any, Tuple[str, ...]] = {}

    for f in raw_files:
        iterate_rows(f, hash_mode, hash_method, True, sample_limit, raw_set, sample_raw_map)
    for f in proc_files:
        iterate_rows(f, hash_mode, hash_method, True, sample_limit, proc_set, sample_proc_map)

    missing = raw_set - proc_set
    extra = proc_set - raw_set

    sample_missing = []
    sample_extra = []
    for k in list(missing)[:sample_limit]:
        if k in sample_raw_map:
            sample_missing.append(sample_raw_map[k])
        elif isinstance(k, tuple):
            sample_missing.append(k)
    for k in list(extra)[:sample_limit]:
        if k in sample_proc_map:
            sample_extra.append(sample_proc_map[k])
        elif isinstance(k, tuple):
            sample_extra.append(k)

    return {
        "raw_unique_rows": len(raw_set),
        "processed_unique_rows": len(proc_set),
        "missing_rows_count": len(missing),
        "extra_rows_count": len(extra),
        "sample_missing_rows": sample_missing,
        "sample_extra_rows": sample_extra,
        "mode": mode,
        "hash_method": hash_method if hash_mode else None,
    }


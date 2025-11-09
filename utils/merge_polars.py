import os
import glob
import polars as pl
from typing import List, Dict, Any


def _find_csv_files(input_folder: str, pattern: str = "*.csv", recursive: bool = True) -> List[str]:
    search = os.path.join(input_folder, "**", pattern) if recursive else os.path.join(input_folder, pattern)
    return sorted(glob.glob(search, recursive=recursive))


def _concat_lazy_frames(paths: List[str], infer_schema_length: int = 1000) -> pl.LazyFrame:
    lfs = []
    for p in paths:
        try:
            lf = pl.scan_csv(p, infer_schema_length=infer_schema_length)
            lfs.append(lf)
        except Exception:
            continue
    if not lfs:
        raise FileNotFoundError("No readable CSV files found.")
    try:
        return pl.concat(lfs, how="diagonal_relaxed")
    except Exception:
        # Fallback for older Polars
        return pl.concat(lfs, how="diagonal")


def merge_shuffle_partitioned(input_folder: str,
                              output_folder: str,
                              pattern: str = "*.csv",
                              num_parts: int = 20,
                              seed: int = 42,
                              infer_schema_length: int = 1000,
                              recursive: bool = True) -> Dict[str, Any]:
    os.makedirs(output_folder, exist_ok=True)
    files = _find_csv_files(input_folder, pattern=pattern, recursive=recursive)
    if not files:
        return {"error": f"No CSV files found under {input_folder} with pattern {pattern}"}
    lf = _concat_lazy_frames(files, infer_schema_length=infer_schema_length)
    # Add a bucket column based on a stable hash of all columns + seed
    lf_aug = lf.with_columns([
        pl.hash(pl.lit(seed), *[pl.col(c) for c in lf.columns]).alias("__h"),
    ]).with_columns([
        (pl.col("__h") % pl.lit(num_parts)).alias("__bucket")
    ])
    created: List[str] = []
    total_rows = 0
    for b in range(num_parts):
        out_path = os.path.join(output_folder, f"merged_shuffled_part_{b+1}.csv")
        lf_b = lf_aug.filter(pl.col("__bucket") == b).drop(["__h", "__bucket"])
        # stream writing
        lf_b.sink_csv(out_path)
        created.append(out_path)
        # Optionally count rows per part (cheap metadata)
        try:
            cnt = lf_b.select(pl.count()).collect().item()
            total_rows += int(cnt)
        except Exception:
            pass
    return {
        "created_files": created,
        "parts": num_parts,
        "total_rows": total_rows,
        "input_files": files,
    }


def merge_shuffle_single(input_folder: str,
                         output_file: str,
                         temp_folder: str,
                         pattern: str = "*.csv",
                         num_parts: int = 20,
                         seed: int = 42,
                         infer_schema_length: int = 1000,
                         recursive: bool = True) -> Dict[str, Any]:
    # First write partitioned shards, then combine into single CSV lazily
    meta = merge_shuffle_partitioned(
        input_folder=input_folder,
        output_folder=temp_folder,
        pattern=pattern,
        num_parts=num_parts,
        seed=seed,
        infer_schema_length=infer_schema_length,
        recursive=recursive,
    )
    if meta.get("error"):
        return meta
    # Combine shards into a single file in random order (seed order)
    shards = meta["created_files"][:]
    # Deterministic shuffle by hashing names with seed
    order_df = pl.DataFrame({"path": shards})
    order_df = order_df.with_columns([
        (pl.col("path").hash(seed=seed)).alias("_k")
    ]).sort("_k")
    ordered_paths = order_df["path"].to_list()

    lfs = [pl.scan_csv(p, infer_schema_length=infer_schema_length) for p in ordered_paths]
    try:
        lf_all = pl.concat(lfs, how="diagonal_relaxed")
    except Exception:
        lf_all = pl.concat(lfs, how="diagonal")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    lf_all.sink_csv(output_file)
    meta["single_file"] = output_file
    return meta


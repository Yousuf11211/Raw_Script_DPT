# utils/data_quality.py

import polars as pl
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Optional


def _float_columns(schema: Dict[str, pl.datatypes.DataType]) -> List[str]:
    return [c for c, dt in schema.items() if dt in (pl.Float32, pl.Float64, pl.Int64, pl.Int32, pl.UInt32, pl.UInt64)]


def analyze_inf_columns(lf: pl.LazyFrame) -> Tuple[int, pd.DataFrame]:
    """
    Eagerly computes per-column INF counts for numeric columns and returns a report.

    Returns: (total_rows, pandas_df with columns: Feature, InfCount, InfPercent)
    """
    st.info("Analyzing 'inf' values across numeric columns (eager scan)...")

    total_rows = lf.select(pl.count()).collect().item()
    if total_rows == 0:
        return 0, pd.DataFrame(columns=["Feature", "InfCount", "InfPercent"])

    numeric_cols = _float_columns(lf.schema)
    if not numeric_cols:
        st.warning("No numeric columns detected to check for 'inf'.")
        return total_rows, pd.DataFrame(columns=["Feature", "InfCount", "InfPercent"])

    # Build expressions to count INF per numeric column
    inf_exprs = [pl.col(c).cast(pl.Float64, strict=False).is_infinite().sum().alias(c) for c in numeric_cols]
    inf_counts_df = lf.select(inf_exprs).collect()

    # Convert to long format pandas report
    counts_pd = (
        inf_counts_df
        .transpose(include_header=True, header_name="Feature", column_names=["InfCount"])  # one row -> long
        .to_pandas()
    )
    counts_pd["InfPercent"] = counts_pd["InfCount"].astype(float) / max(total_rows, 1) * 100.0
    counts_pd.sort_values("InfPercent", ascending=False, inplace=True)
    return total_rows, counts_pd


def drop_inf_columns_lazy(lf: pl.LazyFrame, threshold_percent: float) -> Tuple[pl.LazyFrame, List[str]]:
    """
    Drops columns lazily if their INF percentage exceeds the provided threshold (0-100).

    Returns: (new_lazyframe, dropped_columns)
    """
    total_rows, report_df = analyze_inf_columns(lf)
    if report_df.empty or total_rows == 0:
        return lf, []

    to_drop = report_df.loc[report_df["InfPercent"] > threshold_percent, "Feature"].tolist()
    if not to_drop:
        return lf, []

    st.warning(f"Dropping {len(to_drop)} columns exceeding {threshold_percent:.2f}% INF: {', '.join(to_drop[:10])}{' ...' if len(to_drop) > 10 else ''}")
    return lf.drop(to_drop), to_drop


def impute_inf_with_median(lf: pl.LazyFrame, columns: Optional[List[str]] = None) -> Tuple[pl.LazyFrame, Dict[str, float]]:
    """
    Replaces +/-inf values with the column median for the selected numeric columns.
    If columns is None, uses all numeric columns that contain any INF.

    Returns: (new_lazyframe, medians_dict)
    """
    total_rows, report_df = analyze_inf_columns(lf)
    if report_df.empty:
        return lf, {}

    if columns is None:
        columns = report_df.loc[report_df["InfCount"] > 0, "Feature"].tolist()

    if not columns:
        return lf, {}

    # Compute medians eagerly ignoring +/-inf
    median_exprs = [
        pl.when(pl.col(c).cast(pl.Float64, strict=False).is_infinite())
        .then(None)
        .otherwise(pl.col(c).cast(pl.Float64, strict=False))
        .alias(c)
        for c in columns
    ]
    # Collect one-row DataFrame with numeric columns sanitized (inf -> null), then compute medians
    sanitized = lf.select(median_exprs)
    medians_row = sanitized.select([pl.col(c).median().alias(c) for c in columns]).collect()
    medians: Dict[str, float] = {c: medians_row[c][0] for c in columns}

    # Build with_columns replacements lazily
    replacements = []
    for c in columns:
        m = medians.get(c)
        if m is None or pd.isna(m):
            # If median cannot be computed, skip replacement for that column
            continue
        repl = (
            pl.when(pl.col(c).cast(pl.Float64, strict=False).is_infinite())
            .then(pl.lit(float(m)))
            .otherwise(pl.col(c))
            .alias(c)
        )
        replacements.append(repl)

    if not replacements:
        return lf, {}

    st.success(f"Prepared lazy imputation for {len(replacements)} columns (replace +/-inf with medians).")
    return lf.with_columns(replacements), medians


# ---- Constant / Low-Variance Analysis ----

def unique_counts_report(lf: pl.LazyFrame) -> pd.DataFrame:
    """Return a pandas DataFrame with columns: Feature, UniqueCount for all columns."""
    st.info("Computing unique value counts per column (eager aggregation)...")
    try:
        unique_df = lf.select(pl.all().n_unique()).collect()
        report = unique_df.transpose(include_header=True, header_name="Feature", column_names=["UniqueCount"]).to_pandas()
        report["UniqueCount"] = report["UniqueCount"].astype(int)
        return report
    except Exception as e:
        st.error(f"Failed to compute unique counts: {e}")
        return pd.DataFrame(columns=["Feature", "UniqueCount"])


def analyze_constant_low_variance(lf: pl.LazyFrame, low_var_threshold: int) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Identify constant columns (n_unique == 1) and low-variance columns (2..threshold).

    Returns: (constant_cols, low_var_cols, unique_counts_df)
    """
    report = unique_counts_report(lf)
    if report.empty:
        return [], [], report
    constant_cols = report.loc[report["UniqueCount"] == 1, "Feature"].tolist()
    low_var_cols = report.loc[(report["UniqueCount"] >= 2) & (report["UniqueCount"] <= int(low_var_threshold)), "Feature"].tolist()
    return constant_cols, low_var_cols, report


def drop_columns_lazy(lf: pl.LazyFrame, columns: List[str]) -> pl.LazyFrame:
    """Drop provided columns lazily."""
    if not columns:
        return lf
    return lf.drop(columns)


# ---- Mixed-Type Analysis ----

def analyze_mixed_types(lf: pl.LazyFrame) -> pd.DataFrame:
    """
    For each column, compute counts of value categories: NaN (null or NaN), inf, integer, float, string.
    Returns a pandas DataFrame with rows per Feature and columns: NaN, inf, integer, float, string.
    """
    st.info("Analyzing mixed types per column (eager aggregation)...")
    exprs = []
    for col in lf.columns:
        c = pl.col(col)
        num = c.cast(pl.Float64, strict=False)
        # base masks
        is_null = c.is_null()
        is_nan = num.is_nan()
        is_inf = num.is_infinite()
        is_num_not_null = num.is_not_null()
        # integer if numeric, not inf, not nan, and floor equals value
        is_integer = is_num_not_null & (~is_inf) & (~is_nan) & (num.floor() == num)
        # float if numeric, not integer, not inf, not nan
        is_float = is_num_not_null & (~is_inf) & (~is_nan) & (~(num.floor() == num))
        # string if not null and numeric cast failed
        is_string = (~is_null) & num.is_null()
        # NaN: treat nulls OR numeric NaN as NaN bucket
        is_NaN_bucket = is_null | is_nan

        exprs.extend([
            is_NaN_bucket.sum().alias(f"NaN__{col}"),
            is_inf.sum().alias(f"inf__{col}"),
            is_integer.sum().alias(f"integer__{col}"),
            is_float.sum().alias(f"float__{col}"),
            is_string.sum().alias(f"string__{col}"),
        ])

    out = lf.select(exprs).collect()
    # Convert to long pandas format per feature
    df = out.to_pandas()
    # df columns like 'NaN__col', 'inf__col'...
    rows = []
    for col in lf.columns:
        row = { 'Feature': col }
        for k in ['NaN', 'inf', 'integer', 'float', 'string']:
            key = f"{k}__{col}"
            row[k] = int(df.iloc[0][key]) if key in df.columns else 0
        rows.append(row)
    result = pd.DataFrame(rows)
    return result


def coerce_columns_to_numeric(lf: pl.LazyFrame, columns: List[str], dtype: pl.DataType = pl.Float64) -> pl.LazyFrame:
    """Lazily cast selected columns to a numeric type (default Float64) with strict=False."""
    if not columns:
        return lf
    return lf.with_columns([pl.col(c).cast(dtype, strict=False).alias(c) for c in columns if c in lf.columns])

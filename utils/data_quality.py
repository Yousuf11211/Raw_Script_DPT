# utils/data_quality.py

import polars as pl
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Optional
import ipaddress


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


def coerce_columns_to_datetime(lf: pl.LazyFrame, columns: List[str], fmt: Optional[str] = None) -> pl.LazyFrame:
    """Lazily parse selected columns as Datetime. If fmt provided, use it; else attempt flexible ISO parsing.
    Falls back to leaving column unchanged if parsing fails silently (strict=False)."""
    if not columns:
        return lf
    parsed = []
    for c in columns:
        if c in lf.columns:
            col_utf8 = pl.col(c).cast(pl.Utf8, strict=False)
            if fmt:
                parsed.append(col_utf8.str.strptime(pl.Datetime, format=fmt, strict=False, exact=False).alias(c))
            else:
                # try multiple common layouts by chaining coalesce; if first fails returns nulls
                attempt = (
                    col_utf8.str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S.%f", strict=False, exact=False)
                    .fill_null(col_utf8.str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False, exact=False))
                    .fill_null(col_utf8.str.strptime(pl.Datetime, format="%Y-%m-%d", strict=False, exact=False))
                )
                parsed.append(attempt.alias(c))
    if not parsed:
        return lf
    return lf.with_columns(parsed)


def coerce_ipv4_to_integer(lf: pl.LazyFrame, columns: List[str]) -> pl.LazyFrame:
    """Convert dotted-decimal IPv4 addresses to UInt32 integers lazily for selected columns.
    Non-IPv4 strings become null."""
    if not columns:
        return lf
    exprs = []
    for c in columns:
        if c in lf.columns:
            s = pl.col(c).cast(pl.Utf8, strict=False).str.split_exact(".", 3)
            expr = (
                (s.struct.field("field_0").cast(pl.UInt32) * pl.lit(256**3)) +
                (s.struct.field("field_1").cast(pl.UInt32) * pl.lit(256**2)) +
                (s.struct.field("field_2").cast(pl.UInt32) * pl.lit(256)) +
                (s.struct.field("field_3").cast(pl.UInt32))
            ).alias(c).cast(pl.UInt32, strict=False)
            exprs.append(expr)
    if not exprs:
        return lf
    return lf.with_columns(exprs)

# ---- Encoding Candidates Analysis ----

def analyze_encoding_candidates(lf: pl.LazyFrame, datetime_patterns: Optional[List[str]] = None, sample_size: int = 100) -> pd.DataFrame:
    """Robustly analyze columns and flag encoding vs numeric vs datetime without raising parse errors.
    Datetime detection uses sampling + pattern matching.
    Returns DataFrame with columns: Feature, UniqueCount, NonNullCount, HasString, IsNumeric, IsDatetime,
    NeedsEncoding, CardinalityLabel, SuggestedEncoding."""
    st.info("Analyzing columns for encoding candidates (robust sampling)...")
    if datetime_patterns is None:
        datetime_patterns = [
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]

    # Unique & null counts
    unique_df = lf.select(pl.all().n_unique()).collect()
    unique_pd = unique_df.transpose(include_header=True, header_name="Feature", column_names=["UniqueCount"]).to_pandas()
    null_df = lf.select(pl.all().null_count()).collect()
    null_pd = null_df.transpose(include_header=True, header_name="Feature", column_names=["NullCount"]).to_pandas()

    # Numeric cast failures (treat as string presence proxy)
    cast_fail_exprs = [pl.col(c).cast(pl.Float64, strict=False).is_null().sum().alias(c) for c in lf.columns]
    cast_fail_df = lf.select(cast_fail_exprs).collect()
    cast_fail_pd = cast_fail_df.transpose(include_header=True, header_name="Feature", column_names=["PostCastNullCount"]).to_pandas()

    merged = unique_pd.merge(null_pd, on="Feature").merge(cast_fail_pd, on="Feature")
    total_rows = lf.select(pl.count()).collect().item()

    # Pre-sample columns for datetime detection
    from datetime import datetime
    def is_datetime_like(samples: List[str]) -> bool:
        if not samples:
            return False
        success = 0
        total = 0
        for val in samples:
            total += 1
            parsed = False
            for pat in datetime_patterns:
                try:
                    datetime.strptime(val, pat)
                    parsed = True
                    break
                except Exception:
                    continue
            if parsed:
                success += 1
        return (total > 0) and (success / total >= 0.9)

    # Collect samples per column (limited) - do individually to avoid huge memory
    column_samples: Dict[str, List[str]] = {}
    for col in lf.columns:
        try:
            sample_series = (
                lf.select(pl.col(col).drop_nulls().cast(pl.Utf8, strict=False).head(sample_size))
                .collect()
                .to_pandas()
            )
            if not sample_series.empty:
                column_samples[col] = [s for s in sample_series[col].tolist() if isinstance(s, str)]
            else:
                column_samples[col] = []
        except Exception:
            column_samples[col] = []

    # IP detection helper
    def is_ip_like(samples: List[str]) -> bool:
        if not samples:
            return False
        success = 0
        total = 0
        for val in samples:
            total += 1
            try:
                ipaddress.ip_address(val)
                success += 1
            except Exception:
                continue
        return (total > 0) and (success / total >= 0.9)

    records = []
    for _, row in merged.iterrows():
        feature = row['Feature']
        unique_count = int(row['UniqueCount'])
        null_count = int(row['NullCount'])
        post_cast_null = int(row['PostCastNullCount'])
        non_null = total_rows - null_count
        cast_failures = max(post_cast_null - null_count, 0)
        has_string = cast_failures > 0
        samples = column_samples.get(feature, [])
        is_datetime = is_datetime_like(samples)
        is_ip = is_ip_like(samples)
        is_numeric = (not has_string) and (not is_datetime) and (not is_ip)
        needs_encoding = (not is_numeric) and (not is_datetime) and (not is_ip) and unique_count > 1
        # Cardinality classification & suggestion
        if unique_count <= 1:
            card_label = 'constant'
            suggested = 'drop or ignore'
        elif unique_count == 2:
            card_label = 'binary'
            suggested = 'binary / one-hot'
        elif unique_count <= 20:
            card_label = 'low'
            suggested = 'one-hot'
        elif unique_count <= 100:
            card_label = 'medium'
            suggested = 'label encoding'
        else:
            card_label = 'high'
            suggested = 'embedding / target encoding'
        if is_numeric:
            suggested = 'none'
            needs_encoding = False
        if is_datetime:
            suggested = 'datetime parse'
            needs_encoding = False
        if is_ip:
            suggested = 'ip parse'
            needs_encoding = False
        records.append({
            'Feature': feature,
            'UniqueCount': unique_count,
            'NonNullCount': non_null,
            'HasString': has_string,
            'IsNumeric': is_numeric,
            'IsDatetime': is_datetime,
            'IsIP': is_ip,
            'NeedsEncoding': needs_encoding,
            'CardinalityLabel': card_label,
            'SuggestedEncoding': suggested
        })

    df = pd.DataFrame(records).sort_values(['NeedsEncoding','CardinalityLabel','UniqueCount'], ascending=[False, True, True])
    return df


def sample_string_values(lf: pl.LazyFrame, column: str, max_samples: int = 10) -> list:
    """
    Return up to max_samples unique string values from the given column (excluding nulls and numeric values).
    """
    try:
        # Cast to float, find where cast fails (is null but original is not null)
        col_expr = pl.col(column)
        num_expr = col_expr.cast(pl.Float64, strict=False)
        mask = (~col_expr.is_null()) & num_expr.is_null()
        # Get unique string values
        df = lf.filter(mask).select(col_expr).unique().limit(max_samples).collect().to_pandas()
        return df[column].dropna().astype(str).tolist()
    except Exception as e:
        st.error(f"Failed to sample string values for {column}: {e}")
        return []


def map_requested_columns(existing_columns: List[str], requested: List[str], normalize: bool = True) -> Tuple[List[str], List[str]]:
    """
    Map requested column names to actual existing column names using case-insensitive matching.
    Returns (found_actual_names, not_found_requested_names).
    When normalize=True, trims whitespace and lowercases requested names before matching.
    """
    if not requested:
        return [], []
    existing_map = {c.lower(): c for c in existing_columns}
    found: List[str] = []
    not_found: List[str] = []
    for name in requested:
        key = name.strip().lower() if normalize else name
        actual = existing_map.get(key)
        if actual is not None:
            if actual not in found:
                found.append(actual)
        else:
            not_found.append(name)
    return found, not_found

import os
import polars as pl
import pandas as pd
from typing import List, Dict, Tuple, Optional

try:
    import matplotlib.pyplot as plt  # optional
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

NUMERIC_TYPES = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64}


def get_numeric_columns(lf: pl.LazyFrame) -> List[str]:
    return [c for c, t in lf.schema.items() if t in NUMERIC_TYPES]


def analyze_iqr_outliers(lf: pl.LazyFrame, multiplier: float = 1.5, sample_rows: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """Compute IQR lower/upper bounds and outlier counts per numeric column.
    Returns (pandas summary df, bounds map {col: (lower, upper)})."""
    numeric_cols = get_numeric_columns(lf)
    if not numeric_cols:
        return pd.DataFrame(columns=["Feature","Q1","Q3","Lower","Upper","OutlierCount","OutlierPct"]), {}

    # Collect needed columns (optionally sample rows) for quantile efficiency
    to_collect = lf.select([pl.col(c) for c in numeric_cols])
    if sample_rows is not None:
        to_collect = to_collect.limit(sample_rows)
    df_num = to_collect.collect()

    records = []
    bounds: Dict[str, Tuple[float, float]] = {}
    total_rows = df_num.height
    for col in numeric_cols:
        s = df_num[col]
        if s.n_unique() <= 1:
            continue
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        # Build boolean mask and count outliers robustly
        mask = (s < lower) | (s > upper)
        try:
            outliers = int(mask.sum())  # newer Polars supports sum on boolean
        except Exception:
            outliers = int((mask.cast(pl.Int8)).sum())
        pct = (outliers / total_rows * 100.0) if total_rows else 0.0
        bounds[col] = (lower, upper)
        records.append({
            "Feature": col,
            "Q1": q1,
            "Q3": q3,
            "Lower": lower,
            "Upper": upper,
            "OutlierCount": outliers,
            "OutlierPct": pct
        })

    summary_df = pd.DataFrame(records).sort_values("OutlierPct", ascending=False).reset_index(drop=True)
    return summary_df, bounds


def remove_outliers_lazy(lf: pl.LazyFrame, bounds: Dict[str, Tuple[float, float]], mode: str = "remove", columns: Optional[List[str]] = None) -> pl.LazyFrame:
    """Apply outlier filtering lazily.
    mode: 'remove' -> drop outlier rows; 'keep_only' -> keep only outliers; 'cap' -> winsorize values to bounds.
    columns: subset to operate on (defaults to all bounds keys)."""
    if not bounds:
        return lf
    cols = columns or list(bounds.keys())

    if mode == "cap":
        exprs = []
        for c in cols:
            if c not in lf.columns or c not in bounds:
                continue
            lower, upper = bounds[c]
            exprs.append(pl.col(c).clip(lower_bound=lower, upper_bound=upper).alias(c))
        if not exprs:
            return lf
        return lf.with_columns(exprs)

    # Build row filter
    filters = []
    for c in cols:
        if c not in lf.columns or c not in bounds:
            continue
        lower, upper = bounds[c]
        if mode == "remove":
            filters.append((pl.col(c) >= lower) & (pl.col(c) <= upper))
        elif mode == "keep_only":
            filters.append((pl.col(c) < lower) | (pl.col(c) > upper))
    if not filters:
        return lf

    if mode == "remove":
        combined = filters[0]
        for f in filters[1:]:
            combined = combined & f
    else:  # keep_only
        combined = filters[0]
        for f in filters[1:]:
            combined = combined | f
    return lf.filter(combined)


def generate_outlier_plot(df: pd.DataFrame, column: str, lower: float, upper: float, out_path: str) -> bool:
    if not HAS_PLOT:
        return False
    if column not in df.columns:
        return False
    try:
        series = df[column].dropna()
        plt.figure(figsize=(10, 6))
        if series.nunique() > 50:
            series.hist(bins=50, color="steelblue", edgecolor="black")
            plt.title(f"Histogram: {column}")
        else:
            counts = series.value_counts().sort_index()
            plt.bar(counts.index, counts.values, color="steelblue")
            plt.title(f"Value Counts: {column}")
        plt.axvline(lower, color='red', linestyle='--', label='Lower Bound')
        plt.axvline(upper, color='green', linestyle='--', label='Upper Bound')
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.legend(loc='upper right')
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        return True
    except Exception:
        return False


def collect_for_plots(lf: pl.LazyFrame, columns: List[str], max_rows: int = 250_000) -> pd.DataFrame:
    subset_cols = [c for c in columns if c in lf.columns]
    if not subset_cols:
        return pd.DataFrame()
    return lf.select([pl.col(c) for c in subset_cols]).limit(max_rows).collect().to_pandas()

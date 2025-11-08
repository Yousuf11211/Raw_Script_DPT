import polars as pl
import streamlit as st
from typing import Dict, Tuple, Optional, List, Any
import pandas as pd

# Function to count the number of Columns, and duplicate columns(Remove if any) print them optionally save the results
def get_duplicate_columns(file_path: str) -> Tuple[int, Optional[List[str]]]:
    """
    Reads only the header of the file to identify column count and duplicates.
    This is EAGER (collects header), but header size is minimal.

    Returns: (total_columns, [list of duplicate column names])
    """
    try:
        # Read only the first row (header) with Polars
        header_df = pl.read_csv(file_path, n_rows=1).clear()
        columns = header_df.columns
        count = len(columns)

        # Check for duplicates
        seen_cols = set()
        duplicate_cols = []
        for col in columns:
            if col in seen_cols and col not in duplicate_cols:
                duplicate_cols.append(col)
            seen_cols.add(col)

        duplicate_to_report = sorted(duplicate_cols) if duplicate_cols else None

        return count, duplicate_to_report

    except pl.exceptions.ComputeError as e:
        # Handle empty file or severe parsing issues
        st.error(f"Error reading header for duplicate check: {e}")
        return 0, None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return 0, None


def drop_duplicate_columns_lazy(lf: pl.LazyFrame, duplicate_cols: Optional[List[str]]) -> pl.LazyFrame:
    """
    Lazily drops duplicate columns, keeping only the first occurrence.

    Note: Polars automatically handles duplicate names by appending '_duplicated'
    when reading, but if we need a custom "first-occurrence" rule, we must
    replicate the Pandas indexing logic or rely on Polars' naming convention
    and then rename/select columns manually if needed.

    For simplicity and performance, this Polars version assumes we drop all
    columns that appeared AFTER the first one.
    """
    if not duplicate_cols:
        return lf

    # Eagerly collect all column names from the LazyFrame (still just metadata)
    all_cols = lf.columns

    # Identify indices to keep (based on first occurrence)
    keep_indices = []
    cleaned_cols = []

    # We rebuild the column list, keeping only the first instance of each name
    for col in all_cols:
        if col not in cleaned_cols:
            cleaned_cols.append(col)

    st.info(f"Applying lazy drop. Keeping {len(cleaned_cols)} unique columns.")

    # The Polars way: Select only the unique column names in order
    return lf.select(cleaned_cols)



# Function to count the number of Rows and Duplicate Rows, then Delete Duplicate Rows save new CSV
def get_row_and_duplicate_counts(lf: pl.LazyFrame) -> Tuple[int, int]:
    """
    Executes an aggregation to count total rows and duplicate rows.

    CRITICAL: This forces an EAGER operation (.collect()) on the whole file,
    but only for a single aggregation, which is faster than loading the whole DataFrame.

    Returns: (total_rows, duplicate_rows_count)
    """
    st.info("Calculating total rows and duplicates (may take a moment for huge files)...")
    try:
        # Calculate total rows
        total_rows = lf.select(pl.count()).collect().item()

        # Calculate unique rows, then find the difference
        # We must collect the result of the aggregation
        unique_rows_count = lf.unique().select(pl.count()).collect().item()

        duplicate_rows_count = total_rows - unique_rows_count

        return total_rows, duplicate_rows_count

    except Exception as e:
        st.error(f"Error during duplicate row count aggregation: {e}")
        return 0, 0


def drop_duplicate_rows_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Lazily adds the instruction to drop duplicate rows (keeping the first occurrence).
    """
    # Polars' unique() method defaults to keeping the first occurrence.
    # The actual cleaning happens later during .collect()
    return lf.unique()


# Note: This function is EAGER (uses .collect()) because you need the final counts/percentages
# to display in the UI. We run a minimal aggregation plan to keep it memory efficient.

def get_missing_and_infinite_report(lf: pl.LazyFrame) -> Tuple[int, pd.DataFrame]:
    """
    Calculates the count and percentage of missing (null) and infinite (inf)
    values for all columns in the LazyFrame.

    Returns: (total_rows_count, report_df_pandas)
    """
    st.info("Calculating missing/inf values across all columns...")

    # 1. Get total rows (Eager count is efficient)
    total_rows = lf.select(pl.count()).collect().item()
    if total_rows == 0:
        return 0, pd.DataFrame()

    # 2. Build the Aggregation Plan

    # a) Missing (Null) Counts: Polars has a direct method: lf.null_count()
    null_counts_lf = lf.null_count()

    # b) Infinite Counts: Polars does not have a direct inf_count. We must build an expression.
    # We only check for inf on floating-point columns (pl.Float32, pl.Float64)
    inf_expressions = [
        pl.col(col).is_infinite().sum().alias(f"inf_{col}")
        for col, dtype in lf.schema.items() if dtype in (pl.Float32, pl.Float64)
    ]

    # Combine null counts and inf counts into a single aggregation
    if inf_expressions:
        # If there are float columns, collect both nulls and infs
        full_report_df = null_counts_lf.select(inf_expressions).collect()
    else:
        # If no float columns, just collect nulls
        full_report_df = null_counts_lf.collect()

    # 3. Format the Report (Convert to Pandas for easier reporting/UI display)

    # The result is a single-row Polars DataFrame. Convert and transpose it.
    report_df = full_report_df.transpose(include_header=True, header_name="Feature", column_names=["Value"]).to_pandas()
    report_df.set_index("Feature", inplace=True)
    report_df.rename(columns={"Value": "Count"}, inplace=True)

    # 4. Process the Report

    report_data = []

    for feature_name, row_data in report_df.iterrows():
        count = row_data['Count']

        # Check if this is a NULL count or an INF count
        if feature_name.endswith('_null_count'):
            original_col = feature_name[:-11]
            report_type = 'Missing (NULL)'
        elif feature_name.startswith('inf_'):
            original_col = feature_name[4:]
            report_type = 'Infinite (INF)'
        else:
            continue  # Skip non-count columns

        if count > 0:
            percentage = (count / total_rows) * 100 if total_rows > 0 else 0
            report_data.append({
                'Feature': original_col,
                'Type': report_type,
                'Count': count,
                'Percentage': round(percentage, 4)
            })

    # Final Report DataFrame (Pandas)
    final_report_df = pd.DataFrame(report_data)

    return total_rows, final_report_df


# --- CONSTANTS FROM ORIGINAL SCRIPT ---
NEVER_NEGATIVE_KEYWORDS = [
    'port', 'duration', 'count', 'bytes', 'size', 'rate', 'percentage',
    'variance', 'std', 'total', 'max', 'min', 'median', 'mode', 'mean',
    'iat', 'active', 'idle', 'bulk', 'handshake', 'subflow'
]
CAN_BE_NEGATIVE_KEYWORDS = ['skew', 'cov', 'delta']
PORT_COLUMNS = ['src_port', 'dst_port']


# --- HELPER FUNCTION TO IDENTIFY TARGET COLUMNS ---
def _identify_target_columns(lf: pl.LazyFrame) -> Dict[str, list]:
    """Identifies columns that must be non-negative based on keywords."""
    target_cols = {
        'non_negative': [],
        'port_range': []
    }

    for col in lf.columns:
        col_lower = col.lower()

        # Skip columns that can be negative
        if any(kw in col_lower for kw in CAN_BE_NEGATIVE_KEYWORDS):
            continue

        # Identify non-negative columns
        if any(kw in col_lower for kw in NEVER_NEGATIVE_KEYWORDS):
            target_cols['non_negative'].append(col)

        # Identify port columns
        if col_lower in PORT_COLUMNS:
            target_cols['port_range'].append(col)

    return target_cols


# --- CORE VALIDATION & FILTERING FUNCTION ---
def get_validation_report_and_filter_plan(lf: pl.LazyFrame) -> Tuple[pl.LazyFrame, Dict[str, Any]]:
    """
    Identifies rows with invalid data, generates a report (EAGER), and returns
    a new LazyFrame with the filter applied (LAZY).

    Returns: (filtered_lazyframe, validation_report_dict)
    """
    st.info("Building validation filter plan...")

    # 1. Prepare for EAGER statistics collection
    # Use updated Polars method (with_row_index) due to deprecation of with_row_count
    lf_indexed = lf.with_row_index(name="row_index")

    # Check for 'label' column presence
    label_col = next((col for col in lf.columns if col.lower() == 'label'), None)

    # 2. Build the combined filter expression (The MASK for INVALID rows)

    invalid_mask = pl.lit(False)
    target_cols = _identify_target_columns(lf)

    # a. Non-Negative Check (e.g., duration < 0)
    for col in target_cols['non_negative']:
        # Ensure conversion to numeric, handling potential non-numeric data
        expr = pl.col(col).cast(pl.Float64, strict=False) < 0
        invalid_mask = invalid_mask | expr
        st.caption(f"Added filter: {col} < 0")

    # b. Port Range Check (e.g., port > 65535 or port < 0)
    for col in target_cols['port_range']:
        # Port must be between 0 and 65535, inclusive
        expr = ~pl.col(col).cast(pl.UInt16, strict=False).is_between(0, 65535)
        invalid_mask = invalid_mask | expr
        st.caption(f"Added filter: {col} not in [0, 65535]")

    # 3. Collect Statistics on INVALID Rows (EAGER Operation)
    # We collect just the index and label for the report, NOT the whole DataFrame
    if label_col:
        cols_to_collect = [pl.col("row_index"), pl.col(label_col).alias("label")]
    else:
        cols_to_collect = [pl.col("row_index")]

    invalid_rows_df = (
        lf_indexed
        .filter(invalid_mask)
        .select(cols_to_collect)
        .collect()  # <-- EAGER: Executes the plan to find invalid rows
    )

    # 4. Generate Report
    report = {
        'invalid_count': invalid_rows_df.height,
        'invalid_indices': invalid_rows_df["row_index"].to_list() if invalid_rows_df.height > 0 else [],
        'label_breakdown': invalid_rows_df["label"].value_counts().to_pandas().set_index('label')['count'].to_dict()
        if label_col and invalid_rows_df.height > 0 else {}
    }

    # 5. Apply the filter to the original LazyFrame (LAZY Operation)
    # We apply the inverse mask (~invalid_mask) to keep only VALID rows.
    filtered_lf = lf.filter(~invalid_mask)

    st.success(f"Filter plan created. Identified {report['invalid_count']:,} rows for removal.")
    return filtered_lf, report
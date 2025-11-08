import polars as pl
import streamlit as st
from typing import Dict, Tuple, Optional, List
import pandas as pd
from functools import reduce

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
    # We only check for inf on floating-point columns (pl.Float64)
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



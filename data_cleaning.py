import polars as pl
import streamlit as st
from typing import Dict, Tuple, Optional, List

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


# TODO add missing value handling



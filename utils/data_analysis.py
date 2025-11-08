# utils/data_analysis.py

import polars as pl
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional


# Note: This function is EAGER because calculating the full distribution requires scanning ALL rows.
# Polars makes this EAGER aggregation highly optimized.

def get_class_distribution_report(lf: pl.LazyFrame) -> Tuple[pd.DataFrame, Any]:
    """
    Calculates the class distribution for the 'Label' column using Polars.

    Returns: (summary_df_pandas, matplotlib_figure)
    """
    st.info("Calculating full class distribution (This runs an aggregation across the entire dataset)...")

    # 1. Identify the 'Label' column (case-insensitive check on the schema)
    label_col = None
    for col in lf.columns:
        if col.lower() == "label":
            label_col = col
            break

    if label_col is None:
        st.warning("No 'Label' column found for class distribution analysis.")
        return pd.DataFrame({'Label': ['N/A'], 'Count': [0]}), None

    try:
        # 2. Polars Aggregation: Group by the label column and count occurrences.
        # This is the most efficient way to get counts for a huge file.
        summary_df_polars = (
            lf.group_by(label_col)
            .agg(pl.count().alias("Count"))
            .sort("Count", descending=True)
            .collect()  # EAGER step: execute the aggregation plan
        )

        # 3. Convert to Pandas for Visualization
        summary_df = summary_df_polars.to_pandas()

        # 4. Generate Visualization (Matplotlib Figure)
        fig, ax = plt.subplots(figsize=(10, 6))
        summary_df.set_index(label_col)["Count"].plot(kind="bar", ax=ax)
        ax.set_title("Class Distribution (Benign vs. Attack Types)")
        ax.set_ylabel("Number of Samples")
        ax.set_xlabel("Label")
        plt.xticks(rotation=45)
        plt.tight_layout()

        return summary_df, fig

    except Exception as e:
        st.error(f"Error during class distribution calculation: {e}")
        return pd.DataFrame(), None

# Define the DOMINANCE RANGES required for your report
DOMINANCE_RANGES = [
        (0.95, 1.01, "95-100%"), (0.90, 0.95, "90-95%"),
        (0.80, 0.90, "80-90%"), (0.70, 0.80, "70-80%"),
        (0.60, 0.70, "60-70%"), (0.50, 0.60, "50-60%"),
    ]

def get_dominance_report(lf: pl.LazyFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyzes the LazyFrame for value dominance in all columns and calculates
        the total count for each unique value. This function is EAGER.

        Returns: (dominance_summary_df, label_distribution_df)
        """
        st.info("Calculating dominance and label distribution (heavy aggregation)...")

        # 1. Get total rows (used for percentage calculation)
        total_rows = lf.select(pl.count()).collect().item()
        if total_rows == 0:
            return pd.DataFrame(), pd.DataFrame()

        # --- A. Dominance Calculation ---

        dominance_data = []

        # Iterate through all columns to calculate value counts and dominance ratio
        for col in lf.columns:
            # Polars EAGER aggregation: Group by the column and count, then collect.
            value_counts_df = (
                lf.group_by(col)
                .agg(pl.count().alias("Count"))
                .sort("Count", descending=True)
                .collect()
            )

            if value_counts_df.height == 0:
                continue

            # Calculate dominance ratio (count of the most common value / total rows)
            most_common_count = value_counts_df["Count"][0]
            ratio = most_common_count / total_rows

            # Find the correct dominance bucket
            dominance_label = "Below 50%"
            for low, high, label in DOMINANCE_RANGES:
                if low <= ratio < high:
                    dominance_label = label
                    break

            dominance_data.append({
                'Feature': col,
                'Most Common Value': value_counts_df[col][0],
                'Count': most_common_count,
                'Ratio': round(ratio * 100, 2),
                'Dominance Range': dominance_label,
                'Value_Counts_DF': value_counts_df.to_pandas()  # Store counts for detailed view
            })

        # Create the final dominance report DataFrame
        dominance_summary_df = pd.DataFrame(dominance_data)

        # --- B. Global Label Distribution (Task 1 requirement) ---

        label_col = None
        for col in lf.columns:
            if col.lower() == "label":
                label_col = col
                break

        if label_col:
            # Polars EAGER aggregation for labels
            label_df_polars = (
                lf.group_by(label_col)
                .agg(pl.count().alias("Count"))
                .sort("Count", descending=True)
                .collect()
            )
            label_df = label_df_polars.to_pandas()
            total_labels = label_df['Count'].sum()
            label_df['Percentage'] = (label_df['Count'] / total_labels) * 100
        else:
            label_df = pd.DataFrame({'Label': ['N/A'], 'Count': [0], 'Percentage': [0]})

        st.success("Dominance and label analysis complete.")
        return dominance_summary_df, label_df


def get_value_label_breakdown(lf: pl.LazyFrame, feature: str, top_n: int = 10) -> pd.DataFrame:
    """
    For a given feature (column), compute the count of each value and how those counts
    distribute across labels. Returns a pandas DataFrame sorted by total count desc.

    Columns: Value, Total, <label1>, <label2>, ...
    """
    # Identify label column
    label_col: Optional[str] = None
    for c in lf.columns:
        if c.lower() == 'label':
            label_col = c
            break
    if label_col is None:
        st.warning("No 'Label' column found to compute breakdown.")
        # Just return value counts
        vc = (
            lf.group_by(feature).agg(pl.count().alias('Total')).sort('Total', descending=True).limit(top_n).collect()
        )
        return vc.to_pandas().rename(columns={feature: 'Value'})

    # Compute counts grouped by (feature, label)
    grouped = (
        lf.group_by([feature, label_col])
        .agg(pl.count().alias('Count'))
        .collect()
    )

    # Pivot to wide format: rows are feature values, columns are labels
    pivoted = grouped.pivot(index=feature, columns=label_col, values='Count', aggregate_fn='first').fill_null(0)

    # Add total column
    pivoted = pivoted.with_columns(pl.sum_horizontal(pl.all().exclude(feature)).alias('Total'))
    # Sort by total desc and limit
    pivoted = pivoted.sort('Total', descending=True).limit(top_n)

    return pivoted.to_pandas().rename(columns={feature: 'Value'})

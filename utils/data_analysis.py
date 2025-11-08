# utils/data_analysis.py

import polars as pl
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any


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
# utils/io_utils.py
import os
import streamlit as st
import polars as pl
from datetime import datetime
from typing import Optional


def default_output_path(base_input_path: Optional[str], suffix: str = "cleaned") -> str:
    """Build a default output CSV path from an input CSV path and a suffix."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_input_path and base_input_path.lower().endswith('.csv'):
        folder = os.path.dirname(base_input_path)
        name = os.path.splitext(os.path.basename(base_input_path))[0]
        return os.path.join(folder, f"{name}_{suffix}_{ts}.csv")
    # fallback to CWD
    return os.path.join(os.getcwd(), f"dataset_{suffix}_{ts}.csv")


def write_lazyframe_to_csv(lf: pl.LazyFrame, path: str) -> bool:
    """
    Attempt to write a LazyFrame to CSV efficiently. Tries sink_csv when available,
    otherwise falls back to collect(streaming=True) and DataFrame.write_csv.

    Returns True on success.
    """
    try:
        # Prefer lazy sink if available (low memory)
        if hasattr(lf, 'sink_csv'):
            lf.sink_csv(path)
            return True
        # Fallback: collect and write (may be memory intensive)
        df = lf.collect(streaming=True)
        df.write_csv(path)
        return True
    except Exception as e:
        st.error(f"Failed to write CSV to {path}: {e}")
        return False


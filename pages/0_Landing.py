import streamlit as st
from utils.ui_helpers import initialize_state, inject_global_styles, render_top_nav

st.set_page_config(page_title="Thesis Data Tool", layout="wide", initial_sidebar_state="collapsed")
initialize_state()
inject_global_styles()
render_top_nav(current_page="0_Landing")

st.title("Thesis Data Processing & Modeling Workbench")
st.caption("Developer: Syed Yousuf Uddin")

st.markdown(
    """
This application streamlines end-to-end network / tabular security dataset workflows.

Use the top navigation to pick a main area (Data Cleaning, Data Processing, Data Analysis, Model Training, Model Testing, Utilities). The submenu below the top bar shows tools available in the selected area.

Key capabilities:
- Data Cleaning: Validation, INF handling, dominance reports, low-variance pruning, mixed-type repair, encoding prep, column deletion.
- Data Processing: Class balancing, dataset downscaling, separate & save sets, merge & shuffle (Polars), outlier detection.
- Data Analysis: Feature importance & per-label analysis.
- Model Training: Hyperparameter tuning, isolation forest training, attack model train & test.
- Model Testing: Isolation Forest testing and SHAP explanations.
- Utilities: Raw vs processed comparison, frontend test batch generator.
"""
)

st.markdown("---")

cf = st.session_state.get('current_file_path')
lf = st.session_state.get('current_lazy_frame')
if cf:
    st.write(f"Active file: `{cf}`")
if lf is not None:
    try:
        import polars as pl
        rows = lf.select(pl.count()).collect().item()
        st.write(f"Loaded lazyframe rows: {rows:,}")
    except Exception:
        st.write("Lazyframe loaded (row count unavailable).")
else:
    st.info("No dataset loaded yet. Load one from any tool header when needed.")

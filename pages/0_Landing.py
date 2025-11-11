import streamlit as st
from utils.ui_helpers import initialize_state, inject_global_styles, render_top_nav

st.set_page_config(page_title="Thesis Data Tool", layout="wide", initial_sidebar_state="collapsed")
initialize_state()
inject_global_styles()
render_top_nav(current_page="pages/0_Landing.py")

st.title("Thesis Data Processing & Modeling Workbench")
st.caption("Developer: Syed Yousuf Uddin")

st.markdown("""
This application streamlines end-to-end network / tabular security dataset workflows:

**Data Cleaning**: Validation, INF handling, dominance reports, low-variance pruning, mixed-type repair, encoding prep, column deletion.
**Data Processing**: Class balancing, dataset downscaling, separating benign/attack sets, large-scale merge & shuffle with Polars, outlier detection.
**Data Analysis**: Feature importance & per-label importance exploration.
**Model Training**: Hyperparameter tuning, isolation forest benign model training, attack classifier training & evaluation.
**Model Testing**: Isolation Forest evaluation and SHAP explainability for tree models.
**Utilities**: Raw vs processed folder comparison and frontend test batch generation.

Use the category buttons in the top navigation to reach a hub for each area, then pick a specific tool.
""")

st.markdown("### Quick Hub Access")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Data Cleaning Hub", use_container_width=True):
        st.switch_page("pages/DataCleaning/hub.py")
    if st.button("Data Processing Hub", use_container_width=True):
        st.switch_page("pages/DataProcessing/hub.py")
with col2:
    if st.button("Data Analysis Hub", use_container_width=True):
        st.switch_page("pages/DataAnalysis/hub.py")
    if st.button("Model Training Hub", use_container_width=True):
        st.switch_page("pages/ModelTraining/hub.py")
with col3:
    if st.button("Model Testing Hub", use_container_width=True):
        st.switch_page("pages/ModelTesting/hub.py")
    if st.button("Utilities Hub", use_container_width=True):
        st.switch_page("pages/Utilities/hub.py")

st.markdown("---")
st.markdown("### Current Session State Summary")
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
